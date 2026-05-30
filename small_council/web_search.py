from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from typing import Any, Protocol

from .config import resolve_project_path


DEFAULT_SEARCH_ENGINES = ["bing", "wikipedia", "wikidata", "github", "stackoverflow"]


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    error: str | None = None
    disabled: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.disabled


class SearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        max_results: int,
        engines: list[str] | None = None,
    ) -> list[SearchResult]:
        ...


class SearchError(RuntimeError):
    pass


class SearchTimeout(SearchError):
    pass


class SearchParseError(SearchError):
    pass


class SearchRateLimitError(SearchError):
    pass


class SearchServerError(SearchError):
    pass


def web_search_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("search")
    legacy = config.get("webSearch")
    if not isinstance(raw, dict):
        raw = legacy if isinstance(legacy, dict) else {}

    engines = raw.get("defaultEngines", DEFAULT_SEARCH_ENGINES)
    if not isinstance(engines, list):
        engines = DEFAULT_SEARCH_ENGINES
    default_engines = [str(engine).strip() for engine in engines if str(engine).strip()]

    return {
        "enabled": bool(raw.get("enabled", True)),
        "provider": str(raw.get("provider", "searxng")),
        "baseUrl": str(raw.get("baseUrl", "http://localhost:8080")),
        "timeoutSeconds": float(raw.get("timeoutSeconds", 15)),
        "maxResults": int(raw.get("maxResults", 8)),
        "cacheTtlSeconds": float(raw.get("cacheTtlSeconds", 900)),
        "minDelaySeconds": float(raw.get("minDelaySeconds", 3)),
        "maxConcurrentRequests": int(raw.get("maxConcurrentRequests", 1)),
        "defaultEngines": default_engines or DEFAULT_SEARCH_ENGINES[:],
    }


def create_search_provider(config: dict[str, Any]) -> SearchProvider | None:
    search_config = web_search_config(config)
    if search_config["provider"] == "searxng":
        return SearxngSearchProvider(search_config)
    return None


def create_search_worker(config: dict[str, Any]) -> SearchWorker | None:
    search_config = web_search_config(config)
    provider = create_search_provider(config)
    if provider is None:
        return None
    return SearchWorker(search_config, provider)


def search_enabled(config: dict[str, Any]) -> bool:
    search_config = web_search_config(config)
    return bool(search_config.get("enabled")) and create_search_provider(config) is not None


class SearxngSearchProvider:
    name = "searxng"

    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config.get("baseUrl", "http://localhost:8080")).rstrip("/")
        self.timeout_seconds = float(config.get("timeoutSeconds", 15))

    def search(
        self,
        query: str,
        max_results: int,
        engines: list[str] | None = None,
    ) -> list[SearchResult]:
        params: dict[str, str] = {
            "q": query,
            "format": "json",
        }
        selected_engines = _normalize_engines(engines)
        if selected_engines:
            params["engines"] = ",".join(selected_engines)
        encoded = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{self.base_url}/search?{encoded}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise SearchTimeout(f"SearXNG search timed out for query: {query}") from exc
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise SearchRateLimitError(f"SearXNG rate-limited query: {query}") from exc
            if 500 <= exc.code <= 599:
                raise SearchServerError(
                    f"SearXNG returned HTTP {exc.code} for query: {query}"
                ) from exc
            raise SearchError(f"SearXNG returned HTTP {exc.code} for query: {query}") from exc
        except OSError as exc:
            raise SearchError(f"SearXNG search failed for query {query!r}: {exc}") from exc
        try:
            payload = json.loads(raw)
        except JSONDecodeError as exc:
            raise SearchParseError("SearXNG returned malformed JSON.") from exc
        return parse_searxng_results(payload, max_results)


class SearchWorker:
    def __init__(
        self,
        config: dict[str, Any],
        provider: SearchProvider,
        *,
        monotonic=time.monotonic,
        sleep=asyncio.sleep,
    ) -> None:
        self.config = config
        self.provider = provider
        self.enabled = bool(config.get("enabled", True))
        self.default_engines = _normalize_engines(config.get("defaultEngines"))
        self.default_max_results = max(0, int(config.get("maxResults", 8)))
        self.cache_ttl_seconds = max(0.0, float(config.get("cacheTtlSeconds", 900)))
        self.min_delay_seconds = max(0.0, float(config.get("minDelaySeconds", 3)))
        max_concurrent = max(1, int(config.get("maxConcurrentRequests", 1)))
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[tuple[str, tuple[str, ...], int], tuple[float, list[SearchResult]]] = {}
        self._in_flight: dict[tuple[str, tuple[str, ...], int], asyncio.Task[list[SearchResult]]] = {}
        self._lock = asyncio.Lock()
        self._delay_lock = asyncio.Lock()
        self._last_outbound_at: float | None = None
        self._monotonic = monotonic
        self._sleep = sleep

    async def search(
        self,
        query: str,
        engines: list[str] | None = None,
        max_results: int | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> SearchResponse:
        query_text = str(query).strip()
        if not self.enabled:
            _record(events, status="disabled", query=query_text)
            return SearchResponse(query=query_text, results=[], error="search unavailable: disabled", disabled=True)
        if not query_text:
            _record(events, status="empty_query", query=query_text)
            return SearchResponse(query=query_text, results=[])

        selected_engines = _normalize_engines(engines) or self.default_engines
        selected_max = self.default_max_results if max_results is None else max(0, int(max_results))
        key = _search_key(query_text, selected_engines, selected_max)

        cached = await self._cached(key, events, query_text)
        if cached is not None:
            return SearchResponse(query=query_text, results=cached)

        async with self._lock:
            task = self._in_flight.get(key)
            if task is not None:
                _record(events, status="in_flight_join", query=query_text, engines=selected_engines, maxResults=selected_max)
            else:
                task = asyncio.create_task(
                    self._run_outbound(query_text, selected_engines, selected_max, events)
                )
                self._in_flight[key] = task
        try:
            results = await task
            return SearchResponse(query=query_text, results=results)
        except Exception as exc:
            _record(events, status="failed", query=query_text, error=str(exc), errorType=type(exc).__name__)
            return SearchResponse(query=query_text, results=[], error=f"search unavailable: {exc}")
        finally:
            async with self._lock:
                if self._in_flight.get(key) is task:
                    self._in_flight.pop(key, None)

    async def _cached(
        self,
        key: tuple[str, tuple[str, ...], int],
        events: list[dict[str, Any]] | None,
        query: str,
    ) -> list[SearchResult] | None:
        if self.cache_ttl_seconds <= 0:
            _record(events, status="cache_disabled", query=query)
            return None
        now = self._monotonic()
        async with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                _record(events, status="cache_miss", query=query)
                return None
            expires_at, results = cached
            if expires_at <= now:
                self._cache.pop(key, None)
                _record(events, status="cache_miss", query=query, reason="expired")
                return None
            _record(events, status="cache_hit", query=query, results=len(results))
            return list(results)

    async def _run_outbound(
        self,
        query: str,
        engines: list[str],
        max_results: int,
        events: list[dict[str, Any]] | None,
    ) -> list[SearchResult]:
        _record(events, status="concurrency_waiting", query=query)
        async with self._semaphore:
            _record(events, status="concurrency_acquired", query=query)
            try:
                await self._wait_for_min_delay(events, query)
                _record(
                    events,
                    status="outgoing",
                    provider=self.provider.name,
                    query=query,
                    engines=engines,
                    maxResults=max_results,
                )
                results = await asyncio.to_thread(
                    self.provider.search, query, max_results, engines
                )
                _record(events, status="ok", query=query, results=len(results))
                await self._store_cache(query, engines, max_results, results)
                return results
            except SearchTimeout:
                _record(events, status="timeout", query=query)
                raise
            except SearchParseError:
                _record(events, status="malformed_json", query=query)
                raise
            except SearchRateLimitError:
                _record(events, status="http_429", query=query)
                raise
            except SearchServerError as exc:
                _record(events, status="http_5xx", query=query, error=str(exc))
                raise
            except SearchError as exc:
                _record(events, status="network_failure", query=query, error=str(exc))
                raise
            finally:
                _record(events, status="concurrency_released", query=query)

    async def _wait_for_min_delay(
        self, events: list[dict[str, Any]] | None, query: str
    ) -> None:
        async with self._delay_lock:
            now = self._monotonic()
            if self._last_outbound_at is None:
                self._last_outbound_at = now
                wait_seconds = 0.0
            else:
                next_allowed_at = self._last_outbound_at + self.min_delay_seconds
                wait_seconds = max(0.0, next_allowed_at - now)
                self._last_outbound_at = now + wait_seconds
        if wait_seconds:
            _record(events, status="min_delay_waiting", query=query, seconds=wait_seconds)
            await self._sleep(wait_seconds)

    async def _store_cache(
        self,
        query: str,
        engines: list[str],
        max_results: int,
        results: list[SearchResult],
    ) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        key = _search_key(query, engines, max_results)
        expires_at = self._monotonic() + self.cache_ttl_seconds
        async with self._lock:
            self._cache[key] = (expires_at, list(results))


def parse_searxng_results(payload: dict[str, Any], max_results: int) -> list[SearchResult]:
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return []
    parsed: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or item.get("href") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        source = _source_for_result(item)
        score = _score_for_result(item)
        if not title and not url and not snippet:
            continue
        parsed.append(
            SearchResult(title=title, url=url, snippet=snippet, source=source, score=score)
        )
        if len(parsed) >= max(0, max_results):
            break
    return parsed


def write_search_log(config: dict[str, Any], phase: str, member_name: str, events: list[dict[str, Any]]) -> None:
    logs_dir = resolve_project_path(config["runtime"]["logs_path"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"search-{phase}-{member_name.lower()}.log"
    log_path.write_text(json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def search_context_block(searches: list[dict[str, Any]], snippet_length: int | None = None) -> str:
    compact: list[dict[str, Any]] = []
    max_snippet = 500 if snippet_length is None else max(0, int(snippet_length))
    for search in searches:
        item: dict[str, Any] = {"query": search.get("query", "")}
        if search.get("error"):
            item["message"] = str(search["error"])
            compact.append(item)
            continue
        results = []
        for result in search.get("results", []):
            if isinstance(result, SearchResult):
                result = result.to_dict()
            if not isinstance(result, dict):
                continue
            snippet = str(result.get("snippet") or "")
            if max_snippet and len(snippet) > max_snippet:
                snippet = snippet[: max_snippet - 1].rstrip() + "..."
            results.append(
                {
                    "title": str(result.get("title") or ""),
                    "url": str(result.get("url") or ""),
                    "source": str(result.get("source") or ""),
                    "snippet": snippet,
                }
            )
        item["results"] = results
        if not results:
            item["message"] = "search returned no results"
        compact.append(item)
    return "\n\nWeb search results:\n" + json.dumps(compact, indent=2, sort_keys=True)


def _search_key(query: str, engines: list[str], max_results: int) -> tuple[str, tuple[str, ...], int]:
    normalized_query = " ".join(query.lower().split())
    return (normalized_query, tuple(_normalize_engines(engines)), max(0, int(max_results)))


def _normalize_engines(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        return []
    engines: list[str] = []
    seen: set[str] = set()
    for item in raw:
        engine = str(item).strip()
        key = engine.lower()
        if not engine or key in seen:
            continue
        seen.add(key)
        engines.append(engine)
    return engines


def _source_for_result(item: dict[str, Any]) -> str:
    engines = item.get("engines")
    if isinstance(engines, list) and engines:
        return ", ".join(str(engine) for engine in engines if str(engine).strip())
    for key in ("engine", "source", "category"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _score_for_result(item: dict[str, Any]) -> float | None:
    value = item.get("score")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _record(events: list[dict[str, Any]] | None, **payload: Any) -> None:
    if events is not None:
        events.append(payload)
