from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from typing import Any, Protocol

from .config import resolve_project_path


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    score: float | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    provider: str = ""
    error: str | None = None
    disabled: bool = False
    fallback_provider: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.disabled


@dataclass(frozen=True)
class SearchFetchResponse:
    url: str
    title: str
    content: str
    links: list[str]
    provider: str
    raw: dict[str, Any] | None = None


class SearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        max_results: int,
        engines: list[str] | None = None,
    ) -> SearchResponse:
        ...


class SearchError(RuntimeError):
    pass


class UnknownSearchProviderError(SearchError):
    pass


class SearchProviderUnavailableError(SearchError):
    pass


class SearchConfigurationError(SearchError):
    pass


class SearchAuthenticationError(SearchError):
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
    if not isinstance(raw, dict):
        raw = {}
    searxng = raw.get("searxng") if isinstance(raw.get("searxng"), dict) else {}
    ollama = raw.get("ollama") if isinstance(raw.get("ollama"), dict) else {}

    searxng_base_url = searxng.get("baseUrl", "http://localhost:8080")
    ollama_base_url = ollama.get("baseUrl", "https://ollama.com")

    return {
        "enabled": bool(raw.get("enabled", True)),
        "provider": str(raw.get("provider", "searxng")).strip().lower(),
        "allowFallback": bool(raw.get("allowFallback", False)),
        "fallbackProvider": str(raw.get("fallbackProvider", "searxng")).strip().lower(),
        "timeoutSeconds": float(raw.get("timeoutSeconds", 30)),
        "maxResults": int(raw.get("maxResults", 10)),
        "maxQueriesPerMember": int(raw.get("maxQueriesPerMember", 2)),
        "cacheTtlSeconds": float(raw.get("cacheTtlSeconds", 900)),
        "minDelaySeconds": float(raw.get("minDelaySeconds", 3)),
        "maxConcurrentRequests": int(raw.get("maxConcurrentRequests", 3)),
        "searxng": {
            "baseUrl": str(searxng_base_url),
        },
        "ollama": {
            "baseUrl": str(ollama_base_url),
            "apiKey": ollama.get("apiKey"),
            "apiKeyEnv": str(ollama.get("apiKeyEnv", "OLLAMA_API_KEY")),
            "searchEndpoint": str(ollama.get("searchEndpoint", "/api/web_search")),
            "fetchEndpoint": str(ollama.get("fetchEndpoint", "/api/web_fetch")),
        },
    }


def create_search_provider(
    config: dict[str, Any],
    provider_name: str | None = None,
) -> SearchProvider | None:
    search_config = web_search_config(config)
    selected = str(provider_name or search_config["provider"]).strip().lower()
    if selected == "searxng":
        return SearxngSearchProvider(search_config)
    if selected == "ollama":
        return OllamaCloudSearchProvider(search_config)
    raise UnknownSearchProviderError(f"Unknown search provider: {selected}")


def create_search_worker(config: dict[str, Any]) -> SearchWorker | None:
    search_config = web_search_config(config)
    try:
        provider = create_search_provider(config)
    except SearchError:
        return None
    if provider is None:
        return None
    fallback_provider = None
    if search_config.get("allowFallback"):
        fallback_name = str(search_config.get("fallbackProvider", "")).strip().lower()
        if fallback_name and fallback_name != provider.name:
            try:
                fallback_provider = create_search_provider(config, fallback_name)
            except SearchError:
                fallback_provider = None
    return SearchWorker(search_config, provider, fallback_provider=fallback_provider)


def search_enabled(config: dict[str, Any]) -> bool:
    search_config = web_search_config(config)
    try:
        return bool(search_config.get("enabled")) and create_search_provider(config) is not None
    except SearchError:
        return False


class SearxngSearchProvider:
    name = "searxng"

    def __init__(self, config: dict[str, Any]) -> None:
        provider_config = config.get("searxng") if isinstance(config.get("searxng"), dict) else {}
        self.base_url = str(provider_config.get("baseUrl", "http://localhost:8080")).rstrip("/")
        self.timeout_seconds = float(config.get("timeoutSeconds", 30))

    def search(
        self,
        query: str,
        max_results: int,
        engines: list[str] | None = None,
    ) -> SearchResponse:
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
            headers={
                "Accept": "application/json",
                "X-Forwarded-For": "127.0.0.1",
                "X-Real-IP": "127.0.0.1",
            },
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
        return SearchResponse(
            query=query,
            results=parse_searxng_results(payload, max_results),
            provider=self.name,
        )


class OllamaCloudSearchProvider:
    name = "ollama"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        provider_config = config.get("ollama") if isinstance(config.get("ollama"), dict) else {}
        self.base_url = str(provider_config.get("baseUrl", "https://ollama.com")).rstrip("/")
        self.api_key = _configured_api_key(provider_config)
        self.search_endpoint = _endpoint_path(provider_config.get("searchEndpoint", "/api/web_search"))
        self.fetch_endpoint = _endpoint_path(provider_config.get("fetchEndpoint", "/api/web_fetch"))
        self.timeout_seconds = float(config.get("timeoutSeconds", 30))

    def search(
        self,
        query: str,
        max_results: int,
        engines: list[str] | None = None,
    ) -> SearchResponse:
        payload = self._request_json(
            self.search_endpoint,
            {"query": query, "max_results": max(0, min(10, int(max_results)))},
        )
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise SearchParseError("Ollama web search returned malformed results.")
        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            if not title and not url and not snippet:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                    raw=dict(item),
                )
            )
            if len(results) >= max(0, int(max_results)):
                break
        return SearchResponse(query=query, results=results, provider=self.name)

    def fetch(self, url: str, *, timeout: int | None = None) -> SearchFetchResponse:
        payload = self._request_json(
            self.fetch_endpoint,
            {"url": url},
            timeout_seconds=self.timeout_seconds if timeout is None else float(timeout),
        )
        links = payload.get("links", [])
        if not isinstance(links, list):
            links = []
        return SearchFetchResponse(
            url=url,
            title=str(payload.get("title") or ""),
            content=str(payload.get("content") or ""),
            links=[str(link) for link in links],
            provider=self.name,
            raw=payload,
        )

    def _request_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise SearchAuthenticationError(
                "Ollama web search requires an API key. Set search.ollama.apiKey or OLLAMA_API_KEY."
            )
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise SearchTimeout("Ollama web search timed out.") from exc
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise SearchAuthenticationError("Ollama web search authentication failed.") from exc
            if exc.code == 429:
                raise SearchRateLimitError("Ollama web search was rate-limited.") from exc
            if 500 <= exc.code <= 599:
                raise SearchServerError(f"Ollama web search returned HTTP {exc.code}.") from exc
            raise SearchError(f"Ollama web search returned HTTP {exc.code}.") from exc
        except OSError as exc:
            raise SearchProviderUnavailableError(f"Ollama web search failed: {exc}") from exc
        try:
            decoded = json.loads(raw)
        except JSONDecodeError as exc:
            raise SearchParseError("Ollama web search returned malformed JSON.") from exc
        if not isinstance(decoded, dict):
            raise SearchParseError("Ollama web search returned a non-object response.")
        return decoded


class SearchWorker:
    def __init__(
        self,
        config: dict[str, Any],
        provider: SearchProvider,
        *,
        fallback_provider: SearchProvider | None = None,
        monotonic=time.monotonic,
        sleep=asyncio.sleep,
    ) -> None:
        self.config = config
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.enabled = bool(config.get("enabled", True))
        self.default_engines: list[str] = []
        self.default_max_results = max(0, int(config.get("maxResults", 10)))
        self.timeout_seconds = max(0.0, float(config.get("timeoutSeconds", 30)))
        self.cache_ttl_seconds = max(0.0, float(config.get("cacheTtlSeconds", 900)))
        self.min_delay_seconds = max(0.0, float(config.get("minDelaySeconds", 3)))
        max_concurrent = max(1, int(config.get("maxConcurrentRequests", 3)))
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[tuple[str, tuple[str, ...], int], tuple[float, list[SearchResult]]] = {}
        self._in_flight: dict[tuple[str, tuple[str, ...], int], asyncio.Task[SearchResponse]] = {}
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
            return SearchResponse(query=query_text, results=[], provider=self.provider.name, error="search unavailable: disabled", disabled=True)
        if not query_text:
            _record(events, status="empty_query", query=query_text)
            return SearchResponse(query=query_text, results=[], provider=self.provider.name)

        selected_engines = _normalize_engines(engines) or self.default_engines
        selected_max = self.default_max_results if max_results is None else max(0, int(max_results))
        key = _search_key(query_text, selected_engines, selected_max)

        cached = await self._cached(key, events, query_text)
        if cached is not None:
            return SearchResponse(query=query_text, results=cached, provider=self.provider.name)

        async with self._lock:
            task = self._in_flight.get(key)
            if task is not None:
                _record(events, status="in_flight_join", query=query_text, engines=selected_engines, maxResults=selected_max)
            else:
                task = asyncio.create_task(
                    self._run_outbound(query_text, selected_engines, selected_max, events)
                )
                self._in_flight[key] = task
                task.add_done_callback(
                    lambda completed, task_key=key: asyncio.create_task(
                        self._forget_in_flight(task_key, completed)
                    )
                )
        try:
            if self.timeout_seconds > 0:
                return await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout_seconds)
            return await task
        except asyncio.TimeoutError:
            message = f"timed out after {self.timeout_seconds:g}s"
            _record(events, status="timeout", query=query_text, seconds=self.timeout_seconds)
            return SearchResponse(
                query=query_text,
                results=[],
                provider=self.provider.name,
                error=f"search unavailable: {message}",
            )
        except Exception as exc:
            _record(events, status="failed", query=query_text, error=str(exc), errorType=type(exc).__name__)
            return SearchResponse(query=query_text, results=[], provider=self.provider.name, error=f"search unavailable: {exc}")

    async def _forget_in_flight(
        self, key: tuple[str, tuple[str, ...], int], task: asyncio.Task[SearchResponse]
    ) -> None:
        if task.done() and not task.cancelled():
            task.exception()
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
    ) -> SearchResponse:
        _record(events, status="concurrency_waiting", query=query)
        async with self._semaphore:
            _record(events, status="concurrency_acquired", query=query)
            started_at = self._monotonic()
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
                response = await self._call_provider(self.provider, query, max_results, engines)
                duration = self._monotonic() - started_at
                _record(events, status="ok", query=query, provider=response.provider, results=len(response.results), durationSeconds=duration)
                await self._store_cache(query, engines, max_results, response.results)
                return response
            except SearchTimeout as exc:
                _record(events, status="timeout", query=query)
                return await self._fallback_or_raise(query, engines, max_results, events, exc)
            except SearchParseError as exc:
                _record(events, status="malformed_json", query=query)
                return await self._fallback_or_raise(query, engines, max_results, events, exc)
            except SearchRateLimitError as exc:
                _record(events, status="http_429", query=query)
                return await self._fallback_or_raise(query, engines, max_results, events, exc)
            except SearchServerError as exc:
                _record(events, status="http_5xx", query=query, error=str(exc))
                return await self._fallback_or_raise(query, engines, max_results, events, exc)
            except SearchError as exc:
                _record(events, status="network_failure", query=query, error=str(exc))
                return await self._fallback_or_raise(query, engines, max_results, events, exc)
            finally:
                _record(events, status="concurrency_released", query=query)

    async def _call_provider(
        self,
        provider: SearchProvider,
        query: str,
        max_results: int,
        engines: list[str],
    ) -> SearchResponse:
        return await asyncio.to_thread(provider.search, query, max_results, engines)

    async def _fallback_or_raise(
        self,
        query: str,
        engines: list[str],
        max_results: int,
        events: list[dict[str, Any]] | None,
        exc: SearchError,
    ) -> SearchResponse:
        if self.fallback_provider is None:
            raise exc
        _record(
            events,
            status="fallback",
            query=query,
            provider=self.provider.name,
            fallbackProvider=self.fallback_provider.name,
        )
        response = await self._call_provider(self.fallback_provider, query, max_results, engines)
        response = SearchResponse(
            query=response.query,
            results=response.results,
            provider=self.provider.name,
            error=response.error,
            disabled=response.disabled,
            fallback_provider=response.provider,
        )
        _record(events, status="fallback_ok", query=query, fallbackProvider=response.fallback_provider, results=len(response.results))
        await self._store_cache(query, engines, max_results, response.results)
        return response

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


def _configured_api_key(config: dict[str, Any]) -> str:
    explicit = config.get("apiKey")
    if explicit:
        return str(explicit)
    env_name = str(config.get("apiKeyEnv") or "OLLAMA_API_KEY").strip()
    return os.environ.get(env_name, "").strip() if env_name else ""


def _endpoint_path(value: Any) -> str:
    path = str(value or "").strip()
    if not path:
        raise SearchConfigurationError("Ollama endpoint path must not be empty.")
    return path if path.startswith("/") else f"/{path}"


def _record(events: list[dict[str, Any]] | None, **payload: Any) -> None:
    if events is not None:
        events.append(payload)
