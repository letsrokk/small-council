from __future__ import annotations

import asyncio
import socket
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch

from small_council.web_search import (
    DEFAULT_SEARCH_ENGINES,
    SearchError,
    SearchParseError,
    SearchRateLimitError,
    SearchServerError,
    SearchResult,
    SearchTimeout,
    SearchWorker,
    SearxngSearchProvider,
    parse_searxng_results,
    web_search_config,
)


def _worker_config(**overrides):
    config = {
        "enabled": True,
        "provider": "searxng",
        "baseUrl": "http://localhost:8080",
        "timeoutSeconds": 1,
        "maxResults": 2,
        "cacheTtlSeconds": 900,
        "minDelaySeconds": 0,
        "maxConcurrentRequests": 10,
        "defaultEngines": ["bing", "github"],
    }
    config.update(overrides)
    return config


class FakeProvider:
    name = "fake"

    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.calls: list[tuple[str, int, list[str] | None]] = []
        self.active = 0
        self.max_active = 0

    def search(
        self, query: str, max_results: int, engines: list[str] | None = None
    ) -> list[SearchResult]:
        self.calls.append((query, max_results, engines))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                import time

                time.sleep(self.delay)
            return [
                SearchResult(
                    title=f"Result {query}",
                    url="https://example.com",
                    snippet="Snippet",
                    source="engine",
                )
            ][:max_results]
        finally:
            self.active -= 1


class SearxngSearchTests(unittest.TestCase):
    def test_search_sends_searxng_json_request(self) -> None:
        provider = SearxngSearchProvider(
            {"baseUrl": "http://localhost:8080", "timeoutSeconds": 1}
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self):
                return b'{"results":[{"title":"Example","url":"https://example.com"}]}'

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            results = provider.search("latest movies", 5, ["bing", "github"])

        parsed = urllib.parse.urlparse(captured["url"])
        params = urllib.parse.parse_qs(parsed.query)
        self.assertEqual("http://localhost:8080/search", f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
        self.assertEqual(["latest movies"], params["q"])
        self.assertEqual(["json"], params["format"])
        self.assertEqual(["bing,github"], params["engines"])
        self.assertEqual(1, captured["timeout"])
        self.assertEqual("Example", results[0].title)

    def test_successful_response_parsing(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "Useful snippet",
                    "engine": "duckduckgo",
                    "score": "1.5",
                },
                {
                    "title": "Second",
                    "url": "https://second.example",
                    "snippet": "Alternate snippet",
                    "engines": ["brave", "google"],
                },
            ]
        }

        results = parse_searxng_results(payload, 2)

        self.assertEqual("Example", results[0].title)
        self.assertEqual("https://example.com", results[0].url)
        self.assertEqual("Useful snippet", results[0].snippet)
        self.assertEqual("duckduckgo", results[0].source)
        self.assertEqual(1.5, results[0].score)
        self.assertEqual("brave, google", results[1].source)

    def test_empty_result_set(self) -> None:
        self.assertEqual([], parse_searxng_results({"results": []}, 5))

    def test_result_limit(self) -> None:
        payload = {
            "results": [
                {"title": "One", "url": "https://one.example"},
                {"title": "Two", "url": "https://two.example"},
            ]
        }

        results = parse_searxng_results(payload, 1)

        self.assertEqual(["One"], [result.title for result in results])

    def test_http_timeout(self) -> None:
        provider = SearxngSearchProvider(
            {"baseUrl": "http://localhost:8080", "timeoutSeconds": 0.01}
        )

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaises(SearchTimeout):
                provider.search("latest movies", 5)

    def test_http_429(self) -> None:
        provider = SearxngSearchProvider({"baseUrl": "http://localhost:8080"})
        error = urllib.error.HTTPError("url", 429, "rate limited", {}, None)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SearchError):
                provider.search("latest movies", 5)

    def test_http_5xx(self) -> None:
        provider = SearxngSearchProvider({"baseUrl": "http://localhost:8080"})
        error = urllib.error.HTTPError("url", 503, "unavailable", {}, None)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SearchError):
                provider.search("latest movies", 5)

    def test_malformed_json(self) -> None:
        provider = SearxngSearchProvider(
            {"baseUrl": "http://localhost:8080", "timeoutSeconds": 1}
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self):
                return b"{not json"

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaises(SearchParseError):
                provider.search("latest movies", 5)

    def test_network_error(self) -> None:
        provider = SearxngSearchProvider({"baseUrl": "http://localhost:8080"})

        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            with self.assertRaises(SearchError):
                provider.search("latest movies", 5)


class SearchConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = web_search_config({})

        self.assertTrue(config["enabled"])
        self.assertEqual("searxng", config["provider"])
        self.assertEqual("http://localhost:8080", config["baseUrl"])
        self.assertEqual(15, config["timeoutSeconds"])
        self.assertEqual(8, config["maxResults"])
        self.assertEqual(900, config["cacheTtlSeconds"])
        self.assertEqual(3, config["minDelaySeconds"])
        self.assertEqual(1, config["maxConcurrentRequests"])
        self.assertEqual(DEFAULT_SEARCH_ENGINES, config["defaultEngines"])

    def test_legacy_web_search_fallback(self) -> None:
        config = web_search_config({"webSearch": {"enabled": False, "maxResults": 3}})

        self.assertFalse(config["enabled"])
        self.assertEqual(3, config["maxResults"])


class SearchWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_engines_from_config(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(), provider)

        await worker.search("python")

        self.assertEqual(["bing", "github"], provider.calls[0][2])

    async def test_per_query_engine_override(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(), provider)

        await worker.search("python", engines=["wikipedia"])

        self.assertEqual(["wikipedia"], provider.calls[0][2])

    async def test_cache_hit(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(), provider)
        events: list[dict] = []

        await worker.search("python", events=events)
        await worker.search(" python ", events=events)

        self.assertEqual(1, len(provider.calls))
        self.assertIn("cache_hit", [event["status"] for event in events])

    async def test_cache_expiration(self) -> None:
        now = 100.0

        def monotonic():
            return now

        provider = FakeProvider()
        worker = SearchWorker(_worker_config(cacheTtlSeconds=1), provider, monotonic=monotonic)
        await worker.search("python")
        now = 102.0
        await worker.search("python")

        self.assertEqual(2, len(provider.calls))

    async def test_caching_disabled_with_zero_ttl(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(cacheTtlSeconds=0), provider)

        await worker.search("python")
        await worker.search("python")

        self.assertEqual(2, len(provider.calls))

    async def test_duplicate_concurrent_query_joins_one_in_flight_request(self) -> None:
        provider = FakeProvider(delay=0.05)
        worker = SearchWorker(_worker_config(cacheTtlSeconds=0), provider)

        await asyncio.gather(worker.search("python"), worker.search(" python "))

        self.assertEqual(1, len(provider.calls))

    async def test_different_queries_do_not_share_in_flight_request(self) -> None:
        provider = FakeProvider(delay=0.05)
        worker = SearchWorker(_worker_config(cacheTtlSeconds=0), provider)

        await asyncio.gather(worker.search("python"), worker.search("rust"))

        self.assertEqual(2, len(provider.calls))

    async def test_max_concurrent_requests_one(self) -> None:
        provider = FakeProvider(delay=0.05)
        worker = SearchWorker(
            _worker_config(cacheTtlSeconds=0, maxConcurrentRequests=1), provider
        )

        await asyncio.gather(worker.search("python"), worker.search("rust"))

        self.assertEqual(1, provider.max_active)

    async def test_min_delay_between_outbound_requests(self) -> None:
        now = 100.0
        sleeps: list[float] = []

        def monotonic():
            return now

        async def sleep(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        provider = FakeProvider()
        worker = SearchWorker(
            _worker_config(cacheTtlSeconds=0, minDelaySeconds=3, maxConcurrentRequests=1),
            provider,
            monotonic=monotonic,
            sleep=sleep,
        )

        await worker.search("python")
        await worker.search("rust")

        self.assertEqual([3], sleeps)

    async def test_timeout_handling(self) -> None:
        class TimeoutProvider(FakeProvider):
            def search(self, query, max_results, engines=None):
                raise SearchTimeout("timed out")

        worker = SearchWorker(_worker_config(), TimeoutProvider())
        response = await worker.search("python")

        self.assertFalse(response.ok)
        self.assertIn("timed out", response.error or "")

    async def test_malformed_json_handling(self) -> None:
        class ParseProvider(FakeProvider):
            def search(self, query, max_results, engines=None):
                raise SearchParseError("bad json")

        worker = SearchWorker(_worker_config(), ParseProvider())
        response = await worker.search("python")

        self.assertFalse(response.ok)
        self.assertIn("bad json", response.error or "")

    async def test_http_429_handling(self) -> None:
        class RateLimitProvider(FakeProvider):
            def search(self, query, max_results, engines=None):
                raise SearchRateLimitError("rate limited")

        worker = SearchWorker(_worker_config(), RateLimitProvider())
        response = await worker.search("python")

        self.assertFalse(response.ok)
        self.assertIn("rate limited", response.error or "")

    async def test_http_5xx_handling(self) -> None:
        class ServerErrorProvider(FakeProvider):
            def search(self, query, max_results, engines=None):
                raise SearchServerError("server error")

        worker = SearchWorker(_worker_config(), ServerErrorProvider())
        response = await worker.search("python")

        self.assertFalse(response.ok)
        self.assertIn("server error", response.error or "")

    async def test_disabled_search(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(enabled=False), provider)

        response = await worker.search("python")

        self.assertTrue(response.disabled)
        self.assertEqual(0, len(provider.calls))


if __name__ == "__main__":
    unittest.main()
