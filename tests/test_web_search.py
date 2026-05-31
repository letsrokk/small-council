from __future__ import annotations

import asyncio
import json
import os
import socket
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch

from small_council.web_search import (
    OllamaCloudSearchProvider,
    SearchAuthenticationError,
    SearchError,
    SearchParseError,
    SearchRateLimitError,
    SearchResponse,
    SearchServerError,
    SearchResult,
    SearchTimeout,
    SearchWorker,
    SearxngSearchProvider,
    UnknownSearchProviderError,
    create_search_provider,
    parse_searxng_results,
    web_search_config,
)


def _worker_config(**overrides):
    config = {
        "enabled": True,
        "provider": "searxng",
        "timeoutSeconds": 1,
        "maxResults": 2,
        "maxQueriesPerMember": 2,
        "cacheTtlSeconds": 900,
        "minDelaySeconds": 0,
        "maxConcurrentRequests": 10,
        "searxng": {
            "baseUrl": "http://localhost:8080",
        },
    }
    config.update(overrides)
    return config


class FakeProvider:
    name = "searxng"

    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.calls: list[tuple[str, int, list[str] | None]] = []
        self.active = 0
        self.max_active = 0

    def search(
        self, query: str, max_results: int, engines: list[str] | None = None
    ) -> SearchResponse:
        self.calls.append((query, max_results, engines))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                import time

                time.sleep(self.delay)
            results = [
                SearchResult(
                    title=f"Result {query}",
                    url="https://example.com",
                    snippet="Snippet",
                    source="engine",
                )
            ][:max_results]
            return SearchResponse(query=query, results=results, provider=self.name)
        finally:
            self.active -= 1


class SearxngSearchTests(unittest.TestCase):
    def test_search_sends_searxng_json_request(self) -> None:
        provider = SearxngSearchProvider(
            {"searxng": {"baseUrl": "http://localhost:8080"}, "timeoutSeconds": 1}
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
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = provider.search("latest movies", 5, ["bing", "github"])

        parsed = urllib.parse.urlparse(captured["url"])
        params = urllib.parse.parse_qs(parsed.query)
        self.assertEqual("http://localhost:8080/search", f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
        self.assertEqual(["latest movies"], params["q"])
        self.assertEqual(["json"], params["format"])
        self.assertEqual(["bing,github"], params["engines"])
        self.assertEqual("application/json", captured["headers"]["Accept"])
        self.assertEqual("127.0.0.1", captured["headers"]["X-forwarded-for"])
        self.assertEqual("127.0.0.1", captured["headers"]["X-real-ip"])
        self.assertEqual(1, captured["timeout"])
        self.assertEqual("searxng", response.provider)
        self.assertEqual("Example", response.results[0].title)

    def test_search_omits_engines_when_not_explicit(self) -> None:
        provider = SearxngSearchProvider(
            {"searxng": {"baseUrl": "http://localhost:8080"}, "timeoutSeconds": 1}
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self):
                return b'{"results":[]}'

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            provider.search("latest movies", 5)

        params = urllib.parse.parse_qs(urllib.parse.urlparse(captured["url"]).query)
        self.assertNotIn("engines", params)

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
            {"searxng": {"baseUrl": "http://localhost:8080"}, "timeoutSeconds": 0.01}
        )

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaises(SearchTimeout):
                provider.search("latest movies", 5)

    def test_http_429(self) -> None:
        provider = SearxngSearchProvider({"searxng": {"baseUrl": "http://localhost:8080"}})
        error = urllib.error.HTTPError("url", 429, "rate limited", {}, None)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SearchError):
                provider.search("latest movies", 5)

    def test_http_5xx(self) -> None:
        provider = SearxngSearchProvider({"searxng": {"baseUrl": "http://localhost:8080"}})
        error = urllib.error.HTTPError("url", 503, "unavailable", {}, None)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SearchError):
                provider.search("latest movies", 5)

    def test_malformed_json(self) -> None:
        provider = SearxngSearchProvider(
            {"searxng": {"baseUrl": "http://localhost:8080"}, "timeoutSeconds": 1}
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
        provider = SearxngSearchProvider({"searxng": {"baseUrl": "http://localhost:8080"}})

        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            with self.assertRaises(SearchError):
                provider.search("latest movies", 5)


class SearchConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = web_search_config({})

        self.assertTrue(config["enabled"])
        self.assertEqual("searxng", config["provider"])
        self.assertEqual(30, config["timeoutSeconds"])
        self.assertEqual(10, config["maxResults"])
        self.assertEqual(2, config["maxQueriesPerMember"])
        self.assertEqual(900, config["cacheTtlSeconds"])
        self.assertEqual(3, config["minDelaySeconds"])
        self.assertEqual(3, config["maxConcurrentRequests"])
        self.assertNotIn("baseUrl", config)
        self.assertEqual({"baseUrl": "http://localhost:8080"}, config["searxng"])
        self.assertFalse(config["allowFallback"])
        self.assertEqual("searxng", config["fallbackProvider"])

    def test_nested_provider_config(self) -> None:
        config = web_search_config(
            {
                "search": {
                    "provider": "ollama",
                    "ollama": {"baseUrl": "https://ollama.example", "apiKeyEnv": "TEST_KEY"},
                    "searxng": {
                        "baseUrl": "http://searxng:8080",
                    },
                }
            }
        )

        self.assertEqual("ollama", config["provider"])
        self.assertEqual("https://ollama.example", config["ollama"]["baseUrl"])
        self.assertEqual("TEST_KEY", config["ollama"]["apiKeyEnv"])
        self.assertEqual("http://searxng:8080", config["searxng"]["baseUrl"])


class SearchProviderFactoryTests(unittest.TestCase):
    def test_default_provider_is_searxng(self) -> None:
        provider = create_search_provider({})

        self.assertIsInstance(provider, SearxngSearchProvider)

    def test_configured_provider_is_ollama(self) -> None:
        provider = create_search_provider({"search": {"provider": "ollama"}})

        self.assertIsInstance(provider, OllamaCloudSearchProvider)

    def test_invalid_provider_raises(self) -> None:
        with self.assertRaises(UnknownSearchProviderError):
            create_search_provider({"search": {"provider": "missing"}})


class OllamaCloudSearchTests(unittest.TestCase):
    def test_missing_api_key(self) -> None:
        provider = OllamaCloudSearchProvider(
            web_search_config({"search": {"ollama": {"apiKeyEnv": "MISSING_OLLAMA_TEST_KEY"}}})
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SearchAuthenticationError):
                provider.search("python", 3)

    def test_search_sends_authenticated_request_and_parses_results(self) -> None:
        provider = OllamaCloudSearchProvider(
            web_search_config(
                {
                    "search": {
                        "timeoutSeconds": 2,
                        "ollama": {"baseUrl": "https://ollama.example", "apiKey": "secret"},
                    }
                }
            )
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self):
                return json.dumps(
                    {
                        "results": [
                            {
                                "title": "Example",
                                "url": "https://example.com",
                                "content": "Snippet",
                            }
                        ]
                    }
                ).encode("utf-8")

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = provider.search("python", 3)

        self.assertEqual("https://ollama.example/api/web_search", captured["url"])
        self.assertEqual(2, captured["timeout"])
        self.assertEqual("Bearer secret", captured["headers"]["Authorization"])
        self.assertEqual({"query": "python", "max_results": 3}, captured["payload"])
        self.assertEqual("ollama", response.provider)
        self.assertEqual("Example", response.results[0].title)
        self.assertEqual("Snippet", response.results[0].snippet)
        self.assertEqual("https://example.com", response.results[0].raw["url"])

    def test_fetch_sends_authenticated_request_and_parses_response(self) -> None:
        provider = OllamaCloudSearchProvider(
            web_search_config({"search": {"ollama": {"baseUrl": "https://ollama.example", "apiKey": "secret"}}})
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self):
                return b'{"title":"Example","content":"Page text","links":["https://example.com/a"]}'

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = provider.fetch("https://example.com")

        self.assertEqual("https://ollama.example/api/web_fetch", captured["url"])
        self.assertEqual({"url": "https://example.com"}, captured["payload"])
        self.assertEqual("Example", response.title)
        self.assertEqual("Page text", response.content)
        self.assertEqual(["https://example.com/a"], response.links)

    def test_auth_failure(self) -> None:
        provider = OllamaCloudSearchProvider(
            web_search_config({"search": {"ollama": {"apiKey": "secret"}}})
        )
        error = urllib.error.HTTPError("url", 401, "unauthorized", {}, None)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SearchAuthenticationError):
                provider.search("python", 3)

    def test_timeout(self) -> None:
        provider = OllamaCloudSearchProvider(
            web_search_config({"search": {"ollama": {"apiKey": "secret"}}})
        )

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaises(SearchTimeout):
                provider.search("python", 3)

    def test_malformed_json(self) -> None:
        provider = OllamaCloudSearchProvider(
            web_search_config({"search": {"ollama": {"apiKey": "secret"}}})
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
                provider.search("python", 3)


class SearchWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_search_uses_provider_engines(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(), provider)

        await worker.search("python")

        self.assertEqual([], provider.calls[0][2])

    async def test_per_query_engine_override(self) -> None:
        provider = FakeProvider()
        worker = SearchWorker(_worker_config(), provider)

        await worker.search("python", engines=["wikipedia"])

        self.assertEqual(["wikipedia"], provider.calls[0][2])

    async def test_searxng_default_engines_do_not_apply_to_ollama(self) -> None:
        provider = FakeProvider()
        provider.name = "ollama"
        worker = SearchWorker(_worker_config(), provider)

        await worker.search("python")

        self.assertEqual([], provider.calls[0][2])

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

    async def test_hard_timeout_returns_unsuccessful_response(self) -> None:
        provider = FakeProvider(delay=0.05)
        events: list[dict] = []
        worker = SearchWorker(_worker_config(timeoutSeconds=0.01), provider)

        response = await worker.search("python", events=events)

        self.assertFalse(response.ok)
        self.assertEqual([], response.results)
        self.assertIn("timed out after 0.01s", response.error or "")
        self.assertIn("timeout", [event["status"] for event in events])

    async def test_hard_timeout_does_not_poison_in_flight_cache(self) -> None:
        provider = FakeProvider(delay=0.02)
        worker = SearchWorker(_worker_config(timeoutSeconds=0.01), provider)

        first = await worker.search("python")
        await asyncio.sleep(0.05)
        second = await worker.search("python")

        self.assertFalse(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(1, len(provider.calls))

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

    async def test_fallback_disabled_by_default(self) -> None:
        class FailingProvider(FakeProvider):
            name = "primary"

            def search(self, query, max_results, engines=None):
                raise SearchServerError("down")

        worker = SearchWorker(_worker_config(), FailingProvider())

        response = await worker.search("python")

        self.assertFalse(response.ok)
        self.assertIsNone(response.fallback_provider)
        self.assertIn("down", response.error or "")

    async def test_fallback_provider_used_when_configured(self) -> None:
        class FailingProvider(FakeProvider):
            name = "primary"

            def search(self, query, max_results, engines=None):
                raise SearchServerError("down")

        fallback = FakeProvider()
        fallback.name = "fallback"
        events: list[dict] = []
        worker = SearchWorker(
            _worker_config(allowFallback=True),
            FailingProvider(),
            fallback_provider=fallback,
        )

        response = await worker.search("python", events=events)

        self.assertTrue(response.ok)
        self.assertEqual("primary", response.provider)
        self.assertEqual("fallback", response.fallback_provider)
        self.assertEqual(["fallback", "fallback_ok"], [event["status"] for event in events if event["status"].startswith("fallback")])

    async def test_searxng_fallback_uses_provider_engines(self) -> None:
        class FailingProvider(FakeProvider):
            name = "ollama"

            def search(self, query, max_results, engines=None):
                raise SearchServerError("down")

        fallback = FakeProvider()
        events: list[dict] = []
        worker = SearchWorker(
            _worker_config(),
            FailingProvider(),
            fallback_provider=fallback,
        )

        await worker.search("python", events=events)

        self.assertEqual([], fallback.calls[0][2])


if __name__ == "__main__":
    unittest.main()
