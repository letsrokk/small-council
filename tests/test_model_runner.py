from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from small_council import model_runner
from small_council.state import Member
from small_council.web_search import SearchResponse, SearchResult, SearchWorker


def _config(enabled: bool = True) -> dict:
    return {
        "runtime": {"logs_path": "./runtime/logs", "temp_path": "./runtime/temp"},
        "model_providers": {"ollama": {"enabled": True}},
        "search": {
            "enabled": enabled,
            "provider": "searxng",
            "timeoutSeconds": 1,
            "maxResults": 2,
            "maxQueriesPerMember": 2,
            "cacheTtlSeconds": 0,
            "minDelaySeconds": 0,
            "maxConcurrentRequests": 1,
            "searxng": {
                "baseUrl": "http://localhost:8080",
            },
        },
    }


class FakeSearchProvider:
    name = "searxng"

    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.queries: list[str] = []

    def search(
        self, query: str, max_results: int, engines: list[str] | None = None
    ) -> SearchResponse:
        self.queries.append(query)
        if self.delay:
            import time

            time.sleep(self.delay)
        results = [
            SearchResult(
                title="Result",
                url="https://example.com",
                snippet="Search snippet",
                source="engine",
            )
        ]
        return SearchResponse(query=query, results=results[:max_results], provider=self.name)


class ModelRunnerSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_web_search_calls_provider_once(self) -> None:
        member = Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(payload={})))

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
        ):
            await model_runner.run_member(
                _config(enabled=False), member, "prompt", Path("schema.json"), "research", True
            )

        provider.run.assert_awaited_once_with(
            member, "prompt", Path("schema.json"), "research", False
        )

    async def test_enabled_web_search_injects_structured_results(self) -> None:
        member = Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock())
        provider.run.side_effect = [
            SimpleNamespace(payload={"queries": ["latest restaurants Budapest"]}),
            SimpleNamespace(payload={"recommendation": "Pick one"}),
        ]
        search_provider = FakeSearchProvider()

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(
                model_runner,
                "create_search_worker",
                return_value=SearchWorker(_config()["search"], search_provider),
            ),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_member(
                _config(),
                member,
                "The user asks: 'latest restaurants Budapest'",
                Path("schema.json"),
                "research",
                True,
            )

        self.assertEqual(["latest restaurants Budapest"], search_provider.queries)
        search_plan_prompt = provider.run.await_args_list[0].args[1]
        self.assertIn("Use 1 to 2 concise search queries", search_plan_prompt)
        final_prompt = provider.run.await_args_list[1].args[1]
        self.assertIn("Web search results", final_prompt)
        self.assertIn("Search snippet", final_prompt)
        self.assertEqual(Path("schema.json"), provider.run.await_args_list[1].args[2])

    async def test_search_plan_truncates_to_configured_query_limit(self) -> None:
        member = Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock())
        provider.run.side_effect = [
            SimpleNamespace(payload={"queries": ["latest one", "latest two", "latest three"]}),
            SimpleNamespace(payload={"recommendation": "Pick one"}),
        ]
        search_provider = FakeSearchProvider()

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(
                model_runner,
                "create_search_worker",
                return_value=SearchWorker(_config()["search"], search_provider),
            ),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_member(
                _config(),
                member,
                "The user asks: 'latest restaurants Budapest'",
                Path("schema.json"),
                "research",
                True,
            )

        self.assertEqual(["latest one", "latest two"], search_provider.queries)

    async def test_search_failure_still_runs_final_prompt(self) -> None:
        member = Member("Bram", "qwen3:4b", "skeptical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock())
        provider.run.side_effect = [
            SimpleNamespace(payload={"queries": ["latest news"]}),
            SimpleNamespace(payload={"recommendation": "Fallback"}),
        ]
        search_provider = SimpleNamespace(
            name="searxng",
            search=lambda query, max_results, engines=None: (_ for _ in ()).throw(RuntimeError("down")),
        )

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(
                model_runner,
                "create_search_worker",
                return_value=SearchWorker(_config()["search"], search_provider),
            ),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_member(
                _config(),
                member,
                "The user asks: 'latest news'",
                Path("schema.json"),
                "research",
                True,
            )

        final_prompt = provider.run.await_args_list[1].args[1]
        self.assertIn("search unavailable: down", final_prompt)

    async def test_empty_search_plan_falls_back_for_post_cutoff_year(self) -> None:
        member = Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock())
        provider.run.side_effect = [
            SimpleNamespace(payload={"queries": []}),
            SimpleNamespace(payload={"recommendation": "Pick one"}),
        ]
        search_provider = FakeSearchProvider()

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(
                model_runner,
                "create_search_worker",
                return_value=SearchWorker(_config()["search"], search_provider),
            ),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_member(
                _config(),
                member,
                "The user asks: 'Who won the 2025 Eurovision Song Contest?'",
                Path("schema.json"),
                "research",
                True,
            )

        self.assertEqual(["Who won the 2025 Eurovision Song Contest?"], search_provider.queries)

    async def test_empty_search_plan_falls_back_for_current_question(self) -> None:
        member = Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock())
        provider.run.side_effect = [
            SimpleNamespace(payload={"queries": []}),
            SimpleNamespace(payload={"recommendation": "Pick one"}),
        ]
        search_provider = FakeSearchProvider()

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(
                model_runner,
                "create_search_worker",
                return_value=SearchWorker(_config()["search"], search_provider),
            ),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_member(
                _config(),
                member,
                "The user asks: 'What is the best current phone plan?'",
                Path("schema.json"),
                "research",
                True,
            )

        self.assertEqual(["What is the best current phone plan?"], search_provider.queries)

    async def test_empty_search_plan_does_not_force_search_for_stable_choice(self) -> None:
        member = Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama")
        provider = SimpleNamespace(run=AsyncMock())
        provider.run.return_value = SimpleNamespace(payload={"recommendation": "Pick one"})
        search_provider = FakeSearchProvider()

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=provider),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(
                model_runner,
                "create_search_worker",
                return_value=SearchWorker(_config()["search"], search_provider),
            ),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_member(
                _config(),
                member,
                "The user asks: 'What movie should I watch: Alien or Arrival?'",
                Path("schema.json"),
                "research",
                True,
            )

        self.assertEqual([], search_provider.queries)
        provider.run.assert_awaited_once()
        final_prompt = provider.run.await_args.args[1]
        self.assertNotIn("Web search results", final_prompt)

    async def test_run_many_shares_one_search_worker(self) -> None:
        members = [
            Member("Aurelia", "qwen3:4b", "practical", False, "now", provider="ollama"),
            Member("Bram", "qwen3:4b", "skeptical", False, "now", provider="ollama"),
        ]
        search_provider = FakeSearchProvider(delay=0.05)
        worker = SearchWorker(_config()["search"], search_provider)

        class FakeModelProvider:
            async def run(self, member, prompt, schema_path, phase, web_search=False):
                if phase.endswith("search-plan"):
                    return SimpleNamespace(payload={"queries": ["latest news"]})
                return SimpleNamespace(payload={"recommendation": member.name})

        with (
            patch.object(model_runner, "provider_config", return_value={"enabled": True}),
            patch.object(model_runner, "create_provider", return_value=FakeModelProvider()),
            patch.object(model_runner, "search_enabled", return_value=True),
            patch.object(model_runner, "create_search_worker", return_value=worker),
            patch.object(model_runner, "write_search_log"),
        ):
            await model_runner.run_many(
                _config(),
                [
                    (members[0], "The user asks: 'latest news'", Path("schema.json"), "research", True),
                    (members[1], "The user asks: 'latest news'", Path("schema.json"), "research", True),
                ],
            )

        self.assertEqual(["latest news"], search_provider.queries)


if __name__ == "__main__":
    unittest.main()
