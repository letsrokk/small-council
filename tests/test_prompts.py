from __future__ import annotations

import unittest

from small_council.prompts import (
    discussion_prompt,
    discussion_round_prompt,
    research_prompt,
    search_plan_prompt,
)
from small_council.state import Member


def _member() -> Member:
    return Member(
        name="Aurelia",
        model="qwen3:4b",
        personality="practical",
        is_president=False,
        created_at="now",
        provider="ollama",
    )


class PromptSearchGuidanceTests(unittest.TestCase):
    def test_research_prompt_mentions_search_worker(self) -> None:
        prompt = research_prompt(_member(), "What movie should I watch tonight?")

        self.assertIn("shared Search Worker", prompt)
        self.assertIn("current, external, or missing information", prompt)
        self.assertIn("after your training cutoff", prompt)
        self.assertIn("Do not invent freshness-sensitive details", prompt)

    def test_search_plan_prompt_asks_for_queries_when_information_is_needed(self) -> None:
        prompt = search_plan_prompt(
            _member(),
            "The user asks: 'latest restaurants Budapest'",
        )

        self.assertIn("shared Search Worker", prompt)
        self.assertIn("Use 1 to 3 concise search queries", prompt)
        self.assertIn("after your training cutoff", prompt)
        self.assertIn("Past dates can still require search", prompt)
        self.assertIn("Use 0 queries only", prompt)

    def test_discussion_prompt_does_not_add_search_worker_guidance(self) -> None:
        prompt = discussion_prompt(
            _member(),
            "What movie should I watch tonight?",
            [
                {
                    "proposer": "Aurelia",
                    "recommendation": "Pick one",
                    "short_reasoning": "Reason",
                    "pros": [],
                    "cons": [],
                    "confidence": 0.5,
                }
            ],
        )

        self.assertNotIn("shared Search Worker", prompt)

    def test_discussion_round_prompt_mentions_search_worker_when_enabled(self) -> None:
        prompt = discussion_round_prompt(
            _member(),
            "What movie should I watch tonight?",
            [],
            [],
            1,
            2,
            web_search_enabled=True,
        )

        self.assertIn("shared Search Worker", prompt)
        self.assertIn("current, external, or missing information", prompt)
        self.assertIn("after your training cutoff", prompt)
        self.assertIn("Do not invent freshness-sensitive details", prompt)

    def test_discussion_round_prompt_includes_uncertainty_when_search_disabled(self) -> None:
        prompt = discussion_round_prompt(
            _member(),
            "Pick the best current streaming plan.",
            [],
            [],
            1,
            2,
            web_search_enabled=False,
        )

        self.assertIn("Search is unavailable or forbidden", prompt)


if __name__ == "__main__":
    unittest.main()
