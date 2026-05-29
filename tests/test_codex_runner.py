from __future__ import annotations

import unittest
from pathlib import Path

from small_council.codex_runner import (
    CodexRunError,
    CodexUsageLimitError,
    _codex_error_for_exit,
)
from small_council.state import Member


class CodexErrorClassificationTests(unittest.TestCase):
    def test_usage_limit_exit_is_non_retryable(self) -> None:
        member = Member(
            name="Aurelia",
            model="gpt-5.4-mini",
            personality="practical",
            is_president=False,
            created_at="now",
        )

        error = _codex_error_for_exit(
            member,
            "research",
            1,
            "",
            "ERROR: You've hit your usage limit. Try again at May 30th, 2026 10:28 PM.",
            Path("runtime/logs/research-aurelia.log"),
        )

        self.assertIsInstance(error, CodexUsageLimitError)
        self.assertFalse(error.retryable)
        self.assertEqual("May 30th, 2026 10:28 PM", error.retry_after)
        self.assertIn("usage limit", str(error))
        self.assertIn("Log:", str(error))

    def test_generic_exit_is_retryable(self) -> None:
        member = Member(
            name="Bram",
            model="gpt-5.4-mini",
            personality="skeptical",
            is_president=False,
            created_at="now",
        )

        error = _codex_error_for_exit(
            member,
            "vote",
            1,
            "",
            "temporary network failure",
            Path("runtime/logs/vote-bram.log"),
        )

        self.assertIsInstance(error, CodexRunError)
        self.assertNotIsInstance(error, CodexUsageLimitError)
        self.assertTrue(error.retryable)
        self.assertIn("Bram", str(error))


if __name__ == "__main__":
    unittest.main()
