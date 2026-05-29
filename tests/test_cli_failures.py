from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from small_council import cli
from small_council.codex_runner import CodexRunError, CodexUsageLimitError
from small_council.secretary import LocalSecretary
from small_council.state import Member


def _member(name: str = "Aurelia") -> Member:
    return Member(
        name=name,
        model="gpt-5.4-mini",
        personality="practical",
        is_president=False,
        created_at="now",
    )


class CliFailureHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_member_retry_succeeds_after_retryable_failure(self) -> None:
        member = _member()
        config = {"codex": {"retries": 2, "retry_base_delay_seconds": 0}}
        result = object()
        failure = CodexRunError(
            "temporary failure",
            member_name=member.name,
            phase="research",
            log_path=Path("runtime/logs/research-aurelia.log"),
            retryable=True,
        )
        secretary = LocalSecretary(io.StringIO())
        await secretary.start("Pick dinner")

        with (
            patch.object(cli, "run_member", new=AsyncMock(side_effect=[failure, result])) as run,
            patch.object(cli.asyncio, "sleep", new=AsyncMock()) as sleep,
        ):
            actual = await cli._run_member_with_retries(
                config, member, "prompt", Path("schema.json"), "research", False, secretary
            )

        self.assertIs(actual, result)
        self.assertEqual(2, run.await_count)
        self.assertEqual(1, sleep.await_count)
        self.assertIn("retrying (1/3)", secretary.stream.getvalue())

    async def test_usage_limit_failure_is_not_retried(self) -> None:
        member = _member()
        config = {"codex": {"retries": 2, "retry_base_delay_seconds": 0}}
        failure = CodexUsageLimitError(
            "Codex usage limit reached while running Aurelia in research.",
            member_name=member.name,
            phase="research",
            log_path=Path("runtime/logs/research-aurelia.log"),
            retryable=False,
        )
        secretary = LocalSecretary(io.StringIO())
        await secretary.start("Pick dinner")

        with (
            patch.object(cli, "run_member", new=AsyncMock(side_effect=failure)) as run,
            patch.object(cli.asyncio, "sleep", new=AsyncMock()) as sleep,
        ):
            with self.assertRaises(CodexUsageLimitError):
                await cli._run_member_with_retries(
                    config, member, "prompt", Path("schema.json"), "research", False, secretary
                )

        self.assertEqual(1, run.await_count)
        self.assertEqual(0, sleep.await_count)
        self.assertIn("Aurelia failed research", secretary.stream.getvalue())

    async def test_retryable_failure_aborts_after_configured_retries(self) -> None:
        member = _member()
        config = {"codex": {"retries": 1, "retry_base_delay_seconds": 0}}
        failure = CodexRunError(
            "temporary failure",
            member_name=member.name,
            phase="vote",
            log_path=Path("runtime/logs/vote-aurelia.log"),
            retryable=True,
        )
        secretary = LocalSecretary(io.StringIO())
        await secretary.start("Pick dinner")

        with (
            patch.object(cli, "run_member", new=AsyncMock(side_effect=failure)) as run,
            patch.object(cli.asyncio, "sleep", new=AsyncMock()) as sleep,
        ):
            with self.assertRaises(CodexRunError):
                await cli._run_member_with_retries(
                    config, member, "prompt", Path("schema.json"), "vote", False, secretary
                )

        self.assertEqual(2, run.await_count)
        self.assertEqual(1, sleep.await_count)
        self.assertIn("retrying (1/2)", secretary.stream.getvalue())
        self.assertIn("Aurelia failed vote: temporary failure", secretary.stream.getvalue())


class CliMainFailureMessageTests(unittest.TestCase):
    def test_usage_limit_main_message_is_concise_without_auth_hint(self) -> None:
        members = [_member()]
        stdout = io.StringIO()
        stderr = io.StringIO()
        config = {
            "storage": {
                "leaderboard_path": "./storage/leaderboard.json",
                "council_state_path": "./storage/council-state.json",
            },
            "runtime": {"temp_path": "./runtime/temp", "logs_path": "./runtime/logs"},
            "council": {
                "discussion_rounds": 2,
                "runoff_rounds": 3,
                "secretary": {},
                "diversity_mode": "balanced",
            },
        }
        failure = CodexUsageLimitError(
            "Codex usage limit reached while running Aurelia in research.",
            member_name="Aurelia",
            phase="research",
            log_path=Path("runtime/logs/research-aurelia.log"),
            retryable=False,
        )

        with (
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "_ensure_dirs", return_value=None),
            patch.object(cli, "ensure_state", return_value=members),
            patch.object(cli, "_maybe_resize_members", return_value=members),
            patch.object(cli, "write_agent_files", return_value=None),
            patch.object(cli, "select_renderer", return_value=None),
            patch.object(cli, "_run_decision", new=AsyncMock(side_effect=failure)),
            patch.object(cli.sys, "stdout", stdout),
            patch.object(cli.sys, "stderr", stderr),
        ):
            exit_code = cli.main(["--json-output", "Pick dinner"])

        self.assertEqual(1, exit_code)
        rendered = stderr.getvalue()
        self.assertIn("Council failed: Codex usage limit reached", rendered)
        self.assertIn("Log: runtime/logs/research-aurelia.log", rendered)
        self.assertNotIn("codex login", rendered)
        self.assertNotIn("You've hit your usage limit", rendered)


if __name__ == "__main__":
    unittest.main()
