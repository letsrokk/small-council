from __future__ import annotations

import argparse
import io
import unittest

from small_council.cli import _secretary_config
from small_council.secretary import LocalSecretary, ModelBackedSecretary


class LocalSecretaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_report_emits_only_when_requested(self) -> None:
        stream = io.StringIO()
        secretary = LocalSecretary(stream)
        await secretary.start("Pick dinner")
        output_after_start = stream.getvalue()

        secretary.set_phase("collecting independent research")
        self.assertEqual(output_after_start, stream.getvalue())

        self.assertTrue(await secretary.report_milestone("initial proposals complete"))
        self.assertIn("collecting independent research", stream.getvalue())
        await secretary.stop()

    async def test_renderer_receives_secretary_and_member_events(self) -> None:
        class RecordingRenderer:
            def __init__(self) -> None:
                self.calls = []

            def start_run(self, context) -> None:
                self.calls.append(("start_run", context.question, context.phase))

            def update_phase(self, phase: str) -> None:
                self.calls.append(("update_phase", phase))

            def secretary_message(self, message: str, event_type: str = "milestone") -> None:
                self.calls.append(("secretary_message", message))

            def member_event(self, member_name: str, event_type: str, message: str, payload=None) -> None:
                self.calls.append(("member_event", member_name, event_type, message, payload or {}))

            def member_status(self, member_name: str, status: str) -> None:
                self.calls.append(("member_status", member_name, status))

        renderer = RecordingRenderer()
        secretary = LocalSecretary(io.StringIO(), renderer=renderer)
        await secretary.start("Pick dinner")
        secretary.set_phase("collecting independent research")
        secretary.diversity_lanes_assigned({"Aurelia": "safest pick"}, "balanced")
        secretary.recommendation_done("Aurelia", "Pick sushi")
        secretary.discussion_round_started(1)
        secretary.discussion_message_done(1, "Aurelia", "Fine.", "Pick sushi", "Pick ramen", True)
        secretary.final_recommendation_done("Aurelia", "Pick ramen")
        secretary.vote_done("Aurelia", "Pick ramen", 0)
        secretary.grouping_done([])
        await secretary.report_milestone("initial proposals complete")

        self.assertTrue(any(call[0] == "update_phase" for call in renderer.calls))
        self.assertTrue(any(call[:3] == ("member_event", "Aurelia", "proposal_ready") for call in renderer.calls))
        self.assertTrue(any(call[:3] == ("member_event", "Aurelia", "discussion_reply") for call in renderer.calls))
        self.assertTrue(any(call[:3] == ("member_event", "Aurelia", "vote") for call in renderer.calls))
        self.assertTrue(any(call[0] == "secretary_message" and "initial proposals complete" in call[1] for call in renderer.calls))
        await secretary.stop()

    async def test_immediate_event_updates_emit_short_local_reports(self) -> None:
        stream = io.StringIO()
        secretary = LocalSecretary(stream)
        await secretary.start("Pick dinner")
        secretary.recommendation_done("Aurelia", "Pick sushi")
        self.assertIn("Aurelia finished research.", stream.getvalue())

        secretary.discussion_round_started(1)
        self.assertIn("Discussion round 1 started.", stream.getvalue())

        secretary.discussion_message_done(1, "Aurelia", "Fine.", "Pick sushi", "Pick ramen", True)
        self.assertIn("Aurelia finished discussion round 1.", stream.getvalue())

        secretary.final_recommendation_done("Aurelia", "Pick ramen")
        self.assertIn("Aurelia finalized a proposal.", stream.getvalue())

        secretary.vote_done("Aurelia", "Pick ramen", 0)
        self.assertIn("Aurelia voted in the initial vote.", stream.getvalue())

        await secretary.stop()

    async def test_immediate_event_updates_can_be_disabled(self) -> None:
        stream = io.StringIO()
        secretary = LocalSecretary(stream, immediate_updates=False)
        await secretary.start("Pick dinner")
        baseline = stream.getvalue()

        secretary.recommendation_done("Aurelia", "Pick sushi")
        secretary.discussion_round_started(1)
        secretary.discussion_message_done(1, "Aurelia", "Fine.", "Pick sushi", "Pick ramen", True)
        secretary.final_recommendation_done("Aurelia", "Pick ramen")
        secretary.vote_done("Aurelia", "Pick ramen", 0)
        secretary.diversity_lanes_assigned({"Aurelia": "safest pick"}, "balanced")
        secretary.grouping_done([])

        self.assertEqual(baseline, stream.getvalue())

        secretary.set_phase("collecting independent research")
        self.assertTrue(await secretary.report_milestone("initial proposals complete"))
        self.assertIn("collecting independent research", stream.getvalue())
        await secretary.stop()

    async def test_agent_failure_emits_immediate_retry_and_final_updates(self) -> None:
        class RecordingRenderer:
            def __init__(self) -> None:
                self.statuses = []
                self.messages = []

            def member_status(self, member_name: str, status: str) -> None:
                self.statuses.append((member_name, status))

            def secretary_message(self, message: str, event_type: str = "milestone") -> None:
                self.messages.append(message)

        stream = io.StringIO()
        renderer = RecordingRenderer()
        secretary = LocalSecretary(stream, renderer=renderer)
        await secretary.start("Pick dinner")

        secretary.agent_run_failed("Aurelia", "research", "temporary failure", True, 1, 3)
        secretary.agent_run_failed("Aurelia", "research", "temporary failure", False, 3, 3)

        output = "\n".join(renderer.messages)
        self.assertIn("Aurelia failed research; retrying (1/3): temporary failure", output)
        self.assertIn("Aurelia failed research: temporary failure", output)
        self.assertIn(("Aurelia", "retrying"), renderer.statuses)
        self.assertIn(("Aurelia", "failed"), renderer.statuses)
        await secretary.stop()


class ModelBackedSecretaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_report_invoked_per_milestone(self) -> None:
        calls = []

        async def reporter(config, model, prompt, schema_path, phase):
            calls.append((model, prompt, schema_path, phase))
            return {"message": "Aurelia has finished a proposal."}

        stream = io.StringIO()
        secretary = ModelBackedSecretary({}, "gpt-5.4-mini", "balanced", stream, reporter)
        await secretary.start("Pick dinner")

        secretary.recommendation_done("Aurelia", "Pick sushi")
        self.assertEqual(0, len(calls))
        self.assertIn("Aurelia finished research.", stream.getvalue())

        secretary.set_phase("collecting independent research")
        self.assertTrue(await secretary.report_milestone("initial proposals complete"))
        self.assertEqual(1, len(calls))
        self.assertIn("initial proposals complete", calls[0][1])
        self.assertIn("Aurelia has finished a proposal.", stream.getvalue())
        await secretary.stop()

    async def test_model_failure_falls_back_to_local_reports(self) -> None:
        calls = []

        async def reporter(config, model, prompt, schema_path, phase):
            calls.append(phase)
            raise RuntimeError("model unavailable")

        stream = io.StringIO()
        secretary = ModelBackedSecretary({}, "gpt-5.4-mini", "balanced", stream, reporter)
        await secretary.start("Pick dinner")
        secretary.set_phase("collecting independent research")

        self.assertTrue(await secretary.report_milestone("initial proposals complete"))
        self.assertIn("falling back to local reports", stream.getvalue())
        self.assertIn("collecting independent research", stream.getvalue())

        secretary.set_phase("threaded discussion")
        self.assertTrue(await secretary.report_milestone("discussion round 1 complete"))
        self.assertEqual(1, len(calls))
        self.assertIn("threaded discussion", stream.getvalue())
        await secretary.stop()

    async def test_model_backed_secretary_can_disable_immediate_updates(self) -> None:
        calls = []

        async def reporter(config, model, prompt, schema_path, phase):
            calls.append(phase)
            return {"message": "Initial proposals are in."}

        stream = io.StringIO()
        secretary = ModelBackedSecretary(
            {},
            "gpt-5.4-mini",
            "balanced",
            stream,
            immediate_updates=False,
            model_reporter=reporter,
        )
        await secretary.start("Pick dinner")
        baseline = stream.getvalue()

        secretary.recommendation_done("Aurelia", "Pick sushi")
        secretary.discussion_round_started(1)
        secretary.final_recommendation_done("Aurelia", "Pick ramen")
        secretary.vote_done("Aurelia", "Pick ramen", 0)
        self.assertEqual(0, len(calls))
        self.assertEqual(baseline, stream.getvalue())

        secretary.set_phase("collecting independent research")
        self.assertTrue(await secretary.report_milestone("initial proposals complete"))
        self.assertEqual(1, len(calls))
        self.assertIn("Initial proposals are in.", stream.getvalue())
        await secretary.stop()


class SecretaryConfigTests(unittest.TestCase):
    def test_secretary_config_uses_model_defaults(self) -> None:
        args = argparse.Namespace(
            secretary=None,
            secretary_verbosity=None,
        )
        config = {"secretary": {"model": "gpt-5.4-mini"}}

        secretary_config = _secretary_config(config, args)

        self.assertEqual("model", secretary_config.mode)
        self.assertEqual("balanced", secretary_config.verbosity)
        self.assertEqual("gpt-5.4-mini", secretary_config.model)

    def test_secretary_config_accepts_local_and_high_verbosity(self) -> None:
        args = argparse.Namespace(
            secretary="local",
            secretary_verbosity="high",
        )
        config = {"secretary": {}}

        secretary_config = _secretary_config(config, args)

        self.assertEqual("local", secretary_config.mode)
        self.assertEqual("high", secretary_config.verbosity)

    def test_secretary_config_defaults_immediate_updates_on(self) -> None:
        args = argparse.Namespace(
            secretary=None,
            secretary_verbosity=None,
        )
        config = {"secretary": {}}

        secretary_config = _secretary_config(config, args)

        self.assertTrue(secretary_config.immediate_updates)

    def test_secretary_config_can_disable_immediate_updates(self) -> None:
        args = argparse.Namespace(
            secretary=None,
            secretary_verbosity=None,
            secretary_immediate_updates=False,
        )
        config = {"secretary": {}}

        secretary_config = _secretary_config(config, args)

        self.assertFalse(secretary_config.immediate_updates)


if __name__ == "__main__":
    unittest.main()
