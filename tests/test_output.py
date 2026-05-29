from __future__ import annotations

import argparse
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from small_council import cli
from small_council.output import (
    PlainRenderer,
    MemberViewport,
    RichRenderer,
    render_json_decision,
    render_leaderboard_text,
    render_members_text,
    select_renderer,
)
from small_council.state import Member


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class FakePipe(io.StringIO):
    def isatty(self) -> bool:
        return False


class OutputSelectionTests(unittest.TestCase):
    def test_auto_selection_uses_rich_only_for_interactive_tty(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakeTTY(), stderr=FakeTTY(), stdin=FakeTTY())
        self.assertIsInstance(renderer, RichRenderer)

    def test_auto_selection_falls_back_to_plain_for_redirected_output(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakePipe(), stderr=FakeTTY(), stdin=FakeTTY())
        self.assertIsInstance(renderer, PlainRenderer)

    def test_plain_flag_forces_plain_renderer(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=True, rich_output=False)
        renderer = select_renderer(args, stdout=FakeTTY(), stderr=FakeTTY())
        self.assertIsInstance(renderer, PlainRenderer)

    def test_rich_flag_forces_rich_renderer(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=True)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakePipe(), stderr=FakePipe())
        self.assertIsInstance(renderer, RichRenderer)

    def test_json_flag_bypasses_renderers(self) -> None:
        args = SimpleNamespace(json_output=True, plain_output=False, rich_output=True)
        renderer = select_renderer(args, stdout=FakeTTY(), stderr=FakeTTY(), stdin=FakeTTY())
        self.assertIsNone(renderer)

    def test_auto_selection_accepts_tty_stdin_with_one_tty_output(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakeTTY(), stderr=FakePipe(), stdin=FakeTTY())
        self.assertIsInstance(renderer, RichRenderer)


class OutputRenderingTests(unittest.TestCase):
    def test_rich_renderer_debounces_refreshes(self) -> None:
        class DummyLive:
            def __init__(self) -> None:
                self.update_calls = 0
                self.refresh_calls = 0

            def update(self, renderable, refresh=False):
                self.update_calls += 1

            def refresh(self):
                self.refresh_calls += 1

        renderer = RichRenderer(rich_module=object())
        renderer._live = DummyLive()
        renderer._render = lambda: "renderable"

        with patch("small_council.output.time.monotonic", side_effect=[1.0, 1.01, 1.02, 1.20]):
            renderer.member_status("Aurelia", "running research")
            renderer.member_status("Aurelia", "completed")
            renderer.member_status("Bram", "running research")

        self.assertEqual(3, renderer._live.update_calls)
        self.assertEqual(1, renderer._live.refresh_calls)

    def test_member_viewport_tracks_focus_and_window(self) -> None:
        viewport = MemberViewport(total_count=7, window_size=3, focus_index=0, top_index=0)

        self.assertEqual((0, 3), viewport.visible_range())
        self.assertTrue(viewport.move_focus(1))
        self.assertEqual(1, viewport.focus_index)
        self.assertEqual((0, 3), viewport.visible_range())
        self.assertTrue(viewport.move_focus(2))
        self.assertEqual(3, viewport.focus_index)
        self.assertEqual((1, 4), viewport.visible_range())
        self.assertTrue(viewport.jump(1))
        self.assertEqual(5, viewport.focus_index)
        self.assertEqual((3, 6), viewport.visible_range())
        self.assertTrue(viewport.go_end())
        self.assertEqual(6, viewport.focus_index)
        self.assertEqual((4, 7), viewport.visible_range())
        self.assertTrue(viewport.go_home())
        self.assertEqual(0, viewport.focus_index)
        self.assertEqual((0, 3), viewport.visible_range())

    def test_member_viewport_keeps_focused_card_visible_after_resize(self) -> None:
        viewport = MemberViewport(total_count=7, window_size=3, focus_index=5, top_index=3)

        viewport.set_window_size(1)

        self.assertEqual(5, viewport.focus_index)
        self.assertEqual((5, 6), viewport.visible_range())

    def test_rich_renderer_handles_navigation_keys(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._viewport.set_total_count(6)
        renderer._viewport.set_window_size(3)

        self.assertTrue(renderer._handle_key_chunk(b"\x1b[B"))
        self.assertEqual(1, renderer._viewport.focus_index)
        self.assertTrue(renderer._handle_key_chunk(b"j"))
        self.assertEqual(2, renderer._viewport.focus_index)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[6~"))
        self.assertEqual(4, renderer._viewport.focus_index)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[H"))
        self.assertEqual(0, renderer._viewport.focus_index)

    def test_rich_renderer_switches_layout_on_narrow_terminals(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=90, height=30))
        self.assertTrue(renderer._should_stack_vertically())
        self.assertLess(renderer._left_column_width(), 30)

    def test_member_and_leaderboard_text_remain_readable(self) -> None:
        members = [
            Member(
                name="Aurelia",
                model="gpt-5.4-mini",
                personality="practical",
                is_president=True,
                created_at="now",
                total_wins=2,
                total_proposals=3,
                total_votes_cast=4,
                tie_break_victories=1,
            )
        ]
        leaderboard = [
            {
                "member": "Aurelia",
                "model": "gpt-5.4-mini",
                "personality": "practical",
                "president": True,
                "total_wins": 2,
                "total_proposals": 3,
                "win_rate": 0.667,
                "vote_participation": 4,
                "tie_break_victories": 1,
            }
        ]

        members_text = render_members_text(members)
        leaderboard_text = render_leaderboard_text(leaderboard)

        self.assertIn("Council Members", members_text)
        self.assertIn("Aurelia", members_text)
        self.assertIn("Leaderboard", leaderboard_text)
        self.assertIn("Win Rate", leaderboard_text)

    def test_json_decision_stays_machine_readable(self) -> None:
        payload = {"a": 1, "b": {"c": 2}}
        rendered = render_json_decision(payload)
        self.assertEqual(json.loads(rendered), payload)
        self.assertNotIn("\x1b[", rendered)

    def test_cli_still_prints_final_answer_on_stdout(self) -> None:
        members = [
            Member(
                name="Aurelia",
                model="gpt-5.4-mini",
                personality="practical",
                is_president=True,
                created_at="now",
            )
        ]
        payload = {"final_output": "Watch Train Dreams tonight."}
        stdout = io.StringIO()
        stderr = io.StringIO()
        renderer = PlainRenderer(stdout=io.StringIO(), stderr=io.StringIO())
        config = {
            "storage": {"leaderboard_path": "./storage/leaderboard.json", "council_state_path": "./storage/council-state.json"},
            "runtime": {"temp_path": "./runtime/temp", "logs_path": "./runtime/logs"},
            "council": {"discussion_rounds": 2, "runoff_rounds": 3, "secretary": {}, "diversity_mode": "balanced"},
        }

        with (
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "_ensure_dirs", return_value=None),
            patch.object(cli, "ensure_state", return_value=members),
            patch.object(cli, "_maybe_resize_members", return_value=members),
            patch.object(cli, "write_agent_files", return_value=None),
            patch.object(cli, "select_renderer", return_value=renderer),
            patch.object(cli, "_run_decision", new=AsyncMock(return_value=payload)),
            patch.object(cli.sys, "stdout", stdout),
            patch.object(cli.sys, "stderr", stderr),
        ):
            exit_code = cli.main(["What should I watch tonight?"])

        self.assertEqual(0, exit_code)
        self.assertEqual("Watch Train Dreams tonight.\n", stdout.getvalue())

    def test_cli_json_output_stays_machine_readable(self) -> None:
        members = [
            Member(
                name="Aurelia",
                model="gpt-5.4-mini",
                personality="practical",
                is_president=True,
                created_at="now",
            )
        ]
        payload = {"final_output": "Watch Train Dreams tonight.", "a": 1}
        stdout = io.StringIO()
        stderr = io.StringIO()
        config = {
            "storage": {"leaderboard_path": "./storage/leaderboard.json", "council_state_path": "./storage/council-state.json"},
            "runtime": {"temp_path": "./runtime/temp", "logs_path": "./runtime/logs"},
            "council": {"discussion_rounds": 2, "runoff_rounds": 3, "secretary": {}, "diversity_mode": "balanced"},
        }

        with (
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "_ensure_dirs", return_value=None),
            patch.object(cli, "ensure_state", return_value=members),
            patch.object(cli, "_maybe_resize_members", return_value=members),
            patch.object(cli, "write_agent_files", return_value=None),
            patch.object(cli, "_run_decision", new=AsyncMock(return_value=payload)),
            patch.object(cli.sys, "stdout", stdout),
            patch.object(cli.sys, "stderr", stderr),
        ):
            exit_code = cli.main(["--json-output", "What should I watch tonight?"])

        self.assertEqual(0, exit_code)
        rendered = stdout.getvalue()
        self.assertEqual(json.loads(rendered), payload)
        self.assertNotIn("\x1b[", rendered)


if __name__ == "__main__":
    unittest.main()
