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
    MemberViewState,
    RichRenderer,
    RunContext,
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
    def test_default_selection_uses_rich_for_interactive_tty(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakeTTY(), stderr=FakeTTY(), stdin=FakeTTY())
        self.assertIsInstance(renderer, RichRenderer)

    def test_default_selection_uses_rich_for_redirected_output(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakePipe(), stderr=FakeTTY(), stdin=FakeTTY())
        self.assertIsInstance(renderer, RichRenderer)

    def test_default_selection_falls_back_to_plain_when_rich_is_unavailable(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", side_effect=RuntimeError("missing rich")):
            renderer = select_renderer(args, stdout=FakePipe(), stderr=FakePipe(), stdin=FakePipe())
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

    def test_default_selection_uses_rich_with_one_tty_output(self) -> None:
        args = SimpleNamespace(json_output=False, plain_output=False, rich_output=False)
        with patch("small_council.output._load_rich", return_value=object()):
            renderer = select_renderer(args, stdout=FakeTTY(), stderr=FakePipe(), stdin=FakeTTY())
        self.assertIsInstance(renderer, RichRenderer)


class OutputRenderingTests(unittest.TestCase):
    def _members(self, count: int = 5) -> list[Member]:
        return [
            Member(
                name=f"Member {index}",
                model="gpt-5.4-mini",
                personality="practical",
                is_president=index == 1,
                created_at="now",
            )
            for index in range(1, count + 1)
        ]

    def test_rich_renderer_debounces_refreshes(self) -> None:
        class DummyLive:
            def __init__(self) -> None:
                self.update_calls = 0
                self.refresh_calls = 0

            def update(self, renderable, refresh=False):
                self.update_calls += 1

            def refresh(self):
                self.refresh_calls += 1

            def stop(self):
                return None

        renderer = RichRenderer(rich_module=object())
        renderer._live = DummyLive()
        renderer._render = lambda: "renderable"

        with patch("small_council.output.time.monotonic", side_effect=[1.0, 1.01, 1.02, 1.20]):
            renderer.member_status("Aurelia", "running research")
            renderer.member_status("Aurelia", "completed")
            renderer.member_status("Bram", "running research")

        self.assertEqual(3, renderer._live.update_calls)
        self.assertEqual(1, renderer._live.refresh_calls)

    def test_rich_live_refreshes_only_on_events(self) -> None:
        import rich
        import rich.box  # noqa: F401
        import rich.console  # noqa: F401
        import rich.layout  # noqa: F401
        import rich.live  # noqa: F401
        import rich.panel  # noqa: F401
        import rich.text  # noqa: F401

        output = FakeTTY()
        renderer = RichRenderer(stdout=output, stderr=output, rich_module=rich, enable_keyboard=False)

        renderer.start_run(
            RunContext(
                question="Pick dinner",
                member_count=1,
                diversity_mode="balanced",
                secretary_mode="local",
                discussion_rounds=2,
                runoff_round_limit=3,
                web_search_enabled=False,
            )
        )
        try:
            self.assertFalse(renderer._live.auto_refresh)
        finally:
            renderer.close()

    def test_rich_renderer_bulk_seeds_all_members_before_activity(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = self._members(5)

        renderer.seed_members(members)

        self.assertEqual(
            ["Member 1", "Member 2", "Member 3", "Member 4", "Member 5"],
            sorted(renderer._member_state),
        )
        self.assertEqual("President", renderer._member_state["Member 1"].role)
        self.assertEqual("gpt-5.4-mini", renderer._member_state["Member 5"].model)
        self.assertEqual("queued", renderer._member_state["Member 5"].status)
        self.assertEqual([], renderer._member_state["Member 5"].events)
        self.assertEqual("", renderer._member_state["Member 5"].latest_activity)

    def test_rich_renderer_bulk_seed_refreshes_once(self) -> None:
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

        renderer.seed_members(self._members(5))

        self.assertEqual(1, renderer._live.update_calls)
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
        self.assertEqual(5, renderer._viewport.focus_index)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[5~"))
        self.assertEqual(0, renderer._viewport.focus_index)

    def test_rich_renderer_switches_active_sections_with_left_and_right(self) -> None:
        renderer = RichRenderer(rich_module=object())

        self.assertEqual("members", renderer._active_section)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[D"))
        self.assertEqual("secretary", renderer._active_section)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[C"))
        self.assertEqual("members", renderer._active_section)

    def test_up_down_scroll_secretary_when_secretary_is_active(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=20))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        for index in range(20):
            renderer._add_secretary_line(f"update {index}")
        renderer._active_section = "secretary"

        self.assertTrue(renderer._handle_key_chunk(b"\x1b[A"))
        self.assertGreater(renderer._secretary_scroll_offset, 0)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[B"))
        self.assertEqual(0, renderer._secretary_scroll_offset)

    def test_page_keys_jump_secretary_to_oldest_and_newest(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=20))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        for index in range(20):
            renderer._add_secretary_line(f"update {index}")
        renderer._active_section = "secretary"

        self.assertTrue(renderer._handle_key_chunk(b"\x1b[5~"))
        self.assertEqual(renderer._max_secretary_scroll_offset(), renderer._secretary_scroll_offset)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[6~"))
        self.assertEqual(0, renderer._secretary_scroll_offset)

    def test_home_end_match_page_keys_in_secretary_section(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=20))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        for index in range(20):
            renderer._add_secretary_line(f"update {index}")
        renderer._active_section = "secretary"

        self.assertTrue(renderer._handle_key_chunk(b"\x1b[H"))
        self.assertEqual(renderer._max_secretary_scroll_offset(), renderer._secretary_scroll_offset)
        self.assertTrue(renderer._handle_key_chunk(b"\x1b[F"))
        self.assertEqual(0, renderer._secretary_scroll_offset)

    def test_secretary_new_messages_auto_follow_only_at_tail(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=20))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        for index in range(20):
            renderer._add_secretary_line(f"update {index}")

        renderer._secretary_scroll_offset = 0
        renderer._add_secretary_line("newest")
        self.assertEqual(0, renderer._secretary_scroll_offset)

        renderer._scroll_secretary(2)
        previous = renderer._secretary_scroll_offset
        renderer._add_secretary_line("newer still")
        self.assertGreater(renderer._secretary_scroll_offset, previous)

    def test_final_mode_exits_only_on_escape_or_enter(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._viewport.set_total_count(3)
        renderer._awaiting_final_key = True

        self.assertTrue(renderer._handle_key_chunk(b"\x1b[B"))
        self.assertFalse(renderer._final_key_pressed.is_set())
        renderer._handle_key_chunk(b"\x1b")
        self.assertTrue(renderer._final_key_pressed.is_set())

        renderer._final_key_pressed.clear()
        renderer._handle_key_chunk(b"\r")
        self.assertTrue(renderer._final_key_pressed.is_set())

    def test_rich_renderer_switches_layout_on_narrow_terminals(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=90, height=30))
        self.assertTrue(renderer._should_stack_vertically())
        self.assertLess(renderer._left_column_width(), 30)

    def test_model_backed_secretary_gets_at_least_thirty_percent_width(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=30))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            secretary_verbosity="balanced",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )

        self.assertGreaterEqual(renderer._left_column_width(), 36)

    def test_long_lines_are_soft_wrapped_for_rich_panels(self) -> None:
        renderer = RichRenderer(rich_module=object())
        long_line = "Latest: " + "choose the practical neighborhood option " * 5

        wrapped = renderer._wrap_line(long_line, 32)

        self.assertGreater(len(wrapped), 1)
        self.assertTrue(all(len(line) <= 32 for line in wrapped))

    def test_secretary_history_keeps_more_than_ten_lines_but_stays_bounded(self) -> None:
        renderer = RichRenderer(rich_module=object())

        for index in range(120):
            renderer._add_secretary_line(f"update {index}")

        self.assertEqual(100, len(renderer._secretary_history))
        self.assertEqual("update 20", renderer._secretary_history[0])
        self.assertEqual("update 119", renderer._secretary_history[-1])

    def test_tall_secretary_panel_shows_more_than_eight_history_lines(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=40))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        for index in range(20):
            renderer._add_secretary_line(f"update {index}")
        budget = renderer._secretary_content_height() - 1 - renderer._wrapped_line_count(
            "Phase: starting", renderer._secretary_content_width()
        )

        visible = renderer._visible_secretary_history_lines(renderer._secretary_content_width(), budget)

        self.assertGreater(len(visible), 8)
        self.assertEqual("update 19", visible[-1])

    def test_short_secretary_panel_prioritizes_newest_lines_that_fit(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=90, height=30))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        for index in range(10):
            renderer._add_secretary_line(f"update {index}")
        budget = renderer._secretary_content_height() - 1 - renderer._wrapped_line_count(
            "Phase: starting", renderer._secretary_content_width()
        )

        visible = renderer._visible_secretary_history_lines(renderer._secretary_content_width(), budget)

        self.assertEqual(["update 5", "update 6", "update 7", "update 8", "update 9"], visible)

    def test_secretary_history_fit_counts_wrapped_visual_lines(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._add_secretary_line("alpha beta gamma delta epsilon zeta")

        visible = renderer._visible_secretary_history_lines(width=12, max_lines=2)

        self.assertEqual(2, len(visible))
        self.assertTrue(all(len(line) <= 12 for line in visible))

    def test_member_stack_can_scroll_last_member_into_view_with_tall_cards(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=34))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        members = [
            MemberViewState(
                name=f"Member {index}",
                latest_detail="A longer update that wraps over multiple visual lines. " * 3,
            )
            for index in range(1, 6)
        ]
        renderer._viewport.set_total_count(len(members))
        renderer._viewport.go_end()

        start, end = renderer._visible_member_range(members, renderer._member_card_budget())

        self.assertLess(start, end)
        self.assertEqual(len(members), end)

    def test_member_range_includes_one_partial_card_when_space_remains(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = [MemberViewState(name=f"Member {index}") for index in range(1, 5)]
        renderer._estimated_member_card_height = lambda state: 10

        member_range = renderer._visible_member_window(members, card_budget=25)

        self.assertEqual((0, 3), (member_range.start, member_range.end))
        self.assertEqual(2, member_range.full_count)
        self.assertTrue(member_range.has_partial)

    def test_member_range_does_not_include_partial_card_without_leftover_space(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = [MemberViewState(name=f"Member {index}") for index in range(1, 5)]
        renderer._estimated_member_card_height = lambda state: 10

        member_range = renderer._visible_member_window(members, card_budget=20)

        self.assertEqual((0, 2), (member_range.start, member_range.end))
        self.assertEqual(2, member_range.full_count)
        self.assertFalse(member_range.has_partial)

    def test_partial_card_does_not_block_scrolling_to_next_member(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = [MemberViewState(name=f"Member {index}") for index in range(1, 6)]
        renderer._estimated_member_card_height = lambda state: 10
        renderer._viewport.set_total_count(len(members))

        first_range = renderer._visible_member_window(members, card_budget=35)
        renderer._viewport.window_size = first_range.full_count
        renderer._viewport.focus_index = 3

        self.assertTrue(renderer._viewport.move_focus(1))
        next_range = renderer._visible_member_window(members, card_budget=35)

        self.assertEqual(4, renderer._viewport.focus_index)
        self.assertEqual(len(members), next_range.end)
        self.assertGreaterEqual(renderer._viewport.focus_index, next_range.start)
        self.assertLess(renderer._viewport.focus_index, next_range.end)

    def test_focused_partial_member_is_shifted_into_full_range(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = [MemberViewState(name=f"Member {index}") for index in range(1, 6)]
        renderer._estimated_member_card_height = lambda state: 10
        renderer._viewport.set_total_count(len(members))
        renderer._viewport.focus_index = 3

        member_range = renderer._visible_member_window(members, card_budget=35)

        self.assertEqual(3, renderer._viewport.focus_index)
        self.assertLess(renderer._viewport.focus_index, member_range.full_end)
        self.assertEqual((1, 5), (member_range.start, member_range.end))
        self.assertEqual(4, member_range.partial_index)

    def test_last_member_focus_is_fully_visible_not_only_partial(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = [MemberViewState(name=f"Member {index}") for index in range(1, 6)]
        renderer._estimated_member_card_height = lambda state: 10
        renderer._viewport.set_total_count(len(members))
        renderer._viewport.focus_index = 4

        member_range = renderer._visible_member_window(members, card_budget=35)

        self.assertEqual(len(members), member_range.end)
        self.assertLess(renderer._viewport.focus_index, member_range.full_end)
        self.assertIsNone(member_range.partial_index)

    def test_last_scroll_start_includes_partial_card_when_space_remains(self) -> None:
        renderer = RichRenderer(rich_module=object())
        members = [MemberViewState(name=f"Member {index}") for index in range(1, 5)]
        renderer._estimated_member_card_height = lambda state: 10

        start = renderer._last_scroll_start(members, card_budget=25)

        self.assertEqual(1, start)

    def test_seeded_members_can_scroll_to_last_member_on_short_terminal(self) -> None:
        renderer = RichRenderer(rich_module=object())
        renderer._console = SimpleNamespace(size=SimpleNamespace(width=120, height=24))
        renderer.context = RunContext(
            question="Pick dinner",
            member_count=5,
            diversity_mode="balanced",
            secretary_mode="model",
            discussion_rounds=2,
            runoff_round_limit=3,
            web_search_enabled=False,
        )
        renderer.seed_members(self._members(5))
        members = renderer._ordered_members()
        renderer._viewport.set_total_count(len(members))
        renderer._viewport.go_end()

        start, end = renderer._visible_member_range(members, renderer._member_card_budget())

        self.assertEqual(5, len(renderer._member_state))
        self.assertLess(start, end)
        self.assertEqual(len(members), end)

    def test_final_decision_is_added_to_secretary_history_with_close_prompt(self) -> None:
        class DummyLive:
            def __init__(self) -> None:
                self.renderable = None
                self.refresh_calls = 0

            def update(self, renderable, refresh=False):
                self.renderable = renderable

            def refresh(self):
                self.refresh_calls += 1

            def stop(self):
                return None

        renderer = RichRenderer(rich_module=object())
        live = DummyLive()
        renderer._live = live
        renderer._render = lambda: "renderable"

        renderer.final_decision({"final_output": "Watch Train Dreams tonight."})
        renderer.close()

        history = "\n".join(renderer._secretary_history)
        self.assertIn("Final decision: Watch Train Dreams tonight.", history)
        self.assertIn("Press Esc or Enter to close.", history)
        self.assertEqual(1, live.refresh_calls)

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
        renderer_stderr = io.StringIO()
        renderer = PlainRenderer(stdout=io.StringIO(), stderr=renderer_stderr)
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
        self.assertIn("Secretary", renderer_stderr.getvalue())
        self.assertIn("Final decision: Watch Train Dreams tonight.", renderer_stderr.getvalue())

    def test_cli_suppresses_duplicate_final_answer_when_renderer_owns_final_output(self) -> None:
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
        renderer_stderr = io.StringIO()
        renderer = PlainRenderer(stdout=io.StringIO(), stderr=renderer_stderr)
        renderer.suppresses_final_stdout = True
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
        self.assertEqual("", stdout.getvalue())
        self.assertIn("Final decision: Watch Train Dreams tonight.", renderer_stderr.getvalue())

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
        self.assertIn("Final decision: Watch Train Dreams tonight.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
