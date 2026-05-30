from __future__ import annotations

import argparse
import os
import json
import select
import sys
import time
import threading
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TextIO


@dataclass(frozen=True)
class RunContext:
    question: str
    member_count: int
    diversity_mode: str
    secretary_mode: str
    discussion_rounds: int
    runoff_round_limit: int
    web_search_enabled: bool
    phase: str = "starting"
    secretary_verbosity: str = "balanced"


@dataclass(frozen=True)
class CouncilEvent:
    timestamp: datetime
    source: str
    source_type: Literal["system", "secretary", "member"]
    event_type: str
    phase: str | None
    message: str
    member_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemberViewState:
    name: str
    role: str = "Member"
    status: str = "queued"
    model: str = ""
    lane: str = ""
    phase: str = ""
    latest_activity: str = ""
    latest_detail: str = ""
    events: list[str] = field(default_factory=list)

    def add_event(self, line: str, limit: int = 5) -> None:
        self.events.append(line)
        if len(self.events) > limit:
            del self.events[:-limit]


@dataclass(frozen=True)
class MemberRange:
    start: int
    end: int
    full_count: int
    has_partial: bool = False

    @property
    def full_end(self) -> int:
        return self.start + self.full_count

    @property
    def partial_index(self) -> int | None:
        if not self.has_partial:
            return None
        return self.end - 1


@dataclass
class MemberViewport:
    total_count: int = 0
    window_size: int = 1
    focus_index: int = 0
    top_index: int = 0

    def set_total_count(self, total_count: int) -> None:
        self.total_count = max(0, int(total_count))
        if self.total_count == 0:
            self.focus_index = 0
            self.top_index = 0
            return
        self.focus_index = max(0, min(self.focus_index, self.total_count - 1))
        self._ensure_visible()

    def set_window_size(self, window_size: int) -> None:
        self.window_size = max(1, int(window_size))
        self._ensure_visible()

    def move_focus(self, delta: int) -> bool:
        if self.total_count == 0:
            return False
        next_index = max(0, min(self.total_count - 1, self.focus_index + int(delta)))
        changed = next_index != self.focus_index
        self.focus_index = next_index
        self._ensure_visible()
        return changed

    def jump(self, delta_pages: int) -> bool:
        step = max(1, self.window_size - 1)
        return self.move_focus(step * int(delta_pages))

    def go_home(self) -> bool:
        if self.total_count == 0:
            return False
        changed = self.focus_index != 0
        self.focus_index = 0
        self._ensure_visible()
        return changed

    def go_end(self) -> bool:
        if self.total_count == 0:
            return False
        changed = self.focus_index != self.total_count - 1
        self.focus_index = self.total_count - 1
        self._ensure_visible()
        return changed

    def visible_range(self) -> tuple[int, int]:
        end = min(self.total_count, self.top_index + self.window_size)
        return self.top_index, end

    def visible_count(self) -> int:
        start, end = self.visible_range()
        return max(0, end - start)

    def _ensure_visible(self) -> None:
        if self.total_count == 0:
            self.top_index = 0
            return
        if self.focus_index < self.top_index:
            self.top_index = self.focus_index
        elif self.focus_index >= self.top_index + self.window_size:
            self.top_index = self.focus_index - self.window_size + 1
        self._clamp_top()

    def _clamp_top(self) -> None:
        if self.total_count == 0:
            self.top_index = 0
            return
        max_top = max(0, self.total_count - self.window_size)
        self.top_index = max(0, min(self.top_index, max_top))


class BaseRenderer:
    suppresses_final_stdout = False
    _secretary_history_limit = 100

    def __init__(self, stdout: TextIO | None = None, stderr: TextIO | None = None) -> None:
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.context: RunContext | None = None
        self._started_at = time.monotonic()
        self._secretary_history: list[str] = []
        self._member_state: dict[str, MemberViewState] = {}
        self._final_payload: dict[str, Any] | None = None

    def start_run(self, context: RunContext) -> None:
        self.context = context

    def seed_members(self, members: list[Any]) -> None:
        for member in members:
            state = self._member_state.setdefault(member.name, MemberViewState(name=member.name))
            state.role = "President" if member.is_president else "Member"
            state.model = str(member.model)
            state.status = "queued"
            state.phase = "starting"

    def update_phase(self, phase: str) -> None:
        if self.context:
            self.context = RunContext(**{**self.context.__dict__, "phase": phase})
        for state in self._member_state.values():
            state.phase = phase
        self._add_secretary_line(f"Phase: {phase}")

    def secretary_message(self, message: str, event_type: str = "milestone") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        lines = [line.rstrip() for line in str(message).splitlines() if line.strip()]
        if not lines:
            return
        self._add_secretary_line(f"[{timestamp}] {lines[0]}")
        for line in lines[1:]:
            self._add_secretary_line(line)
        self._render_secretary_message(timestamp, lines)

    def member_event(
        self,
        member_name: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        state = self._member_state.setdefault(member_name, MemberViewState(name=member_name))
        payload = payload or {}
        if payload.get("role"):
            state.role = str(payload["role"])
        if payload.get("model"):
            state.model = str(payload["model"])
        if payload.get("lane"):
            state.lane = str(payload["lane"])
        if payload.get("phase"):
            state.phase = str(payload["phase"])
        if payload.get("status"):
            state.status = str(payload["status"])
        if event_type == "proposal_ready":
            state.latest_activity = "proposal ready"
            state.latest_detail = str(message).strip()
        elif event_type == "discussion_reply":
            state.latest_activity = "discussion reply received"
            state.latest_detail = str(message).strip()
        elif event_type == "final_proposal":
            state.latest_activity = "final proposal ready"
            state.latest_detail = str(message).strip()
        elif event_type == "vote":
            state.latest_activity = "vote recorded"
            state.latest_detail = str(message).strip()
        elif event_type == "lane_assigned":
            state.latest_activity = "diversity lane assigned"
            state.latest_detail = str(message).strip()
        else:
            state.latest_activity = event_type.replace("_", " ")
            state.latest_detail = str(message).strip()
        state.add_event(f"{datetime.now().strftime('%H:%M:%S')} {state.latest_activity}")
        self._render_member_event(member_name, event_type, message, payload)

    def member_status(self, member_name: str, status: str) -> None:
        state = self._member_state.setdefault(member_name, MemberViewState(name=member_name))
        state.status = str(status)
        state.add_event(f"{datetime.now().strftime('%H:%M:%S')} status: {status}")
        self._render_member_status(member_name, status)

    def render_members(self, members: list[Any]) -> None:
        raise NotImplementedError

    def render_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def final_decision(self, payload: dict[str, Any]) -> None:
        self._final_payload = dict(payload)

    def error(self, message: str) -> None:
        print(message, file=self.stderr)

    def close(self) -> None:
        return None

    def _add_secretary_line(self, line: str) -> None:
        self._secretary_history.append(line)
        self._secretary_history = self._secretary_history[-self._secretary_history_limit:]

    def _render_secretary_message(self, timestamp: str, lines: list[str]) -> None:
        raise NotImplementedError

    def _render_member_event(
        self, member_name: str, event_type: str, message: str, payload: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def _render_member_status(self, member_name: str, status: str) -> None:
        raise NotImplementedError


class PlainRenderer(BaseRenderer):
    def start_run(self, context: RunContext) -> None:
        super().start_run(context)
        print("Small Council", file=self.stderr)
        print(
            f"Request: {context.question}",
            file=self.stderr,
        )
        print(
            "Members: "
            f"{context.member_count} | Diversity: {context.diversity_mode} | "
            f"Secretary: {context.secretary_mode}/{context.secretary_verbosity} | "
            f"Search: {'enabled' if context.web_search_enabled else 'disabled'}",
            file=self.stderr,
        )
        print(f"Phase: {context.phase}", file=self.stderr)
        print(file=self.stderr)

    def _render_secretary_message(self, timestamp: str, lines: list[str]) -> None:
        print(f"[{timestamp}] Secretary", file=self.stderr)
        for line in lines:
            print(line, file=self.stderr)
        print(file=self.stderr, flush=True)

    def _render_member_event(
        self, member_name: str, event_type: str, message: str, payload: dict[str, Any]
    ) -> None:
        return None

    def _render_member_status(self, member_name: str, status: str) -> None:
        return None

    def final_decision(self, payload: dict[str, Any]) -> None:
        super().final_decision(payload)
        self.secretary_message("\n".join(final_decision_announcement_lines(payload)))

    def render_members(self, members: list[Any]) -> None:
        print(render_members_text(members), file=self.stdout)

    def render_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        print(render_leaderboard_text(rows), file=self.stdout)


class RichRenderer(BaseRenderer):
    suppresses_final_stdout = True
    _refresh_interval = 0.08

    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        rich_module: Any | None = None,
        force_terminal: bool = False,
        enable_keyboard: bool | None = None,
    ) -> None:
        super().__init__(stdout=stdout, stderr=stderr)
        self._rich = rich_module
        self._force_terminal = force_terminal
        self._console = None
        self._live = None
        self._viewport = MemberViewport()
        self._keyboard_enabled = enable_keyboard
        self._keyboard_thread: threading.Thread | None = None
        self._keyboard_stop = threading.Event()
        self._stdin_fd: int | None = None
        self._stdin_termios: list[Any] | None = None
        self._keyboard_controls = "Keys: Left/Right switch, Up/Down scroll, Esc/Enter close at end"
        self._active_section = "members"
        self._secretary_scroll_offset = 0
        self._last_refresh_at = 0.0
        self._refresh_pending = False
        self._awaiting_final_key = False
        self._final_key_pressed = threading.Event()

    def start_run(self, context: RunContext) -> None:
        super().start_run(context)
        rich = self._rich or _load_rich()
        self._rich = rich
        live_stream = self._live_stream()
        self._console = rich.console.Console(
            file=live_stream,
            force_terminal=self._force_terminal,
            color_system="auto",
            soft_wrap=True,
            highlight=False,
            width=None,
        )
        self._live = rich.live.Live(
            self._render(),
            console=self._console,
            auto_refresh=False,
            screen=False,
            transient=False,
        )
        self._live.start()
        self._maybe_start_keyboard_loop()

    def close(self) -> None:
        self._wait_for_final_key()
        self._stop_keyboard_loop()
        if self._live is not None:
            if self._refresh_pending:
                self._live.refresh()
            self._live.stop()
            self._live = None

    def secretary_message(self, message: str, event_type: str = "milestone") -> None:
        super().secretary_message(message, event_type=event_type)
        self._request_refresh()

    def update_phase(self, phase: str) -> None:
        super().update_phase(phase)
        self._request_refresh()

    def member_event(
        self,
        member_name: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().member_event(member_name, event_type, message, payload)
        self._request_refresh()

    def member_status(self, member_name: str, status: str) -> None:
        super().member_status(member_name, status)
        self._request_refresh()

    def seed_members(self, members: list[Any]) -> None:
        super().seed_members(members)
        self._request_refresh(force=True)

    def render_members(self, members: list[Any]) -> None:
        rich = self._rich or _load_rich()
        if self._console is None:
            self._console = rich.console.Console(file=self.stdout, force_terminal=True, width=None)
        table = self._members_table(members)
        self._console.print(table)

    def render_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        rich = self._rich or _load_rich()
        if self._console is None:
            self._console = rich.console.Console(file=self.stdout, force_terminal=True, width=None)
        table = self._leaderboard_table(rows)
        self._console.print(table)

    def final_decision(self, payload: dict[str, Any]) -> None:
        super().final_decision(payload)
        for line in final_decision_announcement_lines(payload):
            self._add_secretary_line(line)
        self._add_secretary_line("Press Esc or Enter to close.")
        self._awaiting_final_key = True
        self._final_key_pressed.clear()
        self._request_refresh(force=True)

    def error(self, message: str) -> None:
        if self._console is not None:
            self._console.print(f"[red]{message}[/red]")
        else:
            super().error(message)

    def _render_secretary_message(self, timestamp: str, lines: list[str]) -> None:
        return None

    def _render_member_event(
        self, member_name: str, event_type: str, message: str, payload: dict[str, Any]
    ) -> None:
        return None

    def _render_member_status(self, member_name: str, status: str) -> None:
        return None

    def _request_refresh(self, force: bool = False) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=False)
            now = time.monotonic()
            if force or self._last_refresh_at == 0.0 or now - self._last_refresh_at >= self._refresh_interval:
                self._live.refresh()
                self._last_refresh_at = now
                self._refresh_pending = False
            else:
                self._refresh_pending = True

    def _add_secretary_line(self, line: str) -> None:
        was_scrolled = self._secretary_scroll_offset > 0
        super()._add_secretary_line(line)
        if was_scrolled:
            self._secretary_scroll_offset += self._wrapped_line_count(
                line, self._secretary_content_width()
            )
        self._clamp_secretary_scroll()

    def _render(self):
        rich = self._rich or _load_rich()
        layout = rich.layout.Layout()
        layout.split_column(
            rich.layout.Layout(self._render_header(), name="header", size=5),
            rich.layout.Layout(name="body", ratio=1),
        )
        body = layout["body"]
        if self._should_stack_vertically():
            body.split_column(
                rich.layout.Layout(self._render_secretary_panel(), name="secretary", size=9),
                rich.layout.Layout(self._render_member_stack(), name="members", ratio=1),
            )
        else:
            left_size = self._left_column_width()
            body.split_row(
                rich.layout.Layout(self._render_secretary_panel(), name="secretary", size=left_size),
                rich.layout.Layout(self._render_member_stack(), name="members", ratio=1),
            )
        return layout

    def _render_header(self):
        rich = self._rich or _load_rich()
        text = rich.text.Text()
        if self.context is not None:
            text.append("Small Council\n", style="bold")
            text.append(f"Request: {self.context.question}\n")
            text.append(
                "Members: "
                f"{self.context.member_count} | Diversity: {self.context.diversity_mode} | "
                f"Secretary: {self.context.secretary_mode}/{self.context.secretary_verbosity} | "
                f"Search: {'enabled' if self.context.web_search_enabled else 'disabled'}\n"
            )
            text.append(f"Phase: {self.context.phase}")
        else:
            text.append("Small Council", style="bold")
        return rich.panel.Panel(text, border_style="cyan", padding=(0, 1))

    def _render_secretary_panel(self):
        rich = self._rich or _load_rich()
        body = rich.text.Text()
        width = self._secretary_content_width()
        content_height = self._secretary_content_height()
        used_lines = 1
        body.append("Secretary\n", style="bold")
        if self.context is not None:
            self._append_wrapped(body, f"Phase: {self.context.phase}", width, style="dim")
            used_lines += self._wrapped_line_count(f"Phase: {self.context.phase}", width)
        if self._secretary_history:
            history_budget = max(1, content_height - used_lines)
            for line in self._visible_secretary_history_lines(width, history_budget):
                body.append(f"{line}\n")
        else:
            body.append("Waiting for updates...")
        border = "yellow" if self._active_section == "secretary" else "magenta"
        title = "Secretary active" if self._active_section == "secretary" else None
        return rich.panel.Panel(body, border_style=border, title=title, padding=(0, 1))

    def _render_member_stack(self):
        rich = self._rich or _load_rich()
        members = self._ordered_members()
        self._viewport.set_total_count(len(members))
        card_budget = self._member_card_budget()
        member_range = self._visible_member_window(members, card_budget)
        start, end = member_range.start, member_range.end
        window_size = max(1, member_range.full_count)
        self._viewport.window_size = window_size
        visible = members[start:end]
        cards: list[Any] = []
        if members:
            status = f"Members {start + 1}-{end} of {len(members)}"
            if len(members) > window_size:
                status += f" | {self._keyboard_controls}"
            cards.append(rich.text.Text(status, style="bold"))
        else:
            cards.append(rich.text.Text("No member activity yet.", style="dim"))
        for offset, state in enumerate(visible, start=start):
            cards.append(self._render_member_card(state, focused=(offset == self._viewport.focus_index)))
        if not visible and not members:
            cards.append(rich.panel.Panel("Waiting for council members...", border_style="green"))
        border = "yellow" if self._active_section == "members" else "green"
        title = "Members active" if self._active_section == "members" else None
        return rich.panel.Panel(rich.console.Group(*cards), border_style=border, title=title, padding=(0, 1))

    def _render_member_card(self, state: MemberViewState, focused: bool = False):
        rich = self._rich or _load_rich()
        text = rich.text.Text()
        width = self._member_card_content_width()
        text.append(f"{state.name}\n", style="bold")
        self._append_wrapped(text, f"Role: {state.role}", width)
        if state.model:
            self._append_wrapped(text, f"Model: {state.model}", width)
        if state.lane:
            self._append_wrapped(text, f"Lane: {state.lane}", width)
        self._append_wrapped(text, f"Status: {state.status}", width)
        phase = state.phase or (self.context.phase if self.context else "")
        if phase:
            self._append_wrapped(text, f"Phase: {phase}", width)
        self._append_wrapped(
            text,
            f"Latest: {state.latest_detail or state.latest_activity or 'No activity yet.'}",
            width,
        )
        text.append("Events:\n")
        if state.events:
            for item in state.events[-3:]:
                self._append_wrapped(text, item, width)
        else:
            text.append("Waiting for activity\n")
        border = "yellow" if focused else "blue"
        title = f"> {state.name} <" if focused else state.name
        return rich.panel.Panel(text, border_style=border, title=title, padding=(0, 1))

    def _ordered_members(self) -> list[MemberViewState]:
        members = list(self._member_state.values())
        members.sort(key=lambda item: item.name.lower())
        return members

    def _should_stack_vertically(self) -> bool:
        if self._console is None:
            return False
        width = getattr(self._console.size, "width", 0) or 0
        return width < 100

    def _left_column_width(self) -> int:
        if self._console is None:
            return 40
        width = max(80, getattr(self._console.size, "width", 80) or 80)
        target_ratio = 0.3 if self.context and self.context.secretary_mode == "model" else 0.2
        return max(24, min(width - 32, int(width * target_ratio)))

    def _body_height(self) -> int:
        if self._console is None:
            return 24
        height = getattr(self._console.size, "height", 24) or 24
        return max(12, height - 7)

    def _member_stack_height(self) -> int:
        if self._should_stack_vertically():
            return max(4, self._body_height() - 9)
        return self._body_height()

    def _member_card_budget(self) -> int:
        return max(1, self._member_stack_height() - 3)

    def _visible_member_range(self, members: list[MemberViewState], card_budget: int) -> tuple[int, int]:
        member_range = self._visible_member_window(members, card_budget)
        return member_range.start, member_range.end

    def _visible_member_window(self, members: list[MemberViewState], card_budget: int) -> MemberRange:
        if not members:
            return MemberRange(0, 0, 1)
        self._viewport.set_total_count(len(members))
        self._viewport._clamp_top()
        start = self._viewport.top_index
        if self._viewport.focus_index < start:
            start = self._viewport.focus_index

        member_range = self._fit_members_from(start, members, card_budget)
        while (
            not self._member_range_fully_contains_focus(member_range)
            and start < self._viewport.focus_index
        ):
            start += 1
            member_range = self._fit_members_from(start, members, card_budget)

        max_start = self._last_scroll_start(members, card_budget)
        start = min(start, max_start)
        if self._viewport.focus_index < start:
            start = self._viewport.focus_index
        member_range = self._fit_members_from(start, members, card_budget)
        while (
            not self._member_range_fully_contains_focus(member_range)
            and start < self._viewport.focus_index
        ):
            start += 1
            member_range = self._fit_members_from(start, members, card_budget)
        self._viewport.top_index = start
        return member_range

    def _member_range_fully_contains_focus(self, member_range: MemberRange) -> bool:
        return member_range.start <= self._viewport.focus_index < member_range.full_end

    def _fit_members_from(
        self, start: int, members: list[MemberViewState], card_budget: int
    ) -> MemberRange:
        used = 0
        end = start
        full_count = 0
        has_partial = False
        for state in members[start:]:
            height = self._estimated_member_card_height(state)
            if end > start and used + height > card_budget:
                if used < card_budget:
                    end += 1
                    has_partial = True
                break
            used += height
            end += 1
            full_count += 1
            if used >= card_budget:
                break
        end = max(start + 1, end)
        return MemberRange(start, end, max(1, full_count), has_partial)

    def _last_scroll_start(self, members: list[MemberViewState], card_budget: int) -> int:
        used = 0
        start = len(members) - 1
        for index in range(len(members) - 1, -1, -1):
            height = self._estimated_member_card_height(members[index])
            if index < len(members) - 1 and used + height > card_budget:
                if used < card_budget:
                    start = index
                break
            used += height
            start = index
            if used >= card_budget:
                break
        return max(0, start)

    def _estimated_member_card_height(self, state: MemberViewState) -> int:
        width = self._member_card_content_width()
        lines = 1
        lines += self._wrapped_line_count(f"Role: {state.role}", width)
        if state.model:
            lines += self._wrapped_line_count(f"Model: {state.model}", width)
        if state.lane:
            lines += self._wrapped_line_count(f"Lane: {state.lane}", width)
        lines += self._wrapped_line_count(f"Status: {state.status}", width)
        phase = state.phase or (self.context.phase if self.context else "")
        if phase:
            lines += self._wrapped_line_count(f"Phase: {phase}", width)
        lines += self._wrapped_line_count(
            f"Latest: {state.latest_detail or state.latest_activity or 'No activity yet.'}",
            width,
        )
        lines += 1
        if state.events:
            for item in state.events[-3:]:
                lines += self._wrapped_line_count(item, width)
        else:
            lines += 1
        return lines + 2

    def _secretary_content_width(self) -> int:
        if self._should_stack_vertically():
            return self._terminal_width() - 6
        return self._left_column_width() - 4

    def _secretary_content_height(self) -> int:
        if self._should_stack_vertically():
            panel_height = 9
        else:
            panel_height = self._body_height()
        return max(1, panel_height - 2)

    def _visible_secretary_history_lines(self, width: int, max_lines: int) -> list[str]:
        visual_lines = self._secretary_visual_lines(width)
        if not visual_lines:
            return []
        max_lines = max(1, int(max_lines))
        max_offset = max(0, len(visual_lines) - max_lines)
        offset = max(0, min(self._secretary_scroll_offset, max_offset))
        self._secretary_scroll_offset = offset
        end = len(visual_lines) - offset
        start = max(0, end - max_lines)
        return visual_lines[start:end]

    def _secretary_visual_lines(self, width: int) -> list[str]:
        visual_lines: list[str] = []
        for line in self._secretary_history:
            visual_lines.extend(self._wrap_line(line, width))
        return visual_lines

    def _secretary_history_budget(self) -> int:
        width = self._secretary_content_width()
        used_lines = 1
        if self.context is not None:
            used_lines += self._wrapped_line_count(f"Phase: {self.context.phase}", width)
        return max(1, self._secretary_content_height() - used_lines)

    def _max_secretary_scroll_offset(self) -> int:
        return max(
            0,
            len(self._secretary_visual_lines(self._secretary_content_width()))
            - self._secretary_history_budget(),
        )

    def _clamp_secretary_scroll(self) -> None:
        self._secretary_scroll_offset = max(
            0, min(self._secretary_scroll_offset, self._max_secretary_scroll_offset())
        )

    def _scroll_secretary(self, delta: int) -> bool:
        previous = self._secretary_scroll_offset
        self._secretary_scroll_offset = max(
            0, min(self._secretary_scroll_offset + int(delta), self._max_secretary_scroll_offset())
        )
        return self._secretary_scroll_offset != previous

    def _member_card_content_width(self) -> int:
        width = self._terminal_width()
        if not self._should_stack_vertically():
            width -= self._left_column_width()
        return max(12, width - 8)

    def _terminal_width(self) -> int:
        if self._console is None:
            return 80
        return max(40, getattr(self._console.size, "width", 80) or 80)

    def _append_wrapped(self, text, line: str, width: int, style: str | None = None) -> None:
        for wrapped in self._wrap_line(line, width):
            text.append(f"{wrapped}\n", style=style)

    def _wrapped_line_count(self, line: str, width: int) -> int:
        return len(self._wrap_line(line, width))

    def _wrap_line(self, line: str, width: int) -> list[str]:
        width = max(10, int(width))
        return textwrap.wrap(
            str(line),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        ) or [""]

    def _maybe_start_keyboard_loop(self) -> None:
        if self._keyboard_enabled is False:
            return
        if self._keyboard_enabled is None and not self._stdin_isatty():
            return
        if not self._stdin_isatty():
            return
        if self._keyboard_thread is not None:
            return
        try:
            import termios
            import tty
        except ModuleNotFoundError:
            return
        try:
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            self._stdin_fd = None
            self._stdin_termios = None
            return
        self._keyboard_stop.clear()
        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()

    def _stop_keyboard_loop(self) -> None:
        self._keyboard_stop.set()
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=0.5)
            self._keyboard_thread = None
        if self._stdin_fd is not None and self._stdin_termios is not None:
            try:
                import termios

                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_termios)
            except Exception:
                pass
        self._stdin_fd = None
        self._stdin_termios = None

    def _keyboard_loop(self) -> None:
        if self._stdin_fd is None:
            return
        while not self._keyboard_stop.is_set():
            ready, _, _ = select.select([self._stdin_fd], [], [], 0.1)
            if not ready:
                continue
            try:
                chunk = os.read(self._stdin_fd, 32)
            except OSError:
                break
            if not chunk:
                continue
            if self._handle_key_chunk(chunk):
                self._request_refresh()
            if self._awaiting_final_key and self._final_key_pressed.is_set():
                break

    def _wait_for_final_key(self) -> None:
        if not self._awaiting_final_key:
            return
        if self._keyboard_thread is None or self._stdin_fd is None:
            self._awaiting_final_key = False
            return
        while not self._final_key_pressed.wait(timeout=0.1):
            if self._keyboard_thread is None or not self._keyboard_thread.is_alive():
                break
        self._awaiting_final_key = False

    def _handle_key_chunk(self, chunk: bytes) -> bool:
        changed = False
        text = chunk.decode("latin1", errors="ignore")
        index = 0
        while index < len(text):
            key = self._next_key(text, index)
            if key is None:
                index += 1
                continue
            action, consumed = key
            index += consumed
            if action == "exit":
                if self._awaiting_final_key:
                    self._final_key_pressed.set()
                continue
            if action == "left":
                changed = self._set_active_section("secretary") or changed
            elif action == "right":
                changed = self._set_active_section("members") or changed
            elif action == "up":
                changed = self._scroll_active_section(-1) or changed
            elif action == "down":
                changed = self._scroll_active_section(1) or changed
            elif self._active_section == "members" and action in {"page_up", "home"}:
                changed = self._viewport.go_home() or changed
            elif self._active_section == "members" and action in {"page_down", "end"}:
                changed = self._viewport.go_end() or changed
            elif self._active_section == "secretary" and action in {"page_up", "home"}:
                changed = self._jump_secretary_to_start() or changed
            elif self._active_section == "secretary" and action in {"page_down", "end"}:
                changed = self._jump_secretary_to_end() or changed
        return changed

    def _set_active_section(self, section: str) -> bool:
        if section == self._active_section:
            return False
        self._active_section = section
        return True

    def _scroll_active_section(self, delta: int) -> bool:
        if self._active_section == "secretary":
            return self._scroll_secretary(-delta)
        return self._viewport.move_focus(delta)

    def _jump_secretary_to_start(self) -> bool:
        previous = self._secretary_scroll_offset
        self._secretary_scroll_offset = self._max_secretary_scroll_offset()
        return self._secretary_scroll_offset != previous

    def _jump_secretary_to_end(self) -> bool:
        previous = self._secretary_scroll_offset
        self._secretary_scroll_offset = 0
        return self._secretary_scroll_offset != previous

    def _next_key(self, text: str, index: int) -> tuple[str, int] | None:
        chunk = text[index:]
        if not chunk:
            return None
        if chunk.startswith("\x1b[A"):
            return "up", 3
        if chunk.startswith("\x1b[B"):
            return "down", 3
        if chunk.startswith("\x1b[C"):
            return "right", 3
        if chunk.startswith("\x1b[D"):
            return "left", 3
        if chunk.startswith("\x1b[5~"):
            return "page_up", 4
        if chunk.startswith("\x1b[6~"):
            return "page_down", 4
        if chunk.startswith("\x1b[H"):
            return "home", 3
        if chunk.startswith("\x1b[F"):
            return "end", 3
        char = chunk[0]
        if char == "\x1b":
            return "exit", 1
        if char in {"\r", "\n"}:
            return "exit", 1
        if char in {"k", "K"}:
            return "up", 1
        if char in {"j", "J"}:
            return "down", 1
        if char in {"u", "U"}:
            return "page_up", 1
        if char in {"d", "D"}:
            return "page_down", 1
        if char in {"h", "H"}:
            return "home", 1
        if char in {"e", "E"}:
            return "end", 1
        return None

    def _stdin_isatty(self) -> bool:
        try:
            return bool(getattr(sys.stdin, "isatty", lambda: False)())
        except Exception:
            return False

    def _live_stream(self) -> TextIO:
        if self._is_tty(self.stderr):
            return self.stderr
        if self._is_tty(self.stdout):
            return self.stdout
        return self.stderr

    def _is_tty(self, stream: TextIO) -> bool:
        try:
            return bool(getattr(stream, "isatty", lambda: False)())
        except Exception:
            return False

    def _members_table(self, members: list[Any]):
        rich = self._rich or _load_rich()
        table = rich.table.Table(title="Council Members", expand=True, box=rich.box.SIMPLE)
        for header in ["Name", "Role", "Provider", "Model", "Personality", "Wins", "Proposals", "Votes", "Tie Breaks"]:
            table.add_column(header, justify="right" if header in {"Wins", "Proposals", "Votes", "Tie Breaks"} else "left")
        for member in members:
            table.add_row(
                member.name,
                "President" if member.is_president else "Member",
                getattr(member, "provider", "codex"),
                member.model,
                member.personality,
                str(member.total_wins),
                str(member.total_proposals),
                str(member.total_votes_cast),
                str(member.tie_break_victories),
            )
        return table

    def _leaderboard_table(self, rows: list[dict[str, Any]]):
        rich = self._rich or _load_rich()
        table = rich.table.Table(title="Leaderboard", expand=True, box=rich.box.SIMPLE)
        for header in ["Rank", "Member", "Role", "Wins", "Proposals", "Win Rate", "Votes", "Tie Breaks", "Provider", "Model"]:
            table.add_column(header, justify="right" if header in {"Rank", "Wins", "Proposals", "Votes", "Tie Breaks"} else "left")
        for index, row in enumerate(rows, start=1):
            table.add_row(
                str(index),
                row["member"],
                "President" if row["president"] else "Member",
                str(row["total_wins"]),
                str(row["total_proposals"]),
                f"{row['win_rate']:.0%}",
                str(row["vote_participation"]),
                str(row["tie_break_victories"]),
                row.get("provider", "codex"),
                row["model"],
            )
        return table


def select_renderer(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> BaseRenderer | None:
    if getattr(args, "json_output", False):
        return None
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if getattr(args, "plain_output", False):
        return PlainRenderer(stdout=stdout, stderr=stderr)
    try:
        rich = _load_rich()
    except RuntimeError:
        return PlainRenderer(stdout=stdout, stderr=stderr)
    return RichRenderer(stdout=stdout, stderr=stderr, rich_module=rich, force_terminal=True)


def render_human_decision(payload: dict[str, Any]) -> str:
    return str(payload["final_output"]).rstrip()


def final_decision_announcement_lines(payload: dict[str, Any]) -> list[str]:
    final_output = render_human_decision(payload).strip()
    if not final_output:
        return ["Final decision complete."]
    final_lines = [line.strip() for line in final_output.splitlines() if line.strip()]
    return [f"Final decision: {final_lines[0]}", *final_lines[1:]]


def render_json_decision(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def render_members_text(members: list[Any]) -> str:
    rows = []
    for member in members:
        rows.append(
            {
                "Name": member.name,
                "Role": "President" if member.is_president else "Member",
                "Provider": getattr(member, "provider", "codex"),
                "Model": member.model,
                "Personality": member.personality,
                "Wins": member.total_wins,
                "Proposals": member.total_proposals,
                "Votes": member.total_votes_cast,
                "Tie Breaks": member.tie_break_victories,
            }
        )
    return "Council Members\n" + _format_table(
        ["Name", "Role", "Provider", "Model", "Personality", "Wins", "Proposals", "Votes", "Tie Breaks"],
        rows,
        right_align={"Wins", "Proposals", "Votes", "Tie Breaks"},
    )


def render_leaderboard_text(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for rank, row in enumerate(rows, start=1):
        rendered.append(
            {
                "Rank": rank,
                "Member": row["member"],
                "Role": "President" if row["president"] else "Member",
                "Wins": row["total_wins"],
                "Proposals": row["total_proposals"],
                "Win Rate": f"{row['win_rate']:.0%}",
                "Votes": row["vote_participation"],
                "Tie Breaks": row["tie_break_victories"],
                "Provider": row.get("provider", "codex"),
                "Model": row["model"],
            }
        )
    return "Leaderboard\n" + _format_table(
        ["Rank", "Member", "Role", "Wins", "Proposals", "Win Rate", "Votes", "Tie Breaks", "Provider", "Model"],
        rendered,
        right_align={"Rank", "Wins", "Proposals", "Votes", "Tie Breaks"},
    )


def _format_table(
    headers: list[str],
    rows: list[dict[str, Any]],
    right_align: set[str] | None = None,
) -> str:
    right_align = right_align or set()
    rendered_rows = [{header: str(row.get(header, "")) for header in headers} for row in rows]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rendered_rows))
        for header in headers
    }

    def render_row(row: dict[str, str]) -> str:
        cells = []
        for header in headers:
            value = row[header]
            if header in right_align:
                cells.append(value.rjust(widths[header]))
            else:
                cells.append(value.ljust(widths[header]))
        return " | ".join(cells)

    header_row = render_row({header: header for header in headers})
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = [render_row(row) for row in rendered_rows]
    return "\n".join([header_row, separator, *body])


def _load_rich():
    try:
        import rich
        import rich.box  # noqa: F401
        import rich.columns  # noqa: F401
        import rich.console  # noqa: F401
        import rich.layout  # noqa: F401
        import rich.live  # noqa: F401
        import rich.panel  # noqa: F401
        import rich.table  # noqa: F401
        import rich.text  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Rich output was requested but the `rich` package is not installed."
        ) from exc
    return rich
