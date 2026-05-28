from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import TextIO


@dataclass
class SecretaryState:
    question: str
    phase: str = "starting"
    completed_recommendations: list[dict[str, str]] = field(default_factory=list)
    completed_votes: list[dict[str, str]] = field(default_factory=list)
    vote_rounds: list[str] = field(default_factory=list)
    diversity_lanes: dict[str, str] = field(default_factory=dict)
    diversity_mode: str = "balanced"


class Secretary:
    """Local court reporter for long council runs.

    This intentionally does not call a model. It reports only orchestration
    state and model-visible outputs that have already completed.
    """

    def __init__(self, interval_seconds: int, stream: TextIO | None = None) -> None:
        self.interval_seconds = interval_seconds
        self.stream = stream or sys.stderr
        self.state: SecretaryState | None = None
        self._task: asyncio.Task | None = None

    async def start(self, question: str) -> None:
        self.state = SecretaryState(question=question)
        self._write_block(["Secretary", f"Request received: {str(question).strip()}"])
        self._task = asyncio.create_task(self._report_loop())

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def set_phase(self, phase: str) -> None:
        if self.state:
            self.state.phase = phase

    def recommendation_done(self, member: str, recommendation: str) -> None:
        if self.state:
            self.state.completed_recommendations.append(
                {"member": _clean_text(member), "recommendation": _clean_text(recommendation)}
            )

    def vote_done(self, member: str, selected_option: str, round_number: int) -> None:
        if self.state:
            self.state.completed_votes.append(
                {
                    "member": _clean_text(member),
                    "selected_option": _clean_text(selected_option),
                    "round_label": _round_label(round_number),
                }
            )

    def vote_round_done(self, description: str) -> None:
        if self.state:
            self.state.vote_rounds.append(_clean_text(description))

    def diversity_lanes_assigned(self, lanes: dict[str, str], mode: str) -> None:
        if not self.state:
            return
        self.state.diversity_lanes = dict(lanes)
        self.state.diversity_mode = mode
        lines = ["Secretary", f"Diversity mode: {_clean_text(mode)}", "Assigned lanes:"]
        for member, lane in lanes.items():
            lines.append(f"  - {_clean_text(member)}: {_clean_text(lane)}")
        self._write_block(lines)

    def grouping_done(self, groups: list[dict[str, object]]) -> None:
        merged = [group for group in groups if len(group.get("proposers", [])) > 1]
        if not merged:
            self._write_block(["Secretary", "Grouped proposals: no equivalent proposals found"])
            return
        lines = ["Secretary", "Grouped equivalent proposals:"]
        for group in merged:
            proposers = ", ".join(str(proposer) for proposer in group.get("proposers", []))
            lines.append(f"  - {_clean_text(str(group.get('canonical_option', '')))}")
            lines.append(f"    Suggested by: {proposers}")
            lines.append("    Original proposals:")
            original_by_member = zip(
                group.get("proposers", []), group.get("member_recommendations", [])
            )
            for proposer, recommendation in original_by_member:
                lines.append(
                    f"      - {_clean_text(str(proposer))}: {_clean_text(str(recommendation))}"
                )
        self._write_block(lines)

    def _write_block(self, lines: list[str]) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if not lines:
            return
        print(f"[{timestamp}] {lines[0]}", file=self.stream)
        for line in lines[1:]:
            print(line, file=self.stream)
        print(file=self.stream, flush=True)

    async def _report_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            self.report()

    def report(self) -> None:
        if not self.state:
            return
        lines = ["Secretary", f"Phase: {_clean_text(self.state.phase)}"]
        if self.state.completed_recommendations:
            lines.append("Proposals:")
            for item in self.state.completed_recommendations[-3:]:
                lines.append(f"  - {item['member']}: {item['recommendation']}")
        if self.state.completed_votes:
            lines.append("Votes:")
            for item in self.state.completed_votes[-5:]:
                lines.append(
                    f"  - {item['member']} -> {item['selected_option']} [{item['round_label']}]"
                )
        if self.state.vote_rounds:
            lines.append(f"Vote status: {self.state.vote_rounds[-1]}")
        self._write_block(lines)


def _clean_text(value: str) -> str:
    return str(value).strip().rstrip(".;:").strip()


def _round_label(round_number: int) -> str:
    return "initial vote" if round_number == 0 else f"runoff {round_number}"
