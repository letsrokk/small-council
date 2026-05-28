from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

from .config import ROOT
from .prompts import secretary_report_prompt


ModelReporter = Callable[[dict[str, Any], str, str, Path, str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class SecretaryConfig:
    mode: str = "model"
    model: str = "gpt-5.4-mini"
    verbosity: str = "balanced"
    immediate_updates: bool = True


@dataclass
class SecretaryState:
    question: str
    phase: str = "starting"
    draft_recommendations: list[dict[str, str]] = field(default_factory=list)
    final_recommendations: list[dict[str, str]] = field(default_factory=list)
    completed_votes: list[dict[str, str]] = field(default_factory=list)
    vote_rounds: list[str] = field(default_factory=list)
    diversity_lanes: dict[str, str] = field(default_factory=dict)
    diversity_mode: str = "balanced"
    discussion_rounds: list[dict[str, object]] = field(default_factory=list)


class BaseSecretary:
    def __init__(self, stream: TextIO | None = None, immediate_updates: bool = True) -> None:
        self.stream = stream or sys.stderr
        self.state: SecretaryState | None = None
        self._milestone: str = ""
        self._immediate_updates = immediate_updates

    async def start(self, question: str) -> None:
        self.state = SecretaryState(question=question)
        self._write_block(["Secretary", f"Request received: {str(question).strip()}"])

    async def stop(self) -> None:
        return None

    def set_phase(self, phase: str) -> None:
        if self.state:
            self.state.phase = phase

    def recommendation_done(self, member: str, recommendation: str) -> None:
        if self.state:
            self.state.draft_recommendations.append(
                {"member": _clean_text(member), "recommendation": _clean_text(recommendation)}
            )
            self._write_immediate_status(f"{_clean_text(member)} finished research.")

    def final_recommendation_done(self, member: str, recommendation: str) -> None:
        if self.state:
            self.state.final_recommendations.append(
                {"member": _clean_text(member), "recommendation": _clean_text(recommendation)}
            )
            self._write_immediate_status(f"{_clean_text(member)} finalized a proposal.")

    def vote_done(self, member: str, selected_option: str, round_number: int) -> None:
        if self.state:
            self.state.completed_votes.append(
                {
                    "member": _clean_text(member),
                    "selected_option": _clean_text(selected_option),
                    "round_label": _round_label(round_number),
                }
            )
            self._write_immediate_status(
                f"{_clean_text(member)} voted in the {_round_label(round_number)}."
            )

    def vote_round_done(self, description: str) -> None:
        if self.state:
            self.state.vote_rounds.append(_clean_text(description))
            self._write_status(_clean_text(description))

    def discussion_round_started(self, round_number: int) -> None:
        if self.state:
            self.state.discussion_rounds.append(
                {"round_number": round_number, "messages": [], "completed": False}
            )
            self._write_immediate_status(f"Discussion round {round_number} started.")

    def discussion_message_done(
        self,
        round_number: int,
        member: str,
        discussion_reply: str,
        prior_recommendation: str,
        revised_recommendation: str,
        changed: bool,
    ) -> None:
        if not self.state:
            return
        for round_state in self.state.discussion_rounds:
            if round_state.get("round_number") == round_number:
                round_state["messages"].append(
                    {
                        "member": _clean_text(member),
                        "discussion_reply": _clean_text(discussion_reply),
                        "prior_recommendation": _clean_text(prior_recommendation),
                        "revised_recommendation": _clean_text(revised_recommendation),
                        "changed": changed,
                    }
                )
                self._write_immediate_status(
                    f"{_clean_text(member)} finished discussion round {round_number}."
                )
                return

    def discussion_round_done(self, round_number: int) -> None:
        if not self.state:
            return
        for round_state in self.state.discussion_rounds:
            if round_state.get("round_number") == round_number:
                round_state["completed"] = True
                return

    def diversity_lanes_assigned(self, lanes: dict[str, str], mode: str) -> None:
        if not self.state:
            return
        self.state.diversity_lanes = dict(lanes)
        self.state.diversity_mode = mode
        self._write_immediate_status(f"Diversity lanes assigned for {_clean_text(mode)} mode.")

    def grouping_done(self, groups: list[dict[str, object]]) -> None:
        merged = [group for group in groups if len(group.get("proposers", [])) > 1]
        if not merged:
            self._write_immediate_status("Grouped proposals: no equivalent proposals found.")
            return
        self._write_immediate_status(f"Grouped {len(merged)} equivalent proposal set(s).")

    async def report_milestone(self, label: str) -> bool:
        self._milestone = _clean_text(label)
        return await self._emit_report()

    async def _emit_report(self) -> bool:
        raise NotImplementedError

    def _snapshot(self) -> dict[str, Any]:
        if not self.state:
            return {}
        snapshot = asdict(self.state)
        snapshot["milestone"] = self._milestone
        return snapshot

    def _write_block(self, lines: list[str]) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if not lines:
            return
        print(f"[{timestamp}] {lines[0]}", file=self.stream)
        for line in lines[1:]:
            print(line, file=self.stream)
        print(file=self.stream, flush=True)

    def _write_status(self, message: str) -> None:
        self._write_block(["Secretary", message])

    def _write_immediate_status(self, message: str) -> None:
        if self._immediate_updates:
            self._write_status(message)


class LocalSecretary(BaseSecretary):
    """Local court reporter for long council runs.

    This intentionally does not call a model. It reports only orchestration
    state and model-visible outputs that have already completed.
    """

    async def _emit_report(self) -> bool:
        lines = self._local_report_lines()
        if not lines:
            return False
        self._write_block(lines)
        return True

    def _local_report_lines(self) -> list[str]:
        if not self.state:
            return []
        lines = ["Secretary", f"Phase: {_clean_text(self.state.phase)}"]
        if self.state.draft_recommendations:
            lines.append("Draft proposals:")
            for item in self.state.draft_recommendations[-3:]:
                lines.append(f"  - {item['member']}: {item['recommendation']}")
        if self.state.discussion_rounds:
            latest = self.state.discussion_rounds[-1]
            lines.append(f"Discussion round {latest['round_number']}:")
            for message in latest["messages"][-3:]:
                changed = "revised" if message.get("changed") else "kept"
                lines.append(
                    f"  - {message.get('member')}: {changed} -> {message.get('revised_recommendation')}"
                )
                reply = str(message.get("discussion_reply") or "").strip()
                if reply:
                    lines.append(f"    reply: {reply}")
        if self.state.final_recommendations:
            lines.append("Final proposals:")
            for item in self.state.final_recommendations[-3:]:
                lines.append(f"  - {item['member']}: {item['recommendation']}")
        if self.state.completed_votes:
            lines.append("Votes:")
            for item in self.state.completed_votes[-5:]:
                lines.append(
                    f"  - {item['member']} -> {item['selected_option']} [{item['round_label']}]"
                )
        if self.state.vote_rounds:
            lines.append(f"Vote status: {self.state.vote_rounds[-1]}")
        return lines


class ModelBackedSecretary(LocalSecretary):
    def __init__(
        self,
        config: dict[str, Any],
        model: str,
        verbosity: str,
        stream: TextIO | None = None,
        model_reporter: ModelReporter | None = None,
        immediate_updates: bool = True,
    ) -> None:
        super().__init__(stream, immediate_updates=immediate_updates)
        self.config = config
        self.model = model
        self.verbosity = verbosity
        self.model_reporter = model_reporter or _default_model_reporter
        self._fallback_to_local = False
        self._warned_about_fallback = False

    async def _emit_report(self) -> bool:
        if self._fallback_to_local:
            return await super()._emit_report()
        try:
            payload = await self.model_reporter(
                self.config,
                self.model,
                secretary_report_prompt(
                    self.state.question if self.state else "",
                    self._snapshot(),
                    self.verbosity,
                    self._milestone,
                ),
                ROOT / "schemas" / "secretary-report.schema.json",
                f"secretary-{_phase_slug(self._milestone)}",
            )
            message = _clean_model_message(payload.get("message", ""))
            if not message:
                return await super()._emit_report()
            self._write_block(["Secretary", message])
            return True
        except Exception as exc:
            self._fallback_to_local = True
            if not self._warned_about_fallback:
                self._write_block(
                    [
                        "Secretary",
                        f"Model-backed secretary failed; falling back to local reports: {exc}",
                    ]
                )
                self._warned_about_fallback = True
            return await super()._emit_report()


async def _default_model_reporter(
    config: dict[str, Any],
    model: str,
    prompt: str,
    schema_path: Path,
    phase: str,
) -> dict[str, Any]:
    from .codex_runner import run_secretary_model

    return await run_secretary_model(config, model, prompt, schema_path, phase)


def create_secretary(
    config: dict[str, Any],
    secretary_config: SecretaryConfig,
    stream: TextIO | None = None,
) -> BaseSecretary:
    if secretary_config.mode == "model":
        return ModelBackedSecretary(
            config=config,
            model=secretary_config.model,
            verbosity=secretary_config.verbosity,
            stream=stream,
            immediate_updates=secretary_config.immediate_updates,
        )
    return LocalSecretary(stream, immediate_updates=secretary_config.immediate_updates)


Secretary = LocalSecretary


def _clean_text(value: str) -> str:
    return str(value).strip().rstrip(".;:").strip()


def _clean_model_message(value: str) -> str:
    return str(value).strip()


def _phase_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part) or "milestone"


def _round_label(round_number: int) -> str:
    return "initial vote" if round_number == 0 else f"runoff {round_number}"
