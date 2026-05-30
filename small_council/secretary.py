from __future__ import annotations

import sys
import inspect
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

from .config import ROOT
from .output import BaseRenderer
from .prompts import secretary_report_prompt


ModelReporter = Callable[[dict[str, Any], str, str, str, Path, str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class SecretaryConfig:
    mode: str = "model"
    provider: str = "codex"
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
    def __init__(
        self,
        stream: TextIO | None = None,
        immediate_updates: bool = True,
        renderer: BaseRenderer | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.state: SecretaryState | None = None
        self._milestone: str = ""
        self._immediate_updates = immediate_updates
        self._renderer = renderer

    async def start(self, question: str) -> None:
        self.state = SecretaryState(question=question)
        self._emit_secretary_message(f"Request received: {str(question).strip()}")

    async def stop(self) -> None:
        return None

    def set_phase(self, phase: str) -> None:
        if self.state:
            self.state.phase = phase
        if self._renderer:
            self._renderer.update_phase(phase)

    def recommendation_done(self, member: str, recommendation: str) -> None:
        if self.state:
            self.state.draft_recommendations.append(
                {"member": _clean_text(member), "recommendation": _clean_text(recommendation)}
            )
            self._emit_member_event(
                member,
                "proposal_ready",
                recommendation,
                payload={"recommendation": recommendation, "status": "research complete"},
            )
            self._write_immediate_status(f"{_clean_text(member)} finished research.")

    def final_recommendation_done(self, member: str, recommendation: str) -> None:
        if self.state:
            self.state.final_recommendations.append(
                {"member": _clean_text(member), "recommendation": _clean_text(recommendation)}
            )
            self._emit_member_event(
                member,
                "final_proposal",
                recommendation,
                payload={"recommendation": recommendation, "status": "final proposal ready"},
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
            self._emit_member_event(
                member,
                "vote",
                selected_option,
                payload={
                    "selected_option": selected_option,
                    "round_label": _round_label(round_number),
                    "status": "voted",
                },
            )
            self._write_immediate_status(
                f"{_clean_text(member)} voted in the {_round_label(round_number)}."
            )

    def vote_round_done(self, description: str) -> None:
        if self.state:
            self.state.vote_rounds.append(_clean_text(description))
            if self._renderer:
                self._emit_secretary_message(_clean_text(description))
            else:
                self._write_status(_clean_text(description))

    def discussion_round_started(self, round_number: int) -> None:
        if self.state:
            self.state.discussion_rounds.append(
                {"round_number": round_number, "messages": [], "completed": False}
            )
            if self._renderer:
                self._emit_secretary_message(f"Discussion round {round_number} started.")
            else:
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
                self._emit_member_event(
                    member,
                    "discussion_reply",
                    discussion_reply,
                    payload={
                        "discussion_reply": discussion_reply,
                        "prior_recommendation": prior_recommendation,
                        "revised_recommendation": revised_recommendation,
                        "changed": changed,
                        "status": "discussing",
                    },
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
                self._emit_secretary_message(f"Discussion round {round_number} complete.")
                return

    def diversity_lanes_assigned(self, lanes: dict[str, str], mode: str) -> None:
        if not self.state:
            return
        self.state.diversity_lanes = dict(lanes)
        self.state.diversity_mode = mode
        for member_name, lane in lanes.items():
            self._emit_member_event(
                member_name,
                "lane_assigned",
                lane,
                payload={"lane": lane, "mode": mode, "status": "assigned diversity lane"},
            )
        self._write_immediate_status(f"Diversity lanes assigned for {_clean_text(mode)} mode.")

    def grouping_done(self, groups: list[dict[str, object]]) -> None:
        merged = [group for group in groups if len(group.get("proposers", [])) > 1]
        if not merged:
            if self._renderer:
                self._emit_secretary_message("Grouped proposals: no equivalent proposals found.")
            else:
                self._write_immediate_status("Grouped proposals: no equivalent proposals found.")
            return
        if self._renderer:
            self._emit_secretary_message(f"Grouped {len(merged)} equivalent proposal set(s).")
        else:
            self._write_immediate_status(f"Grouped {len(merged)} equivalent proposal set(s).")

    def agent_run_failed(
        self,
        member: str,
        phase: str,
        reason: str,
        retryable: bool,
        attempt: int,
        max_attempts: int,
    ) -> None:
        member = _clean_text(member)
        phase = _clean_text(phase)
        reason = _clean_text(reason)
        will_retry = retryable and attempt < max_attempts
        status = "retrying" if will_retry else "failed"
        if self._renderer:
            self._renderer.member_status(member, status)
        if will_retry:
            self._write_immediate_status(
                f"{member} failed {phase}; retrying ({attempt}/{max_attempts}): {reason}"
            )
        else:
            self._write_immediate_status(f"{member} failed {phase}: {reason}")

    async def report_milestone(self, label: str) -> bool:
        self._milestone = _clean_text(label)
        if self._renderer:
            self._renderer.secretary_message(f"{self._milestone}")
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
        if self._renderer:
            self._renderer.secretary_message("\n".join(lines))
            return
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

    def _emit_secretary_message(self, message: str) -> None:
        if self._renderer:
            self._renderer.secretary_message(message)
        else:
            self._write_status(message)

    def _emit_member_event(
        self,
        member: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._renderer:
            self._renderer.member_event(member, event_type, message, payload=payload)

    def _emit_member_status(self, member: str, status: str) -> None:
        if self._renderer:
            self._renderer.member_status(member, status)


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
        provider: str,
        model: str,
        verbosity: str,
        stream: TextIO | None = None,
        model_reporter: ModelReporter | None = None,
        immediate_updates: bool = True,
        renderer: BaseRenderer | None = None,
    ) -> None:
        if not isinstance(verbosity, str):
            old_model = provider
            old_verbosity = model
            old_stream = verbosity
            old_reporter = stream
            provider = "codex"
            model = old_model
            verbosity = old_verbosity
            stream = old_stream
            if model_reporter is None:
                model_reporter = old_reporter  # type: ignore[assignment]
        super().__init__(stream, immediate_updates=immediate_updates, renderer=renderer)
        self.config = config
        self.provider = provider
        self.model = model
        self.verbosity = verbosity
        self.model_reporter = model_reporter or _default_model_reporter
        self._fallback_to_local = False
        self._warned_about_fallback = False

    async def _emit_report(self) -> bool:
        if self._fallback_to_local:
            return await super()._emit_report()
        try:
            prompt = secretary_report_prompt(
                self.state.question if self.state else "",
                self._snapshot(),
                self.verbosity,
                self._milestone,
            )
            schema = ROOT / "schemas" / "secretary-report.schema.json"
            phase = f"secretary-{_phase_slug(self._milestone)}"
            if _accepts_legacy_reporter(self.model_reporter):
                payload = await self.model_reporter(self.config, self.model, prompt, schema, phase)  # type: ignore[misc]
            else:
                payload = await self.model_reporter(
                    self.config, self.provider, self.model, prompt, schema, phase
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
    provider: str,
    model: str,
    prompt: str,
    schema_path: Path,
    phase: str,
) -> dict[str, Any]:
    from .model_runner import run_secretary_model

    return await run_secretary_model(config, provider, model, prompt, schema_path, phase)


def create_secretary(
    config: dict[str, Any],
    secretary_config: SecretaryConfig,
    stream: TextIO | None = None,
    renderer: BaseRenderer | None = None,
) -> BaseSecretary:
    if secretary_config.mode == "model":
        return ModelBackedSecretary(
            config=config,
            provider=secretary_config.provider,
            model=secretary_config.model,
            verbosity=secretary_config.verbosity,
            stream=stream,
            immediate_updates=secretary_config.immediate_updates,
            renderer=renderer,
        )
    return LocalSecretary(
        stream,
        immediate_updates=secretary_config.immediate_updates,
        renderer=renderer,
    )


Secretary = LocalSecretary


def _clean_text(value: str) -> str:
    return str(value).strip().rstrip(".;:").strip()


def _clean_model_message(value: str) -> str:
    return str(value).strip()


def _phase_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part) or "milestone"


def _accepts_legacy_reporter(reporter: ModelReporter) -> bool:
    try:
        signature = inspect.signature(reporter)
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    return len(positional) == 5


def _round_label(round_number: int) -> str:
    return "initial vote" if round_number == 0 else f"runoff {round_number}"
