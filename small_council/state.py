from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT, read_json, resolve_project_path, write_json
from .model_providers import ModelInfo, effective_model_pool


@dataclass(frozen=True)
class Member:
    name: str
    model: str
    personality: str
    is_president: bool
    created_at: str
    total_proposals: int = 0
    total_wins: int = 0
    total_votes_cast: int = 0
    tie_break_victories: int = 0
    provider: str = "codex"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Member":
        data = dict(payload)
        data.setdefault("provider", "codex")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "personality": self.personality,
            "is_president": self.is_president,
            "created_at": self.created_at,
            "total_proposals": self.total_proposals,
            "total_wins": self.total_wins,
            "total_votes_cast": self.total_votes_cast,
            "tie_break_victories": self.tie_break_victories,
        }


def ensure_state(config: dict[str, Any], reset: bool = False) -> list[Member]:
    state_path = resolve_project_path(config["storage"]["council_state_path"])
    if reset and state_path.exists():
        state_path.unlink()
    payload = read_json(state_path, default=None)
    if payload and payload.get("members"):
        members = [Member.from_dict(item) for item in payload["members"]]
        persist_members(config, members)
        write_agent_files(members)
        return members

    members = _create_members(config)
    members = _apply_model_overrides(config, members)
    persist_members(config, members)
    write_agent_files(members)
    delete_stale_agent_files(members)
    return members


def resize_members(config: dict[str, Any], members: list[Member], target_count: int) -> list[Member]:
    min_members = int(config.get("council", {}).get("min_members", 1))
    if target_count < min_members:
        raise ValueError(f"Council must have at least {min_members} active member.")
    if target_count == len(members):
        persist_members(config, members)
        write_agent_files(members)
        return members
    if target_count < len(members):
        resized = _remove_members(members, target_count)
    else:
        resized = _add_members(config, members, target_count - len(members))
    persist_members(config, resized)
    write_agent_files(resized)
    delete_stale_agent_files(resized)
    return resized


def persist_members(config: dict[str, Any], members: list[Member]) -> None:
    state_path = resolve_project_path(config["storage"]["council_state_path"])
    payload = {
        "version": 1,
        "created_or_updated_at": now_iso(),
        "members": [member.to_dict() for member in members],
    }
    write_json(state_path, payload)
    persist_leaderboard(config, members)


def persist_leaderboard(config: dict[str, Any], members: list[Member]) -> None:
    leaderboard_path = resolve_project_path(config["storage"]["leaderboard_path"])
    rows = []
    for member in sorted(members, key=lambda item: (-item.total_wins, item.name)):
        proposals = member.total_proposals
        rows.append(
            {
                "member": member.name,
                "provider": member.provider,
                "model": member.model,
                "personality": member.personality,
                "president": member.is_president,
                "total_wins": member.total_wins,
                "total_proposals": proposals,
                "win_rate": round(member.total_wins / proposals, 3) if proposals else 0.0,
                "vote_participation": member.total_votes_cast,
                "tie_break_victories": member.tie_break_victories,
            }
        )
    write_json(leaderboard_path, {"updated_at": now_iso(), "leaderboard": rows})


def delete_stale_agent_files(members: list[Member]) -> None:
    definitions_dir = ROOT / "agents" / "definitions"
    if not definitions_dir.exists():
        return
    active_files = {f"{member.name.lower()}.md" for member in members}
    for path in definitions_dir.glob("*.md"):
        if path.name not in active_files:
            path.unlink()


def update_after_decision(
    config: dict[str, Any],
    members: list[Member],
    proposing_members: set[str],
    winning_member: str | None,
    voter_names: set[str],
    tie_breaker_member: str | None,
    winning_members: set[str] | None = None,
) -> list[Member]:
    winning_names = winning_members or ({winning_member} if winning_member else set())
    updated = []
    for member in members:
        data = member.to_dict()
        if member.name in proposing_members:
            data["total_proposals"] += 1
        if member.name in winning_names:
            data["total_wins"] += 1
        if member.name in voter_names:
            data["total_votes_cast"] += 1
        if tie_breaker_member == member.name:
            data["tie_break_victories"] += 1
        updated.append(Member.from_dict(data))
    persist_members(config, updated)
    write_agent_files(updated)
    return updated


def president(members: list[Member]) -> Member:
    for member in members:
        if member.is_president:
            return member
    return members[0]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _create_members(config: dict[str, Any]) -> list[Member]:
    names = config["council"]["member_names"]
    personality_pool = list(config["personality_pool"])
    rng = random.SystemRandom()

    models = _draw_models(config, [], names, rng)
    personalities = _draw_values(rng, personality_pool, len(names), unique=True)
    president_name = rng.choice(names)
    timestamp = now_iso()

    return [
        Member(
            name=name,
            provider=models[index].provider,
            model=models[index].model,
            personality=personalities[index],
            is_president=name == president_name,
            created_at=timestamp,
        )
        for index, name in enumerate(names)
    ]


def _add_members(config: dict[str, Any], members: list[Member], count: int) -> list[Member]:
    rng = random.SystemRandom()
    existing_names = {member.name for member in members}
    names = _next_member_names(config, existing_names, count)
    models = _next_models(config, members, names, rng)
    personalities = _next_personalities(config, members, count, rng)
    timestamp = now_iso()
    additions = [
        Member(
            name=name,
            provider=models[index].provider,
            model=models[index].model,
            personality=personalities[index],
            is_president=False,
            created_at=timestamp,
        )
        for index, name in enumerate(names)
    ]
    return members + _apply_model_overrides(config, additions)


def _remove_members(members: list[Member], target_count: int) -> list[Member]:
    resized = members[:target_count]
    if any(member.is_president for member in resized):
        return resized
    updated = []
    for index, member in enumerate(resized):
        data = member.to_dict()
        data["is_president"] = index == 0
        updated.append(Member.from_dict(data))
    return updated


def _next_member_names(
    config: dict[str, Any], existing_names: set[str], count: int
) -> list[str]:
    configured = list(config["council"].get("member_names", []))
    generated = list(config["council"].get("generated_member_names", []))
    candidates = configured + generated
    names = []
    for candidate in candidates:
        if candidate not in existing_names and candidate not in names:
            names.append(candidate)
            if len(names) == count:
                return names
    index = len(existing_names) + 1
    while len(names) < count:
        candidate = f"Member {index}"
        if candidate not in existing_names and candidate not in names:
            names.append(candidate)
        index += 1
    return names


def _next_models(
    config: dict[str, Any], members: list[Member], names: list[str], rng: random.SystemRandom
) -> list[ModelInfo]:
    return _draw_models(config, members, names, rng)


def _draw_models(
    config: dict[str, Any],
    members: list[Member],
    names: list[str],
    rng: random.SystemRandom,
) -> list[ModelInfo]:
    pool = effective_model_pool(config)
    if not pool:
        raise ValueError("No enabled models are available from configured providers.")
    selected_by_name: dict[str, ModelInfo] = {}
    remaining_names: list[str] = []
    used = {(member.provider, member.model) for member in members}
    for name in names:
        override = _model_override(config, name)
        if override is None:
            remaining_names.append(name)
            continue
        if not _pool_contains(pool, override):
            raise ValueError(
                f"Model override for {name} uses {override.provider}/{override.model}, "
                "which is outside the effective model pool."
            )
        selected_by_name[name] = override
        used.add((override.provider, override.model))

    prefer_unique = bool(
        config.get("model_assignment", {}).get("prefer_unique_models", True)
    )
    allow_duplicates = bool(
        config.get("model_assignment", {}).get("allow_duplicates_when_needed", True)
    )
    available = [
        model for model in pool if (model.provider, model.model) not in used
    ] if prefer_unique else pool[:]
    rng.shuffle(available)
    drawn = available[: len(remaining_names)]
    if len(drawn) < len(remaining_names):
        if not allow_duplicates:
            raise ValueError("Not enough unique enabled models are available for council assignment.")
        while len(drawn) < len(remaining_names):
            drawn.append(rng.choice(pool))
    for name, model in zip(remaining_names, drawn):
        selected_by_name[name] = model
    return [selected_by_name[name] for name in names]


def _next_personalities(
    config: dict[str, Any], members: list[Member], count: int, rng: random.SystemRandom
) -> list[str]:
    pool = list(config["personality_pool"])
    used = {member.personality for member in members}
    available = [personality for personality in pool if personality not in used]
    rng.shuffle(available)
    selected = available[:count]
    while len(selected) < count:
        selected.append(rng.choice(pool))
    return selected


def _apply_model_overrides(config: dict[str, Any], members: list[Member]) -> list[Member]:
    overrides = config.get("model_overrides") or {}
    if not overrides:
        return members
    pool = effective_model_pool(config)
    updated: list[Member] = []
    for member in members:
        override = _model_override(config, member.name)
        if not override:
            updated.append(member)
            continue
        if not _pool_contains(pool, override):
            raise ValueError(
                f"Model override for {member.name} uses {override.provider}/{override.model}, "
                "which is outside the effective model pool."
            )
        data = member.to_dict()
        data["provider"] = override.provider
        data["model"] = override.model
        updated.append(Member.from_dict(data))
    return updated


def _model_override(config: dict[str, Any], name: str) -> ModelInfo | None:
    overrides = config.get("model_overrides") or {}
    if not isinstance(overrides, dict):
        return None
    raw = overrides.get(name)
    if not raw:
        return None
    if isinstance(raw, dict):
        return ModelInfo(
            provider=str(raw.get("provider", "codex")),
            model=str(raw.get("model", "")),
        )
    return ModelInfo(provider="codex", model=str(raw))


def _pool_contains(pool: list[ModelInfo], model: ModelInfo) -> bool:
    return any(item.provider == model.provider and item.model == model.model for item in pool)


def _draw_values(
    rng: random.SystemRandom, values: list[str], count: int, unique: bool
) -> list[str]:
    if unique and count <= len(values):
        shuffled = values[:]
        rng.shuffle(shuffled)
        return shuffled[:count]
    if unique:
        shuffled = values[:]
        rng.shuffle(shuffled)
        selected = shuffled[:]
        while len(selected) < count:
            selected.append(rng.choice(values))
        return selected
    return [rng.choice(values) for _ in range(count)]


def write_agent_files(members: list[Member]) -> None:
    definitions_dir = ROOT / "agents" / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    for member in members:
        role = "President" if member.is_president else "Council Member"
        content = f"""# {member.name}

Role: {role}
Provider: {member.provider}
Model: {member.model}
Personality: {member.personality}

Persistent identity:
- Keep this name, model, and personality unless the local council state is reset or rerolled.
- Reason independently before seeing any other member's recommendation.
- During independent research, use the shared Search Worker for current, external, or missing information instead of guessing freshness-sensitive details.
- Keep private reasoning hidden; provide concise conclusions, tradeoffs, and votes only.
- Prefer practical, light personal decisions over exhaustive analysis.
"""
        (definitions_dir / f"{member.name.lower()}.md").write_text(content, encoding="utf-8")
