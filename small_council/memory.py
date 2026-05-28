from __future__ import annotations

from typing import Any

from .config import read_json, resolve_project_path, write_json
from .state import now_iso


def load_recent_memory(config: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    path = _history_path(config)
    payload = read_json(path, {"decisions": []})
    decisions = payload.get("decisions", [])
    return {"recent_decisions": decisions[-limit:]}


def append_decision_memory(
    config: dict[str, Any],
    question: str,
    winning_option: str | None,
    winning_member: str | None,
    votes: list[dict[str, Any]],
    final_tied_options: list[str] | None = None,
) -> None:
    path = _history_path(config)
    payload = read_json(path, {"decisions": []})
    payload.setdefault("decisions", []).append(
        {
            "timestamp": now_iso(),
            "question": question,
            "winning_option": winning_option,
            "winning_member": winning_member,
            "final_tied_options": final_tied_options or [],
            "votes": [
                {
                    "voter": vote.get("voter"),
                    "selected_option": vote.get("selected_option"),
                    "reason": vote.get("reason"),
                }
                for vote in votes
            ],
        }
    )
    write_json(path, payload)


def _history_path(config: dict[str, Any]):
    memories_path = resolve_project_path(config["storage"]["memories_path"])
    return memories_path / "decision-history.json"
