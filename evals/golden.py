from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import CaseRunResult, EvalCase
from .utils import _load_yaml_like


GOLDEN_FIELDS = {
    "acceptable_winners",
    "unacceptable_winners",
    "required_final_output_terms",
    "forbidden_final_output_terms",
    "required_behaviors",
    "forbidden_behaviors",
    "expected_status",
    "allow_unresolved_tie",
}


@dataclass(frozen=True)
class GoldenOutcome:
    golden_score: int | None
    golden_failures: list[str] = field(default_factory=list)
    golden_pass: bool | None = None


def load_golden_datasets(golden_dir: str | Path) -> dict[str, dict[str, Any]]:
    base = Path(golden_dir)
    datasets: dict[str, dict[str, Any]] = {}
    if not base.exists():
        raise FileNotFoundError(f"Golden dataset directory does not exist: {base}")
    for path in sorted(base.glob("*.yaml")):
        data = _load_yaml_like(path.read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Golden dataset must be a mapping: {path}")
        entries = data.get("entries", data)
        if not isinstance(entries, dict):
            raise ValueError(f"Golden dataset entries must be a mapping: {path}")
        datasets[path.name] = {str(key): _normalize_spec(value, path, str(key)) for key, value in entries.items()}
    return datasets


def resolve_golden(case: EvalCase, datasets: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if case.golden is not None:
        return _normalize_spec(case.golden, Path("<inline>"), case.id)
    if not case.golden_ref:
        return None
    if "#" not in case.golden_ref:
        raise ValueError(f"Case {case.id} golden_ref must use file.yaml#ENTRY_ID")
    filename, entry_id = case.golden_ref.split("#", 1)
    dataset = datasets.get(filename)
    if dataset is None:
        raise ValueError(f"Case {case.id} references missing golden dataset: {filename}")
    spec = dataset.get(entry_id)
    if spec is None:
        raise ValueError(f"Case {case.id} references missing golden entry: {case.golden_ref}")
    return spec


def validate_golden_references(cases: list[EvalCase], datasets: dict[str, dict[str, Any]]) -> None:
    for case in cases:
        resolve_golden(case, datasets)


def golden_score(case: EvalCase, result: CaseRunResult, datasets: dict[str, dict[str, Any]]) -> GoldenOutcome:
    spec = resolve_golden(case, datasets)
    if spec is None:
        return GoldenOutcome(golden_score=None, golden_pass=None)

    payload = result.execution.json_payload or {}
    final_output = _clean(payload.get("final_output"))
    winner = _clean(payload.get("winning_option"))
    status = _clean(payload.get("status")).lower()
    combined = _combined_behavior_text(payload)
    failures: list[str] = []

    allow_unresolved_tie = bool(spec.get("allow_unresolved_tie", False))
    if _is_unresolved(status) and not allow_unresolved_tie:
        failures.append("unresolved_tie_not_allowed")

    expected_status = _strings(spec.get("expected_status"))
    if expected_status and not any(_soft_match(status, expected) for expected in expected_status):
        failures.append(f"status_expected:{','.join(expected_status)}")

    acceptable_winners = _strings(spec.get("acceptable_winners"))
    if acceptable_winners and not (_is_unresolved(status) and allow_unresolved_tie):
        if not winner or not any(_soft_match(winner, candidate) for candidate in acceptable_winners):
            failures.append("winner_not_acceptable")

    for candidate in _strings(spec.get("unacceptable_winners")):
        if winner and _soft_match(winner, candidate):
            failures.append(f"winner_unacceptable:{candidate}")

    for term in _strings(spec.get("required_final_output_terms")):
        if not _soft_match(final_output, term):
            failures.append(f"final_output_missing:{term}")

    for term in _strings(spec.get("forbidden_final_output_terms")):
        if _soft_match(final_output, term):
            failures.append(f"final_output_forbidden:{term}")

    for behavior in _strings(spec.get("required_behaviors")):
        if not _soft_match(combined, behavior):
            failures.append(f"behavior_missing:{behavior}")

    for behavior in _strings(spec.get("forbidden_behaviors")):
        if _soft_match(combined, behavior):
            failures.append(f"behavior_forbidden:{behavior}")

    score = max(0, 100 - (20 * len(failures)))
    return GoldenOutcome(golden_score=score, golden_failures=failures, golden_pass=not failures)


def _normalize_spec(value: Any, path: Path, entry_id: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Golden entry {entry_id} in {path} must be a mapping")
    unknown = set(value) - GOLDEN_FIELDS
    if unknown:
        raise ValueError(
            f"Golden entry {entry_id} in {path} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    normalized = dict(value)
    for key in GOLDEN_FIELDS - {"allow_unresolved_tie"}:
        normalized[key] = _strings(normalized.get(key))
    normalized["allow_unresolved_tie"] = bool(normalized.get("allow_unresolved_tie", False))
    return normalized


def _combined_behavior_text(payload: dict[str, Any]) -> str:
    parts = [
        payload.get("final_output"),
        payload.get("winning_option"),
        payload.get("status"),
        payload.get("draft_recommendations"),
        payload.get("final_recommendations"),
        payload.get("recommendation_groups"),
        payload.get("votes"),
        payload.get("vote_rounds"),
        payload.get("discussion_transcript"),
    ]
    return "\n".join(_stringify(part) for part in parts if part is not None)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _soft_match(text: str, expected: str) -> bool:
    text_norm = _normalize(text)
    expected_norm = _normalize(expected)
    if not expected_norm:
        return True
    if expected_norm in text_norm:
        return True
    expected_tokens = _tokens(expected_norm)
    if not expected_tokens:
        return True
    text_tokens = set(_tokens(text_norm))
    overlap = sum(1 for token in expected_tokens if token in text_tokens)
    threshold = 1 if len(expected_tokens) == 1 else max(2, round(len(expected_tokens) * 0.7))
    return overlap >= threshold


def _tokens(text: str) -> list[str]:
    return [token for token in text.split() if token not in {"a", "an", "and", "or", "the", "to", "of"}]


def _normalize(value: Any) -> str:
    cleaned = _clean(value).casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _is_unresolved(status: str) -> bool:
    normalized = _normalize(status)
    return normalized in {"unresolved", "tie", "tied", "deadlock", "no consensus"}
