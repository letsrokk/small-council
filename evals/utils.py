from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .models import EvalCase


REQUIRED_CASE_FIELDS = {
    "id",
    "name",
    "category",
    "prompt",
    "tags",
    "args",
    "expected_behavior",
    "scoring_focus",
    "hard_failure_rules",
}


def load_suite(path: str | Path) -> list[EvalCase]:
    suite_path = Path(path)
    raw = suite_path.read_text(encoding="utf-8")
    data = _load_yaml_like(raw)
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError(f"Suite must contain a list of cases: {suite_path}")

    cases = [_case_from_mapping(item, suite_path) for item in data]
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case ids: {', '.join(duplicates)}")
    return cases


def _load_yaml_like(raw: str) -> Any:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return json.loads(raw)
    return yaml.safe_load(raw)


def _case_from_mapping(item: Any, suite_path: Path) -> EvalCase:
    if not isinstance(item, dict):
        raise ValueError(f"Case entries must be mappings in {suite_path}")
    missing = REQUIRED_CASE_FIELDS - set(item)
    if missing:
        case_id = item.get("id", "<unknown>")
        raise ValueError(f"Case {case_id} missing fields: {', '.join(sorted(missing))}")
    return EvalCase(
        id=str(item["id"]),
        name=str(item["name"]),
        category=str(item["category"]),
        prompt=str(item["prompt"]),
        tags=[str(value) for value in item["tags"]],
        args=[str(value) for value in item["args"]],
        expected_behavior=[str(value) for value in item["expected_behavior"]],
        scoring_focus=[str(value) for value in item["scoring_focus"]],
        hard_failure_rules=[str(value) for value in item["hard_failure_rules"]],
        golden_ref=str(item["golden_ref"]) if item.get("golden_ref") is not None else None,
        golden=dict(item["golden"]) if isinstance(item.get("golden"), dict) else None,
    )


def filter_cases(
    cases: list[EvalCase],
    case_id: str | None = None,
    category: str | None = None,
    tag: str | None = None,
) -> list[EvalCase]:
    selected = cases
    if case_id:
        selected = [case for case in selected if case.id == case_id]
    if category:
        selected = [case for case in selected if case.category == category]
    if tag:
        selected = [case for case in selected if tag in case.tags]
    return selected


def extract_last_valid_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    decoder = json.JSONDecoder()
    last_payload: dict[str, Any] | None = None
    last_error: str | None = None
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        candidate = stdout[index:].lstrip()
        try:
            payload, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"JSON candidate at byte {index} failed: {exc.msg}"
            continue
        trailing = candidate[end:].strip()
        if isinstance(payload, dict) and (not trailing or _only_noise_after_json(trailing)):
            last_payload = payload
            last_error = None
    if last_payload is not None:
        return last_payload, None
    if last_error:
        return None, last_error
    return None, "No JSON object start found in stdout."


def _only_noise_after_json(trailing: str) -> bool:
    if trailing[:1] in {"}", "]", ","}:
        return False
    return not any(char in "{}[]" for char in trailing)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
