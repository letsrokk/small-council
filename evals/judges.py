from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from small_council.config import ROOT, load_config
from small_council.model_runner import run_member
from small_council.state import Member

from .models import CaseRunResult


DEFAULT_JUDGE_CONFIG_PATH = ROOT / "config" / "judge.yaml"
MAX_TEXT_CHARS = 6000
MAX_ARTIFACT_CHARS = 12000
TEN_POINT_SCALE_ERROR = (
    "Judge score appears to use a 1-10 scale. Scores must be 1-100; "
    "0 is reserved for catastrophic/no usable output."
)


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    score: int | None = None
    passed: bool | None = None
    reasoning: str | None = None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    safety_concerns: list[str] = field(default_factory=list)
    regression_risk: str | None = None
    error: str | None = None


def judge_result(
    result: CaseRunResult,
    provider: str,
    model: str,
    options: dict[str, Any] | None = None,
    timeout_seconds: float = 300,
) -> JudgeResult:
    try:
        return asyncio.run(_judge_result_async(result, provider, model, options or {}, timeout_seconds))
    except Exception as exc:
        return JudgeResult(error=str(exc))


async def _judge_result_async(
    result: CaseRunResult,
    provider: str,
    model: str,
    options: dict[str, Any],
    timeout_seconds: float,
) -> JudgeResult:
    config = _judge_config(provider, model, options)
    member = Member(
        name="Judge",
        provider=provider,
        model=model,
        personality="strict evaluator",
        is_president=False,
        created_at="eval-runtime",
    )
    schema_path = ROOT / "schemas" / "judge.schema.json"
    prompt = build_judge_prompt(result)
    response = await asyncio.wait_for(
        run_member(config, member, prompt, schema_path, "judge", False),
        timeout=timeout_seconds,
    )
    judged = parse_judge_payload(response.payload)
    if judged.error == TEN_POINT_SCALE_ERROR:
        retry_prompt = (
            prompt
            + "\n\nYour previous response used a 1-10 style score. "
            "Re-evaluate from scratch and return a score on the required 1-100 integer scale."
        )
        response = await asyncio.wait_for(
            run_member(config, member, retry_prompt, schema_path, "judge", False),
            timeout=timeout_seconds,
        )
        judged = parse_judge_payload(response.payload)
    return judged


def load_judge_config(path: Path = DEFAULT_JUDGE_CONFIG_PATH) -> JudgeConfig:
    data = load_config(path)
    provider = str(data.get("provider") or "").strip()
    model = str(data.get("model") or "").strip()
    if not provider:
        raise ValueError("config/judge.yaml provider must be a non-empty string.")
    if not model:
        raise ValueError("config/judge.yaml model must be a non-empty string.")
    options = data.get("options")
    return JudgeConfig(provider=provider, model=model, options=_validate_judge_options(options))


def parse_judge_payload(payload: dict[str, Any]) -> JudgeResult:
    try:
        score = int(payload["score"])
        passed = bool(payload["pass"])
    except (KeyError, TypeError, ValueError) as exc:
        return JudgeResult(error=f"Invalid judge payload: {exc}")
    if score < 0 or score > 100:
        return JudgeResult(error=f"Invalid judge score: {score}")
    weaknesses = _strings(payload.get("weaknesses"))
    if 1 <= score <= 10:
        return JudgeResult(error=TEN_POINT_SCALE_ERROR, weaknesses=weaknesses)
    return JudgeResult(
        score=score,
        passed=passed,
        reasoning=str(payload.get("reasoning") or ""),
        strengths=_strings(payload.get("strengths")),
        weaknesses=weaknesses,
        safety_concerns=_strings(payload.get("safety_concerns")),
        regression_risk=str(payload.get("regression_risk") or "unknown"),
    )


def build_judge_prompt(result: CaseRunResult) -> str:
    case = result.case
    payload = result.execution.json_payload or {}
    report_view = {
        "case": {
            "id": case.id,
            "name": case.name,
            "category": case.category,
            "prompt": case.prompt,
            "expected_behavior": case.expected_behavior,
            "scoring_focus": case.scoring_focus,
            "hard_failure_rules": case.hard_failure_rules,
        },
        "deterministic": {
            "score": result.deterministic_score,
            "score_breakdown": _plain_dataclass(result.score_breakdown),
            "validation": _plain_dataclass(result.validation),
        },
        "golden": {
            "score": result.golden_score,
            "pass": result.golden_pass,
            "failures": result.golden_failures,
        },
        "council_output": {
            "final_output": payload.get("final_output"),
            "status": payload.get("status"),
            "winning_option": payload.get("winning_option"),
            "draft_recommendations": payload.get("draft_recommendations"),
            "final_recommendations": payload.get("final_recommendations"),
            "recommendation_groups": payload.get("recommendation_groups"),
            "votes": payload.get("votes"),
            "vote_rounds": payload.get("vote_rounds"),
        },
        "execution": {
            "duration_seconds": result.execution.duration_seconds,
            "exit_code": result.execution.exit_code,
            "timed_out": result.execution.timed_out,
            "stdout_excerpt": _truncate(result.execution.stdout, MAX_TEXT_CHARS),
            "stderr_excerpt": _truncate(result.execution.stderr, MAX_TEXT_CHARS),
        },
        "artifacts": _artifact_excerpts(result.artifact_paths),
    }
    return (
        "You are evaluating a completed Small Council deterministic eval run.\n"
        "Score on a 1-100 integer scale. Do not use a 1-10 scale. "
        "Use 90-100 for excellent, 70-89 for passable/good, 50-69 for weak, "
        "1-49 for severe problems, and 0 only for catastrophic failure or no usable output. "
        "Use the deterministic report, storage snapshots, and runtime logs as evidence.\n"
        "Do not override deterministic hard failures. Score semantic quality, process integrity, "
        "safety, and regression risk. "
        "Return only JSON matching the provided schema.\n\n"
        + json.dumps(report_view, indent=2, ensure_ascii=False, sort_keys=True)
    )


def _judge_config(provider: str, model: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config()
    providers = dict(config.get("model_providers") or {})
    provider_config = dict(providers.get(provider) or {})
    provider_config["enabled"] = True
    if options:
        provider_options = dict(provider_config.get("options") or {})
        provider_options.update(_validate_judge_options(options))
        provider_config["options"] = provider_options
    static_models = [str(item) for item in provider_config.get("static_models") or []]
    if model not in static_models:
        static_models.append(model)
    provider_config["static_models"] = static_models
    providers[provider] = provider_config
    config["model_providers"] = providers
    return config


def _validate_judge_options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("config/judge.yaml options must be a mapping.")
    unknown = set(value) - {"temperature", "seed", "num_ctx"}
    if unknown:
        raise ValueError(
            "config/judge.yaml options has unknown field(s): "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    options: dict[str, Any] = {}
    if "temperature" in value:
        temperature = value["temperature"]
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float))
        ):
            raise ValueError("config/judge.yaml options.temperature must be a number or null.")
        options["temperature"] = float(temperature) if temperature is not None else None
    if "seed" in value:
        seed = value["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("config/judge.yaml options.seed must be an integer or null.")
        options["seed"] = seed
    if "num_ctx" in value:
        num_ctx = value["num_ctx"]
        if num_ctx is not None and (isinstance(num_ctx, bool) or not isinstance(num_ctx, int)):
            raise ValueError("config/judge.yaml options.num_ctx must be an integer or null.")
        options["num_ctx"] = num_ctx
    return options


def _artifact_excerpts(paths: list[str]) -> list[dict[str, str]]:
    excerpts: list[dict[str, str]] = []
    remaining = MAX_ARTIFACT_CHARS
    for raw_path in paths:
        if remaining <= 0:
            break
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            excerpts.append({"path": raw_path, "error": str(exc)})
            continue
        excerpt = _truncate(text, min(remaining, 3000))
        remaining -= len(excerpt)
        excerpts.append({"path": raw_path, "excerpt": excerpt})
    return excerpts


def _plain_dataclass(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _plain_dataclass(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, list):
        return [_plain_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_dataclass(item) for key, item in value.items()}
    return value


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: limit - 20].rstrip() + "\n...[truncated]..."
