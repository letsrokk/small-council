from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    name: str
    category: str
    prompt: str
    tags: list[str]
    args: list[str]
    expected_behavior: list[str]
    scoring_focus: list[str]
    hard_failure_rules: list[str]
    golden_ref: str | None = None
    golden: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvalRunMetadata:
    timestamp: str
    git_commit: str | None
    version_name: str | None
    selected_filters: dict[str, str | None]
    suite_path: str
    repeat: int
    timeout_seconds: float
    council_cmd: str


@dataclass
class CouncilExecution:
    command: list[str]
    stdout: str
    stderr: str
    duration_seconds: float
    exit_code: int | None
    timed_out: bool = False
    json_payload: dict[str, Any] | None = None
    json_error: str | None = None


@dataclass
class ValidationResult:
    valid_json: bool
    required_fields_present: bool
    recommendation_counts_sensible: bool
    winner_exists: bool
    final_answer_aligns_with_winner: bool
    vote_references_valid: bool
    runoff_counts_valid: bool
    diversity_lanes_present: bool
    safety_passed: bool
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    answers_actual_request: int = 0
    practicality: int = 0
    reasoning_quality: int = 0
    tradeoff_awareness: int = 0
    proposal_diversity: int = 0
    internal_consistency: int = 0
    json_schema_validity: int = 0
    safety_resistance: int = 0
    total_before_caps: int = 0
    deterministic_score: int = 0
    applied_caps: list[str] = field(default_factory=list)


@dataclass
class CaseRunResult:
    case: EvalCase
    repeat_index: int
    execution: CouncilExecution
    validation: ValidationResult
    score_breakdown: ScoreBreakdown
    deterministic_score: int
    golden_score: int | None = None
    golden_failures: list[str] = field(default_factory=list)
    golden_pass: bool | None = None
    judge_score: int | None = None
    judge_pass: bool | None = None
    judge_reasoning: str | None = None
    judge_strengths: list[str] = field(default_factory=list)
    judge_weaknesses: list[str] = field(default_factory=list)
    judge_safety_concerns: list[str] = field(default_factory=list)
    judge_regression_risk: str | None = None
    judge_error: str | None = None
    combined_score: int | None = None
    artifact_paths: list[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class EvalReport:
    metadata: EvalRunMetadata
    results: list[CaseRunResult]


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    return value
