from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from small_council.config import load_config, save_config

from .golden import golden_score, load_golden_datasets, resolve_golden, validate_golden_references
from .judges import JudgeConfig, judge_result, load_judge_config
from .models import (
    CaseRunResult,
    CouncilExecution,
    EvalCase,
    EvalReport,
    EvalRunMetadata,
    ScoreBreakdown,
    ValidationResult,
)
from .report import PASS_THRESHOLD, write_json_report, write_markdown_report
from .scorers import score_case, validate_result
from .utils import extract_last_valid_json, filter_cases, git_commit, load_suite


DEFAULT_COUNCIL_CMD = "./council --secretary local"


@dataclass(frozen=True)
class EvalSandbox:
    config_path: Path
    roots: dict[Path, Path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Small Council evals.")
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--tag")
    parser.add_argument("--category")
    parser.add_argument("--suite", default="evals/cases.yaml")
    parser.add_argument("--output", default="evals/reports/latest.json")
    parser.add_argument("--markdown", default="evals/reports/latest.md")
    parser.add_argument("--version-name")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--council-cmd", default=DEFAULT_COUNCIL_CMD)
    parser.add_argument("--golden", action="store_true", help="Run golden validation after deterministic eval.")
    parser.add_argument("--golden-dir", default="evals/golden")
    parser.add_argument("--golden-weight", type=float)
    parser.add_argument("--llm-judge", action="store_true", help="Run LLM judge after deterministic/golden eval.")
    parser.add_argument("--judge-provider")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-timeout-seconds", type=float, default=300)
    parser.add_argument("--judge-weight", type=float)
    parser.add_argument("--compare")
    parser.add_argument("--skip", action="store_true", help="Skip deterministic execution and post-process an existing report.")
    parser.add_argument("--input-report")
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Rerun only cases that failed in the latest report.",
    )
    parser.add_argument(
        "--failed-report",
        default="evals/reports/latest.json",
        help="Report to read failed cases from when --failed-only is used.",
    )
    parser.add_argument("--artifact-dir", default="evals/reports/artifacts")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print validation warnings and stderr snippets for failing cases.",
    )
    args = parser.parse_args(argv)

    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.judge_timeout_seconds <= 0:
        parser.error("--judge-timeout-seconds must be positive")
    if args.golden_weight is not None and args.golden_weight < 0:
        parser.error("--golden-weight must be non-negative")
    if args.judge_weight is not None and args.judge_weight < 0:
        parser.error("--judge-weight must be non-negative")
    if args.skip and not (args.golden or args.llm_judge):
        parser.error("--skip requires --golden and/or --llm-judge")
    if args.skip and args.failed_only:
        parser.error("--failed-only cannot be used with --skip")

    suite_path = Path(args.suite)
    suite_start = time.monotonic()
    if args.skip:
        input_report = Path(args.input_report or args.output)
        report = load_json_report(input_report)
        _enrich_report_cases(report, suite_path)
        report.results = _filter_results(
            report.results,
            case_id=args.case_id,
            category=args.category,
            tag=args.tag,
        )
        if not report.results:
            parser.error("No report results matched the selected filters.")
        cases = report.results
        previous_json_path = Path(args.compare) if args.compare else _previous_path(Path(args.output))
        if Path(args.output) != input_report:
            _backup_previous_report(args.output)
        if Path(args.markdown) != input_report:
            _backup_previous_report(args.markdown)
        _print_skip_start(args, input_report, len(cases))
    else:
        suite_cases = load_suite(suite_path)
        if args.failed_only:
            cases = _failed_cases_from_report(args, suite_cases, suite_path, parser)
        else:
            cases = filter_cases(
                suite_cases,
                case_id=args.case_id,
                category=args.category,
                tag=args.tag,
            )
        if not cases:
            parser.error("No cases matched the selected filters.")
        previous_json_path = Path(args.compare) if args.compare else _backup_previous_report(args.output)
        _backup_previous_report(args.markdown)
        report = _run_deterministic(args, cases, suite_path, suite_start)

    _run_post_processing(args, report)
    _apply_blended_scores(args, report)
    write_json_report(report, args.output)
    write_markdown_report(report, args.markdown)
    comparison = _compare_with_previous(previous_json_path, report)
    _print_summary(args, report.results, suite_start, comparison)
    return 0


def _run_deterministic(
    args: argparse.Namespace,
    cases: list[EvalCase],
    suite_path: Path,
    suite_start: float,
) -> EvalReport:
    metadata = EvalRunMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(),
        version_name=args.version_name,
        selected_filters={
            "case": args.case_id,
            "category": args.category,
            "tag": args.tag,
        },
        suite_path=str(suite_path),
        repeat=args.repeat,
        timeout_seconds=args.timeout_seconds,
        council_cmd=args.council_cmd,
    )
    results: list[CaseRunResult] = []
    total_runs = len(cases) * args.repeat
    run_id = _run_id(metadata.timestamp)
    sandbox = _create_eval_sandbox(Path(args.artifact_dir), run_id)
    _print_start(args, cases, total_runs)
    run_index = 0
    for case in cases:
        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            _print_case_start(args, run_index, total_runs, case, repeat_index, suite_start)
            before = _artifact_snapshot(sandbox.roots)
            execution = execute_case(
                case,
                args.council_cmd,
                args.timeout_seconds,
                config_path=sandbox.config_path,
            )
            artifact_paths = _capture_artifacts(
                before,
                Path(args.artifact_dir),
                run_id,
                case.id,
                repeat_index,
                sandbox.roots,
            )
            validation = validate_result(case, execution)
            score = score_case(case, execution, validation)
            deterministic_score = score.deterministic_score
            result = CaseRunResult(
                case=case,
                repeat_index=repeat_index,
                execution=execution,
                validation=validation,
                score_breakdown=score,
                deterministic_score=deterministic_score,
                artifact_paths=artifact_paths,
                passed=deterministic_score >= PASS_THRESHOLD
                and not validation.hard_failures,
            )
            results.append(result)
            _print_case_result(args, result, suite_start, len(results), total_runs)
    return EvalReport(metadata=metadata, results=results)


def execute_case(
    case,
    council_cmd: str,
    timeout_seconds: float,
    config_path: Path | None = None,
) -> CouncilExecution:
    command = [
        *shlex.split(council_cmd),
        *case.args,
        "--json-output",
        "--plain-output",
        case.prompt,
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=_benchmark_env(config_path),
        )
        duration = time.monotonic() - start
        payload, error = extract_last_valid_json(completed.stdout)
        return CouncilExecution(
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
            exit_code=completed.returncode,
            timed_out=False,
            json_payload=payload,
            json_error=error,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        payload, error = extract_last_valid_json(stdout)
        return CouncilExecution(
            command=command,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            exit_code=None,
            timed_out=True,
            json_payload=payload,
            json_error=error or "Command timed out.",
        )


def _benchmark_env(config_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["SMALL_COUNCIL_BENCHMARK"] = "1"
    if config_path is not None:
        env["SMALL_COUNCIL_CONFIG"] = str(config_path)
    return env


def _run_post_processing(args: argparse.Namespace, report: EvalReport) -> None:
    if args.golden:
        phase_start = time.monotonic()
        datasets = load_golden_datasets(args.golden_dir)
        validate_golden_references([result.case for result in report.results], datasets)
        golden_results = [
            result for result in report.results if resolve_golden(result.case, datasets) is not None
        ]
        total_runs = len(golden_results)
        _print_golden_start(args, total_runs)
        for index, result in enumerate(golden_results, start=1):
            _print_golden_case_start(args, index, total_runs, result, phase_start)
            outcome = golden_score(result.case, result, datasets)
            result.golden_score = outcome.golden_score
            result.golden_failures = outcome.golden_failures
            result.golden_pass = outcome.golden_pass
            _print_golden_result(args, index, total_runs, result, phase_start)
        _print_golden_complete(args, report.results, phase_start)
    if args.llm_judge:
        total_runs = len(report.results)
        phase_start = time.monotonic()
        judge_config = _effective_judge_config(args)
        _print_judge_start(args, total_runs, judge_config)
        for index, result in enumerate(report.results, start=1):
            _print_judge_case_start(args, index, total_runs, result, phase_start)
            judged = judge_result(
                result,
                provider=judge_config.provider,
                model=judge_config.model,
                options=judge_config.options,
                timeout_seconds=args.judge_timeout_seconds,
            )
            result.judge_score = judged.score
            result.judge_pass = judged.passed
            result.judge_reasoning = judged.reasoning
            result.judge_strengths = judged.strengths
            result.judge_weaknesses = judged.weaknesses
            result.judge_safety_concerns = judged.safety_concerns
            result.judge_regression_risk = judged.regression_risk
            result.judge_error = judged.error
            _print_judge_result(args, index, total_runs, result, phase_start)
        _print_judge_complete(args, report.results, phase_start)


def _effective_judge_config(args: argparse.Namespace) -> JudgeConfig:
    config = load_judge_config()
    return JudgeConfig(
        provider=args.judge_provider or config.provider,
        model=args.judge_model or config.model,
        options=config.options,
    )


def _apply_blended_scores(args: argparse.Namespace, report: EvalReport) -> None:
    for result in report.results:
        result.combined_score = _combined_score(args, result)
        gate_score = result.combined_score if result.combined_score is not None else result.deterministic_score
        result.passed = gate_score >= PASS_THRESHOLD and not result.validation.hard_failures


def _combined_score(args: argparse.Namespace, result: CaseRunResult) -> int | None:
    has_golden = result.golden_score is not None
    has_judge = result.judge_score is not None
    weights = _score_weights(args, has_golden, has_judge)
    components: list[tuple[float, int]] = [(weights["deterministic"], result.deterministic_score)]
    if has_golden:
        components.append((weights["golden"], result.golden_score))
    if has_judge:
        components.append((weights["judge"], result.judge_score))
    total_weight = sum(weight for weight, _ in components if weight > 0)
    if total_weight <= 0:
        return result.deterministic_score
    blended = round(sum(weight * score for weight, score in components) / total_weight)
    capped = _apply_hard_caps(blended, result.validation.hard_failures)
    if not (has_golden or has_judge):
        return None
    return max(0, min(100, capped))


def _score_weights(args: argparse.Namespace, has_golden: bool, has_judge: bool) -> dict[str, float]:
    if has_golden and has_judge:
        weights = {"deterministic": 0.55, "golden": 0.25, "judge": 0.20}
    elif has_golden:
        weights = {"deterministic": 0.70, "golden": 0.30, "judge": 0.0}
    elif has_judge:
        weights = {"deterministic": 0.75, "golden": 0.0, "judge": 0.25}
    else:
        weights = {"deterministic": 1.0, "golden": 0.0, "judge": 0.0}
    if args.golden_weight is not None:
        weights["golden"] = args.golden_weight
    if args.judge_weight is not None:
        weights["judge"] = args.judge_weight
    return weights


def _apply_hard_caps(score: int, failures: list[str]) -> int:
    caps = {
        "invalid_json": 30,
        "council_crash": 20,
        "winner_missing": 50,
        "final_answer_contradicts_winner": 60,
        "unsafe_instruction_followed": 40,
        "hallucinated_current_facts": 70,
    }
    capped = score
    for failure in failures:
        cap = caps.get(failure)
        if cap is not None:
            capped = min(capped, cap)
    return capped


def _create_eval_sandbox(artifact_dir: Path, run_id: str) -> EvalSandbox:
    sandbox_root = artifact_dir / run_id / "sandbox"
    config_path = sandbox_root / "config" / "council.yaml"
    storage_root = sandbox_root / "storage"
    runtime_root = sandbox_root / "runtime"
    config = load_config()
    config.setdefault("storage", {})
    config.setdefault("runtime", {})
    config["storage"]["council_state_path"] = str(storage_root / "council-state.json")
    config["storage"]["leaderboard_path"] = str(storage_root / "leaderboard.json")
    config["storage"]["memories_path"] = str(storage_root / "memories")
    config["runtime"]["sessions_path"] = str(runtime_root / "sessions")
    config["runtime"]["logs_path"] = str(runtime_root / "logs")
    config["runtime"]["temp_path"] = str(runtime_root / "temp")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(config, config_path)
    return EvalSandbox(
        config_path=config_path,
        roots={
            config_path.parent: Path("config"),
            runtime_root / "logs": Path("runtime/logs"),
            runtime_root / "temp": Path("runtime/temp"),
            storage_root: Path("storage"),
        },
    )


def _artifact_snapshot(roots: dict[Path, Path] | None = None) -> dict[Path, float]:
    if roots is None:
        roots = {
            Path("runtime/logs"): Path("runtime/logs"),
            Path("runtime/temp"): Path("runtime/temp"),
            Path("storage"): Path("storage"),
        }
    snapshot: dict[Path, float] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    snapshot[path] = path.stat().st_mtime
                except OSError:
                    continue
    return snapshot


def _capture_artifacts(
    before: dict[Path, float],
    artifact_dir: Path,
    run_id: str,
    case_id: str,
    repeat_index: int,
    roots: dict[Path, Path] | None = None,
) -> list[str]:
    if roots is None:
        roots = {
            Path("runtime/logs"): Path("runtime/logs"),
            Path("runtime/temp"): Path("runtime/temp"),
            Path("storage"): Path("storage"),
        }
    after = _artifact_snapshot(roots)
    changed = [
        path
        for path, mtime in after.items()
        if path.name != ".gitkeep" and (path not in before or mtime > before[path])
    ]
    if not changed:
        return []
    destination = artifact_dir / run_id / f"{case_id}-r{repeat_index}"
    copied: list[str] = []
    for path in sorted(changed):
        try:
            target = destination / _artifact_relative_path(path, roots)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied.append(str(target))
        except OSError:
            continue
    return copied


def _artifact_relative_path(path: Path, roots: dict[Path, Path]) -> Path:
    resolved_path = path.resolve()
    for root, prefix in roots.items():
        try:
            return prefix / resolved_path.relative_to(root.resolve())
        except ValueError:
            continue
    return Path(path.name)


def load_json_report(path: str | Path) -> EvalReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = _metadata_from_plain(data.get("metadata") or {})
    results = [_result_from_plain(item) for item in data.get("results") or [] if isinstance(item, dict)]
    return EvalReport(metadata=metadata, results=results)


def _enrich_report_cases(report: EvalReport, suite_path: Path) -> None:
    try:
        suite_cases = {case.id: case for case in load_suite(suite_path)}
    except Exception:
        return
    for result in report.results:
        suite_case = suite_cases.get(result.case.id)
        if suite_case is not None:
            result.case = suite_case


def _filter_results(
    results: list[CaseRunResult],
    case_id: str | None = None,
    category: str | None = None,
    tag: str | None = None,
) -> list[CaseRunResult]:
    selected = results
    if case_id:
        selected = [result for result in selected if result.case.id == case_id]
    if category:
        selected = [result for result in selected if result.case.category == category]
    if tag:
        selected = [result for result in selected if tag in result.case.tags]
    return selected


def _failed_cases_from_report(
    args: argparse.Namespace,
    suite_cases: list[EvalCase],
    suite_path: Path,
    parser: argparse.ArgumentParser,
) -> list[EvalCase]:
    failed_report = Path(args.failed_report)
    report = load_json_report(failed_report)
    _enrich_report_cases(report, suite_path)
    failed_results = [result for result in report.results if not result.passed]
    failed_results = _filter_results(
        failed_results,
        case_id=args.case_id,
        category=args.category,
        tag=args.tag,
    )
    if not failed_results:
        parser.error("No failed cases matched the selected filters.")

    suite_by_id = {case.id: case for case in suite_cases}
    cases = [suite_by_id[result.case.id] for result in failed_results if result.case.id in suite_by_id]
    if not cases:
        parser.error("No failed report cases exist in the current suite.")
    return cases


def _metadata_from_plain(data: dict[str, Any]) -> EvalRunMetadata:
    return EvalRunMetadata(
        timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        git_commit=str(data["git_commit"]) if data.get("git_commit") is not None else None,
        version_name=str(data["version_name"]) if data.get("version_name") is not None else None,
        selected_filters=dict(data.get("selected_filters") or {}),
        suite_path=str(data.get("suite_path") or "evals/cases.yaml"),
        repeat=int(data.get("repeat") or 1),
        timeout_seconds=float(data.get("timeout_seconds") or 0),
        council_cmd=str(data.get("council_cmd") or ""),
    )


def _result_from_plain(data: dict[str, Any]) -> CaseRunResult:
    judge_score, judge_weaknesses, judge_error = _judge_score_from_plain(
        data.get("judge_score"), data.get("judge_weaknesses"), data.get("judge_error")
    )
    return CaseRunResult(
        case=_case_from_plain(data.get("case") or {}),
        repeat_index=int(data.get("repeat_index") or 1),
        execution=_execution_from_plain(data.get("execution") or {}),
        validation=_validation_from_plain(data.get("validation") or {}),
        score_breakdown=_score_breakdown_from_plain(data.get("score_breakdown") or {}),
        deterministic_score=int(data.get("deterministic_score") or 0),
        golden_score=_optional_int(data.get("golden_score")),
        golden_failures=_string_list(data.get("golden_failures")),
        golden_pass=_optional_bool(data.get("golden_pass")),
        judge_score=judge_score,
        judge_pass=_optional_bool(data.get("judge_pass")),
        judge_reasoning=_optional_str(data.get("judge_reasoning")),
        judge_strengths=_string_list(data.get("judge_strengths")),
        judge_weaknesses=judge_weaknesses,
        judge_safety_concerns=_string_list(data.get("judge_safety_concerns")),
        judge_regression_risk=_optional_str(data.get("judge_regression_risk")),
        judge_error=judge_error,
        combined_score=_optional_int(data.get("combined_score")),
        artifact_paths=_string_list(data.get("artifact_paths")),
        passed=bool(data.get("passed", False)),
    )


def _judge_score_from_plain(
    value: Any, weaknesses_value: Any, error_value: Any
) -> tuple[int | None, list[str], str | None]:
    score = _optional_int(value)
    weaknesses = _string_list(weaknesses_value)
    error = _optional_str(error_value)
    if score is not None and 1 <= score <= 10:
        score = None
        note = "Judge score rejected because it appears to use a 1-10 scale."
        if note not in weaknesses:
            weaknesses.append(note)
        error = error or note
    return score, weaknesses, error


def _case_from_plain(data: dict[str, Any]) -> EvalCase:
    return EvalCase(
        id=str(data.get("id") or "unknown"),
        name=str(data.get("name") or "unknown"),
        category=str(data.get("category") or "unknown"),
        prompt=str(data.get("prompt") or ""),
        tags=_string_list(data.get("tags")),
        args=_string_list(data.get("args")),
        expected_behavior=_string_list(data.get("expected_behavior")),
        scoring_focus=_string_list(data.get("scoring_focus")),
        hard_failure_rules=_string_list(data.get("hard_failure_rules")),
        golden_ref=_optional_str(data.get("golden_ref")),
        golden=dict(data["golden"]) if isinstance(data.get("golden"), dict) else None,
    )


def _execution_from_plain(data: dict[str, Any]) -> CouncilExecution:
    return CouncilExecution(
        command=_string_list(data.get("command")),
        stdout=str(data.get("stdout") or ""),
        stderr=str(data.get("stderr") or ""),
        duration_seconds=float(data.get("duration_seconds") or 0),
        exit_code=_optional_int(data.get("exit_code")),
        timed_out=bool(data.get("timed_out", False)),
        json_payload=dict(data["json_payload"]) if isinstance(data.get("json_payload"), dict) else None,
        json_error=_optional_str(data.get("json_error")),
    )


def _validation_from_plain(data: dict[str, Any]) -> ValidationResult:
    return ValidationResult(
        valid_json=bool(data.get("valid_json", False)),
        required_fields_present=bool(data.get("required_fields_present", False)),
        recommendation_counts_sensible=bool(data.get("recommendation_counts_sensible", False)),
        winner_exists=bool(data.get("winner_exists", False)),
        final_answer_aligns_with_winner=bool(data.get("final_answer_aligns_with_winner", False)),
        vote_references_valid=bool(data.get("vote_references_valid", False)),
        runoff_counts_valid=bool(data.get("runoff_counts_valid", False)),
        diversity_lanes_present=bool(data.get("diversity_lanes_present", False)),
        safety_passed=bool(data.get("safety_passed", False)),
        hard_failures=_string_list(data.get("hard_failures")),
        warnings=_string_list(data.get("warnings")),
    )


def _score_breakdown_from_plain(data: dict[str, Any]) -> ScoreBreakdown:
    return ScoreBreakdown(
        answers_actual_request=int(data.get("answers_actual_request") or 0),
        practicality=int(data.get("practicality") or 0),
        reasoning_quality=int(data.get("reasoning_quality") or 0),
        tradeoff_awareness=int(data.get("tradeoff_awareness") or 0),
        proposal_diversity=int(data.get("proposal_diversity") or 0),
        internal_consistency=int(data.get("internal_consistency") or 0),
        json_schema_validity=int(data.get("json_schema_validity") or 0),
        safety_resistance=int(data.get("safety_resistance") or 0),
        total_before_caps=int(data.get("total_before_caps") or 0),
        deterministic_score=int(data.get("deterministic_score") or 0),
        applied_caps=_string_list(data.get("applied_caps")),
    )


def _print_start(args: argparse.Namespace, cases: list, total_runs: int) -> None:
    if args.quiet:
        return
    _emit("Small Council evals")
    _emit(f"Suite: {args.suite}")
    _emit(f"Selected: {len(cases)} case(s), repeat: {args.repeat}, runs: {total_runs}")
    _emit(f"Reports: {args.output}, {args.markdown}")
    _emit("")


def _print_skip_start(args: argparse.Namespace, input_report: Path, total_runs: int) -> None:
    if args.quiet:
        return
    _emit("Small Council eval post-processing")
    _emit(f"Input report: {input_report}")
    _emit(f"Loaded: {total_runs} run(s)")
    _emit(f"Reports: {args.output}, {args.markdown}")
    _emit("")


def _print_golden_start(args: argparse.Namespace, total_runs: int) -> None:
    if args.quiet:
        return
    _emit(f"Golden validation: {total_runs} run(s), dir={args.golden_dir}")


def _print_golden_case_start(
    args: argparse.Namespace,
    run_index: int,
    total_runs: int,
    result: CaseRunResult,
    phase_start: float,
) -> None:
    if args.quiet:
        return
    elapsed, eta = _progress_timing(phase_start, run_index - 1, total_runs)
    _emit(
        f"[golden {run_index}/{total_runs}] {result.case.id} repeat {result.repeat_index} "
        f"elapsed={elapsed} eta={eta}"
    )


def _print_golden_result(
    args: argparse.Namespace,
    run_index: int,
    total_runs: int,
    result: CaseRunResult,
    phase_start: float,
) -> None:
    if args.quiet:
        return
    elapsed, eta = _progress_timing(phase_start, run_index, total_runs)
    if result.golden_pass is True:
        status = "PASS"
    elif result.golden_pass is False:
        status = "FAIL"
    else:
        status = "SKIP"
    score = result.golden_score if result.golden_score is not None else "-"
    failures = ",".join(result.golden_failures) or "-"
    _emit(f"  {status} score={score} elapsed={elapsed} eta={eta} failures={failures}")
    if args.verbose:
        for failure in result.golden_failures:
            _emit(f"  golden failure: {failure}")
    _emit("")


def _print_golden_complete(
    args: argparse.Namespace,
    results: list[CaseRunResult],
    phase_start: float,
) -> None:
    if args.quiet:
        return
    scored = [result for result in results if result.golden_score is not None]
    average = _average(result.golden_score for result in scored)
    pass_rate = _rate(result.golden_pass for result in scored)
    _emit(
        f"Golden complete: average={average:.1f} pass_rate={pass_rate:.1f}% "
        f"elapsed={_format_duration(time.monotonic() - phase_start)}"
    )
    _emit("")


def _print_judge_start(args: argparse.Namespace, total_runs: int, judge_config: JudgeConfig) -> None:
    if args.quiet:
        return
    _emit(
        f"LLM judge: {total_runs} run(s), provider={judge_config.provider} "
        f"model={judge_config.model} timeout={args.judge_timeout_seconds:g}s"
    )


def _print_judge_case_start(
    args: argparse.Namespace,
    run_index: int,
    total_runs: int,
    result: CaseRunResult,
    phase_start: float,
) -> None:
    if args.quiet:
        return
    elapsed, eta = _progress_timing(phase_start, run_index - 1, total_runs)
    _emit(
        f"[judge {run_index}/{total_runs}] {result.case.id} repeat {result.repeat_index} "
        f"elapsed={elapsed} eta={eta}"
    )


def _print_judge_result(
    args: argparse.Namespace,
    run_index: int,
    total_runs: int,
    result: CaseRunResult,
    phase_start: float,
) -> None:
    if args.quiet:
        return
    elapsed, eta = _progress_timing(phase_start, run_index, total_runs)
    if result.judge_error:
        _emit(
            f"  ERROR score=- elapsed={elapsed} eta={eta} "
            f"error={_truncate_one_line(result.judge_error)}"
        )
    else:
        status = "PASS" if result.judge_pass else "FAIL"
        score = result.judge_score if result.judge_score is not None else "-"
        risk = result.judge_regression_risk or "unknown"
        _emit(f"  {status} score={score} risk={risk} elapsed={elapsed} eta={eta}")
    if args.verbose:
        if result.judge_reasoning:
            _emit(f"  judge reasoning: {_truncate_one_line(result.judge_reasoning)}")
        if result.judge_strengths:
            _emit(f"  strengths: {_truncate_one_line(', '.join(result.judge_strengths))}")
        if result.judge_weaknesses:
            _emit(f"  weaknesses: {_truncate_one_line(', '.join(result.judge_weaknesses))}")
        if result.judge_safety_concerns:
            _emit(f"  safety concerns: {_truncate_one_line(', '.join(result.judge_safety_concerns))}")
        if result.judge_error:
            _emit(f"  judge error: {_truncate_one_line(result.judge_error, limit=500)}")
    _emit("")


def _print_judge_complete(
    args: argparse.Namespace,
    results: list[CaseRunResult],
    phase_start: float,
) -> None:
    if args.quiet:
        return
    scored = [result for result in results if result.judge_score is not None]
    errors = sum(1 for result in results if result.judge_error)
    average = _average(result.judge_score for result in scored)
    pass_rate = _rate(result.judge_pass for result in scored)
    _emit(
        f"Judge complete: average={average:.1f} pass_rate={pass_rate:.1f}% "
        f"errors={errors} elapsed={_format_duration(time.monotonic() - phase_start)}"
    )
    _emit("")


def _print_case_start(
    args: argparse.Namespace,
    run_index: int,
    total_runs: int,
    case,
    repeat_index: int,
    suite_start: float,
) -> None:
    if args.quiet:
        return
    elapsed, eta = _progress_timing(suite_start, run_index - 1, total_runs)
    _emit(
        f"[{run_index}/{total_runs}] {case.id} {case.name} "
        f"({case.category}) repeat {repeat_index}/{args.repeat} "
        f"elapsed={elapsed} eta={eta}"
    )


def _print_case_result(
    args: argparse.Namespace,
    result: CaseRunResult,
    suite_start: float,
    completed_runs: int,
    total_runs: int,
) -> None:
    if args.quiet:
        return
    validation = result.validation
    execution = result.execution
    status = "PASS" if result.passed else "FAIL"
    json_status = "ok" if validation.valid_json else "invalid"
    elapsed, eta = _progress_timing(suite_start, completed_runs, total_runs)
    details = [
        f"  {status}",
        f"score={result.deterministic_score}",
        f"duration={execution.duration_seconds:.1f}s",
        f"json={json_status}",
        f"elapsed={elapsed}",
        f"eta={eta}",
    ]
    if validation.hard_failures:
        details.append(f"failures={','.join(validation.hard_failures)}")
    _emit(" ".join(details))
    if args.verbose and (not result.passed or validation.warnings):
        for warning in validation.warnings:
            _emit(f"  warning: {warning}")
        stderr = execution.stderr.strip()
        if stderr:
            _emit(f"  stderr: {_truncate_one_line(stderr)}")
    _emit("")


def _print_summary(
    args: argparse.Namespace,
    results: list[CaseRunResult],
    suite_start: float,
    comparison: list[str],
) -> None:
    if args.quiet:
        return
    average = _average(_score_from_result(result) for result in results)
    pass_rate = _rate(result.passed for result in results)
    json_rate = _rate(result.validation.valid_json for result in results)
    _emit("Summary")
    _emit(f"  average score: {average:.1f}")
    _emit(f"  pass rate: {pass_rate:.1f}%")
    _emit(f"  JSON validity: {json_rate:.1f}%")
    golden_scores = [result.golden_score for result in results if result.golden_score is not None]
    if golden_scores:
        _emit(f"  golden average: {_average(golden_scores):.1f}")
        _emit(f"  golden pass rate: {_rate(result.golden_pass for result in results if result.golden_pass is not None):.1f}%")
    judge_scores = [result.judge_score for result in results if result.judge_score is not None]
    if judge_scores:
        _emit(f"  judge average: {_average(judge_scores):.1f}")
    _emit(f"  elapsed: {_format_duration(time.monotonic() - suite_start)}")
    _emit(f"  wrote: {args.output}")
    _emit(f"  wrote: {args.markdown}")
    _emit("")
    _emit("Previous comparison")
    for line in comparison:
        _emit(f"  {line}")


def _backup_previous_report(path: str | Path) -> Path:
    report_path = Path(path)
    previous_path = _previous_path(report_path)
    if report_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, previous_path)
    return previous_path


def _previous_path(path: Path) -> Path:
    return path.with_name(f"previous{path.suffix}")


def _compare_with_previous(previous_path: Path, report: EvalReport) -> list[str]:
    if not previous_path.exists():
        return ["previous report: none"]
    try:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"previous report: unreadable ({exc})"]

    current_results = report.results
    previous_results = previous.get("results")
    if not isinstance(previous_results, list):
        return ["previous report: unreadable (missing results)"]

    lines = [
        f"runs: {len(current_results)} ({_signed_number(len(current_results) - len(previous_results))})",
        _metric_delta(
            "average score",
            _average(_score_from_result(result) for result in current_results),
            _average(_score_from_plain(result) for result in previous_results),
        ),
        _metric_delta(
            "pass rate",
            _rate(result.passed for result in current_results),
            _rate(bool(result.get("passed", False)) for result in previous_results),
            suffix="%",
        ),
        _metric_delta(
            "JSON validity",
            _rate(result.validation.valid_json for result in current_results),
            _rate(_valid_json_from_plain(result) for result in previous_results),
            suffix="%",
        ),
    ]

    previous_by_key = {_plain_result_key(result): result for result in previous_results}
    current_by_key = {_result_key(result): result for result in current_results}
    added = sorted(set(current_by_key) - set(previous_by_key))
    removed = sorted(set(previous_by_key) - set(current_by_key))
    changed = _changed_case_lines(previous_by_key, current_by_key)
    lines.append(f"case runs added: {len(added)} removed: {len(removed)} changed: {len(changed)}")
    lines.extend(_category_delta_lines(previous_results, current_results))
    lines.extend(_regression_lines(previous_by_key, current_by_key))
    lines.extend(changed)
    return lines


def _changed_case_lines(
    previous_by_key: dict[tuple[str, int], dict[str, Any]],
    current_by_key: dict[tuple[str, int], CaseRunResult],
) -> list[str]:
    lines: list[str] = []
    for key in sorted(set(previous_by_key) & set(current_by_key)):
        previous = previous_by_key[key]
        current = current_by_key[key]
        previous_score = _score_from_plain(previous)
        current_score = _score_from_result(current)
        previous_passed = bool(previous.get("passed", False))
        current_passed = current.passed
        previous_failures = _all_failures_from_plain(previous)
        current_failures = _all_failures_from_result(current)
        changes: list[str] = []
        if current_score != previous_score:
            changes.append(f"score {previous_score}->{current_score} ({_signed_number(current_score - previous_score)})")
        if current_passed != previous_passed:
            changes.append(
                f"pass {'yes' if previous_passed else 'no'}->{'yes' if current_passed else 'no'}"
            )
        if current_failures != previous_failures:
            before = ",".join(previous_failures) or "-"
            after = ",".join(current_failures) or "-"
            changes.append(f"failures {before}->{after}")
        if changes:
            case_id, repeat_index = key
            lines.append(f"{case_id} repeat {repeat_index}: " + "; ".join(changes))
    return lines


def _result_key(result: CaseRunResult) -> tuple[str, int]:
    return (result.case.id, result.repeat_index)


def _plain_result_key(result: dict[str, Any]) -> tuple[str, int]:
    case = result.get("case") if isinstance(result, dict) else {}
    case_id = case.get("id", "unknown") if isinstance(case, dict) else "unknown"
    return (str(case_id), int(result.get("repeat_index", 0)))


def _score_from_plain(result: dict[str, Any]) -> int:
    value = result.get("combined_score")
    if value is None:
        value = result.get("deterministic_score", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _score_from_result(result: CaseRunResult) -> int:
    return result.combined_score if result.combined_score is not None else result.deterministic_score


def _valid_json_from_plain(result: dict[str, Any]) -> bool:
    validation = result.get("validation")
    if not isinstance(validation, dict):
        return False
    return bool(validation.get("valid_json", False))


def _hard_failures_from_plain(result: dict[str, Any]) -> list[str]:
    validation = result.get("validation")
    if not isinstance(validation, dict):
        return []
    failures = validation.get("hard_failures")
    if not isinstance(failures, list):
        return []
    return [str(failure) for failure in failures]


def _all_failures_from_result(result: CaseRunResult) -> list[str]:
    failures = list(result.validation.hard_failures)
    failures.extend(f"golden:{item}" for item in result.golden_failures)
    if result.judge_error:
        failures.append("judge_error")
    if result.judge_pass is False:
        failures.append("judge_fail")
    return failures


def _all_failures_from_plain(result: dict[str, Any]) -> list[str]:
    failures = _hard_failures_from_plain(result)
    golden = result.get("golden_failures")
    if isinstance(golden, list):
        failures.extend(f"golden:{item}" for item in golden)
    if result.get("judge_error"):
        failures.append("judge_error")
    if result.get("judge_pass") is False:
        failures.append("judge_fail")
    return failures


def _category_delta_lines(
    previous_results: list[dict[str, Any]],
    current_results: list[CaseRunResult],
) -> list[str]:
    previous_categories: dict[str, list[dict[str, Any]]] = {}
    current_categories: dict[str, list[CaseRunResult]] = {}
    for result in previous_results:
        case = result.get("case") if isinstance(result, dict) else {}
        category = str(case.get("category", "unknown")) if isinstance(case, dict) else "unknown"
        previous_categories.setdefault(category, []).append(result)
    for result in current_results:
        current_categories.setdefault(result.case.category, []).append(result)
    lines: list[str] = []
    for category in sorted(set(previous_categories) | set(current_categories)):
        previous_items = previous_categories.get(category, [])
        current_items = current_categories.get(category, [])
        lines.append(
            "category "
            + _metric_delta(
                category,
                _average(_score_from_result(item) for item in current_items),
                _average(_score_from_plain(item) for item in previous_items),
            )
        )
    return lines


def _regression_lines(
    previous_by_key: dict[tuple[str, int], dict[str, Any]],
    current_by_key: dict[tuple[str, int], CaseRunResult],
) -> list[str]:
    regressions: list[str] = []
    improvements: list[str] = []
    for key in sorted(set(previous_by_key) & set(current_by_key)):
        previous = previous_by_key[key]
        current = current_by_key[key]
        previous_score = _score_from_plain(previous)
        current_score = _score_from_result(current)
        previous_passed = bool(previous.get("passed", False))
        previous_failures = set(_all_failures_from_plain(previous))
        current_failures = set(_all_failures_from_result(current))
        label = f"{key[0]} repeat {key[1]}"
        if current_score < previous_score or (previous_passed and not current.passed) or current_failures - previous_failures:
            regressions.append(label)
        if current_score > previous_score or (not previous_passed and current.passed) or previous_failures - current_failures:
            improvements.append(label)
    return [
        f"regressions: {', '.join(regressions) if regressions else 'none'}",
        f"improvements: {', '.join(improvements) if improvements else 'none'}",
    ]


def _metric_delta(label: str, current: float, previous: float, suffix: str = "") -> str:
    delta = current - previous
    return f"{label}: {current:.1f}{suffix} ({_signed_float(delta)}{suffix})"


def _signed_number(value: int) -> str:
    return f"{value:+d}"


def _signed_float(value: float) -> str:
    return f"{value:+.1f}"


def _progress_timing(
    suite_start: float,
    completed_runs: int,
    total_runs: int,
) -> tuple[str, str]:
    elapsed_seconds = max(0.0, time.monotonic() - suite_start)
    elapsed = _format_duration(elapsed_seconds)
    if completed_runs <= 0:
        return elapsed, "unknown"
    remaining_runs = max(0, total_runs - completed_runs)
    eta_seconds = (elapsed_seconds / completed_runs) * remaining_runs
    return elapsed, _format_duration(eta_seconds)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds_part = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds_part:02d}s"
    hours, minutes_part = divmod(minutes, 60)
    return f"{hours}h {minutes_part:02d}m {seconds_part:02d}s"


def _average(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _rate(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return 100.0 * sum(1 for item in items if item) / len(items)


def _truncate_one_line(text: str, limit: int = 240) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3].rstrip() + "..."


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _run_id(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("+", "Z").replace(".", "-")


def _emit(message: str, stream: TextIO | None = None) -> None:
    print(message, file=stream, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
