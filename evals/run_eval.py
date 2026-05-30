from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .models import CaseRunResult, CouncilExecution, EvalReport, EvalRunMetadata
from .report import PASS_THRESHOLD, write_json_report, write_markdown_report
from .scorers import score_case, validate_result
from .utils import extract_last_valid_json, filter_cases, git_commit, load_suite


DEFAULT_COUNCIL_CMD = "./council --secretary local"


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

    suite_path = Path(args.suite)
    cases = filter_cases(
        load_suite(suite_path),
        case_id=args.case_id,
        category=args.category,
        tag=args.tag,
    )
    if not cases:
        parser.error("No cases matched the selected filters.")

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
    previous_json_path = _backup_previous_report(args.output)
    _backup_previous_report(args.markdown)
    suite_start = time.monotonic()
    _print_start(args, cases, total_runs)
    run_index = 0
    for case in cases:
        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            _print_case_start(args, run_index, total_runs, case, repeat_index, suite_start)
            execution = execute_case(case, args.council_cmd, args.timeout_seconds)
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
                passed=deterministic_score >= PASS_THRESHOLD
                and not validation.hard_failures,
            )
            results.append(result)
            _print_case_result(args, result, suite_start, len(results), total_runs)

    report = EvalReport(metadata=metadata, results=results)
    write_json_report(report, args.output)
    write_markdown_report(report, args.markdown)
    comparison = _compare_with_previous(previous_json_path, report)
    _print_summary(args, results, suite_start, comparison)
    return 0


def execute_case(case, council_cmd: str, timeout_seconds: float) -> CouncilExecution:
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
            env=_benchmark_env(),
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


def _benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SMALL_COUNCIL_BENCHMARK"] = "1"
    return env


def _print_start(args: argparse.Namespace, cases: list, total_runs: int) -> None:
    if args.quiet:
        return
    _emit("Small Council evals")
    _emit(f"Suite: {args.suite}")
    _emit(f"Selected: {len(cases)} case(s), repeat: {args.repeat}, runs: {total_runs}")
    _emit(f"Reports: {args.output}, {args.markdown}")
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
    average = _average(result.deterministic_score for result in results)
    pass_rate = _rate(result.passed for result in results)
    json_rate = _rate(result.validation.valid_json for result in results)
    _emit("Summary")
    _emit(f"  average score: {average:.1f}")
    _emit(f"  pass rate: {pass_rate:.1f}%")
    _emit(f"  JSON validity: {json_rate:.1f}%")
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
            _average(result.deterministic_score for result in current_results),
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
        current_score = current.deterministic_score
        previous_passed = bool(previous.get("passed", False))
        current_passed = current.passed
        previous_failures = _hard_failures_from_plain(previous)
        current_failures = current.validation.hard_failures
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
    value = result.get("deterministic_score", 0)
    return int(value) if isinstance(value, (int, float)) else 0


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


def _emit(message: str, stream: TextIO | None = None) -> None:
    print(message, file=stream, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
