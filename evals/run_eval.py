from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .models import CaseRunResult, CouncilExecution, EvalReport, EvalRunMetadata
from .report import PASS_THRESHOLD, write_json_report, write_markdown_report
from .scorers import score_case, validate_result
from .utils import extract_last_valid_json, filter_cases, git_commit, load_suite


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
    parser.add_argument("--council-cmd", default="./council")
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
    _print_start(args, cases, total_runs)
    run_index = 0
    for case in cases:
        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            _print_case_start(args, run_index, total_runs, case, repeat_index)
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
            _print_case_result(args, result)

    report = EvalReport(metadata=metadata, results=results)
    write_json_report(report, args.output)
    write_markdown_report(report, args.markdown)
    _print_summary(args, results)
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
) -> None:
    if args.quiet:
        return
    _emit(
        f"[{run_index}/{total_runs}] {case.id} {case.name} "
        f"({case.category}) repeat {repeat_index}/{args.repeat}"
    )


def _print_case_result(args: argparse.Namespace, result: CaseRunResult) -> None:
    if args.quiet:
        return
    validation = result.validation
    execution = result.execution
    status = "PASS" if result.passed else "FAIL"
    json_status = "ok" if validation.valid_json else "invalid"
    details = [
        f"  {status}",
        f"score={result.deterministic_score}",
        f"duration={execution.duration_seconds:.1f}s",
        f"json={json_status}",
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


def _print_summary(args: argparse.Namespace, results: list[CaseRunResult]) -> None:
    if args.quiet:
        return
    average = _average(result.deterministic_score for result in results)
    pass_rate = _rate(result.passed for result in results)
    json_rate = _rate(result.validation.valid_json for result in results)
    _emit("Summary")
    _emit(f"  average score: {average:.1f}")
    _emit(f"  pass rate: {pass_rate:.1f}%")
    _emit(f"  JSON validity: {json_rate:.1f}%")
    _emit(f"  wrote: {args.output}")
    _emit(f"  wrote: {args.markdown}")


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
