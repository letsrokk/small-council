from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .models import CaseRunResult, EvalReport, to_plain_data


PASS_THRESHOLD = 70


def write_json_report(report: EvalReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_plain_data(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(report: EvalReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: EvalReport) -> str:
    results = report.results
    average = _average(_effective_score(result) for result in results)
    deterministic_average = _average(result.deterministic_score for result in results)
    pass_rate = _rate(result.passed for result in results)
    json_rate = _rate(result.validation.valid_json for result in results)
    safety_results = [result for result in results if result.case.category == "safety"]
    safety_rate = _rate(result.validation.safety_passed for result in safety_results)
    has_golden = any(result.golden_score is not None for result in results)
    has_judge = any(result.judge_score is not None or result.judge_error for result in results)
    has_combined = any(result.combined_score is not None for result in results)

    lines = [
        "# Small Council Eval Report",
        "",
        "## Metadata",
        "",
        f"- Timestamp: `{report.metadata.timestamp}`",
        f"- Git commit: `{report.metadata.git_commit or 'unknown'}`",
        f"- Version: `{report.metadata.version_name or 'unspecified'}`",
        f"- Suite: `{report.metadata.suite_path}`",
        f"- Repeat: `{report.metadata.repeat}`",
        f"- Council command: `{report.metadata.council_cmd}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases run | {len(results)} |",
        f"| Average score | {average:.1f} |",
        f"| Deterministic average | {deterministic_average:.1f} |",
        f"| Pass rate | {pass_rate:.1f}% |",
        f"| JSON validity rate | {json_rate:.1f}% |",
        f"| Safety pass rate | {safety_rate:.1f}% |",
        "",
        "## Category Breakdown",
        "",
        "| Category | Runs | Average | Pass rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, category_results in _by_category(results).items():
        lines.append(
            f"| {category} | {len(category_results)} | "
            f"{_average(_effective_score(item) for item in category_results):.1f} | "
            f"{_rate(item.passed for item in category_results):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| Case | Category | Repeat | Deterministic | Combined | Pass | Hard failures |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in results:
        failures = ", ".join(result.validation.hard_failures) or "-"
        combined = str(result.combined_score) if result.combined_score is not None else "-"
        lines.append(
            f"| {result.case.id} | {result.case.category} | {result.repeat_index} | "
            f"{result.deterministic_score} | {combined} | {'yes' if result.passed else 'no'} | {failures} |"
        )

    if has_golden:
        golden_results = [result for result in results if result.golden_score is not None]
        lines.extend(
            [
                "",
                "## Golden Dataset Results",
                "",
                "| Case | Repeat | Score | Pass | Failures |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for result in golden_results:
            failures = ", ".join(result.golden_failures) or "-"
            lines.append(
                f"| {result.case.id} | {result.repeat_index} | {result.golden_score} | "
                f"{'yes' if result.golden_pass else 'no'} | {failures} |"
            )
        lines.extend(
            [
                "",
                f"Golden pass rate: {_rate(result.golden_pass for result in golden_results):.1f}%",
            ]
        )

    if has_judge:
        lines.extend(
            [
                "",
                "## Judge Results",
                "",
                "| Case | Repeat | Score | Strengths | Weaknesses | Safety concerns | Error |",
                "| --- | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for result in results:
            if result.judge_score is None and not result.judge_error:
                continue
            score = str(result.judge_score) if result.judge_score is not None else "-"
            lines.append(
                f"| {result.case.id} | {result.repeat_index} | {score} | "
                f"{_join_cell(result.judge_strengths)} | {_join_cell(result.judge_weaknesses)} | "
                f"{_join_cell(result.judge_safety_concerns)} | {result.judge_error or '-'} |"
            )

    if has_combined:
        lines.extend(
            [
                "",
                "## Combined Results",
                "",
                "| Case | Repeat | Deterministic | Golden | Judge | Blended |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in results:
            lines.append(
                f"| {result.case.id} | {result.repeat_index} | {result.deterministic_score} | "
                f"{_score_cell(result.golden_score)} | {_score_cell(result.judge_score)} | "
                f"{_score_cell(result.combined_score)} |"
            )

    lines.extend(["", "## Top Failures", ""])
    failures = sorted(results, key=_effective_score)[:10]
    if not failures:
        lines.append("No failures.")
    for result in failures:
        if result.passed:
            continue
        details = ", ".join(result.validation.hard_failures or result.validation.warnings) or "Low score."
        lines.append(f"- `{result.case.id}` scored {_effective_score(result)}: {details}")

    return "\n".join(lines).rstrip() + "\n"


def _by_category(results: list[CaseRunResult]) -> dict[str, list[CaseRunResult]]:
    grouped: dict[str, list[CaseRunResult]] = defaultdict(list)
    for result in results:
        grouped[result.case.category].append(result)
    return dict(sorted(grouped.items()))


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


def _effective_score(result: CaseRunResult) -> int:
    return result.combined_score if result.combined_score is not None else result.deterministic_score


def _score_cell(value: int | None) -> str:
    return str(value) if value is not None else "-"


def _join_cell(values: list[str]) -> str:
    return ", ".join(values) if values else "-"
