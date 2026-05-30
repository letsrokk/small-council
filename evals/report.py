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
    average = _average(result.deterministic_score for result in results)
    pass_rate = _rate(result.passed for result in results)
    json_rate = _rate(result.validation.valid_json for result in results)
    safety_results = [result for result in results if result.case.category == "safety"]
    safety_rate = _rate(result.validation.safety_passed for result in safety_results)

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
            f"{_average(item.deterministic_score for item in category_results):.1f} | "
            f"{_rate(item.passed for item in category_results):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| Case | Category | Repeat | Score | Pass | Hard failures |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for result in results:
        failures = ", ".join(result.validation.hard_failures) or "-"
        lines.append(
            f"| {result.case.id} | {result.case.category} | {result.repeat_index} | "
            f"{result.deterministic_score} | {'yes' if result.passed else 'no'} | {failures} |"
        )

    lines.extend(["", "## Top Failures", ""])
    failures = sorted(results, key=lambda item: item.deterministic_score)[:10]
    if not failures:
        lines.append("No failures.")
    for result in failures:
        if result.passed:
            continue
        details = ", ".join(result.validation.hard_failures or result.validation.warnings) or "Low score."
        lines.append(f"- `{result.case.id}` scored {result.deterministic_score}: {details}")

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

