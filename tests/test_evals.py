from __future__ import annotations

import io
import json
import base64
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evals.models import CouncilExecution, EvalReport, EvalRunMetadata
from evals.report import render_markdown, write_json_report
from evals.run_eval import execute_case, main
from evals.scorers import score_case, validate_result
from evals.utils import extract_last_valid_json, filter_cases, load_suite


class EvalSuiteTests(unittest.TestCase):
    def test_cases_load_with_unique_ids_and_smoke_suite(self) -> None:
        cases = load_suite("evals/cases.yaml")
        ids = [case.id for case in cases]

        self.assertEqual(len(ids), len(set(ids)))
        for index in range(1, 11):
            self.assertIn(f"SMOKE{index:02d}", ids)
        self.assertGreaterEqual(len(cases), 40)

    def test_filters_by_case_category_and_tag(self) -> None:
        cases = load_suite("evals/cases.yaml")

        self.assertEqual(["SMOKE01"], [case.id for case in filter_cases(cases, case_id="SMOKE01")])
        self.assertTrue(all(case.category == "safety" for case in filter_cases(cases, category="safety")))
        self.assertTrue(all("showcase" in case.tags for case in filter_cases(cases, tag="showcase")))


class JsonExtractionTests(unittest.TestCase):
    def test_extracts_pure_json(self) -> None:
        payload, error = extract_last_valid_json('{"a": 1, "b": {"c": 2}}')
        self.assertIsNone(error)
        self.assertEqual({"a": 1, "b": {"c": 2}}, payload)

    def test_extracts_last_json_after_noise(self) -> None:
        payload, error = extract_last_valid_json('progress {"bad": }\nfinal\n{"ok": true}')
        self.assertIsNone(error)
        self.assertEqual({"ok": True}, payload)

    def test_reports_invalid_output(self) -> None:
        payload, error = extract_last_valid_json("no json here")
        self.assertIsNone(payload)
        self.assertIn("No JSON", error)


class ScoringTests(unittest.TestCase):
    def test_invalid_json_cap(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        execution = CouncilExecution(
            command=["./council"],
            stdout="oops",
            stderr="",
            duration_seconds=0.1,
            exit_code=0,
            json_payload=None,
            json_error="bad json",
        )

        validation = validate_result(case, execution)
        score = score_case(case, execution, validation)

        self.assertIn("invalid_json", validation.hard_failures)
        self.assertLessEqual(score.deterministic_score, 30)

    def test_mock_payload_scores_and_reports(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        payload = {
            "final_output": "Choose Arrival because it balances wonder, emotion, and thoughtful science-fiction tradeoffs.",
            "status": "resolved",
            "winning_option": "Arrival",
            "draft_recommendations": [
                {"proposer": "A", "recommendation": "Arrival", "short_reasoning": "Thoughtful.", "pros": ["Smart"], "cons": []},
                {"proposer": "B", "recommendation": "Alien", "short_reasoning": "Tense.", "pros": ["Iconic"], "cons": []},
                {"proposer": "C", "recommendation": "Blade Runner 2049", "short_reasoning": "Beautiful.", "pros": ["Visuals"], "cons": []}
            ],
            "final_recommendations": [
                {"proposer": "A", "recommendation": "Arrival", "short_reasoning": "Thoughtful.", "pros": ["Smart"], "cons": []}
            ],
            "recommendation_groups": [{"canonical_option": "Arrival", "proposers": ["A"], "member_recommendations": ["Arrival"]}],
            "votes": [
                {"voter": "A", "selected_option": "Arrival", "round": 0},
                {"voter": "B", "selected_option": "Arrival", "round": 0},
                {"voter": "C", "selected_option": "Arrival", "round": 0}
            ],
            "vote_rounds": [{"round_number": 0, "vote_counts": {"Arrival": 3}, "tied_options": [], "resolved": True, "winning_option": "Arrival"}],
            "leaderboard": [],
            "runoff_rounds": 0,
            "max_runoff_rounds": 3,
            "diversity_mode": "balanced",
            "diversity_lanes": {"A": "mainstream", "B": "mainstream", "C": "mainstream"}
        }
        execution = CouncilExecution(
            command=["./council"],
            stdout=json.dumps(payload),
            stderr="",
            duration_seconds=0.1,
            exit_code=0,
            json_payload=payload,
        )
        validation = validate_result(case, execution)
        score = score_case(case, execution, validation)

        self.assertTrue(validation.valid_json)
        self.assertGreaterEqual(score.deterministic_score, 70)

        result = __import__("evals.models", fromlist=["CaseRunResult"]).CaseRunResult(
            case=case,
            repeat_index=1,
            execution=execution,
            validation=validation,
            score_breakdown=score,
            deterministic_score=score.deterministic_score,
            passed=True,
        )
        report = EvalReport(
            metadata=EvalRunMetadata(
                timestamp="2026-05-30T00:00:00+00:00",
                git_commit="abc",
                version_name="test",
                selected_filters={"case": "SMOKE01"},
                suite_path="evals/cases.yaml",
                repeat=1,
                timeout_seconds=1,
                council_cmd="./council",
            ),
            results=[result],
        )
        markdown = render_markdown(report)
        self.assertIn("Average score", markdown)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_json_report(report, path)
            self.assertEqual("SMOKE01", json.loads(path.read_text())["results"][0]["case"]["id"])


class RunnerTests(unittest.TestCase):
    def test_main_prints_progress_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--case",
                        "SMOKE01",
                        "--timeout-seconds",
                        "5",
                        "--council-cmd",
                        _mock_council_command(_valid_payload()),
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            rendered = stdout.getvalue()
            self.assertEqual(0, exit_code)
            self.assertIn("Small Council evals", rendered)
            self.assertIn("[1/1] SMOKE01 Movie Night (smoke) repeat 1/1", rendered)
            self.assertIn("PASS score=", rendered)
            self.assertIn("Summary", rendered)
            self.assertIn(f"wrote: {output}", rendered)

    def test_main_quiet_suppresses_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--case",
                        "SMOKE01",
                        "--quiet",
                        "--timeout-seconds",
                        "5",
                        "--council-cmd",
                        _mock_council_command(_valid_payload()),
                        "--output",
                        str(Path(tmp) / "latest.json"),
                        "--markdown",
                        str(Path(tmp) / "latest.md"),
                    ]
                )

            self.assertEqual(0, exit_code)
            self.assertEqual("", stdout.getvalue())

    def test_main_verbose_prints_failure_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            command = "python -c \"import sys; print('not json'); print('model failed loudly', file=sys.stderr)\""

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--case",
                        "SMOKE01",
                        "--verbose",
                        "--timeout-seconds",
                        "5",
                        "--council-cmd",
                        command,
                        "--output",
                        str(Path(tmp) / "latest.json"),
                        "--markdown",
                        str(Path(tmp) / "latest.md"),
                    ]
                )

            rendered = stdout.getvalue()
            self.assertEqual(0, exit_code)
            self.assertIn("FAIL score=", rendered)
            self.assertIn("failures=invalid_json", rendered)
            self.assertIn("warning:", rendered)
            self.assertIn("stderr: model failed loudly", rendered)

    def test_execute_case_with_mock_command(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        command = (
            "python -c \"import json, sys; "
            "print('noise'); "
            "print(json.dumps({'final_output':'Choose Arrival','winning_option':'Arrival'}))\""
        )

        execution = execute_case(case, command, timeout_seconds=5)

        self.assertEqual(0, execution.exit_code)
        self.assertEqual("Arrival", execution.json_payload["winning_option"])


def _valid_payload() -> dict:
    return {
        "final_output": "Choose Arrival because it balances wonder, emotion, and thoughtful science-fiction tradeoffs.",
        "status": "resolved",
        "winning_option": "Arrival",
        "draft_recommendations": [
            {"proposer": "A", "recommendation": "Arrival", "short_reasoning": "Thoughtful.", "pros": ["Smart"], "cons": []},
            {"proposer": "B", "recommendation": "Alien", "short_reasoning": "Tense.", "pros": ["Iconic"], "cons": []},
            {"proposer": "C", "recommendation": "Blade Runner 2049", "short_reasoning": "Beautiful.", "pros": ["Visuals"], "cons": []}
        ],
        "final_recommendations": [
            {"proposer": "A", "recommendation": "Arrival", "short_reasoning": "Thoughtful.", "pros": ["Smart"], "cons": []}
        ],
        "recommendation_groups": [{"canonical_option": "Arrival", "proposers": ["A"], "member_recommendations": ["Arrival"]}],
        "votes": [
            {"voter": "A", "selected_option": "Arrival", "round": 0},
            {"voter": "B", "selected_option": "Arrival", "round": 0},
            {"voter": "C", "selected_option": "Arrival", "round": 0}
        ],
        "vote_rounds": [{"round_number": 0, "vote_counts": {"Arrival": 3}, "tied_options": [], "resolved": True, "winning_option": "Arrival"}],
        "leaderboard": [],
        "runoff_rounds": 0,
        "max_runoff_rounds": 3,
        "diversity_mode": "balanced",
        "diversity_lanes": {"A": "mainstream", "B": "mainstream", "C": "mainstream"}
    }


def _mock_council_command(payload: dict) -> str:
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f'python -c "import base64; print(base64.b64decode(\'{encoded}\').decode())"'


if __name__ == "__main__":
    unittest.main()
