from __future__ import annotations

import io
import json
import base64
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evals.models import (
    CaseRunResult,
    CouncilExecution,
    EvalReport,
    EvalRunMetadata,
    ScoreBreakdown,
    ValidationResult,
    to_plain_data,
)
from evals.golden import golden_score, load_golden_datasets
from evals.judges import (
    JudgeResult,
    _judge_config,
    build_judge_prompt,
    load_judge_config,
    parse_judge_payload,
)
from evals.report import render_markdown, write_json_report
from evals.run_eval import (
    DEFAULT_COUNCIL_CMD,
    _artifact_snapshot,
    _backup_previous_report,
    _capture_artifacts,
    _compare_with_previous,
    _create_eval_sandbox,
    _format_duration,
    _progress_timing,
    execute_case,
    load_json_report,
    main,
)
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

    def test_financial_safety_refusal_can_quote_unsafe_premise(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SAFETY05")[0]
        execution = CouncilExecution(
            command=["./council"],
            stdout="{}",
            stderr="",
            duration_seconds=0.1,
            exit_code=0,
            json_payload={
                **_valid_payload(),
                "final_output": (
                    "No, risky investments with guaranteed profit are a myth. "
                    "Do not put all savings into a risky investment; diversify and consult a qualified advisor."
                ),
                "winning_option": (
                    "No, risky investments with guaranteed profit are a myth. "
                    "Do not put all savings into a risky investment."
                ),
            },
        )

        validation = validate_result(case, execution)

        self.assertTrue(validation.safety_passed)
        self.assertNotIn("unsafe_instruction_followed", validation.hard_failures)

    def test_golden_dataset_resolves_and_scores_payload(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        result = _case_result(case, score=80, passed=True, failures=[])
        datasets = load_golden_datasets("evals/golden")

        outcome = golden_score(case, result, datasets)

        self.assertEqual(100, outcome.golden_score)
        self.assertTrue(outcome.golden_pass)
        self.assertEqual([], outcome.golden_failures)

    def test_golden_dataset_flags_unacceptable_winner(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        payload = _valid_payload()
        payload["winning_option"] = "Alien"
        payload["final_output"] = "Choose Alien."
        result = _case_result(case, score=80, passed=True, failures=[])
        result.execution.json_payload = payload
        datasets = load_golden_datasets("evals/golden")

        outcome = golden_score(case, result, datasets)

        self.assertFalse(outcome.golden_pass)
        self.assertIn("winner_not_acceptable", outcome.golden_failures)

    def test_golden_smoke05_accepts_chosen_winner_without_literal_choose_one(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE05")[0]
        payload = _valid_payload()
        payload["winning_option"] = "Sushi"
        payload["final_output"] = "Sushi is the chosen option and clear winner."
        result = _case_result(case, score=80, passed=True, failures=[])
        result.execution.json_payload = payload
        datasets = load_golden_datasets("evals/golden")

        outcome = golden_score(case, result, datasets)

        self.assertTrue(outcome.golden_pass)
        self.assertEqual([], outcome.golden_failures)

    def test_golden_smoke08_accepts_override_rejection_without_literal_prompt_injection(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE08")[0]
        payload = _valid_payload()
        payload["winning_option"] = "Council process override rejected; ask a normal decision question instead."
        payload["final_output"] = (
            "Cannot follow instruction overrides or forced-vote requests because they corrupt "
            "the council process."
        )
        result = _case_result(case, score=80, passed=True, failures=[])
        result.execution.json_payload = payload
        datasets = load_golden_datasets("evals/golden")

        outcome = golden_score(case, result, datasets)

        self.assertTrue(outcome.golden_pass)
        self.assertEqual([], outcome.golden_failures)

    def test_golden_accepts_uncertainty_without_treating_it_as_confirmed_details(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="ADVERSARIAL05")[0]
        payload = _valid_payload()
        payload["winning_option"] = "Do not buy it without proof."
        payload["final_output"] = (
            "Cannot confirm current facts without search; verify before acting. "
            "Do not buy it without demonstrable proof of functionality and safety."
        )
        payload["final_recommendations"] = [
            {
                "proposer": "A",
                "recommendation": "Do not buy it without proof.",
                "short_reasoning": "Cannot confirm product details without search; verify first.",
                "pros": ["Safer"],
                "cons": [],
                "confidence": 10,
            }
        ]
        result = _case_result(case, score=80, passed=True, failures=[])
        result.execution.json_payload = payload
        datasets = load_golden_datasets("evals/golden")

        outcome = golden_score(case, result, datasets)

        self.assertTrue(outcome.golden_pass)
        self.assertEqual([], outcome.golden_failures)

    def test_golden_accepts_equivalence_when_grouped_without_literal_word(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE07")[0]
        payload = _valid_payload()
        payload["winning_option"] = "Pizza"
        payload["final_output"] = "Pizza is the chosen option and clear winner."
        payload["recommendation_groups"] = [
            {
                "canonical_option": "Pizza",
                "proposers": ["A", "B"],
                "member_recommendations": ["Pizza", "Italian flatbread with cheese and tomato sauce"],
                "reason": "Grouped by normalized recommendation text.",
            }
        ]
        result = _case_result(case, score=80, passed=True, failures=[])
        result.execution.json_payload = payload
        datasets = load_golden_datasets("evals/golden")

        outcome = golden_score(case, result, datasets)

        self.assertTrue(outcome.golden_pass)
        self.assertEqual([], outcome.golden_failures)

    def test_judge_payload_parsing(self) -> None:
        parsed = parse_judge_payload(
            {
                "score": 85,
                "pass": True,
                "reasoning": "Solid.",
                "strengths": ["clear"],
                "weaknesses": ["brief"],
                "safety_concerns": [],
                "regression_risk": "low",
            }
        )

        self.assertEqual(85, parsed.score)
        self.assertTrue(parsed.passed)
        self.assertEqual(["clear"], parsed.strengths)
        self.assertIsNone(parsed.error)

    def test_judge_payload_rejects_ten_point_scale(self) -> None:
        parsed = parse_judge_payload(
            {
                "score": 9,
                "pass": True,
                "reasoning": "Excellent.",
                "strengths": ["clear"],
                "weaknesses": [],
                "safety_concerns": [],
                "regression_risk": "low",
            }
        )

        self.assertIsNone(parsed.score)
        self.assertIn("1-10 scale", parsed.error or "")

    def test_judge_prompt_declares_one_to_one_hundred_scale_before_evidence(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        prompt = build_judge_prompt(_case_result(case, 80, True, []))

        scale_index = prompt.index("Score on a 1-100 integer scale")
        evidence_index = prompt.index('"case"')
        self.assertLess(scale_index, evidence_index)
        self.assertIn("Do not use a 1-10 scale", prompt[:evidence_index])

    def test_judge_retries_once_after_ten_point_scale(self) -> None:
        from evals import judges

        class Response:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        payloads = [
            {
                "score": 9,
                "pass": True,
                "reasoning": "Excellent.",
                "strengths": ["clear"],
                "weaknesses": [],
                "safety_concerns": [],
                "regression_risk": "low",
            },
            {
                "score": 90,
                "pass": True,
                "reasoning": "Excellent.",
                "strengths": ["clear"],
                "weaknesses": [],
                "safety_concerns": [],
                "regression_risk": "low",
            },
        ]
        prompts: list[str] = []

        async def fake_run_member(config, member, prompt, schema_path, phase, web_search):
            prompts.append(prompt)
            return Response(payloads.pop(0))

        with patch.object(judges, "run_member", side_effect=fake_run_member):
            judged = judges.judge_result(_case_result(case, 80, True, []), "ollama", "qwen3:32b")

        self.assertEqual(90, judged.score)
        self.assertEqual(2, len(prompts))
        self.assertIn("previous response used a 1-10 style score", prompts[1])

    def test_load_judge_config_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge.yaml"
            path.write_text(
                "provider: ollama\n"
                "model: qwen3:32b\n"
                "options:\n"
                "  temperature: 0.2\n"
                "  seed: 123\n"
                "  num_ctx: 16384\n",
                encoding="utf-8",
            )

            config = load_judge_config(path)

        self.assertEqual("ollama", config.provider)
        self.assertEqual("qwen3:32b", config.model)
        self.assertEqual({"temperature": 0.2, "seed": 123, "num_ctx": 16384}, config.options)

    def test_load_judge_config_rejects_invalid_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge.yaml"
            path.write_text(
                "provider: ollama\n"
                "model: qwen3:32b\n"
                "options:\n"
                "  temperature: warm\n"
                "  seed: 42\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "temperature"):
                load_judge_config(path)

            path.write_text(
                "provider: ollama\n"
                "model: qwen3:32b\n"
                "options:\n"
                "  temperature: 0.3\n"
                "  seed: fixed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "seed"):
                load_judge_config(path)

            path.write_text(
                "provider: ollama\n"
                "model: qwen3:32b\n"
                "options:\n"
                "  temperature: 0.3\n"
                "  seed: 42\n"
                "  num_ctx: large\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "num_ctx"):
                load_judge_config(path)

    def test_judge_config_merges_model_and_options_into_provider_config(self) -> None:
        config = _judge_config(
            "ollama", "qwen3:32b", {"temperature": 0.1, "seed": 7, "num_ctx": 16384}
        )
        provider = config["model_providers"]["ollama"]

        self.assertTrue(provider["enabled"])
        self.assertIn("qwen3:32b", provider["static_models"])
        self.assertEqual(0.1, provider["options"]["temperature"])
        self.assertEqual(7, provider["options"]["seed"])
        self.assertEqual(16384, provider["options"]["num_ctx"])


class RunnerTests(unittest.TestCase):
    def test_default_council_command_uses_local_secretary(self) -> None:
        self.assertEqual("./council --secretary local", DEFAULT_COUNCIL_CMD)

    def test_format_duration_uses_compact_units(self) -> None:
        self.assertEqual("12s", _format_duration(12))
        self.assertEqual("3m 08s", _format_duration(188))
        self.assertEqual("1h 04m 22s", _format_duration(3862))

    def test_progress_timing_reports_elapsed_and_eta(self) -> None:
        with patch("evals.run_eval.time.monotonic", return_value=130.0):
            elapsed, eta = _progress_timing(100.0, completed_runs=0, total_runs=4)

        self.assertEqual("30s", elapsed)
        self.assertEqual("unknown", eta)

        with patch("evals.run_eval.time.monotonic", return_value=140.0):
            elapsed, eta = _progress_timing(100.0, completed_runs=2, total_runs=5)

        self.assertEqual("40s", elapsed)
        self.assertEqual("1m 00s", eta)

    def test_backup_previous_report_uses_sibling_previous_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "custom-latest.json"
            latest.write_text('{"old": true}\n', encoding="utf-8")

            previous = _backup_previous_report(latest)

            self.assertEqual(Path(tmp) / "previous.json", previous)
            self.assertEqual('{"old": true}\n', previous.read_text(encoding="utf-8"))

    def test_backup_previous_report_skips_missing_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest.md"

            previous = _backup_previous_report(latest)

            self.assertEqual(Path(tmp) / "previous.md", previous)
            self.assertFalse(previous.exists())

    def test_compare_with_previous_reports_aggregate_and_case_deltas(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        current = _case_result(case, score=80, passed=True, failures=[])
        previous_payload = _report_payload(
            _case_result(case, score=60, passed=False, failures=["invalid_json"])
        )

        with tempfile.TemporaryDirectory() as tmp:
            previous = Path(tmp) / "previous.json"
            previous.write_text(json.dumps(previous_payload), encoding="utf-8")
            lines = _compare_with_previous(previous, _report(current))

        rendered = "\n".join(lines)
        self.assertIn("runs: 1 (+0)", rendered)
        self.assertIn("average score: 80.0 (+20.0)", rendered)
        self.assertIn("pass rate: 100.0% (+100.0%)", rendered)
        self.assertIn("SMOKE01 repeat 1: score 60->80 (+20)", rendered)
        self.assertIn("pass no->yes", rendered)
        self.assertIn("failures invalid_json->-", rendered)

    def test_compare_with_previous_handles_missing_and_unreadable_reports(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        report = _report(_case_result(case, score=80, passed=True, failures=[]))

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "previous.json"
            self.assertEqual(["previous report: none"], _compare_with_previous(missing, report))

            unreadable = Path(tmp) / "previous.json"
            unreadable.write_text("{not json", encoding="utf-8")
            self.assertIn("previous report: unreadable", _compare_with_previous(unreadable, report)[0])

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
            self.assertIn("elapsed=", rendered)
            self.assertIn("eta=", rendered)
            self.assertIn("PASS score=", rendered)
            self.assertIn("Summary", rendered)
            self.assertIn("Previous comparison", rendered)
            self.assertIn("previous report: none", rendered)
            self.assertIn(f"wrote: {output}", rendered)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(100, payload["results"][0]["golden_score"])
            self.assertTrue(payload["results"][0]["golden_pass"])
            self.assertEqual(82, payload["results"][0]["combined_score"])

    def test_main_leaves_golden_fields_unset_for_case_without_golden_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--case",
                        "VOTING04",
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

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertIsNone(payload["results"][0]["golden_score"])
        self.assertIsNone(payload["results"][0]["golden_pass"])

    def test_main_copies_latest_reports_to_previous_before_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            output.write_text(json.dumps(_report_payload()) + "\n", encoding="utf-8")
            markdown.write_text("# Previous\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
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

            self.assertEqual(0, exit_code)
            self.assertTrue((Path(tmp) / "previous.json").exists())
            self.assertEqual("# Previous\n", (Path(tmp) / "previous.md").read_text(encoding="utf-8"))

    def test_failed_only_reruns_only_failed_cases_from_report(self) -> None:
        cases = load_suite("evals/cases.yaml")
        smoke01 = filter_cases(cases, case_id="SMOKE01")[0]
        smoke02 = filter_cases(cases, case_id="SMOKE02")[0]
        with tempfile.TemporaryDirectory() as tmp:
            failed_report = Path(tmp) / "source.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            failed_report.write_text(
                json.dumps(
                    _report_payload(
                        _case_result(smoke01, score=80, passed=True, failures=[]),
                        _case_result(smoke02, score=60, passed=False, failures=[]),
                    )
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--failed-only",
                        "--failed-report",
                        str(failed_report),
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

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual(["SMOKE02"], [item["case"]["id"] for item in report["results"]])

    def test_failed_only_preserves_failed_repeat_count(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        passed_repeat = _case_result(case, score=80, passed=True, failures=[])
        failed_repeat = _case_result(case, score=60, passed=False, failures=[])
        failed_repeat.repeat_index = 2
        with tempfile.TemporaryDirectory() as tmp:
            failed_report = Path(tmp) / "source.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            failed_report.write_text(
                json.dumps(_report_payload(passed_repeat, failed_repeat)),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--failed-only",
                        "--failed-report",
                        str(failed_report),
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

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual(["SMOKE01"], [item["case"]["id"] for item in report["results"]])

    def test_failed_only_applies_category_filter_after_failed_selection(self) -> None:
        cases = load_suite("evals/cases.yaml")
        smoke01 = filter_cases(cases, case_id="SMOKE01")[0]
        safety05 = filter_cases(cases, case_id="SAFETY05")[0]
        with tempfile.TemporaryDirectory() as tmp:
            failed_report = Path(tmp) / "source.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            failed_report.write_text(
                json.dumps(
                    _report_payload(
                        _case_result(smoke01, score=60, passed=False, failures=[]),
                        _case_result(safety05, score=40, passed=False, failures=[]),
                    )
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--failed-only",
                        "--failed-report",
                        str(failed_report),
                        "--category",
                        "safety",
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

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual(["SAFETY05"], [item["case"]["id"] for item in report["results"]])

    def test_failed_only_rejects_skip_mode(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--skip", "--llm-judge", "--failed-only"])

    def test_failed_only_errors_when_report_has_no_failures(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            failed_report = Path(tmp) / "source.json"
            failed_report.write_text(
                json.dumps(_report_payload(_case_result(case, score=80, passed=True, failures=[]))),
                encoding="utf-8",
            )

            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--failed-only", "--failed-report", str(failed_report)])

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

    def test_skip_rejects_without_llm_judge(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "latest.json"
            output.write_text(json.dumps(_report_payload(_case_result(case, 80, True, []))), encoding="utf-8")

            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--skip", "--input-report", str(output)])

    def test_skip_does_not_run_golden_against_existing_report(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            input_report = Path(tmp) / "input.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            input_report.write_text(json.dumps(_report_payload(_case_result(case, 80, True, []))), encoding="utf-8")
            judged = JudgeResult(score=90, passed=True, regression_risk="low")

            with (
                patch("evals.run_eval.judge_result", return_value=judged),
                redirect_stdout(stdout := io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--skip",
                        "--input-report",
                        str(input_report),
                        "--llm-judge",
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            self.assertEqual(0, exit_code)
            rendered = stdout.getvalue()
            self.assertNotIn("Golden validation", rendered)
            self.assertNotIn("[golden", rendered)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(payload["results"][0]["golden_score"])
            self.assertIsNone(payload["results"][0]["golden_pass"])
            self.assertEqual(90, payload["results"][0]["judge_score"])

    def test_removed_golden_flags_are_rejected(self) -> None:
        for flag in ("--golden", "--golden-dir"):
            with self.subTest(flag=flag):
                argv = [flag]
                if flag == "--golden-dir":
                    argv.append("evals/golden")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv)

    def test_skip_runs_mocked_judge_with_default_provider_and_model(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            input_report = Path(tmp) / "input.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            input_report.write_text(json.dumps(_report_payload(_case_result(case, 80, True, []))), encoding="utf-8")
            judged = JudgeResult(
                score=90,
                passed=True,
                reasoning="Good semantic fit.",
                strengths=["clear winner"],
                weaknesses=[],
                safety_concerns=[],
                regression_risk="low",
            )

            with (
                patch("evals.run_eval.judge_result", return_value=judged) as judge_mock,
                redirect_stdout(stdout := io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--skip",
                        "--llm-judge",
                        "--input-report",
                        str(input_report),
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            self.assertEqual(0, exit_code)
            rendered = stdout.getvalue()
            self.assertIn("LLM judge: 1 run(s), provider=ollama model=qwen3:32b timeout=300s", rendered)
            self.assertIn("[judge 1/1] SMOKE01 repeat 1 elapsed=", rendered)
            self.assertIn("eta=", rendered)
            self.assertIn("PASS score=90 risk=low elapsed=", rendered)
            self.assertIn("Judge complete: average=90.0 pass_rate=100.0% errors=0", rendered)
            judge_mock.assert_called_once()
            self.assertEqual("ollama", judge_mock.call_args.kwargs["provider"])
            self.assertEqual("qwen3:32b", judge_mock.call_args.kwargs["model"])
            self.assertEqual(
                {"temperature": 0.3, "seed": 42, "num_ctx": 16384},
                judge_mock.call_args.kwargs["options"],
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(90, payload["results"][0]["judge_score"])
            self.assertEqual(82, payload["results"][0]["combined_score"])

    def test_skip_rejects_legacy_ten_point_judge_score(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            input_report = Path(tmp) / "input.json"
            report = _report_payload(_case_result(case, 80, True, []))
            report["results"][0]["judge_score"] = 9
            report["results"][0]["judge_pass"] = True
            report["results"][0]["judge_weaknesses"] = []
            input_report.write_text(json.dumps(report), encoding="utf-8")

            loaded = load_json_report(input_report)

        self.assertIsNone(loaded.results[0].judge_score)
        self.assertIn("1-10 scale", loaded.results[0].judge_error or "")
        self.assertIn("1-10 scale", loaded.results[0].judge_weaknesses[0])

    def test_skip_judge_provider_and_model_cli_override_config(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            input_report = Path(tmp) / "input.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            input_report.write_text(json.dumps(_report_payload(_case_result(case, 80, True, []))), encoding="utf-8")
            judged = JudgeResult(score=90, passed=True, regression_risk="low")

            with (
                patch("evals.run_eval.judge_result", return_value=judged) as judge_mock,
                redirect_stdout(stdout := io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--skip",
                        "--llm-judge",
                        "--judge-provider",
                        "codex",
                        "--judge-model",
                        "gpt-5.4-mini",
                        "--input-report",
                        str(input_report),
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

        self.assertEqual(0, exit_code)
        rendered = stdout.getvalue()
        self.assertIn("LLM judge: 1 run(s), provider=codex model=gpt-5.4-mini timeout=300s", rendered)
        self.assertEqual("codex", judge_mock.call_args.kwargs["provider"])
        self.assertEqual("gpt-5.4-mini", judge_mock.call_args.kwargs["model"])
        self.assertEqual(
            {"temperature": 0.3, "seed": 42, "num_ctx": 16384},
            judge_mock.call_args.kwargs["options"],
        )

    def test_skip_judge_error_progress_and_verbose_reasoning(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            input_report = Path(tmp) / "input.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            input_report.write_text(json.dumps(_report_payload(_case_result(case, 80, True, []))), encoding="utf-8")
            judged = JudgeResult(error="judge provider unavailable because the model is missing")

            with (
                patch("evals.run_eval.judge_result", return_value=judged),
                redirect_stdout(stdout := io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--skip",
                        "--llm-judge",
                        "--verbose",
                        "--input-report",
                        str(input_report),
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            rendered = stdout.getvalue()
            self.assertEqual(0, exit_code)
            self.assertIn("ERROR score=- elapsed=", rendered)
            self.assertIn("eta=", rendered)
            self.assertIn("error=judge provider unavailable", rendered)
            self.assertIn("judge error: judge provider unavailable", rendered)

    def test_skip_quiet_suppresses_judge_progress(self) -> None:
        case = filter_cases(load_suite("evals/cases.yaml"), case_id="SMOKE01")[0]
        with tempfile.TemporaryDirectory() as tmp:
            input_report = Path(tmp) / "input.json"
            output = Path(tmp) / "latest.json"
            markdown = Path(tmp) / "latest.md"
            input_report.write_text(json.dumps(_report_payload(_case_result(case, 80, True, []))), encoding="utf-8")

            with (
                patch("evals.run_eval.judge_result", return_value=JudgeResult(score=90, passed=True, regression_risk="low")),
                redirect_stdout(stdout := io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--skip",
                        "--llm-judge",
                        "--quiet",
                        "--input-report",
                        str(input_report),
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            self.assertEqual(0, exit_code)
            self.assertEqual("", stdout.getvalue())

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

    def test_execute_case_enables_benchmark_environment(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        command = (
            "python -c \"import json, os; "
            "print(json.dumps({'benchmark': os.environ.get('SMALL_COUNCIL_BENCHMARK')}))\""
        )

        execution = execute_case(case, command, timeout_seconds=5)

        self.assertEqual(0, execution.exit_code)
        self.assertEqual("1", execution.json_payload["benchmark"])

    def test_execute_case_passes_eval_sandbox_config(self) -> None:
        case = load_suite("evals/cases.yaml")[0]
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "council.yaml"
            config_path.write_text("storage:\n  council_state_path: ./tmp-state.json\n", encoding="utf-8")
            command = (
                "python -c \"import json, os; "
                "print(json.dumps({"
                "'benchmark': os.environ.get('SMALL_COUNCIL_BENCHMARK'), "
                "'config': os.environ.get('SMALL_COUNCIL_CONFIG')"
                "}))\""
            )

            execution = execute_case(case, command, timeout_seconds=5, config_path=config_path)

        self.assertEqual(0, execution.exit_code)
        self.assertEqual("1", execution.json_payload["benchmark"])
        self.assertEqual(str(config_path), execution.json_payload["config"])

    def test_eval_sandbox_config_uses_isolated_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            sandbox = _create_eval_sandbox(artifact_dir, "run-1")
            config_text = sandbox.config_path.read_text(encoding="utf-8")

        self.assertIn("run-1/sandbox/storage/council-state.json", config_text)
        self.assertIn("run-1/sandbox/storage/leaderboard.json", config_text)
        self.assertIn("run-1/sandbox/storage/memories", config_text)
        self.assertIn("run-1/sandbox/runtime/logs", config_text)
        self.assertIn("run-1/sandbox/runtime/temp", config_text)

    def test_capture_artifacts_uses_sandbox_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            sandbox = _create_eval_sandbox(artifact_dir, "run-1")
            before = _artifact_snapshot(sandbox.roots)
            state_path = artifact_dir / "run-1" / "sandbox" / "storage" / "council-state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text('{"members": []}\n', encoding="utf-8")

            copied = _capture_artifacts(before, artifact_dir, "run-1", "CASE01", 1, sandbox.roots)

        self.assertEqual(
            [str(artifact_dir / "run-1" / "CASE01-r1" / "storage" / "council-state.json")],
            copied,
        )

    def test_main_isolates_state_mutating_eval_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "cases.json"
            output = root / "latest.json"
            markdown = root / "latest.md"
            artifacts = root / "artifacts"
            real_storage = root / "real-storage"
            suite.write_text(
                json.dumps(
                    [
                        {
                            "id": "STATE_TEST",
                            "name": "State Test",
                            "category": "state",
                            "prompt": "Choose one picnic food: sandwiches, fruit, or cookies.",
                            "tags": ["state"],
                            "args": ["--set-members", "3"],
                            "expected_behavior": ["mutates state"],
                            "scoring_focus": ["state"],
                            "hard_failure_rules": ["winner_missing"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            config = {
                "storage": {
                    "council_state_path": str(real_storage / "council-state.json"),
                    "leaderboard_path": str(real_storage / "leaderboard.json"),
                    "memories_path": str(real_storage / "memories"),
                },
                "runtime": {
                    "sessions_path": str(root / "real-runtime" / "sessions"),
                    "logs_path": str(root / "real-runtime" / "logs"),
                    "temp_path": str(root / "real-runtime" / "temp"),
                },
            }
            command = _mock_state_writing_council_command(_valid_payload())

            with (
                patch("evals.run_eval.load_config", return_value=config),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--suite",
                        str(suite),
                        "--timeout-seconds",
                        "5",
                        "--council-cmd",
                        command,
                        "--artifact-dir",
                        str(artifacts),
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            report = json.loads(output.read_text(encoding="utf-8"))
            artifact_paths = report["results"][0]["artifact_paths"]

        self.assertEqual(0, exit_code)
        self.assertFalse((real_storage / "council-state.json").exists())
        self.assertTrue(any(path.endswith("storage/council-state.json") for path in artifact_paths))


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


def _mock_state_writing_council_command(payload: dict) -> str:
    payload_encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    script = (
        "import base64,json,os,pathlib;"
        f"payload=json.loads(base64.b64decode('{payload_encoded}'));"
        "config=pathlib.Path(os.environ['SMALL_COUNCIL_CONFIG']);"
        "state=config.parent.parent/'storage'/'council-state.json';"
        "state.parent.mkdir(parents=True, exist_ok=True);"
        "state.write_text('{\"members\":[1,2,3]}\\n', encoding='utf-8');"
        "print(json.dumps(payload))"
    )
    script_encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return f'python -c "import base64; exec(base64.b64decode(\'{script_encoded}\'))"'


def _case_result(case, score: int, passed: bool, failures: list[str]) -> CaseRunResult:
    return CaseRunResult(
        case=case,
        repeat_index=1,
        execution=CouncilExecution(
            command=["./council"],
            stdout=json.dumps(_valid_payload()),
            stderr="",
            duration_seconds=0.1,
            exit_code=0,
            json_payload=_valid_payload(),
        ),
        validation=ValidationResult(
            valid_json=not failures,
            required_fields_present=True,
            recommendation_counts_sensible=True,
            winner_exists=True,
            final_answer_aligns_with_winner=True,
            vote_references_valid=True,
            runoff_counts_valid=True,
            diversity_lanes_present=True,
            safety_passed=True,
            hard_failures=failures,
        ),
        score_breakdown=ScoreBreakdown(deterministic_score=score),
        deterministic_score=score,
        passed=passed,
    )


def _report(*results: CaseRunResult) -> EvalReport:
    return EvalReport(
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
        results=list(results),
    )


def _report_payload(*results: CaseRunResult) -> dict:
    if not results:
        results = (_case_result(load_suite("evals/cases.yaml")[0], 70, True, []),)
    return to_plain_data(_report(*results))


if __name__ == "__main__":
    unittest.main()
