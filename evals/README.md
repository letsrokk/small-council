# Small Council Evals

This package provides a deterministic benchmark harness for Small Council. It treats the app as a black box and invokes the CLI through:

```bash
./council --json-output --plain-output "your prompt"
```

Run the full suite:

```bash
python -m evals.run_eval
```

Run a subset:

```bash
python -m evals.run_eval --case SMOKE01
python -m evals.run_eval --category smoke
python -m evals.run_eval --tag safety
```

Useful options:

```bash
python -m evals.run_eval \
  --suite evals/cases.yaml \
  --output evals/reports/latest.json \
  --markdown evals/reports/latest.md \
  --version-name local-change \
  --repeat 3 \
  --timeout-seconds 600 \
  --council-cmd ./council
```

The framework captures stdout, stderr, duration, exit code, parsed JSON, validation warnings, deterministic score breakdowns, and report metadata. It continues after individual case failures.

## Scoring

Scores are deterministic and total 100:

- `answers_actual_request`: 20
- `practicality`: 15
- `reasoning_quality`: 15
- `tradeoff_awareness`: 10
- `proposal_diversity`: 10
- `internal_consistency`: 10
- `json_schema_validity`: 10
- `safety_resistance`: 10

Hard caps are applied for invalid JSON, crashes, missing winners, final-answer contradiction, unsafe instruction following, and hallucination traps.

The result model already includes `golden_score`, `judge_score`, and `combined_score` for future extensions. They are currently `null`.

