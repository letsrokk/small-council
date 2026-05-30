# Small Council Evals

This package provides a deterministic benchmark harness for Small Council. It treats the app as a black box and invokes the CLI through:

```bash
./council --json-output --plain-output "your prompt"
```

Run the full suite:

```bash
./eval
```

`./eval` is the project-root entrypoint for `python -m evals.run_eval`, so run it from
the repository root. The module form also works when needed:

```bash
python -m evals.run_eval
```

Run a subset:

```bash
./eval --case SMOKE01
./eval --category smoke
./eval --tag safety
```

Useful options:

```bash
./eval \
  --suite evals/cases.yaml \
  --output evals/reports/latest.json \
  --markdown evals/reports/latest.md \
  --version-name local-change \
  --repeat 3 \
  --timeout-seconds 600 \
  --council-cmd ./council
```

The framework captures stdout, stderr, duration, exit code, parsed JSON, validation warnings, deterministic score breakdowns, and report metadata. It continues after individual case failures.

## Progress Output

By default, `./eval` prints progress to stdout:

- suite path, selected case count, repeat count, total runs, and report paths at startup
- one line before each case run
- one PASS/FAIL result line with score, duration, JSON status, and hard failures when present
- a final summary with average score, pass rate, JSON validity, and written report paths

Use `--quiet` for report-only execution with no progress output:

```bash
./eval --quiet
```

Use `--verbose` to include validation warnings and stderr snippets for failing cases:

```bash
./eval --verbose
```

Verbosity levels:

- default: concise progress and per-case PASS/FAIL lines
- `--quiet`: no progress output; JSON and Markdown reports are still written
- `--verbose`: default output plus failure diagnostics

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
