# Small Council Evals

This package provides a deterministic benchmark harness for Small Council. It treats the app as a black box and invokes the CLI through:

```bash
./council --secretary local --json-output --plain-output "your prompt"
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

Re-run only cases that failed in the latest report:

```bash
./eval --failed-only
./eval --failed-only --failed-report evals/reports/previous.json
./eval --failed-only --golden --llm-judge
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
  --council-cmd "./council --secretary local"
```

The framework captures stdout, stderr, duration, exit code, parsed JSON, validation warnings, deterministic score breakdowns, and report metadata. It continues after individual case failures.

## Benchmark Provider Defaults

Eval runs use the local Secretary by default to avoid spending model calls on
progress reports during benchmarks. They also automatically set benchmark mode
for the council subprocess. This leaves normal `config/council.yaml` defaults
unchanged while forcing benchmark-specific provider options:

- Ollama: `temperature: 0.3`, `seed: 42`
- Codex: `reasoning_effort: low`

Benchmark options take precedence over member overrides and provider defaults,
so baseline runs, one-tier-up/down model comparisons, and future golden dataset
evaluations use deterministic settings without per-case configuration.

## Progress Output

By default, `./eval` prints progress to stdout:

- suite path, selected case count, repeat count, total runs, and report paths at startup
- one line before each case run with elapsed time and ETA
- one PASS/FAIL result line with score, duration, JSON status, elapsed time, ETA, and hard failures when present
- when `--golden` is enabled, a golden phase header, one `[golden i/N]` line per run with elapsed time and ETA, one result line, and a completion summary
- when `--llm-judge` is enabled, a judge phase header with provider/model, one `[judge i/N]` line before each judge call with elapsed time and ETA, one result line, and a completion summary
- a final summary with average score, pass rate, JSON validity, total elapsed time, written report paths, and comparison to the previous report when available

Use `--quiet` for report-only execution with no progress output, including golden and judge post-processing:

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
- `--verbose`: default output plus failure diagnostics, golden failure details, and judge reasoning/error excerpts

## Golden Validation and LLM Judge

Golden validation and LLM judging are post-processing phases. The deterministic
eval report is produced first; golden validation runs next when enabled; the
LLM judge runs last when enabled. Deterministic hard failures remain the
authoritative release gate.

Run deterministic eval plus golden validation:

```bash
./eval --case SMOKE01 --golden
```

Run deterministic eval plus golden validation and an LLM judge:

```bash
./eval --case SMOKE01 --golden --llm-judge
```

The judge uses the existing council model provider stack and reads its default
provider/model/options from `config/judge.yaml`:

```yaml
provider: ollama
model: qwen3:32b
options:
  temperature: 0.3
  seed: 42
```

Override provider and model from the CLI when needed:

```bash
./eval --llm-judge --judge-provider codex --judge-model gpt-5.4-mini
```

Re-run post-processing against an existing deterministic report without
executing `./council`:

```bash
./eval --skip --golden --input-report evals/reports/latest.json
./eval --skip --golden --llm-judge --input-report evals/reports/latest.json
```

Useful post-processing options:

```bash
./eval \
  --golden \
  --golden-dir evals/golden \
  --golden-weight 0.30 \
  --llm-judge \
  --judge-timeout-seconds 300 \
  --judge-weight 0.20
```

Fresh deterministic runs use an eval-local config sandbox under
`evals/reports/artifacts/<run-id>/sandbox/`, so cases that resize members or
change persisted settings do not mutate the real project `storage/`,
`runtime/`, or `config/council.yaml`. Modified sandbox `runtime`, `storage`, and
config files are copied into each case artifact directory for later judge
context. When `--skip` is used with an older report that has no artifact paths,
judging still uses the deterministic report data.

## Previous Report Comparison

At startup, `./eval` backs up existing reports before writing the new run:

- `evals/reports/latest.json` is copied to `evals/reports/previous.json`
- `evals/reports/latest.md` is copied to `evals/reports/previous.md`

When custom report paths are used, the backup files are siblings named
`previous` with the same extension. For example:

```bash
./eval --output tmp/latest-smoke.json --markdown tmp/latest-smoke.md
```

backs up to `tmp/previous.json` and `tmp/previous.md` if those latest files
already exist.

After the new reports are written, the final summary compares the new JSON
report against `previous.json`. Pass `--compare path/to/previous.json` to choose
an explicit baseline. It prints aggregate deltas for run count, average score,
pass rate, JSON validity, category deltas, regressions, improvements, and
changed per-case runs.
If no previous JSON report exists, the summary says `previous report: none`.

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

Optional post-processing populates `golden_score`, `judge_score`, and
`combined_score`.

Blending defaults:

- deterministic only: 100% deterministic
- golden only: 70% deterministic, 30% golden
- judge only: 75% deterministic, 25% judge
- golden and judge: 55% deterministic, 25% golden, 20% judge

Custom golden and judge weights are normalized against the deterministic
component. Deterministic hard caps are applied after blending.
