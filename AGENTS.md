# Repository Guidelines

## Project Structure & Module Organization

Small Council is a Python CLI project. Core code lives in `small_council/`; `cli.py` orchestrates runs, while provider, guardrail, decision, memory, output, and search helpers are split into focused modules. Root scripts `./council` and `./eval` wrap the CLI and eval runner.

Tests are in `tests/` and mirror behavior by module, for example `tests/test_model_providers.py` and `tests/test_guardrails.py`. Benchmark code lives in `evals/`; cases are in `evals/cases.yaml`, golden expectations in `evals/golden/`, and generated reports under `evals/reports/`. Config, prompts, agent definitions, and schemas live in `config/`, `prompts/`, `agents/`, `president/`, and `schemas/`. Runtime state is project-local under `runtime/`, `storage/`, and `.codex/`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt`: install runtime dependencies.
- `chmod +x ./council ./eval`: restore executable bits if needed.
- `./council --init`: initialize local config, prompts, state, and agents.
- `./council --doctor`: validate local provider and configuration setup.
- `./council "What movie should I watch tonight?"`: run the application locally.
- `python -m unittest`: run the unit test suite.
- `./eval --case SMOKE01`: run one deterministic benchmark case.
- `./eval`: run the full benchmark suite.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation, type hints where they clarify contracts, and `from __future__ import annotations` in new modules. Use `snake_case` for modules, functions, variables, config keys, and test names; use `PascalCase` for classes. Prefer `pathlib.Path` for filesystem work. No formatter is configured, so match surrounding style and group imports as standard library, third-party, then local.

## Testing Guidelines

The suite uses `unittest`, including `unittest.mock` and `IsolatedAsyncioTestCase` for async paths. Add tests beside related coverage in `tests/test_*.py`; name methods `test_<expected_behavior>`. Keep unit tests independent of real providers by mocking subprocesses, network calls, and clients. For CLI or eval changes, run `python -m unittest` plus a targeted `./eval --case ...` when practical.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, such as `Improve eval reporting and vote resilience`. Follow that style: one focused change per commit, present-tense summary, no trailing punctuation. Pull requests should describe the behavior change, list tests or evals run, call out config changes, and include screenshots only for visible TUI or documentation changes.

## Security & Configuration Tips

Keep secrets out of tracked config. Prefer environment variables such as `OLLAMA_API_KEY` and `CODEX_HOME=$PWD/.codex`. Do not commit generated logs, temp files, memories, or eval artifacts unless they are intentionally curated documentation.
