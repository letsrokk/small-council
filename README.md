# Small Council

A CLI-first personal decision council powered by project-local OpenAI Codex subagents.

![Small Council in Action](docs/screenshots/in-progress.png)

## Run

```bash
chmod +x ./council
./council --init
./council --members
./council "What movie should I watch tonight?"
```

The app stores all council config, prompts, state, logs, temp files, and generated agent definitions inside this project directory.

## Local Codex Auth

The orchestrator runs Codex with `CODEX_HOME=$PWD/.codex` and `--ignore-user-config`, so it does not depend on `~/.codex/config.toml`.

On first use, authenticate Codex into the project-local home:

```bash
CODEX_HOME=$PWD/.codex codex login
```

Then check:

```bash
./council --doctor
```

## Output And Progress

Decision output is human-readable by default. Use JSON when you want a structured payload:

```bash
./council --json-output "What movie should I watch tonight?"
```

The CLI uses Rich TUI output by default for human-readable runs when Rich is installed. Use `--plain-output` for plain text, or `--json-output` for machine-readable JSON. If Rich is unavailable, the CLI falls back to plain text.

In Rich mode, use `Left`/`Right` to switch between the Secretary and council member areas. `Up`/`Down` scrolls the active area; `PageUp`/`Home` and `PageDown`/`End` jump to oldest/newest Secretary updates or first/last council member depending on the active area. After the final decision, use `Esc` or `Enter` to close the TUI. Narrow terminals fall back to a vertical layout so the Secretary and members remain readable.

The Secretary prints short immediate progress updates for completed events on stderr, then milestone summaries after the larger council phases: initial proposals, each discussion round, final proposals, proposal grouping, the initial vote, and any runoff votes.

The Secretary is non-voting and does not count as a council member. A model-backed Secretary is the default. The local Secretary remains available for deterministic/offline runs:

```bash
./council --secretary local "Where should I go for dinner?"
./council --secretary model --secretary-verbosity balanced "Where should I go for dinner?"
./council --no-secretary-immediate-updates "Where should I go for dinner?"
```

Supported model-backed verbosity levels are `low`, `balanced`, and `high`.

After the initial draft proposals, the council enters a threaded discussion phase, revises those drafts, then groups equivalent final proposals before voting.

## Proposal Diversity

The council uses `balanced` proposal diversity by default. Each member gets a recommendation lane during independent research so open-ended questions produce a broader set of options.

```bash
./council --set-diversity high "What should I cook tonight?"
```

Supported modes are `low`, `balanced`, and `high`.

## Files

- `config/council.yaml`: member names, allowed model pool, personality pool, storage paths.
- `storage/council-state.json`: persisted member identities and stats.
- `storage/leaderboard.json`: persisted leaderboard.
- `agents/definitions/`: generated persistent member definitions.
- `runtime/logs/`: Codex subagent run logs.
- `.codex/`: project-local Codex auth/session home.

## Reset

```bash
./council --reset --init
```

This rerolls models, personalities, and President assignment.

## Resize The Council

```bash
./council --set-members 7 --members
./council --add-members 1 --members
./council --remove-members 2 --leaderboard
```

Increasing the council keeps existing members unchanged and creates new persistent members with generated names when the configured names are exhausted. Reducing the council removes members from the end of the active roster and deletes their generated agent definitions and stats.

## Tie Runoffs

If multiple options tie for the highest vote count, the council removes all lower-scoring options and votes again on only the tied options. The council gets 3 runoff rounds by default. If no single winner emerges, the final answer presents all remaining tied options instead of inventing a winner.

Override the runoff limit for a single decision:

```bash
./council --set-runoff-rounds 5 "What should I cook tonight?"
```

Before voting, the President groups effectively identical recommendations into one canonical option. If that grouped option wins, every member who independently proposed it receives a win.

## Model Overrides

Edit `config/council.yaml` to pin a member later:

```yaml
model_overrides:
  Bram: gpt-5.5
```

Overrides must stay inside `model_pool`.

## Model Pool Note

The requested `gpt-5.1-codex-mini` model was not present in the Codex CLI 0.134.0 model catalog available in this workspace. The default pool uses `gpt-5.4-mini`, which is exposed by this Codex installation.
