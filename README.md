# Small Council

A CLI-first personal decision council powered by project-local OpenAI Codex subagents.

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

The local Secretary prints progress updates to stderr while the council works. The first update appears immediately, and follow-up updates appear every 5 seconds by default:

```bash
./council --set-update-interval 10 "Where should I go for dinner?"
```

The Secretary is a non-voting local reporter and does not count as a council member.

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
