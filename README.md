# Small Council

A CLI-first personal decision council powered by project-local OpenAI Codex subagents.

## What It Looks Like

```bash
> ./council "which Oscars 2026 best picture nominee should I watch tonight?"
[20:32:03] Secretary
Request received: which Oscars 2026 best picture nominee should I watch tonight?

[20:32:03] Secretary
Diversity lanes assigned for balanced mode.

[20:33:00] Secretary
Initial proposals are in: five independent nominees have been put forward across the council’s diversity lanes. The current set spans mainstream, underrated, budget-friendly, special-occasion, and contrarian angles, and no vote has been taken yet. We’re still in the research phase, so the next step is comparison rather than a final pick.

[20:33:41] Secretary
Discussion round 1 is complete. The council has compared the four nominee paths against the practical constraints and updated several members’ positions: Cato and Bram stayed on the low-friction option, while Aurelia and Dima moved toward the headline winner, and Echo moved off the contrarian lane. No vote has been taken yet, so the recommendation remains pending final council selection.

[20:34:06] Secretary
Discussion round 2 is complete. The council has now surfaced the main split clearly: practical couch-friendly picks versus the headline prestige answer, with members mostly re-affirming their positions and no final recommendation issued yet. The thread is ready for the next phase of voting or consolidation.

[20:34:12] Secretary
The council has completed the final proposals milestone and finished its discussion rounds. The visible consensus shifted toward a practical, low-friction nominee, while one member kept the prestige-first counterpoint on the table. No vote was taken yet; the next step is final voting/selection.

[20:34:27] Secretary
Proposal grouping is complete. The council has reduced the initial spread of nominee picks into two clear camps: a majority-leaning practical default around "Train Dreams" and a prestige fallback around "One Battle After Another." No vote has been taken yet; the next step is to convert this into a final choice.

[20:35:08] Secretary
Initial vote is complete, and the council has now finished two discussion rounds. The main split is clear: one camp favors the low-friction, shorter Netflix option for a tonight decision, while the other favors the headline prestige pick for Oscars signal. No final answer has been issued yet; the next step is a final council conclusion.

Watch "Train Dreams" tonight.

The council settled it 4 to 1, and the winning recommendation was shared by Aurelia, Bram, Cato, and Dima. Old-school ruling: when the table reaches that kind of consensus, you don’t dither, you pour a drink, dim the lights, and put on the respectable winner.
```

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
