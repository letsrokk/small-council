from __future__ import annotations

import json

from .state import Member


BASE_RULES = """You are a project-local OpenAI Codex subagent in the Small Council.
Use only concise visible reasoning. Do not reveal chain-of-thought or hidden deliberation.
You may use web search when it is useful for current options, local places, movies, products, or recent facts.
If details are missing, make sensible assumptions instead of asking a follow-up unless the decision is impossible.
Be entertaining in your own voice, but keep the output practical.
Return only JSON matching the supplied schema.
"""


def research_prompt(
    member: Member,
    question: str,
    prior_preferences: dict | None = None,
    diversity_lane: str | None = None,
    diversity_mode: str = "balanced",
) -> str:
    prefs = json.dumps(prior_preferences or {}, indent=2)
    lane_text = diversity_lane or "free choice"
    return f"""{BASE_RULES}

Council member:
- Name: {member.name}
- Model: {member.model}
- Personality: {member.personality}
- President: {member.is_president}
- Diversity mode: {diversity_mode}
- Assigned recommendation lane: {lane_text}

Task:
The user asks: {question!r}

Act independently. Research if useful. Produce one concrete recommendation.
Your personality should influence priorities, tone, and risk tolerance.
Fit your recommendation to your assigned lane.
If diversity mode is "low", avoid duplicating only the most obvious pick when a similarly good alternative exists.
If diversity mode is "balanced", make a practical recommendation that clearly differs from what other lanes are likely to produce.
If diversity mode is "high", strongly prefer a distinct category, style, genre, cuisine, neighborhood, game type, or product type unless your lane explicitly calls for the mainstream option.
Do not coordinate with or reference other council members during independent research.

Known local preferences/history:
{prefs}
"""


def discussion_prompt(
    member: Member,
    question: str,
    recommendations: list[dict],
) -> str:
    return f"""{BASE_RULES}

Council member:
- Name: {member.name}
- Model: {member.model}
- Personality: {member.personality}
- President: {member.is_president}

The user asks: {question!r}

Initial recommendations from the council:
{json.dumps(recommendations, indent=2)}

Briefly critique the alternatives. You may revise your preferred option.
Then vote. Prefer not to vote for your own proposal; if you do, include a specific justification.
Abstain only if there is no responsible choice.
"""


def equivalence_prompt(
    president: Member,
    question: str,
    recommendations: list[dict],
) -> str:
    return f"""{BASE_RULES}

You are the President of the Small Council.
President:
- Name: {president.name}
- Personality: {president.personality}

The user asks: {question!r}

Independent recommendations:
{json.dumps(recommendations, indent=2)}

Group recommendations that are effectively the same final decision option.
Be conservative: merge clearly identical final actions, such as the same movie, restaurant, recipe, game, or product.
Do not merge merely similar alternatives.

For each group, provide:
- canonical_option: the best concise option text voters should use
- proposers: member names whose recommendations belong in the group
- member_recommendations: original recommendation strings in the group
- reason: short explanation for why this group is or is not merged
"""


def runoff_prompt(
    member: Member,
    question: str,
    tied_recommendations: list[dict],
    vote_rounds: list[dict],
    runoff_round: int,
    max_runoff_rounds: int,
) -> str:
    return f"""{BASE_RULES}

Council member:
- Name: {member.name}
- Model: {member.model}
- Personality: {member.personality}
- President: {member.is_president}

The user asks: {question!r}

The council vote is tied. This is runoff round {runoff_round} of {max_runoff_rounds}.
All lower-scoring options have been removed. Vote only for one of these remaining tied options:
{json.dumps(tied_recommendations, indent=2)}

Previous vote rounds:
{json.dumps(vote_rounds, indent=2)}

Briefly compare only the remaining tied options, then vote for one of them.
Do not introduce a new option. Do not vote for an eliminated option.
"""


def president_summary_prompt(
    president: Member,
    question: str,
    recommendations: list[dict],
    votes: list[dict],
    winner: dict,
    leaderboard: list[dict],
) -> str:
    return f"""{BASE_RULES}

You are the President of the Small Council.
President:
- Name: {president.name}
- Personality: {president.personality}

The user asks: {question!r}

Recommendations:
{json.dumps(recommendations, indent=2)}

Votes:
{json.dumps(votes, indent=2)}

Computed winner:
{json.dumps(winner, indent=2)}

Leaderboard:
{json.dumps(leaderboard, indent=2)}

Write plain human-readable text in final_output, not JSON or a code block.
Respect the computed winner and votes exactly.
If status is "resolved", present the winning option normally.
If winning_members has more than one member, mention the shared winning proposers by name.
If status is "unresolved_tie", do not invent a winner. Say no single winner emerged after the configured runoff rounds and present all remaining tied options as equally viable choices.
"""
