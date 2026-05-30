from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any

@dataclass
class VoteRound:
    round_number: int
    vote_counts: dict[str, int]
    tied_options: list[str]
    resolved: bool
    winning_option: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "vote_counts": self.vote_counts,
            "tied_options": self.tied_options,
            "resolved": self.resolved,
            "winning_option": self.winning_option,
        }


@dataclass
class DecisionResult:
    status: str
    winning_option: str | None
    winning_member: str | None
    winning_members: list[str]
    tied_options: list[str]
    tie_broken_by: str | None
    tie_break_vote: dict[str, Any] | None
    vote_counts: dict[str, int]
    vote_rounds: list[VoteRound]


def evaluate_vote_round(
    recommendations: list[dict], votes: list[dict], round_number: int
) -> VoteRound:
    valid_options = {rec["recommendation"] for rec in recommendations}
    counts = Counter(vote["selected_option"] for vote in votes if vote.get("selected_option"))
    counts = Counter({option: count for option, count in counts.items() if option in valid_options})
    if not counts:
        tied = sorted(valid_options)
        return VoteRound(
            round_number=round_number,
            vote_counts={},
            tied_options=tied,
            resolved=False,
            winning_option=None,
        )

    top_count = max(counts.values())
    tied = sorted([option for option, count in counts.items() if count == top_count])
    resolved = len(tied) == 1
    return VoteRound(
        round_number=round_number,
        vote_counts=dict(counts),
        tied_options=[] if resolved else tied,
        resolved=resolved,
        winning_option=tied[0] if resolved else None,
    )


def decision_from_rounds(
    recommendations: list[dict],
    vote_rounds: list[VoteRound],
    tie_breaker_member: str | None = None,
    tie_break_vote: dict[str, Any] | None = None,
) -> DecisionResult:
    final_round = vote_rounds[-1]
    if final_round.resolved and final_round.winning_option:
        option = final_round.winning_option
        return DecisionResult(
            status="resolved",
            winning_option=option,
            winning_member=_proposer_for_option(recommendations, option),
            winning_members=_proposers_for_option(recommendations, option),
            tied_options=[],
            tie_broken_by=None,
            tie_break_vote=None,
            vote_counts=final_round.vote_counts,
            vote_rounds=vote_rounds,
        )
    if tie_breaker_member and tie_break_vote:
        option = tie_break_vote.get("selected_option")
        if option in final_round.tied_options:
            return DecisionResult(
                status="resolved",
                winning_option=option,
                winning_member=_proposer_for_option(recommendations, option),
                winning_members=_proposers_for_option(recommendations, option),
                tied_options=[],
                tie_broken_by=tie_breaker_member,
                tie_break_vote=tie_break_vote,
                vote_counts=final_round.vote_counts,
                vote_rounds=vote_rounds,
            )
    return DecisionResult(
        status="unresolved_tie",
        winning_option=None,
        winning_member=None,
        winning_members=[],
        tied_options=final_round.tied_options,
        tie_broken_by=None,
        tie_break_vote=None,
        vote_counts=final_round.vote_counts,
        vote_rounds=vote_rounds,
    )


def filter_recommendations(recommendations: list[dict], options: list[str]) -> list[dict]:
    allowed = set(options)
    return [rec for rec in recommendations if rec.get("recommendation") in allowed]


def normalize_option(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    return normalized.rstrip(".;:!?")


def fallback_recommendation_groups(recommendations: list[dict]) -> list[dict]:
    groups_by_key: dict[str, dict] = {}
    for recommendation in recommendations:
        option = recommendation["recommendation"]
        key = normalize_option(option)
        group = groups_by_key.setdefault(
            key,
            {
                "canonical_option": option,
                "proposers": [],
                "member_recommendations": [],
                "reason": "Matched by normalized recommendation text.",
            },
        )
        group["proposers"].append(recommendation["proposer"])
        group["member_recommendations"].append(option)
    return list(groups_by_key.values())


def validate_recommendation_groups(groups: list[dict], recommendations: list[dict]) -> list[dict]:
    proposer_to_recommendation = {
        recommendation["proposer"]: recommendation["recommendation"]
        for recommendation in recommendations
    }
    seen: set[str] = set()
    validated: list[dict] = []
    for group in groups:
        proposers = [
            proposer
            for proposer in group.get("proposers", [])
            if proposer in proposer_to_recommendation and proposer not in seen
        ]
        if not proposers:
            continue
        seen.update(proposers)
        member_recommendations = [proposer_to_recommendation[proposer] for proposer in proposers]
        canonical = str(group.get("canonical_option") or member_recommendations[0]).strip()
        validated.append(
            {
                "canonical_option": canonical,
                "proposers": proposers,
                "member_recommendations": member_recommendations,
                "reason": str(group.get("reason") or "Grouped by the President.").strip(),
            }
        )
    for recommendation in recommendations:
        proposer = recommendation["proposer"]
        if proposer not in seen:
            validated.append(
                {
                    "canonical_option": recommendation["recommendation"],
                    "proposers": [proposer],
                    "member_recommendations": [recommendation["recommendation"]],
                    "reason": "Kept as a distinct option.",
                }
            )
    return validated


def canonical_recommendations(groups: list[dict]) -> list[dict]:
    recommendations = []
    for group in groups:
        recommendations.append(
            {
                "proposer": group["proposers"][0],
                "proposers": group["proposers"],
                "recommendation": group["canonical_option"],
                "short_reasoning": group["reason"],
                "pros": [],
                "cons": [],
                "confidence": 10,
            }
        )
    return recommendations


def canonicalize_vote(vote: dict[str, Any], groups: list[dict]) -> dict[str, Any]:
    option_map: dict[str, str] = {}
    proposer_map: dict[str, str] = {}
    for group in groups:
        canonical = group["canonical_option"]
        option_map[normalize_option(canonical)] = canonical
        for original in group["member_recommendations"]:
            option_map[normalize_option(original)] = canonical
        for proposer in group["proposers"]:
            proposer_map[proposer] = canonical
    selected = vote.get("selected_option")
    canonical = option_map.get(normalize_option(selected), selected)
    selected_proposer = vote.get("selected_proposer")
    if selected_proposer in proposer_map:
        canonical = proposer_map[selected_proposer]
    updated = dict(vote)
    updated["selected_option"] = canonical
    return updated


def _proposer_for_option(recommendations: list[dict], option: str) -> str | None:
    for rec in recommendations:
        if rec.get("recommendation") == option:
            return rec.get("proposer")
    return None


def _proposers_for_option(recommendations: list[dict], option: str) -> list[str]:
    for rec in recommendations:
        if rec.get("recommendation") == option:
            return list(rec.get("proposers") or [rec.get("proposer")])
    return []


def validate_recommendation(payload: dict[str, Any], member: Member) -> dict[str, Any]:
    payload.setdefault("proposer", member.name)
    payload["proposer"] = member.name
    return payload


def validate_vote(payload: dict[str, Any], member: Member) -> dict[str, Any]:
    payload.setdefault("voter", member.name)
    payload["voter"] = member.name
    return payload
