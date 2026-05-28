from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .codex_runner import CodexUnavailable, codex_doctor, run_member
from .config import ROOT, load_config, read_json, resolve_project_path
from .decision import (
    canonical_recommendations,
    canonicalize_vote,
    decision_from_rounds,
    evaluate_vote_round,
    fallback_recommendation_groups,
    filter_recommendations,
    validate_recommendation_groups,
    validate_recommendation,
    validate_vote,
)
from .formatting import render_fallback
from .memory import append_decision_memory, load_recent_memory
from .prompts import (
    discussion_prompt,
    equivalence_prompt,
    president_summary_prompt,
    research_prompt,
    runoff_prompt,
)
from .secretary import Secretary
from .state import (
    ensure_state,
    persist_leaderboard,
    president,
    resize_members,
    update_after_decision,
    write_agent_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="council", description="Ask the Small Council for a light decision.")
    parser.add_argument("question", nargs="*", help="Decision prompt, for example: What movie should I watch tonight?")
    parser.add_argument("--init", action="store_true", help="Create/load the local council and print members.")
    parser.add_argument("--members", action="store_true", help="List the current persisted council members.")
    parser.add_argument("--leaderboard", action="store_true", help="Print the persisted leaderboard.")
    parser.add_argument("--reset", action="store_true", help="Reset and reroll local council state.")
    parser.add_argument("--set-members", type=_positive_int, metavar="N", help="Resize the council to exactly N members.")
    parser.add_argument("--add-members", type=_positive_int, metavar="N", help="Add N new council members.")
    parser.add_argument("--remove-members", type=_positive_int, metavar="N", help="Remove N council members from the end of the roster.")
    parser.add_argument("--set-runoff-rounds", type=_positive_int, metavar="N", help="Use N runoff rounds for tied decisions in this run.")
    parser.add_argument("--set-update-interval", type=_positive_int, metavar="N", help="Set Secretary progress update interval in seconds for this run.")
    parser.add_argument("--set-diversity", choices=("low", "balanced", "high"), help="Set proposal diversity mode for this run.")
    parser.add_argument("--json-output", action="store_true", help="Print the final decision payload as JSON.")
    parser.add_argument("--no-search", action="store_true", help="Disable Codex web search during independent research.")
    parser.add_argument("--doctor", action="store_true", help="Check local Codex CLI availability.")
    args = parser.parse_args(argv)

    config = load_config()
    _ensure_dirs(config)

    if args.doctor:
        try:
            print(codex_doctor(config))
        except CodexUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    members = ensure_state(config, reset=args.reset)
    try:
        members = _maybe_resize_members(config, members, args, parser)
    except ValueError as exc:
        parser.error(str(exc))
    write_agent_files(members)

    if args.init:
        _print_members(members)
        return 0

    if args.members:
        print(_members_table(members))
        return 0

    if args.leaderboard:
        persist_leaderboard(config, members)
        print(_leaderboard_text(config))
        return 0

    question = " ".join(args.question).strip()
    if not question:
        parser.error("provide a decision prompt, or use --init/--members/--leaderboard/--doctor")

    try:
        payload = asyncio.run(
            _run_decision(
                config,
                members,
                question,
                web_search=not args.no_search,
                max_runoff_rounds=_runoff_rounds(config, args),
                update_interval_seconds=_update_interval(config, args),
                diversity_mode=_diversity_mode(config, args),
            )
        )
    except Exception as exc:
        print(f"Council failed: {exc}", file=sys.stderr)
        print(
            "\nProject-local Codex state is stored in ./.codex. "
            "If this is the first run, authenticate locally with: "
            "CODEX_HOME=$PWD/.codex codex login",
            file=sys.stderr,
        )
        return 1
    if args.json_output:
        print(render_json_decision(payload))
    else:
        print(render_human_decision(payload))
    return 0


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _maybe_resize_members(config: dict, members, args: argparse.Namespace, parser: argparse.ArgumentParser):
    resize_args = [
        args.set_members is not None,
        args.add_members is not None,
        args.remove_members is not None,
    ]
    if sum(resize_args) > 1:
        parser.error("use only one of --set-members, --add-members, or --remove-members")
    if args.set_members is not None:
        return resize_members(config, members, args.set_members)
    if args.add_members is not None:
        return resize_members(config, members, len(members) + args.add_members)
    if args.remove_members is not None:
        return resize_members(config, members, len(members) - args.remove_members)
    return members


def _runoff_rounds(config: dict, args: argparse.Namespace) -> int:
    if args.set_runoff_rounds is not None:
        return args.set_runoff_rounds
    return int(config.get("council", {}).get("runoff_rounds", 3))


def _update_interval(config: dict, args: argparse.Namespace) -> int:
    if args.set_update_interval is not None:
        return args.set_update_interval
    return int(config.get("council", {}).get("update_interval_seconds", 5))


def _diversity_mode(config: dict, args: argparse.Namespace) -> str:
    if args.set_diversity:
        return args.set_diversity
    mode = str(config.get("council", {}).get("diversity_mode", "balanced"))
    if mode not in {"low", "balanced", "high"}:
        raise ValueError(f"Invalid council.diversity_mode: {mode}")
    return mode


async def _run_decision(
    config: dict,
    members,
    question: str,
    web_search: bool,
    max_runoff_rounds: int,
    update_interval_seconds: int,
    diversity_mode: str,
) -> dict:
    research_schema = ROOT / "schemas" / "recommendation.schema.json"
    equivalence_schema = ROOT / "schemas" / "equivalence.schema.json"
    vote_schema = ROOT / "schemas" / "vote.schema.json"
    summary_schema = ROOT / "schemas" / "summary.schema.json"
    secretary = Secretary(update_interval_seconds)
    await secretary.start(question)

    try:
        memory = load_recent_memory(config)
        secretary.set_phase("collecting independent research")
        diversity_lanes = _assign_diversity_lanes(members, diversity_mode)
        secretary.diversity_lanes_assigned(diversity_lanes, diversity_mode)
        research_jobs = [
            (
                member,
                research_prompt(
                    member,
                    question,
                    memory,
                    diversity_lane=diversity_lanes[member.name],
                    diversity_mode=diversity_mode,
                ),
                research_schema,
                "research",
                web_search,
            )
            for member in members
        ]
        research_results = await _run_jobs_with_secretary(
            config,
            research_jobs,
            secretary,
            lambda result: secretary.recommendation_done(
                result.member.name, result.payload.get("recommendation", "recommendation complete")
            ),
        )
        recommendations = [
            validate_recommendation(result.payload, result.member) for result in research_results
        ]

        secretary.set_phase("grouping equivalent proposals")
        recommendation_groups = await _group_recommendations(
            config,
            president(members),
            question,
            recommendations,
            equivalence_schema,
        )
        secretary.grouping_done(recommendation_groups)
        voting_recommendations = canonical_recommendations(recommendation_groups)

        secretary.set_phase("holding discussion and initial vote")
        discussion_jobs = [
            (
                member,
                discussion_prompt(member, question, voting_recommendations),
                vote_schema,
                "vote",
                False,
            )
            for member in members
        ]
        vote_results = await _run_jobs_with_secretary(
            config,
            discussion_jobs,
            secretary,
            lambda result: _secretary_vote_done(result, secretary, recommendation_groups, 0),
        )
        votes = [
            canonicalize_vote(
                _tag_vote(validate_vote(result.payload, result.member), 0), recommendation_groups
            )
            for result in vote_results
        ]
        all_votes = votes[:]
        vote_rounds = [evaluate_vote_round(voting_recommendations, votes, 0)]
        secretary.vote_round_done(_round_summary(vote_rounds[-1]))
        current_recommendations = voting_recommendations

        for runoff_round in range(1, max_runoff_rounds + 1):
            if vote_rounds[-1].resolved:
                break
            secretary.set_phase(f"running runoff round {runoff_round}")
            current_recommendations = filter_recommendations(
                current_recommendations, vote_rounds[-1].tied_options
            )
            runoff_jobs = [
                (
                    member,
                    runoff_prompt(
                        member,
                        question,
                        current_recommendations,
                        [round_result.to_dict() for round_result in vote_rounds],
                        runoff_round,
                        max_runoff_rounds,
                    ),
                    vote_schema,
                    f"runoff-{runoff_round}",
                    False,
                )
                for member in members
            ]
            runoff_results = await _run_jobs_with_secretary(
                config,
                runoff_jobs,
                secretary,
                lambda result, round_number=runoff_round: _secretary_vote_done(
                    result, secretary, recommendation_groups, round_number
                ),
            )
            runoff_votes = [
                canonicalize_vote(
                    _tag_vote(validate_vote(result.payload, result.member), runoff_round),
                    recommendation_groups,
                )
                for result in runoff_results
            ]
            all_votes.extend(runoff_votes)
            vote_rounds.append(evaluate_vote_round(current_recommendations, runoff_votes, runoff_round))
            secretary.vote_round_done(_round_summary(vote_rounds[-1]))

        decision = decision_from_rounds(voting_recommendations, vote_rounds)
        updated = update_after_decision(
            config=config,
            members=members,
            proposing_members={item["proposer"] for item in recommendations},
            winning_member=decision.winning_member,
            voter_names={item["voter"] for item in all_votes},
            tie_breaker_member=None,
            winning_members=set(decision.winning_members),
        )
        leaderboard = read_json(resolve_project_path(config["storage"]["leaderboard_path"]), {})[
            "leaderboard"
        ]
        append_decision_memory(
            config,
            question=question,
            winning_option=decision.winning_option,
            winning_member=decision.winning_member,
            votes=all_votes,
            final_tied_options=decision.tied_options,
        )

        winner_payload = _winner_payload(decision, max_runoff_rounds)
        secretary.set_phase("preparing final summary")
        final_output = await _final_output(
            config,
            president(updated),
            question,
            recommendations,
            all_votes,
            winner_payload,
            leaderboard,
            summary_schema,
            decision,
            max_runoff_rounds,
        )
        secretary.set_phase("finished")
        return {
            "final_output": final_output.rstrip(),
            "status": decision.status,
            "winning_option": decision.winning_option,
            "winning_member": decision.winning_member,
            "winning_members": decision.winning_members,
            "final_tied_options": decision.tied_options,
            "recommendations": recommendations,
            "recommendation_groups": recommendation_groups,
            "votes": all_votes,
            "vote_rounds": [round_result.to_dict() for round_result in decision.vote_rounds],
            "leaderboard": leaderboard,
            "runoff_rounds": max(0, len(decision.vote_rounds) - 1),
            "max_runoff_rounds": max_runoff_rounds,
            "diversity_mode": diversity_mode,
            "diversity_lanes": diversity_lanes,
        }
    finally:
        await secretary.stop()


async def _group_recommendations(
    config: dict,
    pres,
    question: str,
    recommendations: list[dict],
    equivalence_schema,
) -> list[dict]:
    try:
        result = await run_member(
            config,
            pres,
            equivalence_prompt(pres, question, recommendations),
            equivalence_schema,
            "equivalence",
            False,
        )
        groups = result.payload.get("groups", [])
    except Exception:
        groups = fallback_recommendation_groups(recommendations)
    return validate_recommendation_groups(groups, recommendations)


def _secretary_vote_done(result, secretary: Secretary, groups: list[dict], round_number: int) -> None:
    vote = canonicalize_vote(result.payload, groups)
    secretary.vote_done(result.member.name, vote.get("selected_option", "abstain"), round_number)


def _assign_diversity_lanes(members, diversity_mode: str) -> dict[str, str]:
    if diversity_mode == "low":
        lanes = [
            "best overall practical pick",
            "solid alternative to the most obvious pick",
            "value and convenience pick",
            "taste or quality-focused pick",
            "personality-led wildcard that remains practical",
        ]
    else:
        lanes = [
            "safest/mainstream pick",
            "overlooked or underrated pick",
            "budget/convenience pick",
            "high-experience or special-occasion pick",
            "contrarian/novel pick",
        ]
    if diversity_mode == "high":
        lanes = [
            f"{lane}; avoid the most obvious answer unless this lane requires it"
            for lane in lanes
        ]
    return {member.name: lanes[index % len(lanes)] for index, member in enumerate(members)}


async def _run_jobs_with_secretary(config: dict, jobs: list[tuple], secretary: Secretary, on_result):
    tasks = [
        asyncio.create_task(run_member(config, member, prompt, schema, phase, web_search))
        for member, prompt, schema, phase, web_search in jobs
    ]
    results = []
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            on_result(result)
        return results
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _final_output(
    config: dict,
    pres,
    question: str,
    recommendations: list[dict],
    all_votes: list[dict],
    winner_payload: dict,
    leaderboard: list[dict],
    summary_schema,
    decision,
    max_runoff_rounds: int,
) -> str:
    try:
        summary_result = await run_member(
            config,
            pres,
            president_summary_prompt(
                pres, question, recommendations, all_votes, winner_payload, leaderboard
            ),
            summary_schema,
            "summary",
            False,
        )
        return summary_result.payload["final_output"]
    except Exception:
        if decision.status == "unresolved_tie":
            return render_fallback(
                None,
                f"No single winner after {max_runoff_rounds} runoff round(s).",
                all_votes,
                leaderboard,
                tied_options=decision.tied_options,
            )
        why = "Won the council vote"
        if len(decision.vote_rounds) > 1:
            why += f" after {len(decision.vote_rounds) - 1} runoff round(s)"
        return render_fallback(
            decision.winning_option,
            why + ".",
            all_votes,
            leaderboard,
            winning_members=decision.winning_members,
        )


def _winner_payload(decision, max_runoff_rounds: int) -> dict:
    return {
        "status": decision.status,
        "winning_option": decision.winning_option,
        "winning_member": decision.winning_member,
        "winning_members": decision.winning_members,
        "vote_counts": decision.vote_counts,
        "final_tied_options": decision.tied_options,
        "vote_rounds": [round_result.to_dict() for round_result in decision.vote_rounds],
        "runoff_rounds": max(0, len(decision.vote_rounds) - 1),
        "max_runoff_rounds": max_runoff_rounds,
        "tie_broken_by": None,
    }


def _round_summary(round_result) -> str:
    if round_result.resolved:
        return f"round {round_result.round_number}: {_display_option(round_result.winning_option)} leads outright"
    tied = ", ".join(_display_option(option) for option in round_result.tied_options)
    return f"round {round_result.round_number}: tied options are {tied}"


def _display_option(option: str | None) -> str:
    return str(option or "").strip().rstrip(".;:").strip()


def render_human_decision(payload: dict) -> str:
    return payload["final_output"].rstrip()


def render_json_decision(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _tag_vote(vote: dict, round_number: int) -> dict:
    tagged = dict(vote)
    tagged["round"] = round_number
    return tagged


def _ensure_dirs(config: dict) -> None:
    for section in ("storage", "runtime"):
        for key, value in config.get(section, {}).items():
            path = resolve_project_path(value)
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)


def _print_members(members) -> None:
    print(_members_table(members))


def _leaderboard_text(config: dict) -> str:
    payload = read_json(resolve_project_path(config["storage"]["leaderboard_path"]), {"leaderboard": []})
    rows = []
    for rank, row in enumerate(payload["leaderboard"], start=1):
        rows.append(
            {
                "Rank": rank,
                "Member": row["member"],
                "Role": "President" if row["president"] else "Member",
                "Wins": row["total_wins"],
                "Proposals": row["total_proposals"],
                "Win Rate": f"{row['win_rate']:.0%}",
                "Votes": row["vote_participation"],
                "Tie Breaks": row["tie_break_victories"],
                "Model": row["model"],
            }
        )
    return "Leaderboard\n" + _format_table(
        ["Rank", "Member", "Role", "Wins", "Proposals", "Win Rate", "Votes", "Tie Breaks", "Model"],
        rows,
        right_align={"Rank", "Wins", "Proposals", "Votes", "Tie Breaks"},
    )


def _members_table(members) -> str:
    rows = []
    for member in members:
        rows.append(
            {
                "Name": member.name,
                "Role": "President" if member.is_president else "Member",
                "Model": member.model,
                "Personality": member.personality,
                "Wins": member.total_wins,
                "Proposals": member.total_proposals,
                "Votes": member.total_votes_cast,
                "Tie Breaks": member.tie_break_victories,
            }
        )
    return "Council Members\n" + _format_table(
        ["Name", "Role", "Model", "Personality", "Wins", "Proposals", "Votes", "Tie Breaks"],
        rows,
        right_align={"Wins", "Proposals", "Votes", "Tie Breaks"},
    )


def _format_table(
    headers: list[str],
    rows: list[dict],
    right_align: set[str] | None = None,
) -> str:
    right_align = right_align or set()
    rendered_rows = [{header: str(row.get(header, "")) for header in headers} for row in rows]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rendered_rows))
        for header in headers
    }

    def render_row(row: dict[str, str]) -> str:
        cells = []
        for header in headers:
            value = row[header]
            if header in right_align:
                cells.append(value.rjust(widths[header]))
            else:
                cells.append(value.ljust(widths[header]))
        return " | ".join(cells)

    header_row = render_row({header: header for header in headers})
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = [render_row(row) for row in rendered_rows]
    return "\n".join([header_row, separator, *body])
