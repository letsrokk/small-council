from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .codex_runner import (
    CodexRunError,
    CodexUnavailable,
    CodexUsageLimitError,
)
from .config import (
    ROOT,
    _parse_scalar,
    load_config,
    read_json,
    resolve_project_path,
    save_config,
    set_config_value,
)
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
from .model_providers import (
    effective_model_pool,
    parse_parameter_limit,
    provider_report,
)
from .model_runner import run_member
from .output import (
    BaseRenderer,
    RunContext,
    final_decision_announcement_lines,
    render_leaderboard_text,
    render_members_text,
    select_renderer,
)
from .prompts import (
    discussion_prompt,
    discussion_round_prompt,
    equivalence_prompt,
    president_tie_break_prompt,
    president_summary_prompt,
    research_prompt,
    runoff_prompt,
)
from .secretary import BaseSecretary, SecretaryConfig, create_secretary
from .state import (
    ensure_state,
    persist_leaderboard,
    president,
    resize_members,
    update_after_decision,
    write_agent_files,
)
from .web_search import (
    SearchError,
    create_search_provider,
    create_search_worker,
    search_enabled,
    web_search_config,
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
    parser.add_argument("--secretary", choices=("local", "model"), help="Use local or model-backed Secretary progress reports.")
    parser.add_argument("--secretary-verbosity", choices=("low", "balanced", "high"), help="Set model-backed Secretary verbosity for this run.")
    parser.add_argument("--set-diversity", choices=("low", "balanced", "high"), help="Set proposal diversity mode for this run.")
    parser.add_argument("--plain-output", action="store_true", help="Force plain human-readable output.")
    parser.add_argument("--rich-output", action="store_true", help="Force Rich terminal output.")
    parser.add_argument("--json-output", action="store_true", help="Print the final decision payload as JSON.")
    parser.add_argument("--no-search", action="store_true", help="Disable web search during independent research.")
    parser.add_argument("--doctor", action="store_true", help="Check configured model providers.")
    parser.add_argument("--models", action="store_true", help="List discovered, static, and effective models.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Persist a config value.")
    args = parser.parse_args(argv)

    config = load_config()
    _ensure_dirs(config)
    renderer = None

    if args.set:
        try:
            config = _apply_config_sets(config, args.set)
            save_config(config)
        except ValueError as exc:
            parser.error(str(exc))
        return 0

    if args.doctor:
        print(_doctor_text(config))
        return 0

    if args.models:
        print(_models_text(config))
        return 0

    try:
        renderer = select_renderer(args, sys.stdout, sys.stderr, sys.stdin)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    members = ensure_state(config, reset=args.reset)
    try:
        members = _maybe_resize_members(config, members, args, parser)
    except ValueError as exc:
        parser.error(str(exc))
    write_agent_files(members)

    if args.init:
        if renderer:
            renderer.render_members(members)
        else:
            print(render_members_text(members))
        return 0

    if args.members:
        if renderer:
            renderer.render_members(members)
        else:
            print(render_members_text(members))
        return 0

    if args.leaderboard:
        persist_leaderboard(config, members)
        leaderboard = read_json(resolve_project_path(config["storage"]["leaderboard_path"]), {})[
            "leaderboard"
        ]
        if renderer:
            renderer.render_leaderboard(leaderboard)
        else:
            print(render_leaderboard_text(leaderboard))
        return 0

    question = " ".join(args.question).strip()
    if not question:
        parser.error("provide a decision prompt, or use --init/--members/--leaderboard/--doctor")

    try:
        secretary_config = _secretary_config(config, args)
        _validate_secretary_config(config, secretary_config)
        if renderer:
            renderer.start_run(
                RunContext(
                    question=question,
                    member_count=len(members),
                    diversity_mode=_diversity_mode(config, args),
                    secretary_mode=secretary_config.mode,
                    secretary_verbosity=secretary_config.verbosity,
                    discussion_rounds=_discussion_rounds(config),
                    runoff_round_limit=_runoff_rounds(config, args),
                    web_search_enabled=(not args.no_search) and search_enabled(config),
                )
            )
            renderer.seed_members(members)
        payload = asyncio.run(
            _run_decision(
                config,
                members,
                question,
                web_search=not args.no_search,
                max_runoff_rounds=_runoff_rounds(config, args),
                secretary_config=secretary_config,
                diversity_mode=_diversity_mode(config, args),
                renderer=renderer,
            )
        )
    except CodexRunError as exc:
        if renderer:
            renderer.error(f"Council failed: {exc}")
        else:
            print(f"Council failed: {exc}", file=sys.stderr)
        if renderer:
            renderer.close()
        return 1
    except Exception as exc:
        if renderer:
            renderer.error(f"Council failed: {exc}")
        else:
            print(f"Council failed: {exc}", file=sys.stderr)
        if isinstance(exc, CodexUnavailable):
            print(
                "\nProject-local Codex state is stored in ./.codex. "
                "If this is the first run, authenticate locally with: "
                "CODEX_HOME=$PWD/.codex codex login",
                file=sys.stderr,
            )
        if renderer:
            renderer.close()
        return 1
    if renderer:
        renderer.final_decision(payload)
        renderer.close()
    if args.json_output:
        for line in final_decision_announcement_lines(payload):
            print(line, file=sys.stderr)
        print(render_json_decision(payload))
    elif not (renderer and renderer.suppresses_final_stdout):
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


def _secretary_config(config: dict, args: argparse.Namespace) -> SecretaryConfig:
    secretary_config = config.get("secretary", {})
    mode = args.secretary or str(secretary_config.get("mode", "model"))
    if mode not in {"local", "model"}:
        raise ValueError(f"Invalid secretary.mode: {mode}")

    verbosity = args.secretary_verbosity or str(secretary_config.get("verbosity", "balanced"))
    if verbosity not in {"low", "balanced", "high"}:
        raise ValueError(f"Invalid secretary.verbosity: {verbosity}")

    provider = str(secretary_config.get("provider", "codex"))
    model = str(secretary_config.get("model", "gpt-5.4-mini"))
    return SecretaryConfig(
        mode=mode,
        provider=provider,
        model=model,
        verbosity=verbosity,
        immediate_updates=True,
    )


def _diversity_mode(config: dict, args: argparse.Namespace) -> str:
    if args.set_diversity:
        return args.set_diversity
    mode = str(config.get("council", {}).get("diversity_mode", "balanced"))
    if mode not in {"low", "balanced", "high"}:
        raise ValueError(f"Invalid council.diversity_mode: {mode}")
    return mode


def _discussion_rounds(config: dict) -> int:
    return int(config.get("council", {}).get("discussion_rounds", 2))


async def _run_decision(
    config: dict,
    members,
    question: str,
    web_search: bool,
    max_runoff_rounds: int,
    secretary_config: SecretaryConfig,
    diversity_mode: str,
    renderer: BaseRenderer | None = None,
) -> dict:
    research_schema = ROOT / "schemas" / "recommendation.schema.json"
    equivalence_schema = ROOT / "schemas" / "equivalence.schema.json"
    vote_schema = ROOT / "schemas" / "vote.schema.json"
    summary_schema = ROOT / "schemas" / "summary.schema.json"
    discussion_schema = ROOT / "schemas" / "discussion.schema.json"
    secretary = create_secretary(config, secretary_config, renderer=renderer)
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
            renderer,
            lambda result: secretary.recommendation_done(
                result.member.name, result.payload.get("recommendation", "recommendation complete")
            ),
        )
        draft_recommendations = [
            validate_recommendation(result.payload, result.member) for result in research_results
        ]
        current_recommendations = draft_recommendations
        await secretary.report_milestone("initial proposals complete")

        discussion_transcript: list[dict[str, object]] = [
            {
                "type": "draft_proposal",
                "member": item["proposer"],
                "recommendation": item["recommendation"],
                "short_reasoning": item["short_reasoning"],
                "pros": item["pros"],
                "cons": item["cons"],
                "confidence": item["confidence"],
            }
            for item in draft_recommendations
        ]

        discussion_round_count = _discussion_rounds(config)
        secretary.set_phase("threaded discussion")
        discussion_round_payloads: list[dict[str, object]] = []
        for discussion_round in range(1, discussion_round_count + 1):
            secretary.discussion_round_started(discussion_round)
            discussion_jobs = [
                (
                    member,
                    discussion_round_prompt(
                        member,
                        question,
                        current_recommendations,
                        discussion_transcript,
                        discussion_round,
                        discussion_round_count,
                    ),
                    discussion_schema,
                    f"discussion-{discussion_round}",
                    False,
                )
                for member in members
            ]
            discussion_results = await _run_jobs_with_secretary(
                config,
                discussion_jobs,
                secretary,
                renderer,
                lambda result, round_number=discussion_round, baseline=current_recommendations: _secretary_discussion_done(
                    result, secretary, baseline, round_number
                ),
            )
            round_messages = []
            next_recommendations = []
            result_by_member = {result.member.name: result for result in discussion_results}
            for member in members:
                result = result_by_member[member.name]
                discussion_reply = str(result.payload.get("discussion_reply", "")).strip()
                revised_payload = result.payload.get("revised_recommendation", {})
                revised_recommendation = validate_recommendation(revised_payload, result.member)
                prior_recommendation = _recommendation_for_proposer(
                    current_recommendations, member.name
                )
                changed = _normalize_option(
                    revised_recommendation.get("recommendation", "")
                ) != _normalize_option(prior_recommendation.get("recommendation", ""))
                round_messages.append(
                    {
                        "member": member.name,
                        "discussion_reply": discussion_reply,
                        "prior_recommendation": prior_recommendation.get("recommendation"),
                        "revised_recommendation": revised_recommendation.get("recommendation"),
                        "changed": changed,
                    }
                )
                discussion_transcript.append(
                    {
                        "round": discussion_round,
                        "member": member.name,
                        "discussion_reply": discussion_reply,
                        "prior_recommendation": prior_recommendation.get("recommendation"),
                        "revised_recommendation": revised_recommendation,
                        "changed": changed,
                    }
                )
                next_recommendations.append(revised_recommendation)
            current_recommendations = next_recommendations
            discussion_round_payloads.append(
                {
                    "round_number": discussion_round,
                    "messages": round_messages,
                }
            )
            secretary.discussion_round_done(discussion_round)
            await secretary.report_milestone(f"discussion round {discussion_round} complete")

        secretary.set_phase("finalizing proposals")
        for recommendation in current_recommendations:
            secretary.final_recommendation_done(
                recommendation["proposer"], recommendation["recommendation"]
            )
        await secretary.report_milestone("final proposals complete")

        secretary.set_phase("grouping equivalent proposals")
        recommendation_groups = await _group_recommendations(
            config,
            president(members),
            question,
            current_recommendations,
            equivalence_schema,
        )
        secretary.grouping_done(recommendation_groups)
        await secretary.report_milestone("proposal grouping complete")
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
            renderer,
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
        await secretary.report_milestone("initial vote complete")
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
                renderer,
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
            await secretary.report_milestone(f"runoff round {runoff_round} complete")

        pres = president(members)
        tie_break_vote = None
        tie_breaker_member = None
        if not vote_rounds[-1].resolved:
            secretary.set_phase("president tie-break")
            current_recommendations = filter_recommendations(
                current_recommendations, vote_rounds[-1].tied_options
            )
            tie_break_vote = await _president_tie_break(
                config,
                pres,
                question,
                current_recommendations,
                vote_rounds,
                recommendation_groups,
                vote_schema,
                secretary,
                max_runoff_rounds + 1,
            )
            if tie_break_vote:
                all_votes.append(tie_break_vote)
                tie_breaker_member = pres.name

        decision = decision_from_rounds(
            voting_recommendations,
            vote_rounds,
            tie_breaker_member=tie_breaker_member,
            tie_break_vote=tie_break_vote,
        )
        updated = update_after_decision(
            config=config,
            members=members,
            proposing_members={item["proposer"] for item in draft_recommendations},
            winning_member=decision.winning_member,
            voter_names={item["voter"] for item in all_votes},
            tie_breaker_member=decision.tie_broken_by,
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
            tie_broken_by=decision.tie_broken_by,
            tie_break_vote=decision.tie_break_vote,
        )

        winner_payload = _winner_payload(decision, max_runoff_rounds)
        secretary.set_phase("preparing final summary")
        final_output = await _final_output(
            config,
            president(updated),
            question,
            current_recommendations,
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
            "tie_broken_by": decision.tie_broken_by,
            "tie_break_vote": decision.tie_break_vote,
            "draft_recommendations": draft_recommendations,
            "final_recommendations": current_recommendations,
            "discussion_rounds": discussion_round_payloads,
            "discussion_transcript": discussion_transcript,
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


def _recommendation_for_proposer(recommendations: list[dict], proposer: str) -> dict:
    for recommendation in recommendations:
        if recommendation.get("proposer") == proposer:
            return recommendation
    return {
        "proposer": proposer,
        "recommendation": "",
        "short_reasoning": "",
        "pros": [],
        "cons": [],
        "confidence": 0,
    }


def _normalize_option(value: str) -> str:
    return str(value).strip().lower().rstrip(".;:!?")


def _secretary_discussion_done(
    result,
    secretary: BaseSecretary,
    baseline_recommendations: list[dict],
    round_number: int,
) -> None:
    revised = validate_recommendation(result.payload.get("revised_recommendation", {}), result.member)
    prior = _recommendation_for_proposer(baseline_recommendations, result.member.name)
    changed = _normalize_option(revised.get("recommendation", "")) != _normalize_option(
        prior.get("recommendation", "")
    )
    secretary.state and secretary.discussion_message_done(
        round_number,
        result.member.name,
        result.payload.get("discussion_reply", ""),
        prior.get("recommendation", ""),
        revised.get("recommendation", ""),
        changed,
    )


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


async def _president_tie_break(
    config: dict,
    pres,
    question: str,
    tied_recommendations: list[dict],
    vote_rounds: list,
    groups: list[dict],
    vote_schema,
    secretary: BaseSecretary,
    round_number: int,
) -> dict | None:
    tied_options = set(vote_rounds[-1].tied_options)
    try:
        result = await _run_member_with_retries(
            config,
            pres,
            president_tie_break_prompt(
                pres,
                question,
                tied_recommendations,
                [round_result.to_dict() for round_result in vote_rounds],
            ),
            vote_schema,
            "tie-break",
            False,
            secretary,
        )
        vote = canonicalize_vote(
            _tag_vote(validate_vote(result.payload, result.member), round_number), groups
        )
        vote["tie_break"] = True
    except Exception:
        return None
    if vote.get("selected_option") not in tied_options:
        return None
    return vote


def _secretary_vote_done(
    result, secretary: BaseSecretary, groups: list[dict], round_number: int
) -> None:
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


async def _run_jobs_with_secretary(
    config: dict,
    jobs: list[tuple],
    secretary: BaseSecretary,
    renderer: BaseRenderer | None,
    on_result,
):
    tasks = []
    search_worker = create_search_worker(config) if search_enabled(config) else None
    for member, prompt, schema, phase, web_search in jobs:
        if renderer:
            renderer.member_status(member.name, f"running {phase}")
        task = asyncio.create_task(
            _run_member_with_retries(
                config, member, prompt, schema, phase, web_search, secretary, search_worker
            )
        )
        tasks.append(task)
    results = []
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            if renderer:
                renderer.member_status(result.member.name, "completed")
            on_result(result)
        return results
    except Exception:
        if renderer:
            renderer.error("A council member run failed; cancelling remaining tasks.")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_member_with_retries(
    config: dict,
    member,
    prompt,
    schema,
    phase: str,
    web_search: bool,
    secretary: BaseSecretary,
    search_worker=None,
):
    retries = max(0, int(config.get("codex", {}).get("retries", 2)))
    max_attempts = retries + 1
    base_delay = max(0.0, float(config.get("codex", {}).get("retry_base_delay_seconds", 1.0)))
    attempt = 1
    while True:
        try:
            return await run_member(
                config, member, prompt, schema, phase, web_search, search_worker
            )
        except CodexUsageLimitError as exc:
            secretary.agent_run_failed(
                member.name, phase, exc.message, False, attempt, max_attempts
            )
            raise
        except Exception as exc:
            if not hasattr(exc, "retryable"):
                raise
            message = getattr(exc, "message", str(exc))
            retryable = bool(getattr(exc, "retryable", True)) and attempt < max_attempts
            secretary.agent_run_failed(
                member.name, phase, message, retryable, attempt, max_attempts
            )
            if not retryable:
                raise
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            attempt += 1


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
        if decision.tie_broken_by:
            why += f"; tie broken by {decision.tie_broken_by}"
        return render_fallback(
            decision.winning_option,
            why + ".",
            all_votes,
            leaderboard,
            winning_members=decision.winning_members,
            tie_broken_by=decision.tie_broken_by,
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
        "tie_broken_by": decision.tie_broken_by,
        "tie_break_vote": decision.tie_break_vote,
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
    print(render_members_text(members))


def _leaderboard_text(config: dict) -> str:
    payload = read_json(resolve_project_path(config["storage"]["leaderboard_path"]), {"leaderboard": []})
    return render_leaderboard_text(payload["leaderboard"])


def _members_table(members) -> str:
    return render_members_text(members)


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


def _doctor_text(config: dict) -> str:
    rows = []
    lines = ["Model Provider Doctor"]
    for report in provider_report(config):
        rows.append(
            {
                "Provider": report.provider,
                "Enabled": "yes" if report.enabled else "no",
                "Health": report.health,
                "Discovered": len(report.discovered_models),
                "Static": len(report.static_models),
                "Effective": len(report.effective_models),
                "Errors": "; ".join(report.errors),
            }
        )
    lines.append(_format_table(["Provider", "Enabled", "Health", "Discovered", "Static", "Effective", "Errors"], rows))
    secretary = _secretary_config(config, argparse.Namespace(secretary=None, secretary_verbosity=None))
    lines.append("")
    lines.append(f"Secretary: {secretary.provider}/{secretary.model}")
    search_config = web_search_config(config)
    try:
        search_provider = create_search_provider(config)
        search_status = "configured" if search_provider else "unavailable"
    except SearchError as exc:
        search_provider = None
        search_status = f"unavailable ({exc})"
    lines.append(
        "Web search: "
        f"{'enabled' if search_enabled(config) else 'disabled'} "
        f"provider={search_config['provider']} "
        f"baseUrl={_search_provider_base_url(search_config)} "
        f"fallback={'enabled' if search_config.get('allowFallback') else 'disabled'} "
        f"status={search_status}"
    )
    if not effective_model_pool(config):
        lines.append("Validation: no enabled models are available.")
    try:
        _validate_secretary_config(config, secretary)
    except ValueError as exc:
        lines.append(f"Validation: {exc}")
    return "\n".join(lines)


def _models_text(config: dict) -> str:
    lines = ["Models"]
    for report in provider_report(config):
        lines.append("")
        lines.append(f"[{report.provider}] enabled={'yes' if report.enabled else 'no'} health={report.health}")
        rows = []
        all_names = {
            item.model
            for item in [
                *report.discovered_models,
                *report.static_models,
                *report.effective_models,
            ]
        }
        for name in sorted(all_names):
            info = _model_for_name(name, [*report.discovered_models, *report.static_models, *report.effective_models])
            rows.append(
                {
                    "Model": name,
                    "Size": _format_size(info.parameter_count_billion if info else None),
                    "Discovered": "yes" if any(item.model == name for item in report.discovered_models) else "no",
                    "Static": "yes" if any(item.model == name for item in report.static_models) else "no",
                    "Effective": "yes" if any(item.model == name for item in report.effective_models) else "no",
                }
            )
        if rows:
            lines.append(_format_table(["Model", "Size", "Discovered", "Static", "Effective"], rows))
        else:
            lines.append("(no models)")
    secretary = _secretary_config(config, argparse.Namespace(secretary=None, secretary_verbosity=None))
    lines.append("")
    lines.append(f"Secretary: {secretary.provider}/{secretary.model}")
    pool = effective_model_pool(config)
    lines.append("Assignment pool: " + ", ".join(f"{item.provider}/{item.model}" for item in pool))
    return "\n".join(lines)


def _model_for_name(name: str, models) -> object | None:
    for model in models:
        if model.model == name and model.parameter_count_billion is not None:
            return model
    for model in models:
        if model.model == name:
            return model
    return None


def _format_size(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:g}b"


def _apply_config_sets(config: dict, assignments: list[str]) -> dict:
    updated = config
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError("--set expects KEY=VALUE")
        key, raw_value = assignment.split("=", 1)
        value = _parse_scalar(raw_value)
        _validate_set_value(key, value)
        updated = set_config_value(updated, key, value)
    return updated


def _validate_set_value(key: str, value) -> None:
    if key.startswith("webSearch.") or key in {"search.baseUrl", "search.defaultEngines"}:
        raise ValueError(
            f"{key} is no longer supported; use provider-specific search settings."
        )
    if key.endswith(".enabled") or key.endswith(".discover_models") or key.endswith(".allow_unknown_size_models"):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false.")
    if key == "search.enabled" and not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false.")
    if key in {
        "search.timeoutSeconds",
        "search.maxResults",
        "search.minDelaySeconds",
        "search.maxConcurrentRequests",
    }:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be a positive number.")
    if key == "search.cacheTtlSeconds" and (
        not isinstance(value, (int, float)) or value < 0
    ):
        raise ValueError("search.cacheTtlSeconds must be zero or a positive number.")
    if key in {"search.provider", "search.fallbackProvider"} and str(value).strip().lower() not in {"searxng", "ollama"}:
        raise ValueError(f"{key} must be searxng or ollama.")
    if key == "search.allowFallback" and not isinstance(value, bool):
        raise ValueError("search.allowFallback must be true or false.")
    if key in {
        "search.ollama.baseUrl",
        "search.ollama.apiKeyEnv",
        "search.ollama.searchEndpoint",
        "search.ollama.fetchEndpoint",
        "search.searxng.baseUrl",
    }:
        if not str(value).strip():
            raise ValueError(f"{key} must not be empty.")
    if key == "search.searxng.defaultEngines" and not isinstance(value, list):
        raise ValueError("search.searxng.defaultEngines must be a list.")
    if key.endswith(".max_parameters") and value is not None and parse_parameter_limit(value) is None:
        raise ValueError(f"{key} must be a size like 12b or null.")
    if key == "secretary.provider":
        if not str(value).strip():
            raise ValueError("secretary.provider must not be empty.")


def _search_provider_base_url(search_config: dict) -> str:
    if search_config.get("provider") == "ollama":
        provider_config = search_config.get("ollama") or {}
        if isinstance(provider_config, dict):
            return str(provider_config.get("baseUrl", ""))
    provider_config = search_config.get("searxng") or {}
    if isinstance(provider_config, dict):
        return str(provider_config.get("baseUrl", ""))
    return ""


def _validate_secretary_config(config: dict, secretary: SecretaryConfig) -> None:
    if secretary.mode != "model":
        return
    if not any(
        item.provider == secretary.provider and item.model == secretary.model
        for item in effective_model_pool(config)
    ):
        raise ValueError(
            f"Secretary model {secretary.provider}/{secretary.model} is outside the effective model pool."
        )
