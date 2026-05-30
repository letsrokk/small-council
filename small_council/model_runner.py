from __future__ import annotations

import asyncio
import ast
import re
from pathlib import Path
from typing import Any

from .config import ROOT
from .model_providers import create_provider, provider_config
from .prompts import search_plan_prompt
from .state import Member
from .web_search import (
    SearchWorker,
    create_search_worker,
    search_context_block,
    search_enabled,
    web_search_config,
    write_search_log,
)


class ModelProviderUnavailable(RuntimeError):
    pass


async def run_member(
    config: dict[str, Any],
    member: Member,
    prompt: str,
    schema_path: Path,
    phase: str,
    web_search: bool = False,
    search_worker: SearchWorker | None = None,
) -> Any:
    if not provider_config(config, member.provider).get("enabled", False):
        raise ModelProviderUnavailable(f"Model provider {member.provider} is disabled.")
    provider = create_provider(member.provider, config)
    if provider is None:
        raise ModelProviderUnavailable(f"No model provider is available for {member.provider}.")
    if web_search:
        if search_enabled(config):
            worker = search_worker or create_search_worker(config)
            return await _run_member_with_search(config, provider, member, prompt, schema_path, phase, worker)
        return await provider.run(member, prompt, schema_path, phase, False)
    return await provider.run(member, prompt, schema_path, phase, False)


async def run_secretary_model(
    config: dict[str, Any],
    provider_name: str,
    model: str,
    prompt: str,
    schema_path: Path,
    phase: str,
) -> dict[str, Any]:
    secretary = Member(
        name="Secretary",
        provider=provider_name,
        model=model,
        personality="non-voting progress reporter",
        is_president=False,
        created_at="runtime",
    )
    result = await run_member(config, secretary, prompt, schema_path, phase, False)
    return result.payload


async def run_many(
    config: dict[str, Any],
    jobs: list[tuple[Member, str, Path, str, bool]],
) -> list[Any]:
    search_worker = create_search_worker(config) if search_enabled(config) else None
    tasks = [
        run_member(config, member, prompt, schema, phase, web_search, search_worker)
        for member, prompt, schema, phase, web_search in jobs
    ]
    return await asyncio.gather(*tasks)


async def _run_member_with_search(
    config: dict[str, Any],
    provider: Any,
    member: Member,
    prompt: str,
    schema_path: Path,
    phase: str,
    search_worker: SearchWorker | None,
) -> Any:
    search_config = web_search_config(config)
    if search_worker is None:
        return await provider.run(member, prompt, schema_path, phase, False)

    events: list[dict[str, Any]] = []
    queries = await _planned_queries(config, provider, member, prompt, phase, events)
    if not queries:
        write_search_log(config, phase, member.name, events)
        return await provider.run(member, prompt, schema_path, phase, False)

    searches: list[dict[str, Any]] = []
    max_results = int(search_config["maxResults"])
    for query in queries:
        response = await search_worker.search(query=query, max_results=max_results, events=events)
        if not response.ok:
            searches.append({"query": query, "error": response.error, "results": []})
            continue
        searches.append(
            {
                "query": query,
                "results": [result.to_dict() for result in response.results],
            }
        )

    write_search_log(config, phase, member.name, events)
    final_prompt = prompt + search_context_block(searches)
    return await provider.run(member, final_prompt, schema_path, phase, False)


async def _planned_queries(
    config: dict[str, Any],
    provider: Any,
    member: Member,
    prompt: str,
    phase: str,
    events: list[dict[str, Any]],
) -> list[str]:
    schema_path = ROOT / "schemas" / "search-plan.schema.json"
    try:
        result = await provider.run(
            member,
            search_plan_prompt(member, prompt),
            schema_path,
            f"{phase}-search-plan",
            False,
        )
        raw_queries = result.payload.get("queries", [])
        queries = _normalize_queries(raw_queries)
        if not queries:
            queries = _fallback_queries(prompt)
        events.append({"status": "planned", "queries": queries})
        return queries
    except Exception as exc:
        fallback = _fallback_queries(prompt)
        events.append(
            {
                "status": "planning_failed",
                "error": str(exc),
                "queries": fallback,
            }
        )
        return fallback


def _normalize_queries(raw_queries: Any) -> list[str]:
    if not isinstance(raw_queries, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for raw in raw_queries:
        query = str(raw).strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= 3:
            break
    return queries


def _fallback_queries(prompt: str) -> list[str]:
    question = _question_from_prompt(prompt)
    if not question:
        return []
    if _question_needs_search(question):
        return [question]
    return []


def _question_needs_search(question: str) -> bool:
    text = str(question or "")
    if not text.strip():
        return False
    if re.search(
        r"\b("
        r"today|tonight|current|currently|latest|recent|recently|now|near me|local|"
        r"restaurant|news|price|prices|pricing|weather|available|availability|"
        r"product|products|review|reviews|released|release|launch|launched|"
        r"this (?:week|month|year)|last (?:night|week|month|year)|next (?:week|month|year)|"
        r"yesterday|tomorrow|as of|upcoming|who won|who is the current|what happened to"
        r")\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(?:after|since)\s+20(?:2[4-9]|[3-9][0-9])\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\b20(?:2[4-9]|[3-9][0-9])\b", text):
        return True
    return False


def _question_from_prompt(prompt: str) -> str:
    match = re.search(r"The user asks:\s*(.+)", prompt)
    if match:
        raw = match.group(1).strip()
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, str):
                return parsed
        except (SyntaxError, ValueError):
            return raw.strip("'\"")
    return ""
