from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .model_providers import create_provider, provider_config
from .state import Member


class ModelProviderUnavailable(RuntimeError):
    pass


async def run_member(
    config: dict[str, Any],
    member: Member,
    prompt: str,
    schema_path: Path,
    phase: str,
    web_search: bool = False,
) -> Any:
    if not provider_config(config, member.provider).get("enabled", False):
        raise ModelProviderUnavailable(f"Model provider {member.provider} is disabled.")
    provider = create_provider(member.provider, config)
    if provider is None:
        raise ModelProviderUnavailable(f"No model provider is available for {member.provider}.")
    return await provider.run(member, prompt, schema_path, phase, web_search)


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
    tasks = [
        run_member(config, member, prompt, schema, phase, web_search)
        for member, prompt, schema, phase, web_search in jobs
    ]
    return await asyncio.gather(*tasks)
