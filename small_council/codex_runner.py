from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .config import ROOT, resolve_project_path
from .model_providers import (
    ModelInfo,
    codex_catalog_models,
    effective_models_for_provider,
    provider_config,
)
from .state import Member


@dataclass
class CodexResult:
    member: Member
    payload: dict[str, Any]
    stdout: str
    stderr: str


class CodexUnavailable(RuntimeError):
    pass


class CodexRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        member_name: str,
        phase: str,
        log_path: Path | None = None,
        retryable: bool = True,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.member_name = member_name
        self.phase = phase
        self.log_path = log_path
        self.retryable = retryable
        self.retry_after = retry_after

    def __str__(self) -> str:
        parts = [self.message]
        if self.retry_after:
            parts.append(f"Try again at {self.retry_after}.")
        if self.log_path:
            parts.append(f"Log: {self.log_path}")
        return "\n".join(parts)


class CodexUsageLimitError(CodexRunError):
    pass


class CodexOutputError(CodexRunError):
    pass


def codex_doctor(config: dict[str, Any] | None = None) -> str:
    codex = shutil.which("codex")
    if not codex:
        raise CodexUnavailable("The `codex` executable was not found on PATH.")
    result = subprocess.run(
        [codex, "--version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=_codex_env(config),
    )
    return (result.stdout + result.stderr).strip()


class CodexProvider:
    name = "codex"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = provider_config(config, self.name)

    def discover_models(self) -> list[ModelInfo]:
        return codex_catalog_models(self.config)

    def validate_model(self, model: str) -> bool:
        return any(item.model == model for item in list_codex_models(self.config))

    async def run(
        self,
        member: Member,
        prompt: str,
        schema_path: Path,
        phase: str,
        web_search: bool = False,
    ) -> CodexResult:
        return await run_member(self.config, member, prompt, schema_path, phase, web_search)


def list_codex_models(config: dict[str, Any]) -> list[ModelInfo]:
    provider = CodexProvider(config)
    if not provider.provider_config.get("enabled", False):
        return []
    try:
        discovered = provider.discover_models() if provider.provider_config.get("discover_models", True) else []
    except Exception:
        discovered = []
    return effective_models_for_provider("codex", config, discovered)


async def run_member(
    config: dict[str, Any],
    member: Member,
    prompt: str,
    schema_path: Path,
    phase: str,
    web_search: bool = False,
) -> CodexResult:
    codex = shutil.which("codex")
    if not codex:
        raise CodexUnavailable("The `codex` executable was not found on PATH.")

    runtime_temp = resolve_project_path(config["runtime"]["temp_path"])
    runtime_logs = resolve_project_path(config["runtime"]["logs_path"])
    runtime_temp.mkdir(parents=True, exist_ok=True)
    runtime_logs.mkdir(parents=True, exist_ok=True)
    output_path = runtime_temp / f"{phase}-{member.name.lower()}-last.json"
    log_path = runtime_logs / f"{phase}-{member.name.lower()}.log"

    args = [
        codex,
        "--sandbox",
        "read-only",
        "-a",
        "never",
        "-C",
        str(ROOT),
        "-m",
        member.model,
        "-c",
        f"model_reasoning_effort={json.dumps(config.get('codex', {}).get('reasoning_effort', 'medium'))}",
    ]
    if web_search:
        args.append("--search")
    args.extend(
        [
            "exec",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ]
    )
    args.append("-")

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=ROOT,
        env=_codex_env(config),
    )
    stdout_bytes, stderr_bytes = await proc.communicate(prompt.encode("utf-8"))
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    log_path.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")

    if proc.returncode != 0:
        raise _codex_error_for_exit(member, phase, proc.returncode, stdout, stderr, log_path)
    if not output_path.exists():
        raise CodexOutputError(
            f"Codex subagent {member.name} did not write expected output in {phase}.",
            member_name=member.name,
            phase=phase,
            log_path=log_path,
            retryable=True,
        )

    try:
        payload = _parse_json(output_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise CodexOutputError(
            f"Codex subagent {member.name} wrote invalid JSON in {phase}.",
            member_name=member.name,
            phase=phase,
            log_path=log_path,
            retryable=True,
        ) from exc
    return CodexResult(member=member, payload=payload, stdout=stdout, stderr=stderr)


async def run_secretary_model(
    config: dict[str, Any],
    model: str,
    prompt: str,
    schema_path: Path,
    phase: str,
) -> dict[str, Any]:
    secretary = Member(
        name="Secretary",
        provider="codex",
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
) -> list[CodexResult]:
    tasks = [
        run_member(config, member, prompt, schema, phase, web_search)
        for member, prompt, schema, phase, web_search in jobs
    ]
    return await asyncio.gather(*tasks)


def _codex_env(config: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    configured_home = (config or {}).get("codex", {}).get("project_local_home", "./.codex")
    local_home = resolve_project_path(configured_home)
    local_home.mkdir(parents=True, exist_ok=True)
    env["CODEX_HOME"] = str(local_home)
    env["XDG_CONFIG_HOME"] = str(ROOT / "runtime" / "xdg-config")
    env["XDG_CACHE_HOME"] = str(ROOT / "runtime" / "xdg-cache")
    env["XDG_DATA_HOME"] = str(ROOT / "runtime" / "xdg-data")
    return env


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _codex_error_for_exit(
    member: Member,
    phase: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    log_path: Path,
) -> CodexRunError:
    combined = "\n".join([stdout, stderr])
    if _is_usage_limit_error(combined):
        return CodexUsageLimitError(
            f"Codex usage limit reached while running {member.name} in {phase}.",
            member_name=member.name,
            phase=phase,
            log_path=log_path,
            retryable=False,
            retry_after=_retry_after(combined),
        )
    return CodexRunError(
        f"Codex subagent {member.name} failed in {phase} with exit code {returncode}.",
        member_name=member.name,
        phase=phase,
        log_path=log_path,
        retryable=True,
    )


def _is_usage_limit_error(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        "you've hit your usage limit",
        "you have hit your usage limit",
        "usage limit",
        "out of token",
        "out-of-token",
        "quota",
        "credits",
        "credit limit",
    ]
    return any(pattern in lowered for pattern in patterns)


def _retry_after(text: str) -> str | None:
    match = re.search(r"try again at ([^\n.]+(?:\.[^\n.]+)?)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
