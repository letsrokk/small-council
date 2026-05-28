from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ROOT, resolve_project_path
from .state import Member


@dataclass
class CodexResult:
    member: Member
    payload: dict[str, Any]
    stdout: str
    stderr: str


class CodexUnavailable(RuntimeError):
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
        raise RuntimeError(
            f"Codex subagent {member.name} failed in {phase} with exit code {proc.returncode}.\n"
            f"Log: {log_path}\n{stderr.strip()}"
        )
    if not output_path.exists():
        raise RuntimeError(f"Codex did not write expected output file: {output_path}")

    payload = _parse_json(output_path.read_text(encoding="utf-8"))
    return CodexResult(member=member, payload=payload, stdout=stdout, stderr=stderr)


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
