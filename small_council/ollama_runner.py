from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .model_providers import (
    ModelInfo,
    effective_models_for_provider,
    infer_parameter_count,
    provider_config,
)
from .state import Member


@dataclass
class OllamaResult:
    member: Member
    payload: dict[str, Any]
    stdout: str
    stderr: str


class OllamaRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        member_name: str,
        phase: str,
        log_path: Path | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.member_name = member_name
        self.phase = phase
        self.log_path = log_path
        self.retryable = retryable

    def __str__(self) -> str:
        parts = [self.message]
        if self.log_path:
            parts.append(f"Log: {self.log_path}")
        return "\n".join(parts)


class OllamaProvider:
    name = "ollama"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = provider_config(config, self.name)

    def discover_models(self) -> list[ModelInfo]:
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        discovered: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if not name:
                continue
            metadata = dict(item)
            parameter_count = infer_parameter_count(str(name), metadata)
            if parameter_count is None:
                details = self._safe_show(str(name))
                if details:
                    metadata.update(details)
                    parameter_count = infer_parameter_count(str(name), metadata)
            discovered.append(
                ModelInfo(
                    provider=self.name,
                    model=str(name),
                    parameter_count_billion=parameter_count,
                    context_window=_context_window(metadata),
                    metadata=metadata,
                )
            )
        return discovered

    def validate_model(self, model: str) -> bool:
        return any(item.model == model for item in list_ollama_models(self.config))

    async def run(
        self,
        member: Member,
        prompt: str,
        schema_path: Path,
        phase: str,
        web_search: bool = False,
    ) -> OllamaResult:
        return await asyncio.to_thread(
            self._run_sync, member, prompt, schema_path, phase, web_search
        )

    def _run_sync(
        self,
        member: Member,
        prompt: str,
        schema_path: Path,
        phase: str,
        web_search: bool = False,
    ) -> OllamaResult:
        runtime_temp = resolve_project_path(self.config["runtime"]["temp_path"])
        runtime_logs = resolve_project_path(self.config["runtime"]["logs_path"])
        runtime_temp.mkdir(parents=True, exist_ok=True)
        runtime_logs.mkdir(parents=True, exist_ok=True)
        output_path = runtime_temp / f"{phase}-{member.name.lower()}-last.json"
        log_path = runtime_logs / f"{phase}-{member.name.lower()}.log"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        payload = {
            "model": member.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
        }
        options = {
            key: value
            for key, value in (self.provider_config.get("options") or {}).items()
            if value is not None
        }
        if options:
            payload["options"] = options
        try:
            response = self._request_json("POST", "/api/chat", payload)
        except Exception as exc:
            log_path.write_text(str(exc), encoding="utf-8")
            raise OllamaRunError(
                f"Ollama subagent {member.name} failed in {phase}: {exc}",
                member_name=member.name,
                phase=phase,
                log_path=log_path,
                retryable=True,
            ) from exc
        content = str((response.get("message") or {}).get("content") or "").strip()
        log_path.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
        try:
            parsed = _parse_json(content)
        except JSONDecodeError as exc:
            raise OllamaRunError(
                f"Ollama subagent {member.name} wrote invalid JSON in {phase}.",
                member_name=member.name,
                phase=phase,
                log_path=log_path,
                retryable=True,
            ) from exc
        output_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OllamaResult(member=member, payload=parsed, stdout=content, stderr="")

    def _safe_show(self, model: str) -> dict[str, Any]:
        try:
            return self._request_json("POST", "/api/show", {"model": model})
        except Exception:
            return {}

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        base_url = str(self.provider_config.get("base_url", "http://localhost:11434")).rstrip("/")
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
        timeout = float(self.provider_config.get("request_timeout_seconds", 300))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def list_ollama_models(config: dict[str, Any]) -> list[ModelInfo]:
    provider = OllamaProvider(config)
    if not provider.provider_config.get("enabled", False):
        return []
    try:
        discovered = provider.discover_models() if provider.provider_config.get("discover_models", True) else []
    except Exception:
        discovered = []
    return effective_models_for_provider("ollama", config, discovered)


def ollama_doctor(config: dict[str, Any]) -> str:
    provider = OllamaProvider(config)
    try:
        models = provider.discover_models()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {provider.provider_config.get('base_url')}: {exc}") from exc
    return f"Ollama ok: {len(models)} discovered model(s)."


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _context_window(metadata: dict[str, Any]) -> int | None:
    details = metadata.get("details")
    candidates = []
    if isinstance(details, dict):
        candidates.extend([details.get("context_window"), details.get("num_ctx")])
    candidates.extend([metadata.get("context_window"), metadata.get("num_ctx")])
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None
