from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    parameter_count_billion: float | None = None
    context_window: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderReport:
    provider: str
    enabled: bool
    health: str
    discovered_models: list[ModelInfo]
    static_models: list[ModelInfo]
    effective_models: list[ModelInfo]
    errors: list[str] = field(default_factory=list)


class ModelProvider(Protocol):
    name: str

    def discover_models(self) -> list[ModelInfo]:
        ...

    def validate_model(self, model: str) -> bool:
        ...

    async def run(
        self,
        member: Any,
        prompt: str,
        schema_path: Path,
        phase: str,
        web_search: bool = False,
    ) -> Any:
        ...


DEFAULT_CODEX_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"]


def normalize_provider_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    providers = config.get("model_providers")
    if not isinstance(providers, dict):
        providers = {}

    normalized: dict[str, dict[str, Any]] = {
        "codex": _provider_defaults("codex"),
        "ollama": _provider_defaults("ollama"),
    }
    for name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            continue
        base = normalized.get(str(name), _provider_defaults(str(name)))
        base.update(provider_config)
        normalized[str(name)] = base

    if not normalized["codex"].get("static_models"):
        normalized["codex"]["static_models"] = DEFAULT_CODEX_MODELS[:]

    return normalized


def _provider_defaults(name: str) -> dict[str, Any]:
    defaults = {
        "enabled": name == "codex",
        "discover_models": True,
        "enabled_models": [],
        "disabled_models": [],
        "static_models": [],
        "max_parameters": None,
        "allow_unknown_size_models": True,
    }
    if name == "ollama":
        defaults.update(
            {
                "enabled": False,
                "base_url": "http://localhost:11434",
                "request_timeout_seconds": 300,
                "allow_unknown_size_models": False,
                "options": {"temperature": None, "num_ctx": None},
            }
        )
    return defaults


def provider_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    return normalize_provider_config(config).get(provider, _provider_defaults(provider))


def static_model_infos(provider: str, config: dict[str, Any]) -> list[ModelInfo]:
    models = provider_config(config, provider).get("static_models") or []
    return [
        ModelInfo(
            provider=provider,
            model=str(model),
            parameter_count_billion=infer_parameter_count(str(model)),
            metadata={"source": "static"},
        )
        for model in models
    ]


def effective_models_for_provider(
    provider: str,
    config: dict[str, Any],
    discovered: list[ModelInfo],
) -> list[ModelInfo]:
    pconfig = provider_config(config, provider)
    if not pconfig.get("enabled", False):
        return []
    combined = _merge_models([*discovered, *static_model_infos(provider, config)])
    max_parameters = parse_parameter_limit(pconfig.get("max_parameters"))
    if max_parameters is not None:
        allow_unknown = bool(pconfig.get("allow_unknown_size_models", True))
        combined = [
            model
            for model in combined
            if _passes_parameter_filter(model, max_parameters, allow_unknown)
        ]
    enabled_models = {str(item) for item in pconfig.get("enabled_models") or []}
    if enabled_models:
        combined = [model for model in combined if model.model in enabled_models]
    disabled_models = {str(item) for item in pconfig.get("disabled_models") or []}
    if disabled_models:
        combined = [model for model in combined if model.model not in disabled_models]
    return combined


def effective_model_pool(config: dict[str, Any]) -> list[ModelInfo]:
    pool: list[ModelInfo] = []
    for name in normalize_provider_config(config):
        provider = create_provider(name, config)
        if provider is None:
            continue
        discovered = _safe_discover(provider, config)
        pool.extend(effective_models_for_provider(name, config, discovered))
    return pool


def provider_report(config: dict[str, Any]) -> list[ProviderReport]:
    reports: list[ProviderReport] = []
    for name, pconfig in normalize_provider_config(config).items():
        provider = create_provider(name, config)
        discovered: list[ModelInfo] = []
        errors: list[str] = []
        health = "disabled"
        if pconfig.get("enabled", False):
            if provider is None:
                health = "unavailable"
                errors.append(f"No provider implementation for {name}.")
            else:
                try:
                    discovered = provider.discover_models()
                    health = "ok"
                except Exception as exc:
                    health = "degraded"
                    errors.append(str(exc))
        static = static_model_infos(name, config)
        effective = effective_models_for_provider(name, config, discovered)
        reports.append(
            ProviderReport(
                provider=name,
                enabled=bool(pconfig.get("enabled", False)),
                health=health,
                discovered_models=discovered,
                static_models=static,
                effective_models=effective,
                errors=errors,
            )
        )
    return reports


def create_provider(name: str, config: dict[str, Any]) -> ModelProvider | None:
    if name == "codex":
        from .codex_runner import CodexProvider

        return CodexProvider(config)
    if name == "ollama":
        from .ollama_runner import OllamaProvider

        return OllamaProvider(config)
    return None


def validate_provider_model(config: dict[str, Any], provider: str, model: str) -> bool:
    return any(
        item.provider == provider and item.model == model for item in effective_model_pool(config)
    )


def infer_parameter_count(model: str, metadata: dict[str, Any] | None = None) -> float | None:
    metadata = metadata or {}
    for key in (
        "parameter_count_billion",
        "parameters_billion",
        "parameter_size_billion",
    ):
        value = metadata.get(key)
        parsed = _parse_parameter_value(value)
        if parsed is not None:
            return parsed
    details = metadata.get("details")
    if isinstance(details, dict):
        for key in ("parameter_size", "parameters", "parameter_count"):
            parsed = _parse_parameter_value(details.get(key))
            if parsed is not None:
                return parsed
    for key in ("parameter_size", "parameters", "parameter_count"):
        parsed = _parse_parameter_value(metadata.get(key))
        if parsed is not None:
            return parsed
    return _parse_parameter_value(model)


def parse_parameter_limit(value: Any) -> float | None:
    return _parse_parameter_value(value)


def _parse_parameter_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(b|bn|billion)\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(m|mn|million)\b", text)
    if match:
        return float(match.group(1)) / 1000
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    return None


def _passes_parameter_filter(
    model: ModelInfo, max_parameters: float, allow_unknown: bool
) -> bool:
    size = model.parameter_count_billion
    if size is None:
        return allow_unknown
    return size <= max_parameters


def _merge_models(models: list[ModelInfo]) -> list[ModelInfo]:
    merged: dict[tuple[str, str], ModelInfo] = {}
    for model in models:
        key = (model.provider, model.model)
        existing = merged.get(key)
        if existing is None:
            merged[key] = model
            continue
        if existing.parameter_count_billion is None and model.parameter_count_billion is not None:
            merged[key] = model
    return list(merged.values())


def _safe_discover(provider: ModelProvider, config: dict[str, Any]) -> list[ModelInfo]:
    pconfig = provider_config(config, provider.name)
    if not pconfig.get("discover_models", True):
        return []
    try:
        return provider.discover_models()
    except Exception:
        return []


def codex_catalog_models(config: dict[str, Any]) -> list[ModelInfo]:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("The `codex` executable was not found on PATH.")
    from .codex_runner import _codex_env
    from .config import ROOT

    result = subprocess.run(
        [codex, "debug", "models"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=_codex_env(config),
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "Codex model discovery failed.")
    import json

    output = result.stdout.strip()
    if not output.startswith("{"):
        json_start = output.find("{")
        if json_start < 0:
            raise RuntimeError("Codex model discovery did not return JSON.")
        output = output[json_start:]
    payload = json.loads(output)
    models = payload.get("models", [])
    discovered = []
    for item in models:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug") or item.get("id") or item.get("model")
        if not slug:
            continue
        discovered.append(
            ModelInfo(
                provider="codex",
                model=str(slug),
                parameter_count_billion=infer_parameter_count(str(slug), item),
                context_window=item.get("context_window") if isinstance(item.get("context_window"), int) else None,
                metadata=item,
            )
        )
    return discovered
