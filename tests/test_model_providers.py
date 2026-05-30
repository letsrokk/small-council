from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from small_council import cli, state
from small_council.codex_runner import CodexProvider, list_codex_models
from small_council.config import set_config_value
from small_council.model_providers import (
    ModelInfo,
    effective_models_for_provider,
    infer_parameter_count,
)
from small_council.ollama_runner import OllamaProvider, list_ollama_models
from small_council.state import Member


def _config() -> dict:
    return {
        "council": {"member_names": ["Aurelia", "Bram"], "min_members": 1},
        "personality_pool": ["practical", "skeptical"],
        "storage": {
            "council_state_path": "./storage/council-state.json",
            "leaderboard_path": "./storage/leaderboard.json",
        },
        "runtime": {"temp_path": "./runtime/temp", "logs_path": "./runtime/logs"},
        "model_providers": {
            "codex": {
                "enabled": True,
                "discover_models": False,
                "static_models": ["gpt-5.5"],
                "enabled_models": [],
                "disabled_models": [],
                "max_parameters": None,
                "allow_unknown_size_models": True,
            },
            "ollama": {
                "enabled": True,
                "discover_models": False,
                "static_models": ["qwen3:8b", "qwen3:14b", "unknown-local"],
                "enabled_models": [],
                "disabled_models": [],
                "max_parameters": "12b",
                "allow_unknown_size_models": False,
            },
        },
        "model_assignment": {
            "prefer_unique_models": True,
            "allow_duplicates_when_needed": True,
        },
        "model_overrides": {},
    }


class ProviderConfigTests(unittest.TestCase):
    def test_effective_models_apply_size_enabled_and_disabled_filters(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["enabled_models"] = ["qwen3:8b", "qwen3:14b"]
        config["model_providers"]["ollama"]["disabled_models"] = ["qwen3:14b"]
        discovered = [
            ModelInfo("ollama", "qwen3:8b", 8),
            ModelInfo("ollama", "qwen3:14b", 14),
            ModelInfo("ollama", "mystery", None),
        ]

        models = effective_models_for_provider("ollama", config, discovered)

        self.assertEqual(["qwen3:8b"], [item.model for item in models])

    def test_unknown_size_can_be_allowed(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["allow_unknown_size_models"] = True

        models = effective_models_for_provider(
            "ollama", config, [ModelInfo("ollama", "mystery", None)]
        )

        self.assertIn("mystery", [item.model for item in models])

    def test_parameter_size_is_inferred_from_names(self) -> None:
        self.assertEqual(8, infer_parameter_count("qwen3:8b"))
        self.assertEqual(12, infer_parameter_count("mistral-nemo:12b"))
        self.assertEqual(70, infer_parameter_count("llama3.3:70b"))


class DiscoveryTests(unittest.TestCase):
    def test_codex_discovery_falls_back_to_static_models(self) -> None:
        config = _config()
        config["model_providers"]["codex"]["discover_models"] = True

        with patch.object(CodexProvider, "discover_models", side_effect=RuntimeError("nope")):
            models = list_codex_models(config)

        self.assertEqual(["gpt-5.5"], [item.model for item in models])

    def test_ollama_discovery_reads_tags_and_model_details(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["discover_models"] = True
        tags = {"models": [{"name": "qwen3:8b"}, {"name": "llama3.3:70b"}]}
        show = {"details": {"parameter_size": "8B"}}

        with patch.object(OllamaProvider, "_request_json", side_effect=[tags, show, {}]):
            models = list_ollama_models(config)

        self.assertEqual(["qwen3:8b"], [item.model for item in models])


class StateAssignmentTests(unittest.TestCase):
    def test_legacy_member_without_provider_loads_as_codex(self) -> None:
        member = Member.from_dict(
            {
                "name": "Aurelia",
                "model": "gpt-5.4-mini",
                "personality": "practical",
                "is_president": False,
                "created_at": "now",
            }
        )

        self.assertEqual("codex", member.provider)

    def test_new_members_receive_provider_and_model_pairs(self) -> None:
        config = _config()

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[
                ModelInfo("codex", "gpt-5.5"),
                ModelInfo("ollama", "qwen3:8b"),
            ],
        ):
            members = state._create_members(config)

        self.assertEqual(
            {("codex", "gpt-5.5"), ("ollama", "qwen3:8b")},
            {(member.provider, member.model) for member in members},
        )

    def test_model_override_accepts_provider_model_pair(self) -> None:
        config = _config()
        config["model_overrides"] = {"Bram": {"provider": "ollama", "model": "qwen3:8b"}}
        members = [
            Member("Bram", "gpt-5.5", "skeptical", False, "now"),
        ]

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[ModelInfo("ollama", "qwen3:8b")],
        ):
            updated = state._apply_model_overrides(config, members)

        self.assertEqual("ollama", updated[0].provider)
        self.assertEqual("qwen3:8b", updated[0].model)


class ConfigSetTests(unittest.TestCase):
    def test_set_updates_secretary_provider(self) -> None:
        config = {"secretary": {"provider": "codex", "model": "gpt-5.5"}}

        updated = set_config_value(config, "secretary.provider", "ollama")

        self.assertEqual("ollama", updated["secretary"]["provider"])

    def test_cli_set_persists_config(self) -> None:
        config = {
            "storage": {},
            "runtime": {},
            "secretary": {"provider": "codex", "model": "gpt-5.5"},
        }

        with (
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "_ensure_dirs", return_value=None),
            patch.object(cli, "save_config") as save,
            patch.object(cli.sys, "stdout", io.StringIO()),
        ):
            exit_code = cli.main(["--set", "secretary.provider=ollama"])

        self.assertEqual(0, exit_code)
        saved_config = save.call_args.args[0]
        self.assertEqual("ollama", saved_config["secretary"]["provider"])

    def test_secretary_validation_requires_effective_model(self) -> None:
        config = _config()
        secretary = cli.SecretaryConfig(provider="ollama", model="missing")

        with self.assertRaises(ValueError):
            cli._validate_secretary_config(config, secretary)


if __name__ == "__main__":
    unittest.main()
