from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from small_council import cli, state
from small_council.codex_runner import CodexProvider, list_codex_models
from small_council.config import load_config, save_config, set_config_value
from small_council.model_providers import (
    ModelInfo,
    effective_models_for_provider,
    infer_parameter_count,
    provider_config,
    provider_options,
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
    def test_default_council_provider_options(self) -> None:
        config = load_config()

        ollama_options = provider_config(config, "ollama")["options"]
        codex_options = provider_config(config, "codex")["options"]

        self.assertEqual(0.8, ollama_options["temperature"])
        self.assertIsNone(ollama_options["seed"])
        self.assertNotIn("num_ctx", ollama_options)
        self.assertEqual("medium", codex_options["reasoning_effort"])

    def test_missing_provider_options_fall_back_to_built_in_defaults(self) -> None:
        config = {"model_providers": {"ollama": {"enabled": True}, "codex": {"enabled": True}}}

        self.assertEqual(
            {"temperature": 0.8, "seed": None},
            provider_options(config, "ollama"),
        )
        self.assertEqual(
            {"reasoning_effort": "medium"},
            provider_options(config, "codex"),
        )

    def test_benchmark_options_override_provider_and_member_options(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["options"] = {"temperature": 0.9, "seed": 7}
        config["model_providers"]["codex"]["options"] = {"reasoning_effort": "high"}
        config["model_overrides"] = {
            "Aurelia": {
                "provider": "ollama",
                "model": "qwen3:8b",
                "options": {"temperature": 1.0, "seed": 99},
            },
            "Bram": {
                "provider": "codex",
                "model": "gpt-5.5",
                "options": {"reasoning_effort": "xhigh"},
            },
        }

        with patch.dict("os.environ", {"SMALL_COUNCIL_BENCHMARK": "1"}):
            ollama_options = provider_options(config, "ollama", "Aurelia")
            codex_options = provider_options(config, "codex", "Bram")

        self.assertEqual({"temperature": 0.3, "seed": 42}, ollama_options)
        self.assertEqual({"reasoning_effort": "low"}, codex_options)

    def test_member_options_override_provider_options_outside_benchmark_mode(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["options"] = {"temperature": 0.9, "seed": 7}
        config["model_overrides"] = {
            "Aurelia": {
                "provider": "ollama",
                "model": "qwen3:8b",
                "options": {"temperature": 1.0, "seed": 99},
            }
        }

        with patch.dict("os.environ", {}, clear=True):
            options = provider_options(config, "ollama", "Aurelia")

        self.assertEqual({"temperature": 1.0, "seed": 99}, options)

    def test_ollama_options_accept_num_ctx(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["options"] = {
            "temperature": 0.9,
            "seed": 7,
            "num_ctx": 16384,
        }

        with patch.dict("os.environ", {}, clear=True):
            options = provider_options(config, "ollama")

        self.assertEqual({"temperature": 0.9, "seed": 7, "num_ctx": 16384}, options)

    def test_ollama_options_reject_invalid_num_ctx(self) -> None:
        config = _config()
        config["model_providers"]["ollama"]["options"] = {"num_ctx": "large"}

        with self.assertRaisesRegex(ValueError, "num_ctx"):
            provider_options(config, "ollama")

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

    def test_unavailable_member_model_is_repaired(self) -> None:
        config = _config()
        members = [
            Member(
                "Aurelia",
                "missing",
                "practical",
                True,
                "created",
                total_proposals=3,
                total_wins=2,
                total_votes_cast=5,
                tie_break_victories=1,
                provider="ollama",
            ),
            Member("Bram", "gpt-5.5", "skeptical", False, "now", provider="codex"),
        ]

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[
                ModelInfo("codex", "gpt-5.5"),
                ModelInfo("ollama", "qwen3:8b"),
            ],
        ):
            updated = state._repair_unavailable_member_models(config, members)

        repaired = updated[0]
        self.assertEqual("ollama", repaired.provider)
        self.assertEqual("qwen3:8b", repaired.model)
        self.assertEqual("Aurelia", repaired.name)
        self.assertEqual("practical", repaired.personality)
        self.assertTrue(repaired.is_president)
        self.assertEqual("created", repaired.created_at)
        self.assertEqual(3, repaired.total_proposals)
        self.assertEqual(2, repaired.total_wins)
        self.assertEqual(5, repaired.total_votes_cast)
        self.assertEqual(1, repaired.tie_break_victories)
        self.assertIs(updated[1], members[1])

    def test_valid_member_models_are_not_changed(self) -> None:
        config = _config()
        members = [
            Member("Aurelia", "gpt-5.5", "practical", False, "now", provider="codex"),
            Member("Bram", "qwen3:8b", "skeptical", True, "now", provider="ollama"),
        ]

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[
                ModelInfo("codex", "gpt-5.5"),
                ModelInfo("ollama", "qwen3:8b"),
            ],
        ):
            updated = state._repair_unavailable_member_models(config, members)

        self.assertEqual(members, updated)

    def test_multiple_unavailable_models_prefer_unique_replacements(self) -> None:
        config = _config()
        members = [
            Member("Aurelia", "missing-a", "practical", False, "now", provider="codex"),
            Member("Bram", "missing-b", "skeptical", True, "now", provider="ollama"),
            Member("Cato", "gpt-5.5", "direct", False, "now", provider="codex"),
        ]

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[
                ModelInfo("codex", "gpt-5.5"),
                ModelInfo("ollama", "qwen3:8b"),
                ModelInfo("ollama", "llama3.2:3b"),
            ],
        ):
            updated = state._repair_unavailable_member_models(config, members)

        repaired_pairs = {
            (updated[0].provider, updated[0].model),
            (updated[1].provider, updated[1].model),
        }
        self.assertEqual(
            {("ollama", "qwen3:8b"), ("ollama", "llama3.2:3b")},
            repaired_pairs,
        )
        self.assertEqual(("codex", "gpt-5.5"), (updated[2].provider, updated[2].model))

    def test_repair_allows_duplicates_when_unique_models_are_exhausted(self) -> None:
        config = _config()
        members = [
            Member("Aurelia", "missing-a", "practical", False, "now", provider="codex"),
            Member("Bram", "missing-b", "skeptical", True, "now", provider="ollama"),
        ]

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[ModelInfo("ollama", "qwen3:8b")],
        ):
            updated = state._repair_unavailable_member_models(config, members)

        self.assertEqual(
            [("ollama", "qwen3:8b"), ("ollama", "qwen3:8b")],
            [(member.provider, member.model) for member in updated],
        )

    def test_repair_rejects_duplicates_when_disabled(self) -> None:
        config = _config()
        config["model_assignment"]["allow_duplicates_when_needed"] = False
        members = [
            Member("Aurelia", "missing-a", "practical", False, "now", provider="codex"),
            Member("Bram", "missing-b", "skeptical", True, "now", provider="ollama"),
        ]

        with (
            patch.object(
                state,
                "effective_model_pool",
                return_value=[ModelInfo("ollama", "qwen3:8b")],
            ),
            self.assertRaises(ValueError),
        ):
            state._repair_unavailable_member_models(config, members)

    def test_legacy_member_is_repaired_when_codex_model_is_unavailable(self) -> None:
        config = _config()
        members = [
            Member.from_dict(
                {
                    "name": "Aurelia",
                    "model": "missing-codex",
                    "personality": "practical",
                    "is_president": False,
                    "created_at": "now",
                }
            )
        ]

        with patch.object(
            state,
            "effective_model_pool",
            return_value=[ModelInfo("ollama", "qwen3:8b")],
        ):
            updated = state._repair_unavailable_member_models(config, members)

        self.assertEqual("ollama", updated[0].provider)
        self.assertEqual("qwen3:8b", updated[0].model)


class ConfigSetTests(unittest.TestCase):
    def test_load_and_save_config_respect_environment_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "council.yaml"
            path.write_text("secretary:\n  provider: ollama\n", encoding="utf-8")
            with patch.dict("os.environ", {"SMALL_COUNCIL_CONFIG": str(path)}):
                config = load_config()
                config["secretary"]["model"] = "qwen3:8b"
                save_config(config)

            saved = path.read_text(encoding="utf-8")

        self.assertIn("provider: ollama", saved)
        self.assertIn("model: qwen3:8b", saved)

    def test_explicit_config_path_overrides_environment_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "env.yaml"
            explicit_path = Path(tmp) / "explicit.yaml"
            env_path.write_text("secretary:\n  provider: env\n", encoding="utf-8")
            explicit_path.write_text("secretary:\n  provider: explicit\n", encoding="utf-8")
            with patch.dict("os.environ", {"SMALL_COUNCIL_CONFIG": str(env_path)}):
                config = load_config(explicit_path)
                save_config({"secretary": {"provider": "saved"}}, explicit_path)

            env_saved = env_path.read_text(encoding="utf-8")
            explicit_saved = explicit_path.read_text(encoding="utf-8")

        self.assertEqual("explicit", config["secretary"]["provider"])
        self.assertIn("provider: env", env_saved)
        self.assertIn("provider: saved", explicit_saved)

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
