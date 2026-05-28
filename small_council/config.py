from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "council.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the project-local YAML config.

    The parser intentionally supports the small YAML subset used by this
    project so the CLI has no package dependency just to boot.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return _parse_simple_yaml(path.read_text(encoding="utf-8"))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    lines = text.splitlines()
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Invalid YAML list item: {raw_line}")
            parent.append(_parse_scalar(line[2:]))
            continue

        if ":" not in line:
            raise ValueError(f"Invalid YAML line: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_is_list = _next_content_is_list(lines, raw_line)
        container: list[Any] | dict[str, Any] = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def _next_content_is_list(lines: list[str], current_line: str) -> bool:
    index = lines.index(current_line)
    current_indent = len(current_line) - len(current_line.lstrip(" "))
    for line in lines[index + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        return indent > current_indent and line.strip().startswith("- ")
    return False


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
