"""Load YAML/JSON config files for repeatable abliterate run and train-run."""

from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load config from .yaml/.yml or .json. Returns a flat dict (e.g. model, name, num_instructions)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML required for YAML config. Install with: uv add pyyaml"
            ) from None
        data = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json"
        )
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object / YAML mapping")
    return data


def apply_config_to_args(
    args: object,
    config: dict,
    *,
    only_if_default: dict[str, object] | None = None,
) -> None:
    """
    Set attributes on args from config. Optionally only set when current value equals default.

    If only_if_default is provided, for each key in config we set attr only when
    getattr(args, key) == only_if_default.get(key). So CLI overrides config.
    If only_if_default is None, config overwrites all listed keys.
    """
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if only_if_default is not None:
            current = getattr(args, key, None)
            default = only_if_default.get(key)
            if current != default:
                continue
        setattr(args, key, value)
