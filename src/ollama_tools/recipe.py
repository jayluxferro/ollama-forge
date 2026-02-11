"""Model recipe (YAML/JSON) loading for config-driven builds."""

from __future__ import annotations

import json
from pathlib import Path


def load_recipe(path: str | Path) -> dict:
    """Load recipe from .yaml/.yml/.json: name, base or gguf, optional params."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".json",):
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML required for YAML recipes. Install with: uv add pyyaml"
            ) from None
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported recipe format: {suffix}. Use .yaml, .yml, or .json")
    if not isinstance(data, dict):
        raise ValueError("Recipe must be a JSON object / YAML mapping")
    if "name" not in data:
        raise ValueError("Recipe must include 'name'")
    sources = [k for k in ("base", "gguf", "hf_repo") if data.get(k)]
    if len(sources) == 0:
        raise ValueError(
            "Recipe must include one of: 'base' (create-from-base), 'gguf' (path), "
            "or 'hf_repo' (Hugging Face repo id)"
        )
    if len(sources) > 1:
        raise ValueError("Recipe must include only one of: 'base', 'gguf', 'hf_repo'")
    return data
