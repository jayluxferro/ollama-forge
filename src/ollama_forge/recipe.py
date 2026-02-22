"""Model recipe (YAML/JSON) loading for config-driven builds."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# {{ varname }} or ${varname}; varname = [a-zA-Z_][a-zA-Z0-9_]*
_VAR_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}|\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _substitute_string(s: str, context: dict[str, str]) -> str:
    """Replace {{ var }} and ${var} with context[var]; leave unchanged if var missing."""
    def repl(m: re.Match) -> str:
        key = m.group(1) or m.group(2)
        return context.get(key, m.group(0))
    return _VAR_PATTERN.sub(repl, s)


def apply_recipe_variables(recipe: dict) -> dict:
    """
    In-place substitute variables in string values: {{ varname }} and ${varname}.
    Context: os.environ then recipe.variables (recipe variables override env).
    Returns the same dict (modified in place).
    """
    context = {k: str(v) for k, v in os.environ.items()}
    if "variables" in recipe and isinstance(recipe["variables"], dict):
        for k, v in recipe["variables"].items():
            context[k] = str(v)

    def do(obj: object) -> None:
        if isinstance(obj, str):
            return  # will replace at the level that holds the string
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str):
                    obj[k] = _substitute_string(v, context)
                else:
                    do(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = _substitute_string(v, context)
                else:
                    do(v)

    do(recipe)
    return recipe


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
            raise ImportError("PyYAML required for YAML recipes. Install with: uv add pyyaml") from None
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported recipe format: {suffix}. Use .yaml, .yml, or .json")
    if not isinstance(data, dict):
        raise ValueError("Recipe must be a JSON object / YAML mapping")
    apply_recipe_variables(data)
    if "name" not in data:
        raise ValueError("Recipe must include 'name'")
    sources = [k for k in ("base", "gguf", "hf_repo") if data.get(k)]
    if len(sources) == 0:
        raise ValueError(
            "Recipe must include one of: 'base' (create-from-base), 'gguf' (path), or 'hf_repo' (Hugging Face repo id)"
        )
    if len(sources) > 1:
        raise ValueError("Recipe must include only one of: 'base', 'gguf', 'hf_repo'")
    return data
