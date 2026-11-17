"""Task-based model router.

Maps task kinds (code, math, vision, chat, search) to model IDs.
Config is YAML-driven or auto-inferred from the available model names.

Why YAML: different deployments have different models. Hard-coding defaults
would route tasks to models that aren't loaded. Auto-inference from model names
gives a sensible first cut, and the user can tune it with a config file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

TaskKind = Literal["code", "math", "vision", "chat", "search", "default"]

_TASK_KINDS: tuple[TaskKind, ...] = ("code", "math", "vision", "chat", "search", "default")

# Name patterns that signal a model's strength for a task.
_HINTS: dict[str, list[str]] = {
    "code": ["code", "coder", "starcoder", "deepseek-coder", "qwen.*coder", "codellama", "phind"],
    "math": ["math", "qwen.*math", "deepseek.*math", "numina"],
    "vision": ["llava", "vision", "vlm", "bakllava", "minicpm-v", "moondream", "cogvlm", "pixtral"],
    "chat": ["instruct", "chat", "hermes", "dolphin", "openhermes"],
}


def _score(model_id: str, task: str) -> int:
    lower = model_id.lower()
    patterns = _HINTS.get(task, [])
    return sum(1 for p in patterns if re.search(p, lower))


def auto_config(available_models: list[str]) -> dict[str, str]:
    """Pick the best model for each task from the available list by name heuristics."""
    cfg: dict[str, str] = {}
    if not available_models:
        return cfg
    fallback = available_models[0]
    for task in ("code", "math", "vision", "chat", "search"):
        best = max(available_models, key=lambda m: _score(m, task))
        cfg[task] = best if _score(best, task) > 0 else fallback
    cfg["default"] = fallback
    return cfg


class ModelRouter:
    """
    Route a task kind to a model ID.

    Priority: explicit config key > 'default' key > first configured model.
    """

    def __init__(self, config: dict[str, str] | None = None, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            import yaml
            self._cfg: dict[str, str] = yaml.safe_load(Path(config_path).read_text()) or {}
        else:
            self._cfg = dict(config or {})

    def route(self, task: str) -> str | None:
        """Return the model ID for *task*, falling back to 'default' then None."""
        return self._cfg.get(task) or self._cfg.get("default")

    def all_routes(self) -> dict[str, str]:
        return dict(self._cfg)

    def set(self, task: str, model_id: str) -> None:
        self._cfg[task] = model_id

    def save(self, path: str | Path) -> None:
        import yaml
        Path(path).write_text(yaml.dump(self._cfg, default_flow_style=False))

    @classmethod
    def from_available(cls, available_models: list[str]) -> "ModelRouter":
        """Build a router from the auto-inferred config for the given model list."""
        return cls(config=auto_config(available_models))
