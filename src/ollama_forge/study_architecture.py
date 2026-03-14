"""Architecture profile helpers for study and abliterate workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ArchitectureProfile:
    model_name: str
    model_type: str
    arch_class: str
    reasoning_class: str
    num_layers: int
    hidden_size: int
    num_heads: int
    recommended_profile: str
    notes: list[str]


def detect_architecture_profile(handle: Any, model_name: str | None = None) -> ArchitectureProfile:
    config = getattr(handle.model, "config", None)
    model_type = getattr(config, "model_type", "") if config is not None else ""
    name = (model_name or getattr(config, "_name_or_path", "") or getattr(handle, "architecture", "")).lower()
    is_moe = any(token in name for token in ("moe", "mixtral", "a3b", "a22b", "deepseek-v3", "gpt-oss"))
    is_reasoning = any(token in name for token in ("r1", "qwq", "think", "o1", "o3"))
    notes: list[str] = []
    if is_moe:
        notes.append("MoE-like naming detected; prefer conservative, layer-aware interventions.")
    if is_reasoning:
        notes.append("Reasoning-oriented model detected; preserve residual pathways where possible.")
    recommended = "balanced"
    if is_moe or is_reasoning:
        recommended = "safe"
    if is_moe and is_reasoning:
        recommended = "safe"
    return ArchitectureProfile(
        model_name=model_name or getattr(config, "_name_or_path", "") or getattr(handle, "architecture", "unknown"),
        model_type=model_type,
        arch_class="moe" if is_moe else "dense",
        reasoning_class="reasoning" if is_reasoning else "standard",
        num_layers=int(getattr(handle, "num_layers", 0)),
        hidden_size=int(getattr(handle, "hidden_size", 0)),
        num_heads=int(getattr(handle, "num_heads", 0)),
        recommended_profile=recommended,
        notes=notes,
    )
