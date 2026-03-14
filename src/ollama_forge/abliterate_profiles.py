"""Abliteration profiles for easier UX."""

from __future__ import annotations

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
        "description": "Lower-strength defaults that favor preserving capability over maximum refusal removal.",
        "num_instructions": 64,
        "agg": "last_non_special",
        "strength": 0.6,
        "atten_strength": 0.6,
        "mlp_strength": 0.5,
        "norm_preserving": True,
        "per_layer_directions": False,
        "output_only": False,
    },
    "balanced": {
        "description": "General-purpose defaults for most models and iterative experimentation.",
        "num_instructions": 128,
        "agg": "last",
        "strength": 1.0,
        "atten_strength": 1.0,
        "mlp_strength": 0.9,
        "norm_preserving": True,
        "per_layer_directions": False,
        "output_only": False,
    },
    "aggressive": {
        "description": "Highest-strength defaults for stronger refusal removal with more capability risk.",
        "num_instructions": 256,
        "agg": "mean",
        "strength": 1.3,
        "atten_strength": 1.3,
        "mlp_strength": 1.2,
        "norm_preserving": False,
        "per_layer_directions": True,
        "output_only": True,
    },
}


def list_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES.keys())


def get_profiles() -> dict[str, dict[str, Any]]:
    return {name: dict(values) for name, values in _PROFILES.items()}


def get_profile(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    key = name.strip().lower()
    return dict(_PROFILES.get(key, {}))
