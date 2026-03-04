"""Abliteration profiles for easier UX."""

from __future__ import annotations

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
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


def get_profile(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    key = name.strip().lower()
    return dict(_PROFILES.get(key, {}))
