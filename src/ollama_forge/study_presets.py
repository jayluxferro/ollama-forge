"""Reusable preset catalog for generic ablation studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StudyPreset:
    """A reusable ablation recipe."""

    name: str
    key: str
    description: str
    strategies: list[dict[str, Any]]
    metrics: list[str] = field(default_factory=lambda: ["perplexity"])
    max_samples: int = 100
    batch_size: int = 4
    max_length: int = 256
    tags: list[str] = field(default_factory=list)


_PRESETS = [
    StudyPreset(
        name="Quick Scan",
        key="quick",
        description="Fast sanity check over layers and FFNs for a first pass on any model.",
        strategies=[
            {"name": "layer_removal", "params": {}},
            {"name": "ffn_ablation", "params": {}},
        ],
        max_samples=25,
        batch_size=4,
        max_length=128,
        tags=["fast", "general"],
    ),
    StudyPreset(
        name="Full Sweep",
        key="full",
        description="Run all core strategies across layers, heads, FFNs, and embeddings.",
        strategies=[
            {"name": "layer_removal", "params": {}},
            {"name": "head_pruning", "params": {}},
            {"name": "ffn_ablation", "params": {}},
            {"name": "embedding_ablation", "params": {"chunk_size": 48}},
        ],
        max_samples=200,
        batch_size=4,
        max_length=256,
        tags=["thorough", "general"],
    ),
    StudyPreset(
        name="Attention Deep-Dive",
        key="attention",
        description="Focus exclusively on attention heads to find critical and redundant heads.",
        strategies=[{"name": "head_pruning", "params": {}}],
        max_samples=100,
        batch_size=4,
        max_length=256,
        tags=["attention", "heads"],
    ),
    StudyPreset(
        name="Layer Importance",
        key="layers",
        description="Remove layers and FFNs one at a time to map depth importance.",
        strategies=[
            {"name": "layer_removal", "params": {}},
            {"name": "ffn_ablation", "params": {}},
        ],
        max_samples=100,
        batch_size=4,
        max_length=256,
        tags=["layers", "depth"],
    ),
    StudyPreset(
        name="Knowledge Localization",
        key="knowledge",
        description="Target FFNs and embedding chunks to inspect knowledge localization.",
        strategies=[
            {"name": "ffn_ablation", "params": {}},
            {"name": "embedding_ablation", "params": {"chunk_size": 32}},
        ],
        max_samples=150,
        batch_size=4,
        max_length=256,
        tags=["knowledge", "ffn", "embeddings"],
    ),
    StudyPreset(
        name="Pruning Candidates",
        key="pruning",
        description="Find heads and FFNs that can be removed with minimal quality loss.",
        strategies=[
            {"name": "head_pruning", "params": {}},
            {"name": "ffn_ablation", "params": {}},
        ],
        max_samples=100,
        batch_size=4,
        max_length=256,
        tags=["pruning", "compression"],
    ),
    StudyPreset(
        name="Embedding Analysis",
        key="embeddings",
        description="Systematically ablate embedding chunks for representation analysis.",
        strategies=[
            {"name": "embedding_ablation", "params": {"chunk_size": 16}},
        ],
        max_samples=100,
        batch_size=4,
        max_length=256,
        tags=["embeddings", "representation"],
    ),
    StudyPreset(
        name="Jailbreak Analysis",
        key="jailbreak",
        description="Fine-grained preset for locating refusal-mediating components in instruct models.",
        strategies=[
            {"name": "head_pruning", "params": {}},
            {"name": "ffn_ablation", "params": {}},
            {"name": "embedding_ablation", "params": {"chunk_size": 16}},
        ],
        max_samples=400,
        batch_size=4,
        max_length=512,
        tags=["jailbreak", "refusal", "alignment"],
    ),
    StudyPreset(
        name="Guardrail Ablation",
        key="guardrail",
        description="Systematic guardrail localization across layers, heads, FFNs, and embeddings.",
        strategies=[
            {"name": "layer_removal", "params": {}},
            {"name": "head_pruning", "params": {}},
            {"name": "ffn_ablation", "params": {}},
            {"name": "embedding_ablation", "params": {"chunk_size": 24}},
        ],
        max_samples=300,
        batch_size=4,
        max_length=512,
        tags=["safety", "alignment", "guardrails"],
    ),
    StudyPreset(
        name="Robustness Test",
        key="robustness",
        description="Stress-test the model by ablating layers, heads, and FFNs with larger eval sets.",
        strategies=[
            {"name": "layer_removal", "params": {}},
            {"name": "head_pruning", "params": {}},
            {"name": "ffn_ablation", "params": {}},
        ],
        max_samples=500,
        batch_size=8,
        max_length=512,
        tags=["robustness", "thorough"],
    ),
]

STUDY_PRESETS: dict[str, StudyPreset] = {preset.key: preset for preset in _PRESETS}


def get_study_preset(key: str) -> StudyPreset:
    if key not in STUDY_PRESETS:
        available = ", ".join(sorted(STUDY_PRESETS))
        raise KeyError(f"Unknown preset {key!r}. Available: {available}")
    return STUDY_PRESETS[key]


def list_study_presets() -> list[StudyPreset]:
    return list(_PRESETS)
