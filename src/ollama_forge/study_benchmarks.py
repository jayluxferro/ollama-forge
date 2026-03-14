"""Curated benchmark catalog for study workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkPreset:
    key: str
    name: str
    kind: str
    path: str
    description: str


def _data_path(filename: str) -> str:
    return str((Path(__file__).resolve().parent / "security_eval" / "data" / filename).resolve())


_PRESETS = [
    BenchmarkPreset(
        key="sample_prompts",
        name="Sample Security Prompts",
        kind="security_eval",
        path=_data_path("sample_prompts.jsonl"),
        description="Small mixed prompt set for quick refusal/compliance checks.",
    ),
    BenchmarkPreset(
        key="indirect_injection",
        name="Indirect Prompt Injection",
        kind="security_eval",
        path=_data_path("sample_indirect.jsonl"),
        description="Prompt injection flavored sample set.",
    ),
    BenchmarkPreset(
        key="system_extraction",
        name="System Prompt Extraction",
        kind="security_eval",
        path=_data_path("system_prompt_extraction.jsonl"),
        description="Focused system prompt extraction prompt set.",
    ),
    BenchmarkPreset(
        key="wikitext_ppl",
        name="WikiText Perplexity",
        kind="dataset",
        path="wikitext:wikitext-2-raw-v1:test",
        description="General-language perplexity baseline on WikiText-2 test.",
    ),
]

BENCHMARK_PRESETS: dict[str, BenchmarkPreset] = {preset.key: preset for preset in _PRESETS}


def list_benchmark_presets(*, kind: str | None = None) -> list[BenchmarkPreset]:
    presets = list(_PRESETS)
    if kind:
        presets = [preset for preset in presets if preset.kind == kind]
    return presets


def get_benchmark_preset(key: str) -> BenchmarkPreset:
    if key not in BENCHMARK_PRESETS:
        available = ", ".join(sorted(BENCHMARK_PRESETS))
        raise KeyError(f"Unknown benchmark preset {key!r}. Available: {available}")
    return BENCHMARK_PRESETS[key]


def compare_benchmark_runs(primary: dict, secondary: dict) -> dict:
    primary_kpis = primary.get("kpis") or {}
    secondary_kpis = secondary.get("kpis") or {}
    metric_names = sorted(set(primary_kpis) | set(secondary_kpis))
    return {
        "primary_model": primary.get("model"),
        "secondary_model": secondary.get("model"),
        "metrics": {
            name: {"primary": primary_kpis.get(name), "secondary": secondary_kpis.get(name)}
            for name in metric_names
        },
    }
