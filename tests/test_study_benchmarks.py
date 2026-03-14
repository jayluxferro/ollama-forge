"""Tests for study benchmark catalog."""

from ollama_forge.study_benchmarks import get_benchmark_preset, list_benchmark_presets


def test_list_benchmark_presets_returns_entries() -> None:
    presets = list_benchmark_presets()
    assert presets
    assert any(preset.kind == "security_eval" for preset in presets)


def test_get_benchmark_preset_returns_expected_item() -> None:
    preset = get_benchmark_preset("sample_prompts")
    assert preset.kind == "security_eval"
    assert preset.path.endswith(".jsonl")
