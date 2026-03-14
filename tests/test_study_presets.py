"""Tests for ablation study preset definitions."""

from ollama_forge.study_presets import get_study_preset, list_study_presets


def test_list_study_presets_contains_expected_entries() -> None:
    presets = list_study_presets()
    keys = {preset.key for preset in presets}
    assert "quick" in keys
    assert "guardrail" in keys
    assert "robustness" in keys


def test_get_study_preset_returns_expected_preset() -> None:
    preset = get_study_preset("jailbreak")
    assert preset.name == "Jailbreak Analysis"
    assert any(strategy["name"] == "head_pruning" for strategy in preset.strategies)


def test_get_study_preset_rejects_unknown_key() -> None:
    try:
        get_study_preset("missing")
    except KeyError as exc:
        assert "Available" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown preset")
