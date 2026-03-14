"""Tests for study model presets and recommendations."""

from ollama_forge.study_model_presets import (
    detect_hardware_tier,
    list_model_presets,
    recommended_model_presets,
)


def test_list_model_presets_filters_by_tier() -> None:
    presets = list_model_presets(tier="medium")
    assert presets
    assert all(preset.tier == "medium" for preset in presets)


def test_recommended_model_presets_respect_limit() -> None:
    presets = recommended_model_presets(tier="small", limit=3)
    assert len(presets) <= 3


def test_detect_hardware_tier_returns_known_tier() -> None:
    tier, info = detect_hardware_tier()
    assert tier in {"tiny", "small", "medium", "large", "frontier"}
    assert "platform" in info
