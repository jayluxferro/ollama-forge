"""Tests for informed artifact inspection helpers."""

from __future__ import annotations

from ollama_forge.abliterate_informed_reports import compare_informed_artifacts, informed_artifact_markdown


def test_compare_informed_artifacts_tracks_profile_and_strength() -> None:
    payload = compare_informed_artifacts(
        {"run_status": "success", "recommendation": {"profile": "balanced", "strength": 1.0}},
        {"run_status": "failed", "recommendation": {"profile": "aggressive", "strength": 1.3}},
    )
    assert payload["profile"]["a"] == "balanced"
    assert payload["strength"]["b"] == 1.3


def test_informed_artifact_markdown_includes_status() -> None:
    text = informed_artifact_markdown(
        {"run_status": "success", "recommendation": {"profile": "balanced", "strength": 1.0}},
    )
    assert "Run status" in text
