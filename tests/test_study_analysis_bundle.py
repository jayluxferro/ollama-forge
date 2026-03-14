"""Tests for study analysis bundle helpers."""

from ollama_forge.study_analysis_bundle import build_analysis_bundle, update_analysis_bundle


def test_build_analysis_bundle_contains_modules_and_results() -> None:
    bundle = build_analysis_bundle(
        config_path="study.yaml",
        modules=["activation_probe", "logit_lens"],
        results={"activation_probe": {"layers": []}},
    )
    assert bundle["config_path"] == "study.yaml"
    assert bundle["modules"] == ["activation_probe", "logit_lens"]


def test_update_analysis_bundle_merges_results_and_metadata() -> None:
    bundle = build_analysis_bundle(config_path="study.yaml", modules=["a"], results={"a": {"ok": True}})
    updated = update_analysis_bundle(bundle, extra_results={"b": {"ok": False}}, metadata={"run": "x"})
    assert "b" in updated["results"]
    assert updated["metadata"]["run"] == "x"
