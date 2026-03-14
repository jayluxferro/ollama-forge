"""Tests for ablation study config loading."""

from __future__ import annotations

from ollama_forge.study_config import StudyConfig


def test_study_config_expands_preset_defaults() -> None:
    cfg = StudyConfig.from_dict(
        {
            "preset": "quick",
            "model": {"name": "Qwen/Qwen2.5-0.5B"},
            "dataset": {"name": "wikitext", "split": "test"},
        }
    )
    assert cfg.batch_size == 4
    assert cfg.dataset.max_samples == 25
    assert [item.name for item in cfg.strategies] == ["layer_removal", "ffn_ablation"]


def test_study_config_allows_explicit_override_over_preset() -> None:
    cfg = StudyConfig.from_dict(
        {
            "study_preset": "quick",
            "model": {"name": "Qwen/Qwen2.5-0.5B"},
            "dataset": {"name": "wikitext", "split": "test", "max_samples": 5},
            "batch_size": 16,
        }
    )
    assert cfg.batch_size == 16
    assert cfg.dataset.max_samples == 5


def test_study_config_round_trip() -> None:
    cfg = StudyConfig.from_dict(
        {
            "model": {"name": "Qwen/Qwen2.5-0.5B", "dtype": "float16"},
            "dataset": {"name": "wikitext", "split": "test"},
            "strategies": [{"name": "layer_removal", "params": {"count": 1}}],
            "metrics": ["perplexity", "accuracy"],
            "output_dir": "results/custom",
        }
    )
    as_dict = cfg.to_dict()
    assert as_dict["model"]["dtype"] == "float16"
    assert as_dict["strategies"][0]["params"]["count"] == 1
    assert as_dict["output_dir"] == "results/custom"
