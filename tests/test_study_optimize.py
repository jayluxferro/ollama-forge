"""Tests for study optimization helpers."""

from __future__ import annotations

from ollama_forge.study_config import StudyConfig
from ollama_forge.study_optimize import optimize_study_strength


class _FakeHandle:
    num_layers = 1
    num_heads = 1
    hidden_size = 4

    def __init__(self) -> None:
        self.applied = []

    def remove_layer(self, idx: int, strength: float = 1.0) -> None:
        self.applied.append(strength)

    def restore(self) -> None:
        return None


class _FakeEvaluator:
    def __init__(self, *, handle, dataset, metrics, **kwargs):
        self.handle = handle

    def evaluate(self):
        strength = self.handle.applied[-1] if self.handle.applied else 0.0
        return {"perplexity": 10.0 - strength}


def test_optimize_study_strength_selects_best_strength(tmp_path) -> None:
    cfg = StudyConfig.from_dict(
        {
            "model": {"name": "fake/model"},
            "dataset": {"name": "fake", "split": "test"},
            "strategies": [{"name": "layer_removal", "params": {}}],
            "metrics": ["perplexity"],
            "output_dir": str(tmp_path / "base"),
        }
    )
    result = optimize_study_strength(
        cfg,
        strengths=[0.25, 0.5, 1.0],
        metric="perplexity",
        objective="min",
        model_loader=lambda cfg: _FakeHandle(),
        dataset_loader=lambda cfg: [{"text": "x"}],
        evaluator_factory=_FakeEvaluator,
        output_dir=tmp_path / "opt",
    )
    assert result.best_strength == 1.0
