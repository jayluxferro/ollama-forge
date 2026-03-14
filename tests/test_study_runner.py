"""Tests for study planning and runner execution."""

from __future__ import annotations

import json

from ollama_forge.study_config import StudyConfig
from ollama_forge.study_reports import load_study_report
from ollama_forge.study_runner import plan_study, run_study


class _FakeHandle:
    def __init__(self) -> None:
        self.num_layers = 2
        self.num_heads = 2
        self.hidden_size = 8
        self.applied: list[tuple] = []
        self.restores = 0

    def remove_layer(self, idx: int) -> None:
        self.applied.append(("remove_layer", idx))

    def restore(self) -> None:
        self.restores += 1


class _FakeEvaluator:
    def __init__(self, *, handle, dataset, metrics, **kwargs):
        self.handle = handle
        self.dataset = dataset
        self.metrics = metrics

    def evaluate(self) -> dict[str, float]:
        return {"perplexity": float(len(self.handle.applied))}


def test_plan_study_returns_expected_summary() -> None:
    cfg = StudyConfig.from_dict(
        {
            "preset": "quick",
            "model": {"name": "Qwen/Qwen2.5-0.5B"},
            "dataset": {"name": "wikitext", "split": "test"},
        }
    )
    plan = plan_study(cfg)
    payload = plan.to_dict()
    assert payload["model_name"] == "Qwen/Qwen2.5-0.5B"
    assert payload["strategies"][0]["strategy"] == "layer_removal"


def test_run_study_generates_report_files(tmp_path) -> None:
    cfg = StudyConfig.from_dict(
        {
            "model": {"name": "fake/model"},
            "dataset": {"name": "fake-dataset", "split": "test"},
            "strategies": [{"name": "layer_removal", "params": {}}],
            "output_dir": str(tmp_path),
        }
    )

    report = run_study(
        cfg,
        model_loader=lambda model_cfg: _FakeHandle(),
        dataset_loader=lambda dataset_cfg: [{"text": "x"}],
        evaluator_factory=_FakeEvaluator,
        output_dir=tmp_path,
    )

    assert report.baseline_metrics == {"perplexity": 0.0}
    assert len(report.results) == 2
    assert (tmp_path / "study-results.json").is_file()
    assert (tmp_path / "study-results.csv").is_file()
    assert (tmp_path / "study-summary.txt").is_file()
    assert (tmp_path / "study-manifest.json").is_file()
    payload = json.loads((tmp_path / "study-results.json").read_text(encoding="utf-8"))
    assert payload["results"][0]["strategy"] == "layer_removal"
    loaded = load_study_report(tmp_path / "study-results.json")
    assert loaded.model_name == "fake/model"
    assert len(loaded.results) == 2
