"""Tests for local study contribution workflows."""

from __future__ import annotations

from ollama_forge.study_community import (
    aggregate_study_contributions,
    load_study_contributions,
    save_study_contribution,
)
from ollama_forge.study_reports import StudyReport, StudyResult


def _make_report() -> StudyReport:
    report = StudyReport(model_name="demo/model", config={"x": 1})
    report.add_baseline({"perplexity": 5.0})
    report.add_result(StudyResult(
        strategy="layer_removal", component="layer_0", description="x", metrics={"perplexity": 6.0},
    ))
    return report


def test_save_and_load_study_contribution(tmp_path) -> None:
    path = save_study_contribution(_make_report(), output_dir=tmp_path, notes="smoke")
    assert path.is_file()
    records = load_study_contributions(tmp_path)
    assert len(records) == 1
    assert records[0]["notes"] == "smoke"


def test_aggregate_study_contributions(tmp_path) -> None:
    save_study_contribution(_make_report(), output_dir=tmp_path)
    save_study_contribution(_make_report(), output_dir=tmp_path)
    payload = aggregate_study_contributions(load_study_contributions(tmp_path))
    assert payload["demo/model"]["n_reports"] == 2
