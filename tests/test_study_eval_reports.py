"""Tests for external eval report ingestion and comparison."""

from __future__ import annotations

import json

from ollama_forge.study_eval_reports import compare_eval_reports, load_eval_report


def test_load_eval_report_security_eval(tmp_path) -> None:
    path = tmp_path / "security.json"
    path.write_text(json.dumps({"kpis": {"asr_pct": 42.0, "refusal_rate_pct": 58.0}}), encoding="utf-8")
    report = load_eval_report(path)
    assert report.kind == "security_eval"
    assert report.metrics["asr_pct"] == 42.0


def test_load_eval_report_lm_eval(tmp_path) -> None:
    path = tmp_path / "lm.json"
    path.write_text(json.dumps({"results": {"hellaswag": {"acc_norm,none": 0.5}}}), encoding="utf-8")
    report = load_eval_report(path)
    assert report.kind == "lm_eval"
    assert report.metrics["hellaswag.acc_norm,none"] == 0.5


def test_compare_eval_reports_merges_metrics(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"kpis": {"asr_pct": 42.0}}), encoding="utf-8")
    b.write_text(json.dumps({"kpis": {"asr_pct": 10.0, "refusal_rate_pct": 90.0}}), encoding="utf-8")
    payload = compare_eval_reports(load_eval_report(a), load_eval_report(b))
    assert payload["metrics"]["asr_pct"]["a"] == 42.0
    assert payload["metrics"]["refusal_rate_pct"]["b"] == 90.0
