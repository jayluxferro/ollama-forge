"""Tests for informed abliterate run artifacts."""

from ollama_forge.abliterate_informed import build_informed_run_artifact, update_informed_run_artifact


def test_build_informed_run_artifact_captures_sources_and_run() -> None:
    artifact = build_informed_run_artifact(
        analysis_docs=[{"_source_file": "a.json"}, {"_source_file": "b.json"}],
        recommendation={"profile": "balanced"},
        requested_run={"model": "Qwen/Qwen2.5-0.5B-Instruct", "name": "demo"},
    )
    assert artifact["analysis_files"] == ["a.json", "b.json"]
    assert artifact["recommendation"]["profile"] == "balanced"
    assert artifact["requested_run"]["name"] == "demo"


def test_update_informed_run_artifact_adds_status_and_report() -> None:
    artifact = build_informed_run_artifact(
        analysis_docs=[{"_source_file": "a.json"}],
        recommendation={"profile": "balanced"},
        requested_run={"model": "demo/model"},
    )
    updated = update_informed_run_artifact(
        artifact,
        run_status="success",
        report_path="report.json",
        report_payload={"status": {"label": "ok"}},
    )
    assert updated["run_status"] == "success"
    assert updated["report_path"] == "report.json"
    assert updated["report"]["status"]["label"] == "ok"


def test_update_informed_run_artifact_adds_benchmark_and_comparison() -> None:
    artifact = build_informed_run_artifact(
        analysis_docs=[{"_source_file": "a.json"}],
        recommendation={"profile": "balanced"},
        requested_run={"model": "demo/model"},
    )
    updated = update_informed_run_artifact(
        artifact,
        run_status="success",
        benchmark_path="bench.json",
        benchmark_payload={"kpis": {"asr_pct": 20.0}},
        eval_comparison={"metrics": {"asr_pct": {"a": 10.0, "b": 20.0}}},
    )
    assert updated["benchmark_path"] == "bench.json"
    assert updated["benchmark"]["kpis"]["asr_pct"] == 20.0
    assert "eval_comparison" in updated
