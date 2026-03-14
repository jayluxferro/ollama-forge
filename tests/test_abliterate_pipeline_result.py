"""Tests for informed pipeline result artifacts."""

from ollama_forge.abliterate_pipeline import (
    InformedPipelineResult,
    choose_pipeline_pass,
    compare_pipeline_results,
    load_informed_pipeline_result,
    pipeline_markdown,
    save_informed_pipeline_result,
)


def test_informed_pipeline_result_collects_stages() -> None:
    result = InformedPipelineResult()
    result.add_stage("analysis_bundle", "success", output_file="bundle.json")
    result.add_stage("informed_run", "failed", cause="boom")
    result.benchmark_report = "bench.json"
    result.eval_comparison = {"metrics": {"asr_pct": {"a": 10.0, "b": 20.0}}}
    result.second_pass_artifact = "second.json"
    result.second_pass_report = "second-report.json"
    result.second_pass_benchmark = "second-bench.json"
    result.second_pass_benchmark_comparison = {"metrics": {"asr_pct": {"a": 20.0, "b": 35.0}}}
    payload = result.to_dict()
    assert payload["stages"][0]["name"] == "analysis_bundle"
    assert payload["stages"][1]["status"] == "failed"
    assert payload["benchmark_report"] == "bench.json"
    assert payload["second_pass_artifact"] == "second.json"
    assert payload["second_pass_benchmark"] == "second-bench.json"


def test_choose_pipeline_pass_prefers_higher_asr() -> None:
    selected, reason = choose_pipeline_pass(
        first_benchmark={"kpis": {"asr_pct": 20.0, "refusal_rate_pct": 80.0}},
        second_benchmark={"kpis": {"asr_pct": 45.0, "refusal_rate_pct": 55.0}},
    )
    assert selected == "second_pass"
    assert "ASR" in reason


def test_pipeline_markdown_includes_selection() -> None:
    result = InformedPipelineResult(selected_pass="second_pass", selection_reason="better ASR")
    text = pipeline_markdown(result)
    assert "Selected pass" in text
    assert "second_pass" in text


def test_load_and_compare_pipeline_results(tmp_path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    save_informed_pipeline_result(InformedPipelineResult(selected_pass="first_pass"), path_a)
    save_informed_pipeline_result(InformedPipelineResult(selected_pass="second_pass"), path_b)
    payload = compare_pipeline_results(load_informed_pipeline_result(path_a), load_informed_pipeline_result(path_b))
    assert payload["selected_pass"]["a"] == "first_pass"
    assert payload["selected_pass"]["b"] == "second_pass"
