"""Orchestration artifacts for informed abliteration pipelines."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineStage:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class InformedPipelineResult:
    analysis_bundle: str | None = None
    informed_artifact: str | None = None
    run_report: str | None = None
    benchmark_report: str | None = None
    eval_comparison: dict[str, Any] | None = None
    refined_recommendation: dict[str, Any] | None = None
    second_pass_artifact: str | None = None
    second_pass_report: str | None = None
    second_pass_benchmark: str | None = None
    second_pass_benchmark_comparison: dict[str, Any] | None = None
    selected_pass: str | None = None
    selection_reason: str | None = None
    stages: list[PipelineStage] = field(default_factory=list)

    def add_stage(self, name: str, status: str, **details: Any) -> None:
        self.stages.append(PipelineStage(name=name, status=status, details=details))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def save_informed_pipeline_result(result: InformedPipelineResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_informed_pipeline_result(path: str | Path) -> InformedPipelineResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = InformedPipelineResult(
        analysis_bundle=payload.get("analysis_bundle"),
        informed_artifact=payload.get("informed_artifact"),
        run_report=payload.get("run_report"),
        benchmark_report=payload.get("benchmark_report"),
        eval_comparison=payload.get("eval_comparison"),
        refined_recommendation=payload.get("refined_recommendation"),
        second_pass_artifact=payload.get("second_pass_artifact"),
        second_pass_report=payload.get("second_pass_report"),
        second_pass_benchmark=payload.get("second_pass_benchmark"),
        second_pass_benchmark_comparison=payload.get("second_pass_benchmark_comparison"),
        selected_pass=payload.get("selected_pass"),
        selection_reason=payload.get("selection_reason"),
    )
    for stage in payload.get("stages", []):
        result.stages.append(
            PipelineStage(
                name=stage.get("name", ""),
                status=stage.get("status", ""),
                details=stage.get("details", {}) or {},
            )
        )
    return result


def choose_pipeline_pass(
    *,
    first_benchmark: dict[str, Any] | None,
    second_benchmark: dict[str, Any] | None,
) -> tuple[str, str]:
    if first_benchmark is None and second_benchmark is None:
        return "first_pass", "No benchmark data available; defaulting to first pass."
    if first_benchmark is None:
        return "second_pass", "Only second-pass benchmark data is available."
    if second_benchmark is None:
        return "first_pass", "Only first-pass benchmark data is available."

    first_kpis = first_benchmark.get("kpis") or {}
    second_kpis = second_benchmark.get("kpis") or {}

    first_asr = first_kpis.get("asr_pct")
    second_asr = second_kpis.get("asr_pct")
    first_refusal = first_kpis.get("refusal_rate_pct")
    second_refusal = second_kpis.get("refusal_rate_pct")

    if isinstance(first_asr, (int, float)) and isinstance(second_asr, (int, float)):
        if second_asr > first_asr:
            return "second_pass", f"Second pass improved ASR from {first_asr:.1f}% to {second_asr:.1f}%."
        if second_asr < first_asr:
            return "first_pass", f"First pass retained higher ASR ({first_asr:.1f}% vs {second_asr:.1f}%)."

    if isinstance(first_refusal, (int, float)) and isinstance(second_refusal, (int, float)):
        if second_refusal < first_refusal:
            return "second_pass", f"Second pass reduced refusal from {first_refusal:.1f}% to {second_refusal:.1f}%."
        if second_refusal > first_refusal:
            return "first_pass", f"First pass retained lower refusal ({first_refusal:.1f}% vs {second_refusal:.1f}%)."

    return "first_pass", "Benchmarks were tied on primary KPIs; defaulting to first pass."


def pipeline_markdown(result: InformedPipelineResult) -> str:
    lines = ["# Informed Abliteration Pipeline", ""]
    if result.selected_pass:
        lines.append(f"Selected pass: **{result.selected_pass}**")
    if result.selection_reason:
        lines.append(f"Selection reason: {result.selection_reason}")
    lines.append("")
    lines.append("## Artifacts")
    for key in (
        "analysis_bundle",
        "informed_artifact",
        "run_report",
        "benchmark_report",
        "second_pass_artifact",
        "second_pass_report",
        "second_pass_benchmark",
    ):
        value = getattr(result, key)
        if value:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Stages")
    lines.append("| Stage | Status | Details |")
    lines.append("|---|---|---|")
    for stage in result.stages:
        lines.append(f"| {stage.name} | {stage.status} | {json.dumps(stage.details, sort_keys=True)} |")
    return "\n".join(lines) + "\n"


def pipeline_html(result: InformedPipelineResult) -> str:
    rows = []
    for stage in result.stages:
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(stage.name),
                html.escape(stage.status),
                html.escape(json.dumps(stage.details, sort_keys=True)),
            )
        )
    selected = (
        f"<p><strong>Selected pass:</strong> {html.escape(result.selected_pass or '')}</p>"
        if result.selected_pass else ""
    )
    reason = (
        f"<p><strong>Selection reason:</strong> {html.escape(result.selection_reason or '')}</p>"
        if result.selection_reason else ""
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Informed Pipeline</title></head><body>"
        "<h1>Informed Abliteration Pipeline</h1>"
        f"{selected}{reason}"
        "<table border='1'><tr><th>Stage</th><th>Status</th><th>Details</th></tr>"
        f"{''.join(rows)}</table></body></html>"
    )


def save_informed_pipeline_exports(result: InformedPipelineResult, output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    md = root / "informed-pipeline.md"
    html_path = root / "informed-pipeline.html"
    md.write_text(pipeline_markdown(result), encoding="utf-8")
    html_path.write_text(pipeline_html(result), encoding="utf-8")
    return md, html_path


def compare_pipeline_results(a: InformedPipelineResult, b: InformedPipelineResult) -> dict[str, Any]:
    return {
        "selected_pass": {"a": a.selected_pass, "b": b.selected_pass},
        "selection_reason": {"a": a.selection_reason, "b": b.selection_reason},
        "stage_count": {"a": len(a.stages), "b": len(b.stages)},
        "has_benchmark": {"a": bool(a.benchmark_report), "b": bool(b.benchmark_report)},
        "has_second_pass": {"a": bool(a.second_pass_artifact), "b": bool(b.second_pass_artifact)},
    }


def pipeline_comparison_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Informed Pipeline Comparison", "", "| Field | A | B |", "|---|---|---|"]
    for key, values in sorted(payload.items()):
        lines.append(f"| {key} | {values.get('a')} | {values.get('b')} |")
    return "\n".join(lines) + "\n"


def pipeline_comparison_html(payload: dict[str, Any]) -> str:
    rows = []
    for key, values in sorted(payload.items()):
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(key),
                html.escape(str(values.get("a"))),
                html.escape(str(values.get("b"))),
            )
        )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Pipeline Comparison</title></head><body>"
        "<h1>Informed Pipeline Comparison</h1>"
        "<table border='1'><tr><th>Field</th><th>A</th><th>B</th></tr>"
        f"{''.join(rows)}</table></body></html>"
    )


def save_pipeline_comparison(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        path.write_text(pipeline_comparison_markdown(payload), encoding="utf-8")
    elif suffix in (".html", ".htm"):
        path.write_text(pipeline_comparison_html(payload), encoding="utf-8")
    elif suffix == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported pipeline comparison export format: {path}")
    return path
