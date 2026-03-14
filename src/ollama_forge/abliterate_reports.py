"""Reporting and aggregation helpers for abliterate runs and benchmarks."""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 1
RUN_REPORT_KIND = "abliterate_run"
BENCHMARK_REPORT_KIND = "abliterate_benchmark"
CONTRIBUTION_REPORT_KIND = "abliterate_contribution"
DEFAULT_CONTRIB_DIR = "community_results"


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str, *, fallback: str = "abliterate") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text[:80] or fallback


def build_run_report(
    *,
    source_model: str | None,
    resolved_model: str | None,
    ollama_model: str,
    profile: str | None,
    config: dict[str, Any],
    artifacts: dict[str, Any],
    status: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": RUN_REPORT_KIND,
        "timestamp_iso": _timestamp_iso(),
        "source_model": source_model,
        "resolved_model": resolved_model,
        "ollama_model": ollama_model,
        "profile": profile,
        "config": config,
        "artifacts": artifacts,
        "status": status,
        "evaluation": evaluation,
        "notes": notes,
    }


def build_benchmark_report(
    *,
    prompt_set: str,
    output_dir: str,
    primary: dict[str, Any],
    compare: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": BENCHMARK_REPORT_KIND,
        "timestamp_iso": _timestamp_iso(),
        "prompt_set": prompt_set,
        "output_dir": output_dir,
        "primary": primary,
    }
    if compare is not None:
        report["compare"] = compare
        report["deltas"] = _benchmark_deltas(primary.get("kpis") or {}, compare.get("kpis") or {})
    return report


def _benchmark_deltas(primary_kpis: dict[str, Any], compare_kpis: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in (
        "asr_pct",
        "refusal_rate_pct",
        "extraction_rate_pct",
        "avg_latency_sec",
        "benign_refusal_rate_pct",
    ):
        primary_val = primary_kpis.get(key)
        compare_val = compare_kpis.get(key)
        if isinstance(primary_val, (int, float)) and isinstance(compare_val, (int, float)):
            deltas[key] = round(primary_val - compare_val, 4)
    return deltas


def save_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def report_markdown(report: dict[str, Any]) -> str:
    kind = report.get("report_kind")
    if kind == BENCHMARK_REPORT_KIND:
        lines = ["# Abliterate Benchmark Report", ""]
        lines.append(f"Prompt set: {report.get('prompt_set')}")
        primary = report.get("primary") or {}
        compare = report.get("compare") or {}
        lines.append(f"Primary: {primary.get('model')} @ {primary.get('base_url')}")
        if compare:
            lines.append(f"Compare: {compare.get('model')} @ {compare.get('base_url')}")
        deltas = report.get("deltas") or {}
        if deltas:
            lines.append("")
            lines.append("## Deltas")
            for key, value in sorted(deltas.items()):
                lines.append(f"- {key}: {value:.4f}")
        return "\n".join(lines) + "\n"

    lines = ["# Abliterate Run Report", ""]
    lines.append(f"Source model: {report.get('source_model')}")
    lines.append(f"Ollama model: {report.get('ollama_model')}")
    lines.append(f"Profile: {report.get('profile') or 'custom'}")
    status = report.get("status") or {}
    lines.append(f"Status: {status.get('label')}")
    evaluation = report.get("evaluation") or {}
    if evaluation:
        lines.append("")
        lines.append("## Evaluation")
        for key, value in sorted(evaluation.items()):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def report_html(report: dict[str, Any]) -> str:
    kind = report.get("report_kind")
    if kind == BENCHMARK_REPORT_KIND:
        deltas = report.get("deltas") or {}
        rows = "".join(f"<li>{key}: {value:.4f}</li>" for key, value in sorted(deltas.items()))
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Abliterate Benchmark Report</title></head><body>"
            f"<h1>Abliterate Benchmark Report</h1><p>Prompt set: {report.get('prompt_set')}</p>"
            f"<p>Primary: {(report.get('primary') or {}).get('model')}</p>"
            f"<ul>{rows}</ul></body></html>"
        )

    evaluation = report.get("evaluation") or {}
    rows = "".join(f"<li>{key}: {value}</li>" for key, value in sorted(evaluation.items()))
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Abliterate Run Report</title></head><body>"
        f"<h1>Abliterate Run Report</h1><p>Source model: {report.get('source_model')}</p>"
        f"<p>Ollama model: {report.get('ollama_model')}</p>"
        f"<p>Status: {(report.get('status') or {}).get('label')}</p>"
        f"<ul>{rows}</ul></body></html>"
    )


def regenerate_report_exports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = save_report(report, root / "report.json")
    md_path = root / "report.md"
    html_path = root / "report.html"
    md_path.write_text(report_markdown(report), encoding="utf-8")
    html_path.write_text(report_html(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def contribution_filename(report: dict[str, Any]) -> str:
    source = report.get("source_model") or report.get("ollama_model") or "abliterate"
    profile = report.get("profile") or "custom"
    timestamp = report.get("timestamp_iso", "").replace(":", "-")
    return f"{_slug(source)}_{_slug(profile, fallback='custom')}_{timestamp}.json"


def save_contribution(
    report: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_CONTRIB_DIR,
    notes: str = "",
) -> Path:
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": CONTRIBUTION_REPORT_KIND,
        "timestamp_iso": _timestamp_iso(),
        "notes": notes,
        "report": report,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_report(payload, target_dir / contribution_filename(report))


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reports(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = load_report(path)
        except (OSError, json.JSONDecodeError):
            continue
        data["_source_file"] = str(path)
        if data.get("report_kind") == CONTRIBUTION_REPORT_KIND and isinstance(data.get("report"), dict):
            report = dict(data["report"])
            report["_source_file"] = str(path)
            report["_contribution_notes"] = data.get("notes", "")
            records.append(report)
            continue
        if data.get("report_kind") == RUN_REPORT_KIND:
            records.append(data)
    return records


def aggregate_reports(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        evaluation = record.get("evaluation") or {}
        if not evaluation:
            continue
        model_key = record.get("source_model") or record.get("ollama_model") or "unknown"
        profile_key = record.get("profile") or "custom"
        groups.setdefault((model_key, profile_key), []).append(evaluation)

    aggregated: dict[str, dict[str, Any]] = {}
    for (model_key, profile_key), evaluations in groups.items():
        aggregated.setdefault(model_key, {})
        summary: dict[str, Any] = {"n_runs": len(evaluations)}
        for metric_name in ("refusal_rate", "refusal_count", "total"):
            values = [
                float(evaluation[metric_name])
                for evaluation in evaluations
                if isinstance(evaluation.get(metric_name), (int, float))
            ]
            if not values:
                continue
            summary[metric_name] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }
        aggregated[model_key][profile_key] = summary
    return aggregated


def generate_latex_table(
    aggregated: dict[str, dict[str, Any]],
    *,
    metric: str = "refusal_rate",
    min_runs: int = 1,
) -> str:
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\hline",
        "Model & Profile & Metric & Mean & Runs \\\\",
        r"\hline",
    ]
    for model_key in sorted(aggregated):
        for profile_key in sorted(aggregated[model_key]):
            summary = aggregated[model_key][profile_key]
            n_runs = int(summary.get("n_runs", 0))
            metric_summary = summary.get(metric) or {}
            if n_runs < min_runs or not metric_summary:
                continue
            lines.append(
                f"{model_key} & {profile_key} & {metric} & {metric_summary['mean']:.4f} & {n_runs} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)
