"""Ingest and compare external evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalReport:
    kind: str
    source_file: str
    metrics: dict[str, float]
    raw: dict[str, Any]


def load_eval_report(path: str | Path) -> EvalReport:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "kpis" in payload:
        metrics = {
            key: float(value)
            for key, value in (payload.get("kpis") or {}).items()
            if isinstance(value, (int, float))
        }
        return EvalReport(kind="security_eval", source_file=str(source), metrics=metrics, raw=payload)
    if "results" in payload and isinstance(payload.get("results"), dict):
        metrics: dict[str, float] = {}
        for task_name, task_metrics in payload["results"].items():
            if isinstance(task_metrics, dict):
                for metric_name, value in task_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{task_name}.{metric_name}"] = float(value)
        return EvalReport(kind="lm_eval", source_file=str(source), metrics=metrics, raw=payload)
    raise ValueError(f"Unrecognized eval report format: {source}")


def compare_eval_reports(report_a: EvalReport, report_b: EvalReport) -> dict[str, Any]:
    metric_names = sorted(set(report_a.metrics) | set(report_b.metrics))
    return {
        "kind_a": report_a.kind,
        "kind_b": report_b.kind,
        "source_a": report_a.source_file,
        "source_b": report_b.source_file,
        "metrics": {
            name: {"a": report_a.metrics.get(name), "b": report_b.metrics.get(name)}
            for name in metric_names
        },
    }
