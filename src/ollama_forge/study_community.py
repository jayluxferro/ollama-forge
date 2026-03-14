"""Local contribution and aggregation helpers for study reports."""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama_forge.study_reports import StudyReport, load_study_report

CONTRIBUTION_SCHEMA_VERSION = 1
DEFAULT_STUDY_CONTRIB_DIR = "study_results_community"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "study"


def save_study_contribution(
    report: StudyReport,
    *,
    source_report: str | None = None,
    output_dir: str | Path = DEFAULT_STUDY_CONTRIB_DIR,
    notes: str = "",
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{_slug(report.model_name)}_{timestamp}.json"
    payload = {
        "contribution_schema_version": CONTRIBUTION_SCHEMA_VERSION,
        "timestamp": timestamp,
        "notes": notes,
        "source_report": source_report,
        "report": report.to_dict(),
    }
    path = root / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_study_contributions(directory: str | Path = DEFAULT_STUDY_CONTRIB_DIR) -> list[dict[str, Any]]:
    root = Path(directory)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("contribution_schema_version") != CONTRIBUTION_SCHEMA_VERSION:
            continue
        report_payload = payload.get("report")
        if not isinstance(report_payload, dict):
            continue
        report = load_study_report(path) if report_payload.get("model_name") is None else None
        records.append(
            {
                "timestamp": payload.get("timestamp"),
                "notes": payload.get("notes", ""),
                "source_report": payload.get("source_report"),
                "report": report_payload if report is None else report.to_dict(),
                "_source_file": str(path),
            }
        )
    return records


def aggregate_study_contributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        report = record.get("report") or {}
        model_name = report.get("model_name", "unknown")
        grouped.setdefault(model_name, []).append(report)

    aggregated: dict[str, Any] = {}
    for model_name, reports in grouped.items():
        metric_names = sorted(
            {
                metric
                for report in reports
                for metric in (report.get("baseline_metrics") or {}).keys()
            }
        )
        summary: dict[str, Any] = {"n_reports": len(reports)}
        for metric in metric_names:
            values = [
                float((report.get("baseline_metrics") or {}).get(metric))
                for report in reports
                if isinstance((report.get("baseline_metrics") or {}).get(metric), (int, float))
            ]
            if not values:
                continue
            summary[metric] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }
        aggregated[model_name] = summary
    return aggregated
