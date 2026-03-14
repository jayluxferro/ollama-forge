"""Inspection and comparison helpers for informed-run artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def load_informed_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_informed_artifacts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_status": {"a": a.get("run_status"), "b": b.get("run_status")},
        "profile": {
            "a": (a.get("recommendation") or {}).get("profile"),
            "b": (b.get("recommendation") or {}).get("profile"),
        },
        "strength": {
            "a": (a.get("recommendation") or {}).get("strength"),
            "b": (b.get("recommendation") or {}).get("strength"),
        },
        "has_report": {"a": bool(a.get("report")), "b": bool(b.get("report"))},
        "has_benchmark": {"a": bool(a.get("benchmark")), "b": bool(b.get("benchmark"))},
        "has_eval_comparison": {"a": bool(a.get("eval_comparison")), "b": bool(b.get("eval_comparison"))},
    }


def informed_artifact_markdown(artifact: dict[str, Any]) -> str:
    recommendation = artifact.get("recommendation") or {}
    lines = ["# Informed Run Artifact", ""]
    lines.append(f"Run status: {artifact.get('run_status')}")
    lines.append(f"Profile: {recommendation.get('profile')}")
    lines.append(f"Strength: {recommendation.get('strength')}")
    lines.append("")
    lines.append("## Notes")
    for note in recommendation.get("notes", []) or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def informed_artifact_html(artifact: dict[str, Any]) -> str:
    recommendation = artifact.get("recommendation") or {}
    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in recommendation.get("notes", []) or [])
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Informed Artifact</title></head><body>"
        "<h1>Informed Run Artifact</h1>"
        f"<p><strong>Run status:</strong> {html.escape(str(artifact.get('run_status')))}</p>"
        f"<p><strong>Profile:</strong> {html.escape(str(recommendation.get('profile')))}</p>"
        f"<p><strong>Strength:</strong> {html.escape(str(recommendation.get('strength')))}</p>"
        f"<ul>{notes}</ul></body></html>"
    )


def save_informed_artifact_export(artifact: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        path.write_text(informed_artifact_markdown(artifact), encoding="utf-8")
    elif suffix in (".html", ".htm"):
        path.write_text(informed_artifact_html(artifact), encoding="utf-8")
    elif suffix == ".json":
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported informed artifact export format: {path}")
    return path
