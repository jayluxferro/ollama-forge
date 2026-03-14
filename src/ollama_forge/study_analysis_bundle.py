"""Helpers for bundling multiple study analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_analysis_bundle(*, config_path: str, modules: list[str], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_path": config_path,
        "modules": modules,
        "results": results,
    }


def save_analysis_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_analysis_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_analysis_bundle(
    bundle: dict[str, Any],
    *,
    extra_results: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(bundle)
    results = dict(updated.get("results", {}))
    if extra_results:
        results.update(extra_results)
    updated["results"] = results
    if metadata:
        merged_meta = dict(updated.get("metadata", {}))
        merged_meta.update(metadata)
        updated["metadata"] = merged_meta
    return updated
