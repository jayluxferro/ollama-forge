"""Tests for abliterate report export/regeneration helpers."""

from __future__ import annotations

from ollama_forge.abliterate_reports import (
    build_run_report,
    regenerate_report_exports,
    report_markdown,
)


def _make_report():
    return build_run_report(
        source_model="test/model",
        resolved_model="test/model",
        ollama_model="test-abliterated",
        profile="balanced",
        config={"strength": 1.0},
        artifacts={"output_dir": "/tmp/out"},
        status={"label": "ollama_created"},
        evaluation={"refusal_rate": 0.1, "refusal_count": 1, "total": 10},
    )


def test_report_markdown_contains_source_model() -> None:
    text = report_markdown(_make_report())
    assert "test/model" in text


def test_regenerate_report_exports_writes_files(tmp_path) -> None:
    exports = regenerate_report_exports(_make_report(), tmp_path)
    assert set(exports) == {"json", "markdown", "html"}
