"""Tests for study manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

from ollama_forge.study_manifest import build_study_manifest, save_study_manifest


def test_build_manifest_has_required_fields() -> None:
    manifest = build_study_manifest(config={"model": "test"}, artifacts={"report": "r.json"})
    assert manifest["schema_version"] == 2
    assert "timestamp_iso" in manifest
    assert "python_version" in manifest
    assert "platform" in manifest
    assert manifest["config"] == {"model": "test"}
    assert manifest["artifacts"] == {"report": "r.json"}


def test_build_manifest_includes_ollama_forge_version() -> None:
    manifest = build_study_manifest(config={}, artifacts={})
    assert "ollama_forge_version" in manifest
    assert isinstance(manifest["ollama_forge_version"], str)


def test_build_manifest_includes_git_commit() -> None:
    manifest = build_study_manifest(config={}, artifacts={})
    # git_commit is optional (only present if in a git repo)
    if "git_commit" in manifest:
        assert isinstance(manifest["git_commit"], str)
        assert len(manifest["git_commit"]) >= 4


def test_save_manifest_creates_file(tmp_path: Path) -> None:
    manifest = build_study_manifest(config={"x": 1}, artifacts={})
    path = save_study_manifest(manifest, tmp_path / "manifest.json")
    assert path.is_file()
    loaded = json.loads(path.read_text())
    assert loaded["config"] == {"x": 1}
    assert loaded["schema_version"] == 2


def test_save_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    path = save_study_manifest(
        build_study_manifest(config={}, artifacts={}),
        tmp_path / "nested" / "dir" / "manifest.json",
    )
    assert path.is_file()
