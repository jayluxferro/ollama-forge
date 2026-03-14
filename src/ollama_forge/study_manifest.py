"""Reproducibility manifests for study runs."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_hash() -> str | None:
    """Return the short git commit hash of the current repo, or None."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _ollama_forge_version() -> str:
    """Return the installed ollama-forge version."""
    try:
        from importlib.metadata import version

        return version("ollama-forge")
    except Exception:
        return "unknown"


def build_study_manifest(*, config: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_iso": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "ollama_forge_version": _ollama_forge_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
        "artifacts": artifacts,
    }
    git_hash = _git_hash()
    if git_hash:
        manifest["git_commit"] = git_hash
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            manifest["cuda_device_count"] = int(torch.cuda.device_count())
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            manifest["mps_available"] = True
    except Exception:
        pass
    try:
        import transformers

        manifest["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        import datasets

        manifest["datasets_version"] = datasets.__version__
    except Exception:
        pass
    return manifest


def save_study_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
