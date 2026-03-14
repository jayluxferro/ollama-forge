"""Reproducibility manifests for study runs."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_study_manifest(*, config: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "timestamp_iso": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
        "artifacts": artifacts,
    }
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            manifest["cuda_device_count"] = int(torch.cuda.device_count())
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
