"""Model registry — maps model IDs to backend kind, path, and size estimate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BackendKind = Literal["llama_cpp", "mlx_lm", "mlx_vlm", "ollama"]


@dataclass
class ModelRecord:
    model_id: str
    kind: BackendKind
    path: Path | None = None  # None for Ollama pass-through models
    estimated_mb: int = 2000


class ModelRegistry:
    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}

    def register(self, model_id: str, path: Path | None = None, **kw: object) -> ModelRecord:
        """Register a model. If path is given, detect kind and size automatically."""
        if path is not None and path.exists():
            kind = detect_kind(path)
            estimated_mb = estimate_mb(path)
        else:
            kind = kw.get("kind", "ollama")  # type: ignore[assignment]
            estimated_mb = int(kw.get("estimated_mb", 2000))
        rec = ModelRecord(
            model_id=model_id,
            kind=kind,
            path=path,
            estimated_mb=estimated_mb,
        )
        self._records[model_id] = rec
        return rec

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def all(self) -> list[ModelRecord]:
        return list(self._records.values())


def detect_kind(path: Path) -> BackendKind:
    """Infer backend from file/directory structure."""
    if path.suffix.lower() == ".gguf":
        return "llama_cpp"
    if path.is_dir():
        config = path / "config.json"
        if config.exists():
            try:
                with open(config) as f:
                    cfg = json.load(f)
                # Vision models carry a vision_config, vision_tower, or image_token_index
                if any(k in cfg for k in ("vision_config", "vision_tower", "image_token_index")):
                    return "mlx_vlm"
            except Exception:
                pass
            return "mlx_lm"
    return "ollama"


def estimate_mb(path: Path) -> int:
    """Rough GPU memory estimate from file sizes."""
    if path.is_file():
        return max(1, path.stat().st_size // (1024 * 1024))
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
        if total == 0:
            total = sum(f.stat().st_size for f in path.rglob("*.gguf"))
        return max(1, total // (1024 * 1024))
    return 2000
