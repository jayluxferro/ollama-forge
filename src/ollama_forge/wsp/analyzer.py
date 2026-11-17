"""Phase 2: Weight-space analysis — locate obstruction circuits.

Loads MLP projection weights from an MLX safetensors model, runs partial SVD
on each layer, and ranks candidate layers by their structural signature for
each obstruction type.

Supports both quantized models (uint32 + scales/biases) and float16/bfloat16
models without quantization.

The key metrics per layer:
  spectral_gap     — sv[0] / sv[1]: high values mean one dominant direction
                     (potential "bottleneck" circuit that could suppress or
                     loop capability)
  effective_rank   — exp(entropy of normalised singular values): how many
                     dimensions are truly active (low = more concentrated)
  relative_norm    — frobenius norm relative to the median across all same-type
                     layers: unusually large norms flag atypical layers
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class LayerAnalysis:
    layer_idx: int
    proj_type: str          # "gate_proj" | "up_proj" | "down_proj"
    shape: tuple[int, ...]  # original weight shape (rows, cols)
    top_sv: list[float]     # top-k singular values
    spectral_gap: float     # sv[0]/sv[1]
    effective_rank: float   # exp(entropy of normalised sv spectrum)
    frob_norm: float        # frobenius norm of the dequantized weight matrix

    @property
    def layer_position(self) -> float | None:
        return None  # filled in after full scan

    def star_rating(self, n_layers: int, obstruction: str) -> int:
        pos = self.layer_idx / max(n_layers - 1, 1)
        gap = self.spectral_gap
        if obstruction == "loops":
            # Middle-to-upper layers, high spectral gap, down_proj preferred
            depth_score = 1.0 - abs(pos - 0.60) * 2  # peaks at 60% depth
            proj_bonus = 0.3 if self.proj_type == "down_proj" else 0.0
        elif obstruction == "sycophancy":
            depth_score = 1.0 - abs(pos - 0.72) * 2  # peaks at 72% depth
            proj_bonus = 0.2 if self.proj_type in ("gate_proj", "up_proj") else 0.0
        elif obstruction == "verbosity":
            depth_score = 1.0 - abs(pos - 0.45) * 2  # peaks at 45% depth
            proj_bonus = 0.1
        else:
            depth_score = 0.5
            proj_bonus = 0.0
        score = max(0.0, depth_score) + proj_bonus + (gap - 1.0) * 0.4
        if score >= 1.2:
            return 3
        if score >= 0.7:
            return 2
        return 1


@dataclass
class WeightMap:
    model_path: str
    n_layers: int
    quantized: bool
    quantization_bits: int
    elapsed_s: float
    analyses: list[LayerAnalysis] = field(default_factory=list)

    def top_candidates(
        self,
        obstruction: str = "loops",
        proj_type: str | None = "down_proj",
        n: int = 10,
    ) -> list[LayerAnalysis]:
        candidates = [
            a for a in self.analyses
            if proj_type is None or a.proj_type == proj_type
        ]
        candidates.sort(
            key=lambda a: a.star_rating(self.n_layers, obstruction) * 10
            + a.spectral_gap,
            reverse=True,
        )
        return candidates[:n]

    def report(self, obstruction: str = "loops", n: int = 10) -> str:
        lines = [
            f"Weight map — {self.model_path}",
            f"  {self.n_layers} layers  |  "
            f"{'quantized ' + str(self.quantization_bits) + '-bit' if self.quantized else 'fp16/bf16'}  |  "
            f"{self.elapsed_s:.1f}s",
            "",
            f"Top candidates for  {obstruction.upper()}  obstructions:",
            "",
        ]
        candidates = self.top_candidates(obstruction=obstruction, proj_type=None)
        pos_w = max(len(c.proj_type) for c in candidates) if candidates else 10
        for c in candidates:
            pos = c.layer_idx / max(self.n_layers - 1, 1)
            stars = "★" * c.star_rating(self.n_layers, obstruction)
            lines.append(
                f"  Layer {c.layer_idx:3d}  {c.proj_type:{pos_w}s}  "
                f"gap={c.spectral_gap:.2f}  eff_rank={c.effective_rank:.0f}  "
                f"pos={pos:.0%}  {stars}"
            )
        lines += [
            "",
            "Edit guidance: zero the top-1 or top-2 singular directions of",
            "the highest-ranked down_proj layers (forge wsp sharpen — Phase 3).",
        ]
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "model_path": self.model_path,
            "n_layers": self.n_layers,
            "quantized": self.quantized,
            "quantization_bits": self.quantization_bits,
            "elapsed_s": self.elapsed_s,
            "analyses": [
                {
                    "layer_idx": a.layer_idx,
                    "proj_type": a.proj_type,
                    "shape": list(a.shape),
                    "top_sv": a.top_sv,
                    "spectral_gap": a.spectral_gap,
                    "effective_rank": a.effective_rank,
                    "frob_norm": a.frob_norm,
                }
                for a in self.analyses
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> "WeightMap":
        analyses = [
            LayerAnalysis(
                layer_idx=a["layer_idx"],
                proj_type=a["proj_type"],
                shape=tuple(a["shape"]),
                top_sv=a["top_sv"],
                spectral_gap=a["spectral_gap"],
                effective_rank=a["effective_rank"],
                frob_norm=a["frob_norm"],
            )
            for a in d.get("analyses", [])
        ]
        return cls(
            model_path=d["model_path"],
            n_layers=d["n_layers"],
            quantized=d["quantized"],
            quantization_bits=d.get("quantization_bits", 16),
            elapsed_s=d.get("elapsed_s", 0.0),
            analyses=analyses,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _effective_rank(sv: list) -> float:
    import numpy as np
    a = np.array(sv, dtype=np.float64)
    a = a[a > 1e-10]
    if len(a) == 0:
        return 1.0
    p = a / a.sum()
    return float(np.exp(-np.dot(p, np.log(p + 1e-15))))


def _spectral_gap(sv: list) -> float:
    if len(sv) < 2 or sv[1] < 1e-10:
        return float(sv[0]) if len(sv) > 0 else 1.0
    return float(sv[0] / sv[1])


def _svd_metrics(
    arr,
    top_k: int = 16,
    max_rows: int = 512,
) -> tuple[list[float], float, float, float]:
    """Return (top_sv, spectral_gap, effective_rank, frob_norm)."""
    import numpy as np
    frob = float(np.linalg.norm(arr, "fro"))
    # subsample rows for speed (SVD on a row-compressed view)
    if arr.shape[0] > max_rows:
        step = arr.shape[0] // max_rows
        arr_s = arr[::step].copy()
    else:
        arr_s = arr
    sv = np.linalg.svd(arr_s, compute_uv=False)
    gap = _spectral_gap(sv)
    er = _effective_rank(sv)
    return sv[:top_k].tolist(), gap, er, frob


# ── model discovery ───────────────────────────────────────────────────────────

def _find_safetensors_files(model_path: Path) -> list[Path]:
    """Return all safetensors files in model_path, sorted by shard index."""
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        # Look one level deeper (snapshot dirs)
        for child in model_path.iterdir():
            if child.is_dir():
                files = sorted(child.glob("*.safetensors"))
                if files:
                    break
    return files


def _load_all_weights(files: list[Path]) -> dict:
    """Load and merge weights from all shard files."""
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("mlx is required for weight analysis — install with: pip install mlx") from exc

    merged: dict = {}
    for f in files:
        chunk = mx.load(str(f))
        merged.update(chunk)
    return merged


def _read_config(model_path: Path) -> dict:
    for candidate in [
        model_path / "config.json",
        *sorted(model_path.glob("*/config.json")),
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def find_model_path(model_name: str) -> Path | None:
    """Try to resolve a model short-name or HF repo to a local directory."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        return None
    name_slug = "models--" + model_name.replace("/", "--")
    candidate = hf_cache / name_slug
    if candidate.exists():
        # Find the most recent snapshot
        snapshots = sorted((candidate / "snapshots").glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        return snapshots[0] if snapshots else None
    # Fuzzy: find any dir whose name contains both parts of the repo id
    parts = model_name.lower().replace("/", "--").split("--")
    for entry in hf_cache.iterdir():
        if all(p in entry.name.lower() for p in parts):
            snapshots = sorted((entry / "snapshots").glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
            return snapshots[0] if snapshots else None
    return None


# ── main analyser ─────────────────────────────────────────────────────────────

class WeightSpaceAnalyzer:
    """Scan MLP layers and compute spectral obstruction signatures."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def scan(
        self,
        proj_types: list[str] | None = None,
        top_k_sv: int = 16,
        max_rows: int = 512,
        progress_fn: Callable[[str], None] | None = None,
    ) -> WeightMap:
        """
        Load weights and run partial SVD on each MLP projection layer.

        Args:
            proj_types: Which MLP projections to scan.  None = all three.
            top_k_sv:   How many singular values to keep per layer.
            max_rows:   Row subsample limit for speed (SVD on a proxy matrix).
            progress_fn: Optional callback(message) for progress updates.
        """
        import numpy as np

        if proj_types is None:
            proj_types = ["gate_proj", "up_proj", "down_proj"]

        def _progress(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)

        t0 = time.perf_counter()

        # ── 1. find and load weights ──────────────────────────────────────────
        files = _find_safetensors_files(self.model_path)
        if not files:
            raise FileNotFoundError(
                f"No .safetensors files found in {self.model_path}"
            )
        _progress(f"loading {len(files)} shard(s) from {self.model_path.name}")
        weights = _load_all_weights(files)

        # ── 2. read config (quantization, n_layers) ───────────────────────────
        cfg = _read_config(self.model_path)
        q_cfg = cfg.get("quantization", {})
        quantized = "bits" in q_cfg
        q_bits = int(q_cfg.get("bits", 4)) if quantized else 16
        n_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        if n_layers is None:
            # Infer from weight keys
            idxs = set()
            for k in weights:
                parts = k.split(".")
                if "layers" in parts:
                    idx_pos = parts.index("layers")
                    if idx_pos + 1 < len(parts) and parts[idx_pos + 1].isdigit():
                        idxs.add(int(parts[idx_pos + 1]))
            n_layers = max(idxs) + 1 if idxs else 0

        _progress(
            f"{n_layers} layers  |  "
            f"{'quantized ' + str(q_bits) + '-bit' if quantized else 'fp16/bf16'}"
        )

        # ── 3. scan layers ────────────────────────────────────────────────────
        try:
            import mlx.core as mx
            _has_mlx = True
        except ImportError:
            _has_mlx = False

        analyses: list[LayerAnalysis] = []

        for layer_idx in range(n_layers):
            for ptype in proj_types:
                wkey = f"model.layers.{layer_idx}.mlp.{ptype}.weight"
                if wkey not in weights:
                    # Try without "model." prefix
                    wkey = f"layers.{layer_idx}.mlp.{ptype}.weight"
                if wkey not in weights:
                    continue

                w_raw = weights[wkey]

                # Dequantize if needed
                if quantized and _has_mlx and str(w_raw.dtype) == "mlx.core.uint32":
                    skey = wkey.replace(".weight", ".scales")
                    bkey = wkey.replace(".weight", ".biases")
                    scales = weights.get(skey)
                    biases = weights.get(bkey)
                    if scales is not None:
                        w_fp = mx.dequantize(
                            w_raw, scales=scales, biases=biases, bits=q_bits
                        )
                        mx.eval(w_fp)
                        arr = np.array(w_fp.astype(mx.float32))
                    else:
                        # Fallback: treat packed uint32 as float — not ideal
                        arr = np.array(w_raw).view(np.float32)
                else:
                    # Float16/bfloat16 — convert to float32 for SVD
                    if _has_mlx:
                        import mlx.core as mx
                        arr = np.array(w_raw.astype(mx.float32))
                    else:
                        arr = np.array(w_raw).astype(np.float32)

                top_sv, gap, er, frob = _svd_metrics(arr, top_k=top_k_sv, max_rows=max_rows)
                analyses.append(
                    LayerAnalysis(
                        layer_idx=layer_idx,
                        proj_type=ptype,
                        shape=tuple(arr.shape),
                        top_sv=top_sv,
                        spectral_gap=gap,
                        effective_rank=er,
                        frob_norm=frob,
                    )
                )

            if layer_idx % 7 == 6 or layer_idx == n_layers - 1:
                _progress(f"  scanned layers 0–{layer_idx}")

        elapsed = time.perf_counter() - t0
        return WeightMap(
            model_path=str(self.model_path),
            n_layers=n_layers,
            quantized=quantized,
            quantization_bits=q_bits,
            elapsed_s=elapsed,
            analyses=analyses,
        )
