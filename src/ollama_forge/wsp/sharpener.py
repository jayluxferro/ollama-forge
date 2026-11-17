"""Phase 3: Surgical weight editing — zero obstruction-circuit directions.

Takes a WeightMap from Phase 2 and modifies the top-ranked MLP projection
layers by removing their dominant singular directions.  The edit is minimal
by design: zeroing the top-1 or top-2 directions of a few layers typically
touches ≪ 0.1% of total parameters yet can unlock significant latent capability
(cf. Takehi et al. 2026: 7.9% → 48.1% on MATH-500 by editing 0.016% of
Qwen3.5-0.8B weights).

Output: a new directory with:
  model.safetensors  — modified weights (float16; original is not overwritten)
  config.json        — copied from source (quantization entry removed)
  tokenizer*         — all tokenizer files copied from source

Usage:
  sharpen = Sharpener(model_path)
  result  = sharpen.sharpen(weight_map, obstruction="loops", n_layers=3)
  print(result.report())
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class EditRecord:
    layer_idx: int
    proj_type: str
    n_directions: int
    params_edited: int
    total_params: int
    delta_frob: float   # ||W_new - W|| / ||W||

    @property
    def edit_fraction(self) -> float:
        return self.params_edited / max(self.total_params, 1)


@dataclass
class SharpenResult:
    source_path: str
    output_path: str
    obstruction: str
    edits: list[EditRecord] = field(default_factory=list)
    total_params: int = 0
    total_edited: int = 0
    elapsed_s: float = 0.0

    @property
    def edit_fraction(self) -> float:
        return self.total_edited / max(self.total_params, 1)

    def report(self) -> str:
        lines = [
            f"Sharpen result — {Path(self.source_path).name}",
            f"  Obstruction target: {self.obstruction}",
            f"  Output: {self.output_path}",
            f"  Elapsed: {self.elapsed_s:.1f}s",
            "",
            f"  Total params : {self.total_params:,}",
            f"  Params edited: {self.total_edited:,}  ({self.edit_fraction:.4%})",
            "",
            "  Layers edited:",
        ]
        for e in self.edits:
            lines.append(
                f"    layer {e.layer_idx:3d}  {e.proj_type:12s}  "
                f"zeroed {e.n_directions} direction(s)  "
                f"Δfrob={e.delta_frob:.4f}  "
                f"({e.edit_fraction:.4%} of layer)"
            )
        lines += [
            "",
            "Load the sharpened model with mlx_lm:",
            f"  mlx_lm.generate --model {self.output_path} --prompt '...'",
            "",
            "Or serve it with forge api:",
            f"  forge api --mlx-models {self.output_path}:sharpened",
        ]
        return "\n".join(lines)


# ── randomised SVD ────────────────────────────────────────────────────────────

def _randomized_svd(
    A: np.ndarray,
    n_components: int,
    n_iter: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast approximate top-k SVD via randomised range finding.

    Returns (U, s, Vh) with shapes (m, k), (k,), (k, n).
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    k = min(n_components + 10, min(m, n))

    # Normalize before power iteration: prevents float64 overflow on large
    # weight matrices (e.g. 8960×1536 with sv[0] ≈ 300 → ||A||² ≈ 1e9).
    # Singular values scale linearly, so we rescale them back below.
    scale = float(np.linalg.norm(A, "fro"))
    A_n = A / scale if scale > 0 else A

    # Suppress spurious FPU exception warnings from Apple Accelerate BLAS on
    # large (n > 4096) float64 matrices — the results are always finite.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Random projection to find approximate column range
        Omega = rng.standard_normal((n, k))
        Q = A_n @ Omega
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(Q)
            Q = A_n @ (A_n.T @ Q)
        Q, _ = np.linalg.qr(Q)

        # SVD of the small projected matrix
        B = Q.T @ A_n                              # (k, n)
        U_B, s, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_B                                # (m, k)
    s = s * scale                                  # rescale back to original magnitude

    nc = min(n_components, len(s))
    return U[:, :nc], s[:nc], Vh[:nc, :]


# ── core edit ─────────────────────────────────────────────────────────────────

def _zero_top_directions(
    W: np.ndarray,
    n_directions: int,
    sparse_fraction: float = 0.0,
) -> tuple[np.ndarray, int, float]:
    """Remove the top-n singular directions from W.

    If sparse_fraction > 0, only zero the values whose contribution to the
    dominant direction exceeds the top (1 - sparse_fraction) quantile, keeping
    the total edit budget small (similar to the original paper's approach of
    zeroing ~65k values per layer).  sparse_fraction=0 (default) removes the
    full rank-k component (more aggressive; faster to compute).

    Returns (W_modified, n_params_changed, delta_frob_ratio).
    """
    U, s, Vh = _randomized_svd(W, n_components=n_directions)

    W64 = W.astype(np.float64)
    removal = np.zeros_like(W64)
    for i in range(len(s)):
        component = s[i] * np.outer(U[:, i], Vh[i, :])
        if sparse_fraction > 0.0:
            # Keep only the top (1-sparse_fraction) largest-magnitude elements
            thresh = np.quantile(np.abs(component), 1.0 - sparse_fraction)
            component = np.where(np.abs(component) >= thresh, component, 0.0)
        removal += component

    W_new = (W64 - removal).astype(np.float16)
    W_fp16 = W.astype(np.float16)

    changed = int(np.sum(W_new != W_fp16))
    frob_w = float(np.linalg.norm(W64, "fro"))
    frob_delta = float(np.linalg.norm(W_new.astype(np.float64) - W64, "fro"))
    delta_ratio = frob_delta / frob_w if frob_w > 0 else 0.0

    return W_new, changed, delta_ratio


# ── file helpers ──────────────────────────────────────────────────────────────

def _total_param_count(weights: dict) -> int:
    total = 0
    for v in weights.values():
        try:
            import mlx.core as mx
            if isinstance(v, mx.array):
                total += int(np.prod(list(v.shape)))
                continue
        except ImportError:
            pass
        total += int(np.prod(list(v.shape)))
    return total


def _copy_support_files(src: Path, dst: Path) -> None:
    """Copy config.json and all tokenizer files."""
    cfg_src = src / "config.json"
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text())
        (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    # Tokenizer and other non-weight files
    skip_exts = {".safetensors", ".npz", ".gguf", ".npy"}
    for f in src.iterdir():
        if f.suffix in skip_exts:
            continue
        if f.name == "config.json":
            continue  # already handled
        if f.is_file():
            shutil.copy2(f, dst / f.name)


# ── Sharpener ─────────────────────────────────────────────────────────────────

class Sharpener:
    """Apply surgical weight edits to a safetensors model."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def sharpen(
        self,
        weight_map,
        obstruction: str = "loops",
        n_layers: int = 3,
        n_directions: int = 1,
        proj_type: str = "down_proj",
        sparse_fraction: float = 0.0,
        output_path: str | Path | None = None,
        progress_fn: Callable[[str], None] | None = None,
    ) -> SharpenResult:
        """Edit the top-n_layers candidate layers and write a new model.

        Args:
            weight_map:       WeightMap from Phase 2 (or dict with same interface).
            obstruction:      Which obstruction type to target.
            n_layers:         How many candidate layers to edit (top-ranked first).
            n_directions:     How many singular directions to zero per layer.
            proj_type:        Which MLP projection to edit (default: down_proj).
            sparse_fraction:  If >0, only zero the top fraction of values aligned
                              with the dominant direction (keeps edit budget tiny;
                              0.005 ≈ 65k values per layer, matching the paper).
            output_path:   Where to write the sharpened model directory.
                           Defaults to <source_name>-sharp/ beside source.
            progress_fn:   Optional callback(message).
        """
        try:
            import mlx.core as mx
            _has_mlx = True
        except ImportError:
            _has_mlx = False
            raise RuntimeError("mlx is required — install with: pip install mlx") from None

        def _progress(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)

        t0 = time.perf_counter()

        # ── 1. determine output path ──────────────────────────────────────────
        if output_path is None:
            output_path = self.model_path.parent / (self.model_path.name + "-sharp")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # ── 2. load source weights ────────────────────────────────────────────
        from ollama_forge.wsp.analyzer import _find_safetensors_files, _load_all_weights, _read_config

        files = _find_safetensors_files(self.model_path)
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {self.model_path}")

        _progress(f"loading {len(files)} shard(s)...")
        weights = _load_all_weights(files)

        cfg = _read_config(self.model_path)
        q_cfg = cfg.get("quantization", {})
        quantized = "bits" in q_cfg
        q_bits = int(q_cfg.get("bits", 4)) if quantized else 16

        total_params = _total_param_count(weights)

        # ── 3. identify candidate layers ──────────────────────────────────────
        candidates = weight_map.top_candidates(
            obstruction=obstruction,
            proj_type=proj_type,
            n=n_layers,
        )
        if not candidates:
            raise ValueError(
                f"No candidates found for obstruction={obstruction!r}, "
                f"proj_type={proj_type!r}"
            )

        _progress(
            f"targeting {len(candidates)} layer(s): "
            + ", ".join(f"{c.layer_idx}({c.proj_type})" for c in candidates)
        )

        # ── 4. convert all weights to a mutable dict (fp16 mlx arrays) ───────
        # We'll build a new weights dict. Unedited layers stay quantized.
        new_weights: dict = {}
        for k, v in weights.items():
            new_weights[k] = v  # keep original by default

        edits: list[EditRecord] = []
        total_edited = 0

        # ── 5. apply edits ────────────────────────────────────────────────────
        for cand in candidates:
            li = cand.layer_idx
            pt = cand.proj_type
            wkey = f"model.layers.{li}.mlp.{pt}.weight"

            if wkey not in weights:
                _progress(f"  skip layer {li} {pt} — key not found")
                continue

            w_raw = weights[wkey]
            _progress(f"  editing layer {li} {pt}  ({n_directions} direction(s))...")

            # dequantize if needed
            if quantized and str(w_raw.dtype) == "mlx.core.uint32":
                skey = wkey.replace(".weight", ".scales")
                bkey = wkey.replace(".weight", ".biases")
                scales = weights.get(skey)
                biases = weights.get(bkey)
                w_fp = mx.dequantize(w_raw, scales=scales, biases=biases, bits=q_bits)
                mx.eval(w_fp)
                arr = np.array(w_fp.astype(mx.float32))
            else:
                arr = np.array(w_raw.astype(mx.float32))

            # zero top directions
            arr_fp16, n_changed, delta_frob = _zero_top_directions(arr, n_directions, sparse_fraction)
            total_edited += n_changed

            if quantized:
                # Re-quantize edited layer so the output model stays in the same
                # format as the input (uint32 + scales + biases).  mlx_lm can
                # then load the output directly with the original quantization config.
                w_edit_mx = mx.array(arr_fp16.astype(np.float32))
                mx.eval(w_edit_mx)
                q_group = int(q_cfg.get("group_size", 64))
                w_q, new_scales, new_biases = mx.quantize(w_edit_mx, group_size=q_group, bits=q_bits)
                mx.eval(w_q, new_scales, new_biases)
                skey = wkey.replace(".weight", ".scales")
                bkey = wkey.replace(".weight", ".biases")
                new_weights[wkey] = w_q
                new_weights[skey] = new_scales
                new_weights[bkey] = new_biases
            else:
                # Non-quantized model: store as fp16, no scales/biases
                new_weights[wkey] = mx.array(arr_fp16)

            edits.append(EditRecord(
                layer_idx=li,
                proj_type=pt,
                n_directions=n_directions,
                params_edited=n_changed,
                total_params=arr.size,
                delta_frob=delta_frob,
            ))

        # ── 6. save new weights ───────────────────────────────────────────────
        _progress("saving modified weights...")
        # Evaluate all arrays before saving
        mx.eval(*[v for v in new_weights.values() if hasattr(v, "shape")])
        out_weights = str(output_path / "model.safetensors")
        mx.save_safetensors(out_weights, new_weights)

        # ── 7. copy support files ─────────────────────────────────────────────
        _copy_support_files(self.model_path, output_path)
        _progress(f"saved to {output_path}")

        elapsed = time.perf_counter() - t0
        return SharpenResult(
            source_path=str(self.model_path),
            output_path=str(output_path),
            obstruction=obstruction,
            edits=edits,
            total_params=total_params,
            total_edited=total_edited,
            elapsed_s=elapsed,
        )
