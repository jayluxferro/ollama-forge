"""Phase 5: Gradient-Guided Correction (GGC).

Extends Phase 4 BCC with a direct logit-projection term derived from the
LM head weight matrix:

    d_target = mean_i( W_lm[correct_tok_i] − W_lm[predicted_tok_i] )

This is the direction in residual stream space that, when added to the
down_proj output, directly increases the logit for the correct answer token
relative to the predicted (wrong) answer token.

Combined rank-1 correction at each attributed layer l:

    v = −α · (W_l @ d_k_norm) + β · d_target_norm
    W'_l = W_l + outer(v, d_k_norm)

The first term (BCC, Phase 4) dampens W's amplification of the failure-
trigger input direction.  The second term (Phase 5) adds a push toward the
correct answer whenever the failure trigger fires.

α = 0  → pure logit-projection (no BCC suppression)
β = 0  → pure BCC (reduces to Phase 4)
Both   → combined correction (default)

Relationship to ROME/MEMIT: ROME builds d_target from a gradient of a
specific factual memory.  GGC builds it from the LM head's first-order
Taylor approximation, requiring only a matrix lookup — no backward pass.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ollama_forge.wsp.corrector import (
    _PROBE_BANK,
    ProbeResult,
    attribute_failure_layers,
    run_mlx_probe,
)

# ── LM head extraction ────────────────────────────────────────────────────────

def _get_lm_head_fp32(weights: dict) -> np.ndarray | None:
    """Return the LM head weight matrix as float32, shape (vocab_size, d_model).

    Checks lm_head.weight first (some models have an unshared LM head), then
    falls back to model.embed_tokens.weight (tied embeddings, Qwen2 default).
    Dequantizes if the matrix is stored as int4.
    """
    import mlx.core as mx

    for key_prefix in ("lm_head", "model.embed_tokens"):
        wkey = f"{key_prefix}.weight"
        if wkey not in weights:
            continue
        w = weights[wkey]
        if str(w.dtype) == "mlx.core.uint32":
            skey = wkey.replace(".weight", ".scales")
            bkey = wkey.replace(".weight", ".biases")
            if skey in weights and bkey in weights:
                w = mx.dequantize(w, scales=weights[skey], biases=weights[bkey], bits=4)
        mx.eval(w)
        return np.array(w.astype(mx.float32))

    return None


# ── target direction computation ──────────────────────────────────────────────

def compute_lm_head_direction(
    probe_result: ProbeResult,
    tokenizer,
    weights: dict,
) -> np.ndarray | None:
    """Compute d_target: the residual-stream direction that promotes correct tokens.

    For each failure, looks up:
      correct_tok  = first token of probe_result.fail_expected[i]
      predicted_tok = first token of probe_result.fail_responses[i]

    If correct_tok == predicted_tok the example is skipped.

    Returns the mean of (W_lm[correct_tok] - W_lm[predicted_tok]) over valid
    failures, or None if no valid failures exist.
    """
    if not probe_result.fail_expected or not probe_result.fail_responses:
        return None

    W_lm = _get_lm_head_fp32(weights)
    if W_lm is None:
        return None

    vocab_size = W_lm.shape[0]

    def _first_token(text: str) -> int | None:
        """Return first real (non-whitespace) token ID for text."""
        text = text.strip().lower()
        if not text:
            return None
        # Take only the first word to avoid multi-token encoding confusion
        first_word = re.match(r"[\w$.]+", text)
        text = first_word.group(0) if first_word else text[:8]
        try:
            toks = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return None
        return toks[0] if toks else None

    directions: list[np.ndarray] = []

    for expected, response in zip(probe_result.fail_expected, probe_result.fail_responses, strict=False):
        c_tok = _first_token(expected)
        p_tok = _first_token(response)
        if c_tok is None or p_tok is None:
            continue
        if c_tok == p_tok:
            continue
        if c_tok >= vocab_size or p_tok >= vocab_size:
            continue
        d = W_lm[c_tok].astype(np.float64) - W_lm[p_tok].astype(np.float64)
        directions.append(d)

    if not directions:
        return None

    return np.mean(np.stack(directions), axis=0).astype(np.float32)


# ── correction kernel ─────────────────────────────────────────────────────────

def gradient_rank1_correction(
    W: np.ndarray,
    d_k: np.ndarray,
    d_target: np.ndarray | None,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, int, float]:
    """Combined rank-1 correction: BCC suppression + LM head logit projection.

        v = −α · (W @ d_k_norm) + β · d_target_norm
        W′ = W + outer(v, d_k_norm)

    When d_target is None or beta=0 this reduces to pure BCC (Phase 4).
    When alpha=0 it is a pure additive logit-projection correction.

    Returns (W_corrected_fp16, n_params_changed, delta_frob_ratio).
    """
    W64 = W.astype(np.float64)

    d_k_norm_val = np.linalg.norm(d_k) + 1e-12
    d_k_fp64 = (d_k / d_k_norm_val).astype(np.float64)

    # BCC component: suppress failure direction
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v_bcc = -alpha * (W64 @ d_k_fp64)  # (d_model,)

    # LM-head projection component: promote correct answer
    if d_target is not None and beta != 0.0:
        d_t_norm = np.linalg.norm(d_target) + 1e-12
        d_target_fp64 = (d_target / d_t_norm).astype(np.float64)
        v_logit = beta * d_target_fp64        # (d_model,)
    else:
        v_logit = np.zeros(W64.shape[0], dtype=np.float64)

    v = v_bcc + v_logit                       # (d_model,)
    delta = np.outer(v, d_k_fp64)             # (d_model, d_mlp)

    W_new = (W64 + delta).astype(np.float16)
    W_fp16 = W64.astype(np.float16)

    changed = int(np.sum(W_new != W_fp16))
    frob_w = float(np.linalg.norm(W64, "fro"))
    frob_delta = float(np.linalg.norm(W_new.astype(np.float64) - W64, "fro"))
    delta_ratio = frob_delta / frob_w if frob_w > 0 else 0.0

    return W_new, changed, delta_ratio


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class GGCRecord:
    layer_idx: int
    proj_type: str
    alpha: float
    beta: float
    d_k_norm: float
    d_target_norm: float    # 0.0 if LM head direction unavailable
    delta_frob: float
    params_edited: int
    total_params: int

    @property
    def edit_fraction(self) -> float:
        return self.params_edited / max(self.total_params, 1)


@dataclass
class GGCResult:
    source_path: str
    output_path: str
    n_fail: int
    n_pass: int
    n_target_dirs: int    # how many fail examples contributed to d_target
    attributions: list = field(default_factory=list)
    corrections: list[GGCRecord] = field(default_factory=list)
    elapsed_s: float = 0.0

    def report(self) -> str:
        lines = [
            f"GGC result — {Path(self.source_path).name}",
            f"  Probe:  {self.n_fail} fail  |  {self.n_pass} pass",
            f"  LM-head direction: computed from {self.n_target_dirs} failure pair(s)",
            f"  Output: {self.output_path}",
            f"  Elapsed: {self.elapsed_s:.1f}s",
            "",
            "  Attribution scores (top layers):",
        ]
        for a in sorted(self.attributions, key=lambda x: -x.response_magnitude)[:5]:
            lines.append(
                f"    layer {a.layer_idx:3d}  {a.proj_type:12s}  "
                f"response={a.response_magnitude:.4f}  signal={a.d_k_norm:.4f}"
            )
        lines += ["", "  Corrections applied:"]
        for c in self.corrections:
            lines.append(
                f"    layer {c.layer_idx:3d}  {c.proj_type:12s}  "
                f"α={c.alpha:.3f}  β={c.beta:.3f}  "
                f"|d_target|={c.d_target_norm:.3f}  "
                f"Δfrob={c.delta_frob:.4f}  "
                f"({c.edit_fraction:.4%} of layer)"
            )
        total_p = sum(c.total_params for c in self.corrections)
        total_e = sum(c.params_edited for c in self.corrections)
        if total_p:
            lines += [
                "",
                f"  Total params edited: {total_e:,} / {total_p:,}  "
                f"({total_e/total_p:.4%})",
            ]
        return "\n".join(lines)


# ── GradientCorrector ─────────────────────────────────────────────────────────

class GradientCorrector:
    """Apply Phase 5 Gradient-Guided Correction (GGC) to a safetensors model.

    Extends BCC (Phase 4) with an LM head logit-projection term that directly
    targets the residual stream direction associated with the correct answer.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def correct(
        self,
        probe_bank: list[tuple[str, str]] | None = None,
        n_layers: int = 2,
        proj_type: str = "down_proj",
        alpha: float = 0.1,
        beta: float = 0.3,
        min_layer: int = 20,
        output_path: str | Path | None = None,
        progress_fn: Callable[[str], None] | None = None,
    ) -> GGCResult:
        """Run GGC: probe → attribute → gradient-correct → save.

        Args:
            probe_bank:   (prompt, expected) pairs. Defaults to built-in bank.
            n_layers:     how many top-attributed layers to correct (default: 2)
            proj_type:    which MLP projection to edit (default: down_proj)
            alpha:        BCC suppression strength 0–1 (default: 0.1)
            beta:         LM head logit-projection strength (default: 0.3).
                          Set to 0 to get pure BCC.  Set alpha=0 for pure GGC.
            min_layer:    skip layers below this index (default: 20).
                          Phase 5 targets late layers where the LM head signal
                          propagates most directly.
            output_path:  where to save the corrected model
            progress_fn:  optional progress callback
        """
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise RuntimeError("mlx_lm required — pip install mlx-lm") from None

        def _prog(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)

        t0 = time.perf_counter()

        if probe_bank is None:
            probe_bank = _PROBE_BANK

        # ── 1. output path ────────────────────────────────────────────────────
        if output_path is None:
            output_path = self.model_path.parent / (self.model_path.name + "-ggc")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # ── 2. load model ─────────────────────────────────────────────────────
        _prog("loading model for probing…")
        model, tokenizer = mlx_load(str(self.model_path))

        cfg_path = self.model_path / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        n_model_layers = cfg.get("num_hidden_layers", 28)
        q_cfg = cfg.get("quantization", {})
        quantized = "bits" in q_cfg
        q_bits = int(q_cfg.get("bits", 4)) if quantized else 16
        q_group = int(q_cfg.get("group_size", 64))

        # ── 3. probe ──────────────────────────────────────────────────────────
        _prog(f"probing {len(probe_bank)} examples across {n_model_layers} layers…")
        probe_result = run_mlx_probe(
            model=model,
            tokenizer=tokenizer,
            probe_bank=probe_bank,
            layer_indices=list(range(n_model_layers)),
            max_tokens=80,
            progress_fn=_prog,
        )

        if probe_result.n_fail < 2:
            raise ValueError(
                f"Only {probe_result.n_fail} failures — need ≥2.  "
                "Model already handles all probe examples, or expected answers didn't match."
            )
        if probe_result.n_pass < 2:
            raise ValueError(f"Only {probe_result.n_pass} passes — need ≥2.")

        _prog(
            f"probe: {probe_result.n_fail} fail  {probe_result.n_pass} pass  "
            f"(out of {len(probe_bank)})"
        )

        # ── 4. load weights ───────────────────────────────────────────────────
        from ollama_forge.wsp.analyzer import _find_safetensors_files, _load_all_weights

        files = _find_safetensors_files(self.model_path)
        if not files:
            raise FileNotFoundError(f"No .safetensors in {self.model_path}")

        _prog(f"loading {len(files)} shard(s)…")
        weights = _load_all_weights(files)

        # ── 5. compute LM head direction ──────────────────────────────────────
        _prog("computing LM head target direction…")
        d_target = compute_lm_head_direction(probe_result, tokenizer, weights)
        n_target_dirs = len(probe_result.fail_expected)

        if d_target is not None:
            d_target_norm_val = float(np.linalg.norm(d_target))
            _prog(
                f"d_target computed from {n_target_dirs} failure(s)  "
                f"||d_target||={d_target_norm_val:.4f}"
            )
        else:
            d_target_norm_val = 0.0
            _prog("LM head direction unavailable — falling back to pure BCC (β=0)")

        # ── 6. attribute ──────────────────────────────────────────────────────
        _prog(f"attributing failure layers (min_layer={min_layer})…")
        attributions = attribute_failure_layers(
            probe_result=probe_result,
            weights=weights,
            n_layers=n_model_layers,
            proj_type=proj_type,
            top_k=max(n_layers, 5),
            min_layer=min_layer,
        )

        candidates = attributions[:n_layers]
        _prog(
            "top attributed layers: "
            + ", ".join(f"{a.layer_idx}(response={a.response_magnitude:.3f})" for a in candidates)
        )

        # ── 7. apply combined rank-1 corrections ──────────────────────────────
        import mlx.core as mx

        new_weights: dict = dict(weights)
        corrections: list[GGCRecord] = []

        for attr in candidates:
            li = attr.layer_idx
            wkey = f"model.layers.{li}.mlp.{proj_type}.weight"
            if wkey not in weights:
                _prog(f"  skip layer {li} — key not found")
                continue

            d_k = probe_result.h_fail_mean[li] - probe_result.h_pass_mean[li]
            w_raw = weights[wkey]

            if quantized and str(w_raw.dtype) == "mlx.core.uint32":
                skey = wkey.replace(".weight", ".scales")
                bkey = wkey.replace(".weight", ".biases")
                w_fp = mx.dequantize(
                    w_raw,
                    scales=weights.get(skey),
                    biases=weights.get(bkey),
                    bits=q_bits,
                )
                mx.eval(w_fp)
                arr = np.array(w_fp.astype(mx.float32))
            else:
                arr = np.array(w_raw.astype(mx.float32))

            _prog(f"  correcting layer {li} {proj_type}  (α={alpha}, β={beta})…")
            arr_fp16, n_changed, delta_frob = gradient_rank1_correction(
                arr, d_k, d_target, alpha=alpha, beta=beta
            )

            if quantized:
                w_edit = mx.array(arr_fp16.astype(np.float32))
                mx.eval(w_edit)
                w_q, new_scales, new_biases = mx.quantize(w_edit, group_size=q_group, bits=q_bits)
                mx.eval(w_q, new_scales, new_biases)
                skey = wkey.replace(".weight", ".scales")
                bkey = wkey.replace(".weight", ".biases")
                new_weights[wkey] = w_q
                new_weights[skey] = new_scales
                new_weights[bkey] = new_biases
            else:
                new_weights[wkey] = mx.array(arr_fp16)

            corrections.append(GGCRecord(
                layer_idx=li,
                proj_type=proj_type,
                alpha=alpha,
                beta=beta,
                d_k_norm=float(np.linalg.norm(d_k)),
                d_target_norm=d_target_norm_val,
                delta_frob=delta_frob,
                params_edited=n_changed,
                total_params=arr.size,
            ))

        # ── 8. save ───────────────────────────────────────────────────────────
        _prog("saving corrected weights…")
        mx.eval(*[v for v in new_weights.values() if hasattr(v, "shape")])
        mx.save_safetensors(str(output_path / "model.safetensors"), new_weights)

        skip_exts = {".safetensors", ".npz", ".gguf", ".npy"}
        for f in self.model_path.iterdir():
            if f.suffix in skip_exts or not f.is_file():
                continue
            shutil.copy2(f, output_path / f.name)

        _prog(f"saved → {output_path}")

        elapsed = time.perf_counter() - t0
        return GGCResult(
            source_path=str(self.model_path),
            output_path=str(output_path),
            n_fail=probe_result.n_fail,
            n_pass=probe_result.n_pass,
            n_target_dirs=n_target_dirs,
            attributions=attributions,
            corrections=corrections,
            elapsed_s=elapsed,
        )
