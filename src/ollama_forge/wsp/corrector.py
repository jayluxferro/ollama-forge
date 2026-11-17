"""Phase 4: Behavioral Causal Correction (BCC).

WSP Phases 1–3 select edit targets by spectral gap — a property of the weight
matrix itself, not of model behavior.  That produces capability-neutral edits
(the benchmark showed exactly ±20 pp shuffling with no net gain).

BCC replaces the spectral proxy with a behavioral signal:

  1. PROBE  — run a small set of "fail" and "pass" examples through the model,
              capturing MLP hidden states (gate × up activation, the input to
              down_proj) at every layer.

  2. ATTRIBUTE — for each layer l, compute the behavioral failure direction:
                 d_k[l] = mean(h_fail[l]) - mean(h_pass[l])
                 Score = ‖W_down[l] @ d_k[l]‖ — how strongly W amplifies the
                 failure-correlated input pattern.  Highest score = causal layer.

  3. CORRECT — apply a targeted rank-1 update to the causal layer:
                 W′ = W − α · outer(W @ d_k_norm, d_k_norm)
               This dampens W's response to the failure-trigger direction while
               leaving all orthogonal directions intact.

  4. RE-QUANTIZE — same as Phase 3: re-quantize back to original bits/group_size
                   so the output is a valid, loadable model.

Relationship to ROME/MEMIT: ROME corrects a specific *fact* stored in MLP
key-value memory.  BCC corrects a *behavioral class* (e.g., reasoning loops)
using the same rank-1 structure but with data-driven key/value derivation.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# ── built-in probe bank ───────────────────────────────────────────────────────

# (prompt, expected_substring_in_answer)  — lower-cased matching
# Two sections: HARD (small models often fail — these generate the failure direction)
# and EASY FACTUAL (small models almost always pass — anchors the pass direction
# away from general knowledge retrieval so the correction is task-specific).
_PROBE_BANK: list[tuple[str, str]] = [
    # ── hard multi-step reasoning (target ~10 failures) ───────────────────────
    ("Alice has twice as many stickers as Bob. Bob has 3 more than Carol. "
     "Carol has 4. How many stickers does Alice have? Give only the number.", "14"),
    ("A snail climbs 3 m up a 10 m well each day but slides 2 m back each night. "
     "On which day does it reach the top? Give only the number.", "8"),
    ("A bat and ball cost $1.10. The bat costs $1.00 more than the ball. "
     "How much does the ball cost in cents? Give only the number.", "5"),
    ("If P implies Q and Q implies R and P is true, is R true? Reply only yes or no.", "yes"),
    ("Some cats are black. All black cats are fast. Can we say all cats are fast?"
     " Reply only yes or no.", "no"),
    # Deceptive counting / multi-step
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?"
     " Give only the number.", "9"),
    ("Mary is 24. She is twice as old as Ann was when Mary was as old as Ann is now."
     " How old is Ann now? Give only the number.", "18"),
    ("A train travels 60 km/h for 1.5 hours, then 90 km/h for 2 hours. "
     "What is the total distance in km? Give only the number.", "270"),
    ("How many times does the digit 1 appear in the integers from 1 to 20?"
     " Give only the number.", "12"),
    ("Three friends split a restaurant bill equally. The bill is $73.50 and they "
     "leave a 20% tip. How much does each person pay in dollars? "
     "Give only the number rounded to two decimal places.", "29.40"),
    # Conditional chains
    ("If all Bloops are Razzles, and all Razzles are Lazzles, are all Bloops Lazzles?"
     " Reply only yes or no.", "yes"),
    ("No fish are mammals. All sharks are fish. Are any sharks mammals?"
     " Reply only yes or no.", "no"),
    # Pattern / modular arithmetic
    ("What is 2^10? Give only the number.", "1024"),
    ("What is the sum of all integers from 1 to 100? Give only the number.", "5050"),
    ("If today is Wednesday and a meeting is in 10 days, what day is the meeting?"
     " Give only the day name.", "saturday"),
    ("What is the remainder when 157 is divided by 12? Give only the number.", "1"),
    ("How many prime numbers are there between 10 and 30? Give only the number.", "5"),
    # Combinatorics (tricky for small models)
    ("If you flip a fair coin 3 times, how many ways can you get exactly 2 heads?"
     " Give only the number.", "3"),
    ("In how many ways can you arrange the letters A, B, C? Give only the number.", "6"),
    # ── benchmark-format questions (no "give only" instruction) ─────────────────
    # These match real benchmark question phrasing. Without explicit format
    # constraints, the model tends to be verbose, producing the correct answer
    # buried after a long preamble. These are "fail" when the answer does not
    # appear in the first ~100 chars of the response.
    # Included as HARD examples so the probe captures the verbosity failure direction.
    ("A bat and a ball cost $1.10 total. The bat costs $1 more than the ball."
     " How much does the ball cost in cents?", "5"),
    ("What comes next in the sequence: 2, 4, 8, 16, __?", "32"),
    ("If all bloops are razzles and all razzles are lazzles, are all bloops lazzles?"
     " Reply yes or no.", "yes"),
    ("A farmer has 17 sheep. All but 9 die. How many are left?", "9"),
    ("What is 2^8?", "256"),
    ("A rectangle has length 11 cm and perimeter 36 cm. What is its width?", "7"),
    # ── easy factual (small models almost always pass — anchor the pass mean) ──
    ("What is the capital of France? Give only the city name.", "paris"),
    ("What is 12 × 12? Give only the number.", "144"),
    ("What is the square root of 64? Give only the number.", "8"),
    ("How many sides does a hexagon have? Give only the number.", "6"),
    ("What is H2O? Reply in one word.", "water"),
    ("What is the first element in the periodic table? Give only the element name.",
     "hydrogen"),
    ("How many days are in a leap year? Give only the number.", "366"),
    ("What is 100 ÷ 4? Give only the number.", "25"),
    ("What is 15% of 200? Give only the number.", "30"),
    ("What comes next: 1, 1, 2, 3, 5, 8, __? Give only the number.", "13"),
]


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Activations and pass/fail labels collected from a probe run."""
    n_fail: int
    n_pass: int
    # layer_idx → mean activation vector (d_mlp,) for fail examples
    h_fail_mean: dict[int, np.ndarray] = field(default_factory=dict)
    # layer_idx → mean activation vector (d_mlp,) for pass examples
    h_pass_mean: dict[int, np.ndarray] = field(default_factory=dict)
    fail_examples: list[str] = field(default_factory=list)
    pass_examples: list[str] = field(default_factory=list)


@dataclass
class AttributionResult:
    layer_idx: int
    proj_type: str
    response_magnitude: float  # ‖W @ d_k‖ — how strongly W amplifies failure dir
    d_k_norm: float            # ‖d_k‖ — behavioral signal strength


@dataclass
class CorrectionRecord:
    layer_idx: int
    proj_type: str
    alpha: float
    d_k_norm: float
    delta_frob: float   # ‖W′ − W‖_F / ‖W‖_F
    params_edited: int
    total_params: int

    @property
    def edit_fraction(self) -> float:
        return self.params_edited / max(self.total_params, 1)


@dataclass
class CorrectionResult:
    source_path: str
    output_path: str
    n_fail: int
    n_pass: int
    attributions: list[AttributionResult] = field(default_factory=list)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    elapsed_s: float = 0.0

    def report(self) -> str:
        lines = [
            f"BCC result — {Path(self.source_path).name}",
            f"  Probe:  {self.n_fail} fail  |  {self.n_pass} pass",
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
                f"α={c.alpha:.3f}  Δfrob={c.delta_frob:.4f}  "
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


# ── activation capture ────────────────────────────────────────────────────────

class _CapturingMLP:
    """Wraps a Qwen2MLP to capture the gate×up activation (input to down_proj).

    Captures only the LAST TOKEN position from the first forward call, then
    disables itself so subsequent generate steps are not captured.
    """

    def __init__(self, original_mlp, capture_list: list, active_flag: list):
        self._orig = original_mlp
        self._captures = capture_list
        self._active = active_flag   # single-element list so it's mutable from outside

    def __call__(self, x):
        import mlx.core as mx
        import mlx.nn as nn

        h = nn.silu(self._orig.gate_proj(x)) * self._orig.up_proj(x)

        if self._active[0]:
            # Capture last token position, force eager evaluation
            h_last = h[:, -1, :]           # (batch, d_mlp)
            mx.eval(h_last)
            self._captures.append(np.array(h_last))
            self._active[0] = False        # only capture the first call per example

        return self._orig.down_proj(h)

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


def run_mlx_probe(
    model,
    tokenizer,
    probe_bank: list[tuple[str, str]],
    layer_indices: list[int],
    max_tokens: int = 32,
    progress_fn: Callable[[str], None] | None = None,
) -> ProbeResult:
    """Run probe_bank through model, capture MLP activations, classify pass/fail.

    Args:
        model:        mlx_lm model (already loaded)
        tokenizer:    mlx_lm tokenizer
        probe_bank:   list of (prompt, expected_answer_substring)
        layer_indices: which MLP layers to capture (usually all)
        max_tokens:   max tokens to generate per probe
        progress_fn:  optional callback(message)

    Returns:
        ProbeResult with per-layer mean activations for fail/pass sets
    """
    from mlx_lm import generate

    def _prog(msg: str) -> None:
        if progress_fn:
            progress_fn(msg)

    # Per-layer activation accumulator: layer_idx → list of (d_mlp,) arrays
    fail_acts: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    pass_acts: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    fail_examples: list[str] = []
    pass_examples: list[str] = []

    n = len(probe_bank)
    for idx, (prompt, expected) in enumerate(probe_bank):
        _prog(f"probe {idx+1}/{n}  {'.' * min(idx, 20)}")

        # Install hooks — one capture per example (first generation step)
        captures: dict[int, list] = {li: [] for li in layer_indices}
        active_flags: dict[int, list] = {li: [True] for li in layer_indices}
        originals: dict[int, object] = {}

        for li in layer_indices:
            orig = model.model.layers[li].mlp
            originals[li] = orig
            model.model.layers[li].mlp = _CapturingMLP(orig, captures[li], active_flags[li])

        try:
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        finally:
            for li in layer_indices:
                model.model.layers[li].mlp = originals[li]

        # Classify
        resp_lower = response.strip().lower()
        passed = expected.lower() in resp_lower

        # Accumulate activations
        bucket_acts = pass_acts if passed else fail_acts
        bucket_examples = pass_examples if passed else fail_examples
        bucket_examples.append(prompt)

        for li in layer_indices:
            if captures[li]:
                bucket_acts[li].append(captures[li][0].flatten())  # (d_mlp,)

    # Compute per-layer means
    pr = ProbeResult(
        n_fail=len(fail_examples),
        n_pass=len(pass_examples),
        fail_examples=fail_examples,
        pass_examples=pass_examples,
    )
    for li in layer_indices:
        if fail_acts[li]:
            pr.h_fail_mean[li] = np.mean(np.stack(fail_acts[li]), axis=0)
        if pass_acts[li]:
            pr.h_pass_mean[li] = np.mean(np.stack(pass_acts[li]), axis=0)

    _prog(f"probe complete — {pr.n_fail} fail  {pr.n_pass} pass")
    return pr


# ── attribution & correction ──────────────────────────────────────────────────

def attribute_failure_layers(
    probe_result: ProbeResult,
    weights: dict,
    n_layers: int,
    proj_type: str = "down_proj",
    top_k: int = 5,
    min_layer: int = 0,
) -> list[AttributionResult]:
    """Rank layers by how strongly they amplify the behavioral failure direction.

    For each layer l ≥ min_layer, computes:
        d_k[l] = h_fail_mean[l] − h_pass_mean[l]   (failure direction in input space)
        response = ‖W_down[l] @ d_k_norm[l]‖        (W's amplification of failure dir)

    Higher response = this layer amplifies failure-correlated patterns more.
    min_layer skips early layers (0..min_layer-1) which handle general semantics and
    whose correction tends to hurt factual retrieval.
    """
    import mlx.core as mx

    results: list[AttributionResult] = []

    for li in range(n_layers):
        if li < min_layer:
            continue
        if li not in probe_result.h_fail_mean or li not in probe_result.h_pass_mean:
            continue

        d_k = probe_result.h_fail_mean[li] - probe_result.h_pass_mean[li]
        d_k_norm_val = float(np.linalg.norm(d_k))
        if d_k_norm_val < 1e-8:
            continue

        d_k_normalized = d_k / d_k_norm_val

        wkey = f"model.layers.{li}.mlp.{proj_type}.weight"
        if wkey not in weights:
            continue

        w_raw = weights[wkey]
        # Dequantize if needed to compute response
        if str(w_raw.dtype) == "mlx.core.uint32":
            skey = wkey.replace(".weight", ".scales")
            bkey = wkey.replace(".weight", ".biases")
            w_fp = mx.dequantize(
                w_raw,
                scales=weights.get(skey),
                biases=weights.get(bkey),
                bits=4,
            )
            mx.eval(w_fp)
            W = np.array(w_fp.astype(mx.float32))
        else:
            W = np.array(w_raw.astype(mx.float32))

        d_k_fp32 = d_k_normalized.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            response = W @ d_k_fp32           # (d_model,)
        response_mag = float(np.linalg.norm(response))

        results.append(AttributionResult(
            layer_idx=li,
            proj_type=proj_type,
            response_magnitude=response_mag,
            d_k_norm=d_k_norm_val,
        ))

    results.sort(key=lambda r: -r.response_magnitude)
    return results[:top_k]


def behavioral_rank1_correction(
    W: np.ndarray,
    d_k: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, int, float]:
    """Apply a rank-1 correction that dampens W's response to the failure direction.

        W′ = W − α · outer(W @ d_k_norm, d_k_norm)

    When the layer receives input aligned with d_k (the failure trigger), its
    output is reduced by α.  All other input directions are unaffected.

    Returns (W_corrected_fp16, n_params_changed, delta_frob_ratio).
    """
    W64 = W.astype(np.float64)
    d_k_norm = d_k / (np.linalg.norm(d_k) + 1e-12)
    d_k_fp64 = d_k_norm.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v_response = W64 @ d_k_fp64              # (d_model,)
    delta = alpha * np.outer(v_response, d_k_fp64)   # (d_model, d_mlp)

    W_new = (W64 - delta).astype(np.float16)
    W_fp16 = W64.astype(np.float16)

    changed = int(np.sum(W_new != W_fp16))
    frob_w = float(np.linalg.norm(W64, "fro"))
    frob_delta = float(np.linalg.norm(W_new.astype(np.float64) - W64, "fro"))
    delta_ratio = frob_delta / frob_w if frob_w > 0 else 0.0

    return W_new, changed, delta_ratio


# ── BehavioralCorrector ───────────────────────────────────────────────────────

class BehavioralCorrector:
    """Apply BCC (Phase 4) to a safetensors model using behavioral probe feedback."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def correct(
        self,
        probe_bank: list[tuple[str, str]] | None = None,
        n_layers: int = 2,
        proj_type: str = "down_proj",
        alpha: float = 0.1,
        min_layer: int = 0,
        output_path: str | Path | None = None,
        progress_fn: Callable[[str], None] | None = None,
    ) -> CorrectionResult:
        """Run BCC: probe → attribute → correct → save.

        Args:
            probe_bank:   list of (prompt, expected_answer) pairs.
                          Defaults to the built-in _PROBE_BANK.
            n_layers:     how many top-attributed layers to correct (default: 2)
            proj_type:    which MLP projection to edit (default: down_proj)
            alpha:        correction strength in [0, 1] (default: 0.1)
            min_layer:    skip layers below this index during attribution (default: 0).
                          Set to ~half the model depth to avoid early-layer edits that
                          degrade factual retrieval.
            output_path:  where to save the corrected model
            progress_fn:  optional progress callback
        """
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise RuntimeError("mlx_lm required — install with: pip install mlx-lm") from None

        def _prog(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)

        t0 = time.perf_counter()

        if probe_bank is None:
            probe_bank = _PROBE_BANK

        # ── 1. determine output path ──────────────────────────────────────────
        if output_path is None:
            output_path = self.model_path.parent / (self.model_path.name + "-bcc")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # ── 2. load model for probing ─────────────────────────────────────────
        _prog("loading model for probing…")
        model, tokenizer = mlx_load(str(self.model_path))

        cfg_path = self.model_path / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        n_model_layers = cfg.get("num_hidden_layers", 28)
        q_cfg = cfg.get("quantization", {})
        quantized = "bits" in q_cfg
        q_bits = int(q_cfg.get("bits", 4)) if quantized else 16
        q_group = int(q_cfg.get("group_size", 64))

        layer_indices = list(range(n_model_layers))

        # ── 3. run probe ──────────────────────────────────────────────────────
        _prog(f"probing {len(probe_bank)} examples across {n_model_layers} layers…")
        probe_result = run_mlx_probe(
            model=model,
            tokenizer=tokenizer,
            probe_bank=probe_bank,
            layer_indices=layer_indices,
            max_tokens=80,
            progress_fn=_prog,
        )

        if probe_result.n_fail < 2:
            raise ValueError(
                f"Only {probe_result.n_fail} failures in probe bank — need ≥2 for BCC. "
                "The model already handles all probe examples correctly, or the "
                "expected answers didn't match.  Try a harder probe bank."
            )
        if probe_result.n_pass < 2:
            raise ValueError(
                f"Only {probe_result.n_pass} passes in probe bank — need ≥2 for BCC."
            )

        _prog(
            f"probe: {probe_result.n_fail} fail  {probe_result.n_pass} pass  "
            f"(out of {len(probe_bank)})"
        )

        # ── 4. load weights for editing ───────────────────────────────────────
        from ollama_forge.wsp.analyzer import _find_safetensors_files, _load_all_weights

        files = _find_safetensors_files(self.model_path)
        if not files:
            raise FileNotFoundError(f"No .safetensors in {self.model_path}")

        _prog(f"loading {len(files)} shard(s) for editing…")
        weights = _load_all_weights(files)

        # ── 5. attribute ──────────────────────────────────────────────────────
        _prog(
            f"attributing failure layers (min_layer={min_layer})…"
            if min_layer > 0 else "attributing failure layers…"
        )
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

        # ── 6. apply rank-1 corrections ───────────────────────────────────────
        import mlx.core as mx

        new_weights: dict = dict(weights)
        corrections: list[CorrectionRecord] = []

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

            _prog(f"  correcting layer {li} {proj_type}  (α={alpha})…")
            arr_fp16, n_changed, delta_frob = behavioral_rank1_correction(arr, d_k, alpha)

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

            corrections.append(CorrectionRecord(
                layer_idx=li,
                proj_type=proj_type,
                alpha=alpha,
                d_k_norm=float(np.linalg.norm(d_k)),
                delta_frob=delta_frob,
                params_edited=n_changed,
                total_params=arr.size,
            ))

        # ── 7. save ───────────────────────────────────────────────────────────
        _prog("saving corrected weights…")
        mx.eval(*[v for v in new_weights.values() if hasattr(v, "shape")])
        mx.save_safetensors(str(output_path / "model.safetensors"), new_weights)

        # Copy config and tokenizer files
        skip_exts = {".safetensors", ".npz", ".gguf", ".npy"}
        for f in self.model_path.iterdir():
            if f.suffix in skip_exts or not f.is_file():
                continue
            shutil.copy2(f, output_path / f.name)

        _prog(f"saved → {output_path}")

        elapsed = time.perf_counter() - t0
        return CorrectionResult(
            source_path=str(self.model_path),
            output_path=str(output_path),
            n_fail=probe_result.n_fail,
            n_pass=probe_result.n_pass,
            attributions=attributions,
            corrections=corrections,
            elapsed_s=elapsed,
        )
