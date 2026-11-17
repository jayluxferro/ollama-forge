"""Tests for Phase 6: Abliteration (concept direction suppression)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mlx")

from ollama_forge.wsp.abliterator import (
    AbliterationCorrector,
    AbliterationRecord,
    AbliterationResult,
    ResidualProbeResult,
    compute_concept_direction,
    orthogonalise_weight,
    rank_layers_by_separation,
)

# ── orthogonalise_weight ───────────────────────────────────────────────────────

def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def test_orthogonalise_write_removes_concept_from_output() -> None:
    """After write-mode ortho, d_norm · (W' @ x) ≈ 0 for all x."""
    rng = _rng(0)
    d_model, d_in = 16, 32
    W = rng.standard_normal((d_model, d_in)).astype(np.float32)
    d = rng.standard_normal((d_model,)).astype(np.float32)
    d_norm = d / np.linalg.norm(d)

    W_new, changed, delta_frob = orthogonalise_weight(W, d, mode="write")
    W_new_fp32 = W_new.astype(np.float32)

    # For several random inputs, the concept component in output should vanish
    for _ in range(10):
        x = rng.standard_normal((d_in,)).astype(np.float32)
        output_component = float(d_norm @ (W_new_fp32 @ x))
        # fp16 rounding means residual ~0, not exactly 0
        assert abs(output_component) < 1e-2, f"Concept not suppressed: {output_component:.4f}"


def test_orthogonalise_read_removes_concept_from_input() -> None:
    """After read-mode ortho, the concept direction in x has no effect on W' @ x."""
    rng = _rng(1)
    d_out, d_model = 32, 16
    W = rng.standard_normal((d_out, d_model)).astype(np.float32)
    d = rng.standard_normal((d_model,)).astype(np.float32)
    d_norm = (d / np.linalg.norm(d)).astype(np.float32)

    W_new, changed, delta_frob = orthogonalise_weight(W, d, mode="read")
    W_new_fp32 = W_new.astype(np.float32)

    # W' @ (c * d_norm + x_perp) ≈ W' @ x_perp  (concept component contributes 0)
    x_perp = rng.standard_normal((d_model,)).astype(np.float32)
    x_perp -= d_norm * float(d_norm @ x_perp)  # remove d_norm component

    for c in [0.5, 1.0, 2.0, -1.0]:
        x_full = x_perp + c * d_norm
        y_full = (W_new_fp32 @ x_full).astype(np.float64)
        y_perp = (W_new_fp32 @ x_perp).astype(np.float64)
        diff = float(np.linalg.norm(y_full - y_perp))
        # Difference should be near zero (fp16 noise only)
        assert diff < 2e-1, f"c={c}: concept leaks through: ||Δy||={diff:.4f}"


def test_orthogonalise_write_changes_weight() -> None:
    rng = _rng(2)
    W = rng.standard_normal((16, 32)).astype(np.float32)
    d = rng.standard_normal((16,)).astype(np.float32)
    W_new, changed, delta_frob = orthogonalise_weight(W, d, mode="write")
    assert changed > 0
    assert 0.0 < delta_frob <= 1.0
    assert W_new.dtype == np.float16


def test_orthogonalise_read_changes_weight() -> None:
    rng = _rng(3)
    W = rng.standard_normal((32, 16)).astype(np.float32)
    d = rng.standard_normal((16,)).astype(np.float32)
    W_new, changed, delta_frob = orthogonalise_weight(W, d, mode="read")
    assert changed > 0
    assert 0.0 < delta_frob <= 1.0
    assert W_new.dtype == np.float16


def test_orthogonalise_zero_direction_stable() -> None:
    """Near-zero direction should not raise or produce NaN."""
    rng = _rng(4)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d = np.zeros(8, dtype=np.float32)
    W_new, changed, delta_frob = orthogonalise_weight(W, d, mode="write")
    assert not np.any(np.isnan(W_new.astype(np.float32)))


def test_orthogonalise_read_zero_direction_stable() -> None:
    rng = _rng(5)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d = np.zeros(16, dtype=np.float32)
    W_new, changed, delta_frob = orthogonalise_weight(W, d, mode="read")
    assert not np.any(np.isnan(W_new.astype(np.float32)))


def test_orthogonalise_output_dtype() -> None:
    rng = _rng(6)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d = rng.standard_normal((8,)).astype(np.float32)
    W_new, _, _ = orthogonalise_weight(W, d, mode="write")
    assert W_new.dtype == np.float16


# ── compute_concept_direction ──────────────────────────────────────────────────

def _make_probe(
    n_layers: int,
    d_model: int,
    seed: int = 0,
) -> ResidualProbeResult:
    rng = np.random.default_rng(seed)
    pr = ResidualProbeResult(n_suppress=5, n_retain=5)
    for li in range(n_layers):
        pr.h_suppress_mean[li] = rng.standard_normal(d_model).astype(np.float32)
        pr.h_retain_mean[li] = rng.standard_normal(d_model).astype(np.float32)
    return pr


def test_compute_concept_direction_basic() -> None:
    pr = _make_probe(4, 16)
    d = compute_concept_direction(pr, 2)
    assert d is not None
    assert d.shape == (16,)
    assert d.dtype == np.float32
    # Should equal suppress_mean - retain_mean
    expected = pr.h_suppress_mean[2] - pr.h_retain_mean[2]
    np.testing.assert_allclose(d, expected, rtol=1e-5)


def test_compute_concept_direction_missing_layer() -> None:
    pr = ResidualProbeResult(n_suppress=5, n_retain=5)
    # Layer 0 has no activations
    assert compute_concept_direction(pr, 0) is None


def test_compute_concept_direction_missing_retain() -> None:
    pr = ResidualProbeResult(n_suppress=5, n_retain=5)
    pr.h_suppress_mean[0] = np.ones(8, dtype=np.float32)
    # No retain mean for layer 0
    assert compute_concept_direction(pr, 0) is None


# ── rank_layers_by_separation ──────────────────────────────────────────────────

def test_rank_layers_by_separation_sorted_desc() -> None:
    pr = _make_probe(5, 16, seed=42)
    ranked = rank_layers_by_separation(pr, 5, min_layer=0)
    assert len(ranked) == 5
    norms = [n for _, n in ranked]
    assert norms == sorted(norms, reverse=True)


def test_rank_layers_by_separation_respects_min_layer() -> None:
    pr = _make_probe(5, 16, seed=7)
    ranked = rank_layers_by_separation(pr, 5, min_layer=3)
    for li, _ in ranked:
        assert li >= 3


def test_rank_layers_by_separation_no_layers_above_min() -> None:
    pr = _make_probe(5, 16)
    ranked = rank_layers_by_separation(pr, 5, min_layer=10)
    assert ranked == []


# ── ResidualProbeResult / AbliterationRecord / AbliterationResult ─────────────

def test_abliteration_record_edit_fraction() -> None:
    rec = AbliterationRecord(
        layer_idx=15, proj_type="down_proj", mode="write",
        d_concept_norm=1.5, delta_frob=0.02,
        params_edited=50_000, total_params=5_000_000,
    )
    assert abs(rec.edit_fraction - 0.01) < 1e-9


def test_abliteration_result_report() -> None:
    result = AbliterationResult(
        source_path="/fake/model",
        output_path="/fake/model-abliterated",
        n_suppress=15,
        n_retain=15,
        best_layer=20,
        concept_norm=2.34,
        corrections=[
            AbliterationRecord(20, "down_proj", "write", 2.34, 0.021, 500_000, 2_000_000),
            AbliterationRecord(20, "o_proj",    "write", 2.34, 0.019, 300_000, 1_500_000),
        ],
        elapsed_s=18.7,
    )
    report = result.report()
    assert "Abliteration result" in report
    assert "15 suppress" in report
    assert "Best layer: 20" in report
    assert "down_proj" in report
    assert "o_proj" in report
    assert "write" in report


# ── AbliterationCorrector (mocked end-to-end) ─────────────────────────────────

def _make_fake_weights(n_layers: int = 4, d_model: int = 16, d_mlp: int = 32,
                       d_qkv: int = 8) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(0)
    weights = {}
    for li in range(n_layers):
        # MLP
        for pt, shape in [
            ("down_proj", (d_model, d_mlp)),
            ("gate_proj", (d_mlp, d_model)),
            ("up_proj",   (d_mlp, d_model)),
        ]:
            weights[f"model.layers.{li}.mlp.{pt}.weight"] = mx.array(
                rng.standard_normal(shape).astype(np.float16)
            )
        # Attention
        for pt, shape in [
            ("q_proj", (d_qkv, d_model)),
            ("k_proj", (d_qkv, d_model)),
            ("v_proj", (d_qkv, d_model)),
            ("o_proj", (d_model, d_qkv)),
        ]:
            weights[f"model.layers.{li}.self_attn.{pt}.weight"] = mx.array(
                rng.standard_normal(shape).astype(np.float16)
            )
    return weights


def _make_fake_residual_probe(n_layers: int = 4, d_model: int = 16) -> ResidualProbeResult:
    rng = np.random.default_rng(55)
    pr = ResidualProbeResult(
        n_suppress=5, n_retain=5,
        suppress_examples=["p"] * 5,
        retain_examples=["q"] * 5,
    )
    for li in range(n_layers):
        pr.h_suppress_mean[li] = rng.standard_normal(d_model).astype(np.float32)
        pr.h_retain_mean[li] = rng.standard_normal(d_model).astype(np.float32)
    return pr


def test_abliteration_corrector_produces_output(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "tokenizer.json").write_text("{}")

    out_dir = tmp_path / "model-abliterated"
    fake_weights = _make_fake_weights(4)
    probe_result = _make_fake_residual_probe(4)

    saved = {}

    def fake_save(path, weights, metadata=None):
        saved["path"] = path
        saved["n_keys"] = len(weights)

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.abliterator.run_abliteration_probe", return_value=probe_result), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]), \
         patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        corrector = AbliterationCorrector(tmp_path)
        result = corrector.abliterate(
            suppress_bank=["a"] * 5,
            retain_bank=["b"] * 5,
            n_layers=1,
            min_layer=0,
            output_path=out_dir,
        )

    assert result.n_suppress == 5
    assert result.n_retain == 5
    assert len(result.corrections) > 0
    assert saved.get("path") is not None
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()


def test_abliteration_corrector_empty_suppress_raises(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    pr = ResidualProbeResult(n_suppress=0, n_retain=5)

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.abliterator.run_abliteration_probe", return_value=pr), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]):
        corrector = AbliterationCorrector(tmp_path)
        with pytest.raises(ValueError, match="suppress_bank"):
            corrector.abliterate(
                suppress_bank=[],
                retain_bank=["b"] * 5,
                output_path=tmp_path / "out",
            )


def test_abliteration_corrector_proj_types_override(tmp_path: Path) -> None:
    """proj_types override limits orthogonalisation to specified matrices only."""
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    out_dir = tmp_path / "out"
    fake_weights = _make_fake_weights(4)
    probe_result = _make_fake_residual_probe(4)

    saved = {}

    def fake_save(path, weights, metadata=None):
        saved["path"] = path

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.abliterator.run_abliteration_probe", return_value=probe_result), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]), \
         patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        corrector = AbliterationCorrector(tmp_path)
        result = corrector.abliterate(
            suppress_bank=["a"] * 3,
            retain_bank=["b"] * 3,
            n_layers=1,
            min_layer=0,
            proj_types=["down_proj"],  # only down_proj
            output_path=out_dir,
        )

    # Only down_proj corrections should appear
    proj_types_seen = {c.proj_type for c in result.corrections}
    assert proj_types_seen == {"down_proj"}


# ── new: probe mode tests (pure Python, no MLX inference) ────────────────────

def test_capturing_norm_last_token_mode() -> None:
    """_CapturingNorm last_token: captures once at prefill, then deactivates."""
    import sys
    import types

    from ollama_forge.wsp.abliterator import _CapturingNorm

    # Install a minimal mlx.core stub so the norm's __call__ can run
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.eval = lambda x: None
    sys.modules["mlx.core"] = fake_mx

    class _FakeNorm:
        def __call__(self, x):
            return x

    captures: list = []
    active = [True]
    norm = _CapturingNorm(_FakeNorm(), captures, active, mode="last_token")

    # Prefill pass: (1, 3, 4) — should capture last token position and stop
    x = np.ones((1, 3, 4), dtype=np.float32)
    norm(x)

    assert active[0] is False      # deactivated after first capture
    assert len(captures) == 1
    assert captures[0].shape == (4,)  # d_model only

    # Second call — should not capture
    norm(x)
    assert len(captures) == 1


def test_capturing_norm_all_tokens_mode_captures_every_position() -> None:
    """_CapturingNorm all_tokens: active flag never flips; collects every position."""
    import types

    from ollama_forge.wsp.abliterator import _CapturingNorm

    class _FakeNorm:
        def __call__(self, x):
            return x

    # Minimal mlx stub
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.eval = lambda x: None
    import sys
    sys.modules["mlx.core"] = fake_mx

    captures: list = []
    active = [True]
    norm = _CapturingNorm(_FakeNorm(), captures, active, mode="all_tokens")

    # Prefill (1, 3, 4) — should capture 3 positions
    x_prefill = np.ones((1, 3, 4), dtype=np.float32)
    norm(x_prefill)
    assert len(captures) == 3
    assert active[0] is True  # never deactivated in all_tokens mode

    # Autoregressive step (1, 1, 4) — should append 1 more
    x_ar = np.ones((1, 1, 4), dtype=np.float32) * 2
    norm(x_ar)
    assert len(captures) == 4


def test_run_abliteration_probe_full_response_mode() -> None:
    """run_abliteration_probe: full_response mode averages across all captured positions."""
    from ollama_forge.wsp.abliterator import ResidualProbeResult, run_abliteration_probe

    class _MockNorm:
        def __call__(self, x):
            return x

    class _MockLayer:
        def __init__(self):
            self.input_layernorm = _MockNorm()

    class _MockModelInner:
        def __init__(self):
            self.layers = [_MockLayer(), _MockLayer()]

    class _MockModel:
        def __init__(self):
            self.model = _MockModelInner()

    mock_model = _MockModel()

    def fake_generate(model, tokenizer, prompt, max_tokens, verbose):
        # Simulate capturing: each layer captures `max_tokens` positions
        for li in range(2):
            norm = model.model.layers[li].input_layernorm
            if hasattr(norm, "_captures") and norm._active[0]:
                for _ in range(max_tokens):
                    norm._captures.append(np.random.randn(8).astype(np.float32))
                    if norm._mode == "last_token":
                        norm._active[0] = False
                        break

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<chat>prompt</chat>"

    with patch("mlx_lm.generate", side_effect=fake_generate):
        result = run_abliteration_probe(
            model=mock_model,
            tokenizer=mock_tokenizer,
            suppress_bank=["prompt A"],
            retain_bank=["prompt B"],
            layer_indices=[0, 1],
            probe_mode="full_response",
            full_response_tokens=5,
        )

    assert isinstance(result, ResidualProbeResult)
    assert result.n_suppress == 1
    assert result.n_retain == 1


def test_run_abliteration_probe_invalid_mode_raises() -> None:
    """run_abliteration_probe raises ValueError for unknown probe_mode."""
    from ollama_forge.wsp.abliterator import run_abliteration_probe

    with pytest.raises(ValueError, match="probe_mode"):
        run_abliteration_probe(
            model=None,
            tokenizer=None,
            suppress_bank=["a"],
            retain_bank=["b"],
            layer_indices=[0],
            probe_mode="invalid_mode",
        )


def test_abliterate_probe_mode_passed_through(tmp_path: Path) -> None:
    """abliterate() forwards probe_mode and full_response_tokens to run_abliteration_probe."""
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    probe_kwargs_seen: dict = {}

    def fake_probe(**kwargs):
        probe_kwargs_seen.update(kwargs)
        return _make_fake_residual_probe(2)

    fake_weights = _make_fake_weights(2)

    def fake_save(path, weights, metadata=None):
        pass

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.abliterator.run_abliteration_probe", side_effect=fake_probe), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]), \
         patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        corrector = AbliterationCorrector(tmp_path)
        corrector.abliterate(
            suppress_bank=["a"] * 2,
            retain_bank=["b"] * 2,
            n_layers=1,
            min_layer=0,
            output_path=tmp_path / "out",
            probe_mode="full_response",
            full_response_tokens=25,
        )

    assert probe_kwargs_seen.get("probe_mode") == "full_response"
    assert probe_kwargs_seen.get("full_response_tokens") == 25
