"""Tests for Phase 5: Gradient-Guided Correction (GGC)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
pytest.importorskip("mlx")

from ollama_forge.wsp.corrector import ProbeResult
from ollama_forge.wsp.gradient_corrector import (
    GGCRecord,
    GGCResult,
    GradientCorrector,
    _get_lm_head_fp32,
    compute_lm_head_direction,
    gradient_rank1_correction,
)

# ── gradient_rank1_correction ─────────────────────────────────────────────────

def test_gradient_rank1_combined_changes_weight() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 32)).astype(np.float32)
    d_k = rng.standard_normal((32,)).astype(np.float32)
    d_target = rng.standard_normal((16,)).astype(np.float32)

    W_new, changed, delta_frob = gradient_rank1_correction(W, d_k, d_target, alpha=0.1, beta=0.1)

    assert changed > 0
    assert 0.0 < delta_frob <= 1.0
    assert W_new.dtype == np.float16


def test_gradient_rank1_pure_bcc_equivalent() -> None:
    """With beta=0, should behave like BCC (suppresses failure direction)."""
    from ollama_forge.wsp.corrector import behavioral_rank1_correction

    rng = np.random.default_rng(1)
    W = rng.standard_normal((16, 32)).astype(np.float32)
    d_k = rng.standard_normal((32,)).astype(np.float32)

    W_bcc, _, _ = behavioral_rank1_correction(W, d_k, alpha=0.5)
    W_ggc, _, _ = gradient_rank1_correction(W, d_k, d_target=None, alpha=0.5, beta=0.0)

    # Both should give identical fp16 results
    assert np.array_equal(W_bcc, W_ggc)


def test_gradient_rank1_pure_additive() -> None:
    """With alpha=0, the update is purely outer(d_target_norm, d_k_norm).

    The output of W_new @ d_k_norm should have a component in d_target direction.
    """
    rng = np.random.default_rng(2)
    W = rng.standard_normal((16, 32)).astype(np.float32)
    d_k = rng.standard_normal((32,)).astype(np.float32)
    d_target = rng.standard_normal((16,)).astype(np.float32)
    d_target_norm = (d_target / np.linalg.norm(d_target)).astype(np.float32)
    d_k_norm = (d_k / np.linalg.norm(d_k)).astype(np.float32)

    W_new, _, _ = gradient_rank1_correction(W, d_k, d_target, alpha=0.0, beta=1.0)
    W_new_fp32 = W_new.astype(np.float32)

    # The added component outer(d_target_norm, d_k_norm) means:
    # W_new @ d_k_norm has an additional d_target_norm component (of magnitude 1.0)
    # So d_target_norm @ (W_new @ d_k_norm) ≈ 1.0 + (d_target_norm @ W_fp32 @ d_k_norm)
    added_component = float(d_target_norm @ (W_new_fp32 @ d_k_norm))
    original_component = float(d_target_norm @ (W @ d_k_norm))

    assert added_component > original_component, (
        f"Expected additive component to increase the d_target projection: "
        f"after={added_component:.4f} <= before={original_component:.4f}"
    )


def test_gradient_rank1_alpha_zero_beta_zero_noop() -> None:
    rng = np.random.default_rng(3)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d_k = rng.standard_normal((16,)).astype(np.float32)
    d_target = rng.standard_normal((8,)).astype(np.float32)

    W_new, _, delta_frob = gradient_rank1_correction(W, d_k, d_target, alpha=0.0, beta=0.0)
    assert delta_frob < 1e-2  # fp16 rounding noise only


def test_gradient_rank1_output_dtype() -> None:
    rng = np.random.default_rng(4)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d_k = rng.standard_normal((16,)).astype(np.float32)
    d_target = rng.standard_normal((8,)).astype(np.float32)
    W_new, _, _ = gradient_rank1_correction(W, d_k, d_target, alpha=0.1, beta=0.3)
    assert W_new.dtype == np.float16


# ── _get_lm_head_fp32 ─────────────────────────────────────────────────────────

def _make_weights_with_lm_head(vocab: int = 64, d_model: int = 16) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(10)
    w = rng.standard_normal((vocab, d_model)).astype(np.float16)
    return {"lm_head.weight": mx.array(w)}


def _make_weights_with_embed(vocab: int = 64, d_model: int = 16) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(11)
    w = rng.standard_normal((vocab, d_model)).astype(np.float16)
    return {"model.embed_tokens.weight": mx.array(w)}


def test_get_lm_head_fp32_from_lm_head_key() -> None:
    weights = _make_weights_with_lm_head(vocab=64, d_model=16)
    W = _get_lm_head_fp32(weights)
    assert W is not None
    assert W.shape == (64, 16)
    assert W.dtype == np.float32


def test_get_lm_head_fp32_from_embed_tokens() -> None:
    weights = _make_weights_with_embed(vocab=64, d_model=16)
    W = _get_lm_head_fp32(weights)
    assert W is not None
    assert W.shape == (64, 16)


def test_get_lm_head_fp32_no_key() -> None:
    import mlx.core as mx
    weights = {"model.layers.0.mlp.down_proj.weight": mx.zeros((8, 16))}
    W = _get_lm_head_fp32(weights)
    assert W is None


# ── compute_lm_head_direction ─────────────────────────────────────────────────

def _make_probe_result_with_responses(
    fail_expected: list[str],
    fail_responses: list[str],
    n_layers: int = 4,
    d_mlp: int = 32,
) -> ProbeResult:
    rng = np.random.default_rng(42)
    pr = ProbeResult(
        n_fail=len(fail_expected),
        n_pass=5,
        fail_examples=["prompt"] * len(fail_expected),
        fail_expected=fail_expected,
        fail_responses=fail_responses,
    )
    for li in range(n_layers):
        pr.h_fail_mean[li] = rng.standard_normal((d_mlp,)).astype(np.float32)
        pr.h_pass_mean[li] = rng.standard_normal((d_mlp,)).astype(np.float32)
    return pr


def test_compute_lm_head_direction_basic() -> None:
    # Create a fake tokenizer that returns a unique int for the first char
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = lambda text, add_special_tokens=True: [ord(text[0]) % 32]

    weights = _make_weights_with_lm_head(vocab=128, d_model=16)

    pr = _make_probe_result_with_responses(
        fail_expected=["yes", "5"],
        fail_responses=["no answer given", "1 is incorrect"],
    )

    d_target = compute_lm_head_direction(pr, mock_tokenizer, weights)
    assert d_target is not None
    assert d_target.shape == (16,)
    assert d_target.dtype == np.float32


def test_compute_lm_head_direction_same_token_skipped() -> None:
    """Fail examples where correct == predicted should be skipped."""
    mock_tokenizer = MagicMock()
    # Always return token 42 for any input
    mock_tokenizer.encode = lambda text, add_special_tokens=True: [42]

    weights = _make_weights_with_lm_head(vocab=128, d_model=16)
    pr = _make_probe_result_with_responses(
        fail_expected=["yes"],
        fail_responses=["yes"],
    )

    d_target = compute_lm_head_direction(pr, mock_tokenizer, weights)
    # Only 1 pair, and correct == predicted, so no valid direction
    assert d_target is None


def test_compute_lm_head_direction_empty_probe() -> None:
    mock_tokenizer = MagicMock()
    weights = _make_weights_with_lm_head(vocab=128, d_model=16)
    pr = ProbeResult(n_fail=0, n_pass=10)
    d_target = compute_lm_head_direction(pr, mock_tokenizer, weights)
    assert d_target is None


# ── GGCRecord / GGCResult ─────────────────────────────────────────────────────

def test_ggc_record_edit_fraction() -> None:
    rec = GGCRecord(
        layer_idx=25, proj_type="down_proj",
        alpha=0.1, beta=0.3, d_k_norm=1.2, d_target_norm=5.5,
        delta_frob=0.03, params_edited=1000, total_params=100_000,
    )
    assert abs(rec.edit_fraction - 0.01) < 1e-9


def test_ggc_result_report() -> None:
    from ollama_forge.wsp.corrector import AttributionResult
    result = GGCResult(
        source_path="/fake/model",
        output_path="/fake/model-ggc",
        n_fail=10,
        n_pass=25,
        n_target_dirs=8,
        attributions=[AttributionResult(25, "down_proj", 4.2, 1.1)],
        corrections=[GGCRecord(25, "down_proj", 0.1, 0.3, 1.2, 5.5, 0.012, 500, 1_000_000)],
        elapsed_s=22.4,
    )
    report = result.report()
    assert "GGC result" in report
    assert "10 fail" in report
    assert "layer  25" in report
    assert "β=0.300" in report


# ── GradientCorrector (mocked end-to-end) ─────────────────────────────────────

def _make_fake_weights_fp16(n_layers: int = 4, vocab: int = 64, d_model: int = 16) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(0)
    weights = {}
    for li in range(n_layers):
        for pt in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{li}.mlp.{pt}.weight"
            arr = rng.standard_normal((d_model, 32)).astype(np.float16)
            weights[key] = mx.array(arr)
    # LM head (tied embedding)
    lm = rng.standard_normal((vocab, d_model)).astype(np.float16)
    weights["lm_head.weight"] = mx.array(lm)
    return weights


def _make_probe_result_enough_with_responses(n_layers: int = 4) -> ProbeResult:
    rng = np.random.default_rng(99)
    pr = ProbeResult(
        n_fail=5, n_pass=5,
        fail_examples=["prompt"] * 5,
        fail_expected=["yes"] * 5,
        fail_responses=["no"] * 5,
        pass_examples=["p"] * 5,
    )
    for li in range(n_layers):
        pr.h_fail_mean[li] = rng.standard_normal((32,)).astype(np.float32)
        pr.h_pass_mean[li] = rng.standard_normal((32,)).astype(np.float32)
    return pr


def test_gradient_corrector_produces_output(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "tokenizer.json").write_text("{}")

    out_dir = tmp_path / "model-ggc"
    fake_weights = _make_fake_weights_fp16(4)
    probe_result = _make_probe_result_enough_with_responses(4)

    saved = {}

    def fake_save(path, weights, metadata=None):
        saved["path"] = path
        saved["n_keys"] = len(weights)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = lambda text, add_special_tokens=True: [ord(text[0]) % 32 + 1]

    with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)), \
         patch("ollama_forge.wsp.gradient_corrector.run_mlx_probe", return_value=probe_result), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]), \
         patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        corrector = GradientCorrector(tmp_path)
        result = corrector.correct(n_layers=2, alpha=0.1, beta=0.3, min_layer=0,
                                   output_path=out_dir)

    assert result.n_fail == 5
    assert result.n_pass == 5
    assert len(result.corrections) == 2
    assert saved.get("path") is not None
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()


def test_gradient_corrector_too_few_failures(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    pr = ProbeResult(n_fail=1, n_pass=10)
    for li in range(2):
        pr.h_fail_mean[li] = np.zeros(32, dtype=np.float32)
        pr.h_pass_mean[li] = np.zeros(32, dtype=np.float32)

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.gradient_corrector.run_mlx_probe", return_value=pr), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]):
        corrector = GradientCorrector(tmp_path)
        with pytest.raises(ValueError, match="failure"):
            corrector.correct(output_path=tmp_path / "out")
