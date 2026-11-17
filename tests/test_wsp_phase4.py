"""Tests for Phase 4: Behavioral Causal Correction (BCC)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mlx")

from ollama_forge.commands.wsp import _cmd_correct
from ollama_forge.wsp.corrector import (
    _PROBE_BANK,
    BehavioralCorrector,
    CorrectionRecord,
    CorrectionResult,
    ProbeResult,
    attribute_failure_layers,
    behavioral_rank1_correction,
)

# ── behavioral_rank1_correction ───────────────────────────────────────────────

def test_rank1_correction_reduces_trigger_response() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 32)).astype(np.float32)
    d_k = rng.standard_normal((32,)).astype(np.float32)
    d_k_norm = d_k / np.linalg.norm(d_k)

    response_before = float(np.linalg.norm(W @ d_k_norm))
    W_new, changed, delta_frob = behavioral_rank1_correction(W, d_k, alpha=0.5)
    response_after = float(np.linalg.norm(W_new.astype(np.float32) @ d_k_norm))

    # Response to the trigger direction should be reduced
    assert response_after < response_before
    assert changed > 0
    assert 0.0 < delta_frob <= 1.0


def test_rank1_correction_alpha_zero_is_noop() -> None:
    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d_k = rng.standard_normal((16,)).astype(np.float32)
    W_new, changed, delta_frob = behavioral_rank1_correction(W, d_k, alpha=0.0)
    # fp16 rounding of a float32 source introduces ~1e-3 relative noise even with alpha=0
    assert delta_frob < 1e-2


def test_rank1_correction_output_dtype() -> None:
    rng = np.random.default_rng(2)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    d_k = rng.standard_normal((16,)).astype(np.float32)
    W_new, _, _ = behavioral_rank1_correction(W, d_k, alpha=0.1)
    assert W_new.dtype == np.float16


def test_rank1_correction_orthogonal_unchanged() -> None:
    # Direction orthogonal to d_k should be unaffected (up to fp16 precision)
    W = np.eye(4, 4, dtype=np.float32)
    # d_k = first basis vector
    d_k = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # W row 1 is [0,1,0,0] — orthogonal to d_k
    W_new, _, _ = behavioral_rank1_correction(W, d_k, alpha=1.0)
    # Row 1 of W_new should be close to [0,1,0,0]
    assert abs(float(W_new[1, 1]) - 1.0) < 0.01


# ── attribute_failure_layers ──────────────────────────────────────────────────

def _make_probe_result(n_layers: int = 4, d_mlp: int = 32) -> ProbeResult:
    rng = np.random.default_rng(42)
    pr = ProbeResult(n_fail=10, n_pass=10)
    for li in range(n_layers):
        pr.h_fail_mean[li] = rng.standard_normal((d_mlp,)).astype(np.float32)
        pr.h_pass_mean[li] = rng.standard_normal((d_mlp,)).astype(np.float32)
    return pr


def _make_mlx_weights(n_layers: int = 4, d_model: int = 16, d_mlp: int = 32) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(7)
    weights = {}
    for li in range(n_layers):
        for pt in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{li}.mlp.{pt}.weight"
            arr = rng.standard_normal((d_model, d_mlp)).astype(np.float32)
            weights[key] = mx.array(arr)
    return weights


def test_attribute_returns_ranked_list() -> None:
    pr = _make_probe_result(n_layers=4, d_mlp=32)
    weights = _make_mlx_weights(n_layers=4, d_model=16, d_mlp=32)
    results = attribute_failure_layers(pr, weights, n_layers=4, top_k=4)
    assert len(results) == 4
    # Results should be in descending order of response_magnitude
    for i in range(len(results) - 1):
        assert results[i].response_magnitude >= results[i + 1].response_magnitude


def test_attribute_all_projtype_down() -> None:
    pr = _make_probe_result(n_layers=3, d_mlp=32)
    weights = _make_mlx_weights(n_layers=3, d_model=16, d_mlp=32)
    results = attribute_failure_layers(pr, weights, n_layers=3)
    assert all(r.proj_type == "down_proj" for r in results)


def test_attribute_missing_layer_skipped() -> None:
    pr = ProbeResult(n_fail=5, n_pass=5)
    # Only layer 1 has activations
    pr.h_fail_mean[1] = np.ones(16, dtype=np.float32)
    pr.h_pass_mean[1] = np.zeros(16, dtype=np.float32)
    weights = _make_mlx_weights(n_layers=3, d_model=8, d_mlp=16)
    results = attribute_failure_layers(pr, weights, n_layers=3, top_k=5)
    # Only layer 1 should appear
    assert len(results) == 1
    assert results[0].layer_idx == 1


# ── CorrectionRecord / CorrectionResult ──────────────────────────────────────

def test_correction_record_edit_fraction() -> None:
    rec = CorrectionRecord(
        layer_idx=5, proj_type="down_proj",
        alpha=0.1, d_k_norm=1.0, delta_frob=0.03,
        params_edited=1000, total_params=100_000,
    )
    assert abs(rec.edit_fraction - 0.01) < 1e-9


def test_correction_result_report() -> None:
    result = CorrectionResult(
        source_path="/fake/model",
        output_path="/fake/model-bcc",
        n_fail=8,
        n_pass=12,
        corrections=[
            CorrectionRecord(5, "down_proj", 0.1, 1.2, 0.03, 500, 1_000_000),
        ],
        elapsed_s=15.4,
    )
    report = result.report()
    assert "BCC result" in report
    assert "8 fail" in report
    assert "layer   5" in report


# ── probe bank sanity ─────────────────────────────────────────────────────────

def test_probe_bank_nonempty() -> None:
    assert len(_PROBE_BANK) >= 10


def test_probe_bank_tuples() -> None:
    for prompt, expected in _PROBE_BANK:
        assert isinstance(prompt, str) and len(prompt) > 10
        assert isinstance(expected, str) and len(expected) > 0


# ── BehavioralCorrector (mocked) ──────────────────────────────────────────────

def _make_fake_weights_fp16(n_layers: int = 4) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(0)
    weights = {}
    for li in range(n_layers):
        for pt in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{li}.mlp.{pt}.weight"
            arr = rng.standard_normal((16, 32)).astype(np.float16)
            weights[key] = mx.array(arr)
    return weights


def _make_probe_result_enough(n_layers: int = 4) -> ProbeResult:
    rng = np.random.default_rng(99)
    pr = ProbeResult(n_fail=5, n_pass=5, fail_examples=["f"] * 5, pass_examples=["p"] * 5)
    for li in range(n_layers):
        pr.h_fail_mean[li] = rng.standard_normal((32,)).astype(np.float32)
        pr.h_pass_mean[li] = rng.standard_normal((32,)).astype(np.float32)
    return pr


def test_corrector_produces_output(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "tokenizer.json").write_text("{}")

    out_dir = tmp_path / "model-bcc"
    fake_weights = _make_fake_weights_fp16(4)
    probe_result = _make_probe_result_enough(4)

    saved = {}

    def fake_save(path, weights, metadata=None):
        saved["path"] = path
        saved["n_keys"] = len(weights)

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.corrector.run_mlx_probe", return_value=probe_result), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]), \
         patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        corrector = BehavioralCorrector(tmp_path)
        result = corrector.correct(n_layers=2, alpha=0.1, output_path=out_dir)

    assert result.n_fail == 5
    assert result.n_pass == 5
    assert len(result.corrections) == 2
    assert saved.get("path") is not None
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()


def test_corrector_too_few_failures(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    # Only 1 failure — should raise
    pr = ProbeResult(n_fail=1, n_pass=10)
    for li in range(2):
        pr.h_fail_mean[li] = np.zeros(32, dtype=np.float32)
        pr.h_pass_mean[li] = np.zeros(32, dtype=np.float32)

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.corrector.run_mlx_probe", return_value=pr), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]):
        corrector = BehavioralCorrector(tmp_path)
        with pytest.raises(ValueError, match="failure"):
            corrector.correct(output_path=tmp_path / "out")


# ── _cmd_correct ──────────────────────────────────────────────────────────────

def test_cmd_correct_no_model_path() -> None:
    args = argparse.Namespace(
        model="nonexistent-xyz",
        model_path=None,
        n_layers=2,
        proj_type="down_proj",
        alpha=0.1,
        output_path=None,
    )
    with patch("ollama_forge.wsp.analyzer.find_model_path", return_value=None):
        rc = _cmd_correct(argparse.ArgumentParser(), args)
    assert rc == 2


def test_cmd_correct_with_path(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    probe_result = _make_probe_result_enough(2)
    fake_weights = _make_fake_weights_fp16(2)
    out_dir = tmp_path / "bcc"

    def fake_save(path, weights, metadata=None):
        pass

    args = argparse.Namespace(
        model="test-model",
        model_path=str(tmp_path),
        n_layers=2,
        proj_type="down_proj",
        alpha=0.1,
        output_path=str(out_dir),
    )
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("ollama_forge.wsp.corrector.run_mlx_probe", return_value=probe_result), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files",
               return_value=[tmp_path / "model.safetensors"]), \
         patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        rc = _cmd_correct(argparse.ArgumentParser(), args)
    assert rc == 0
