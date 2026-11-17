"""Tests for the wsp Phase 3 sharpener."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("mlx")

from ollama_forge.commands.wsp import _cmd_sharpen
from ollama_forge.wsp.analyzer import LayerAnalysis, WeightMap
from ollama_forge.wsp.sharpener import (
    EditRecord,
    Sharpener,
    SharpenResult,
    _randomized_svd,
    _zero_top_directions,
)

# ── _randomized_svd ───────────────────────────────────────────────────────────

def test_randomized_svd_shape() -> None:
    rng = np.random.default_rng(1)
    A = rng.standard_normal((64, 32)).astype(np.float32)
    U, s, Vh = _randomized_svd(A, n_components=4)
    assert U.shape == (64, 4)
    assert s.shape == (4,)
    assert Vh.shape == (4, 32)


def test_randomized_svd_rank1_recovery() -> None:
    # Rank-1 matrix: outer product of two vectors
    u = np.array([[1, 2, 3, 4]], dtype=np.float32).T  # (4,1)
    v = np.array([[1, -1, 2]], dtype=np.float32)        # (1,3)
    A = 5.0 * u @ v  # (4,3), rank 1
    U, s, Vh = _randomized_svd(A, n_components=1, n_iter=8)
    # Reconstruction should be close
    A_rec = s[0] * np.outer(U[:, 0], Vh[0, :])
    assert np.allclose(A, A_rec, atol=0.1)


def test_randomized_svd_singular_values_descending() -> None:
    rng = np.random.default_rng(42)
    A = rng.standard_normal((32, 20)).astype(np.float32)
    _, s, _ = _randomized_svd(A, n_components=5)
    for i in range(len(s) - 1):
        assert s[i] >= s[i + 1] - 1e-4


# ── _zero_top_directions ──────────────────────────────────────────────────────

def test_zero_top_directions_reduces_norm() -> None:
    rng = np.random.default_rng(7)
    W = rng.standard_normal((32, 16)).astype(np.float32)
    W_new, changed, delta_frob = _zero_top_directions(W, n_directions=1)
    # Zeroing the top direction must reduce the Frobenius norm
    assert np.linalg.norm(W_new, "fro") < np.linalg.norm(W, "fro")
    assert changed > 0
    assert 0.0 < delta_frob <= 1.0


def test_zero_top_directions_rank1_becomes_zero() -> None:
    # Rank-1 matrix — zeroing the single direction should give ~0
    u = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    v = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    W = np.outer(u, v).astype(np.float32)
    W_new, _, _ = _zero_top_directions(W, n_directions=1)
    assert np.linalg.norm(W_new.astype(np.float32), "fro") < 0.5


def test_zero_top_directions_output_dtype() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 8)).astype(np.float32)
    W_new, _, _ = _zero_top_directions(W, n_directions=1)
    assert W_new.dtype == np.float16


# ── EditRecord ────────────────────────────────────────────────────────────────

def test_edit_record_fraction() -> None:
    er = EditRecord(
        layer_idx=15, proj_type="down_proj",
        n_directions=1, params_edited=200,
        total_params=1000, delta_frob=0.05,
    )
    assert abs(er.edit_fraction - 0.2) < 1e-6


# ── SharpenResult ─────────────────────────────────────────────────────────────

def test_sharpen_result_report() -> None:
    r = SharpenResult(
        source_path="/fake/model",
        output_path="/fake/model-sharp",
        obstruction="loops",
        edits=[EditRecord(15, "down_proj", 1, 500, 2_000_000, 0.03)],
        total_params=100_000_000,
        total_edited=500,
        elapsed_s=12.3,
    )
    report = r.report()
    assert "/fake/model" in report
    assert "0.0005%" in report  # 500/100_000_000
    assert "layer  15" in report


def test_sharpen_result_edit_fraction() -> None:
    r = SharpenResult(
        source_path="x", output_path="y", obstruction="loops",
        total_params=1_000_000, total_edited=100,
    )
    assert abs(r.edit_fraction - 1e-4) < 1e-10


# ── Sharpener (mocked) ────────────────────────────────────────────────────────

def _make_weight_map_with_path(tmp_path: Path, n_layers: int = 4) -> WeightMap:
    analyses = [
        LayerAnalysis(
            layer_idx=i,
            proj_type="down_proj",
            shape=(32, 16),
            top_sv=[2.0 + i * 0.1, 1.5],
            spectral_gap=1.5 + i * 0.05,
            effective_rank=10.0,
            frob_norm=50.0,
        )
        for i in range(n_layers)
    ]
    return WeightMap(
        model_path=str(tmp_path),
        n_layers=n_layers,
        quantized=False,
        quantization_bits=16,
        elapsed_s=1.0,
        analyses=analyses,
    )


def _make_fake_fp16_weights(n_layers: int = 4) -> dict:
    import mlx.core as mx
    rng = np.random.default_rng(0)
    weights = {}
    for i in range(n_layers):
        for ptype in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{i}.mlp.{ptype}.weight"
            arr = rng.standard_normal((32, 16)).astype(np.float16)
            weights[key] = mx.array(arr)
    return weights


def test_sharpener_produces_output_file(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}")

    out_dir = tmp_path / "model-sharp"
    weight_map = _make_weight_map_with_path(tmp_path, n_layers=4)

    fake_weights = _make_fake_fp16_weights(4)

    # Mock both _load_all_weights and mx.save_safetensors
    saved = {}

    def fake_save(path, weights, metadata=None):
        saved["path"] = path
        saved["n_keys"] = len(weights)

    with patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files", return_value=[tmp_path / "model.safetensors"]), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        sharpener = Sharpener(tmp_path)
        result = sharpener.sharpen(
            weight_map=weight_map,
            obstruction="loops",
            n_layers=2,
            n_directions=1,
            output_path=out_dir,
        )

    assert result.total_edited > 0
    assert len(result.edits) == 2
    assert saved.get("path") is not None
    # config.json should be copied from source (quantization key preserved)
    assert (out_dir / "config.json").exists()
    cfg_out = json.loads((out_dir / "config.json").read_text())
    assert "num_hidden_layers" in cfg_out
    # tokenizer.json should be copied
    assert (out_dir / "tokenizer.json").exists()


def test_sharpener_missing_safetensors(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    weight_map = _make_weight_map_with_path(tmp_path, n_layers=2)
    with patch("ollama_forge.wsp.analyzer._find_safetensors_files", return_value=[]):
        sharpener = Sharpener(tmp_path)
        with pytest.raises(FileNotFoundError):
            sharpener.sharpen(weight_map, output_path=tmp_path / "out")


# ── _cmd_sharpen ──────────────────────────────────────────────────────────────

def test_cmd_sharpen_no_model_path() -> None:
    args = argparse.Namespace(
        model="nonexistent-xyz",
        model_path=None,
        weight_map=None,
        obstruction="loops",
        n_layers=2,
        n_directions=1,
        proj_type="down_proj",
        output_path=None,
    )
    with patch("ollama_forge.wsp.analyzer.find_model_path", return_value=None):
        rc = _cmd_sharpen(argparse.ArgumentParser(), args)
    assert rc == 2


def test_cmd_sharpen_with_weight_map_json(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    weight_map = _make_weight_map_with_path(tmp_path, n_layers=4)
    wm_json = tmp_path / "wmap.json"
    wm_json.write_text(json.dumps(weight_map.to_json()))
    out_dir = tmp_path / "sharp"

    fake_weights = _make_fake_fp16_weights(4)

    def fake_save(path, weights, metadata=None):
        pass

    args = argparse.Namespace(
        model="test-model",
        model_path=str(tmp_path),
        weight_map=str(wm_json),
        obstruction="loops",
        n_layers=2,
        n_directions=1,
        proj_type="down_proj",
        output_path=str(out_dir),
    )

    with patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=fake_weights), \
         patch("ollama_forge.wsp.analyzer._find_safetensors_files", return_value=[tmp_path / "model.safetensors"]), \
         patch("mlx.core.save_safetensors", side_effect=fake_save):
        rc = _cmd_sharpen(argparse.ArgumentParser(), args)
    assert rc == 0
