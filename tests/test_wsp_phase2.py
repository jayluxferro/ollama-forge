"""Tests for the wsp Phase 2 analyzer (WeightSpaceAnalyzer, WeightMap, etc.)."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
pytest.importorskip("mlx")

from ollama_forge.wsp.analyzer import (
    LayerAnalysis,
    WeightMap,
    WeightSpaceAnalyzer,
    _effective_rank,
    _spectral_gap,
    _svd_metrics,
    find_model_path,
)
from ollama_forge.commands.wsp import _cmd_locate


# ── _effective_rank ───────────────────────────────────────────────────────────

def test_effective_rank_uniform() -> None:
    # Uniform singular values → maximum effective rank = n
    sv = [1.0] * 10
    er = _effective_rank(sv)
    assert 9.0 <= er <= 10.1


def test_effective_rank_concentrated() -> None:
    # One dominant value → effective rank close to 1
    sv = [100.0] + [0.001] * 9
    er = _effective_rank(sv)
    assert er < 1.5


def test_effective_rank_empty() -> None:
    assert _effective_rank([]) == 1.0


# ── _spectral_gap ─────────────────────────────────────────────────────────────

def test_spectral_gap_basic() -> None:
    assert abs(_spectral_gap([4.0, 2.0]) - 2.0) < 1e-6


def test_spectral_gap_single() -> None:
    assert _spectral_gap([3.0]) == 3.0


def test_spectral_gap_zero_second() -> None:
    result = _spectral_gap([5.0, 0.0])
    assert result == 5.0  # avoids division by zero


# ── _svd_metrics ──────────────────────────────────────────────────────────────

def test_svd_metrics_shape() -> None:
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((64, 32)).astype(np.float32)
    top_sv, gap, er, frob = _svd_metrics(arr, top_k=8, max_rows=512)
    assert len(top_sv) == 8
    assert gap >= 1.0
    assert er >= 1.0
    assert frob > 0.0


def test_svd_metrics_subsampling() -> None:
    # With max_rows=4 we subsample; result should still be valid
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((200, 50)).astype(np.float32)
    top_sv, gap, er, frob = _svd_metrics(arr, top_k=4, max_rows=4)
    assert len(top_sv) == 4
    assert all(v >= 0.0 for v in top_sv)


def test_svd_metrics_identity_high_gap() -> None:
    # Rank-1 matrix has spectral gap = sv[0]/sv[1] ≫ 1
    arr = np.zeros((32, 32), dtype=np.float32)
    arr[0, 0] = 10.0
    arr[1, 1] = 0.001
    top_sv, gap, er, _ = _svd_metrics(arr, top_k=4, max_rows=512)
    assert gap > 5.0
    assert er < 2.0


# ── LayerAnalysis ─────────────────────────────────────────────────────────────

def test_layer_analysis_star_rating_loops() -> None:
    # Layer at 60% depth, high spectral gap → should get 3 stars for loops
    a = LayerAnalysis(
        layer_idx=16, proj_type="down_proj",
        shape=(4096, 1024), top_sv=[5.0, 2.0], spectral_gap=2.5,
        effective_rank=50.0, frob_norm=100.0,
    )
    assert a.star_rating(28, "loops") == 3


def test_layer_analysis_star_rating_early_layer() -> None:
    # Layer 0 (position 0%) → low score for loops (loops tend to be mid-network)
    a = LayerAnalysis(
        layer_idx=0, proj_type="down_proj",
        shape=(4096, 1024), top_sv=[2.0, 1.9], spectral_gap=1.05,
        effective_rank=200.0, frob_norm=50.0,
    )
    assert a.star_rating(28, "loops") == 1


# ── WeightMap ─────────────────────────────────────────────────────────────────

def _make_weight_map() -> WeightMap:
    analyses = [
        LayerAnalysis(
            layer_idx=i,
            proj_type="down_proj",
            shape=(4096, 1024),
            top_sv=[2.0 + i * 0.1, 1.5],
            spectral_gap=1.0 + i * 0.05,
            effective_rank=200.0 - i * 2,
            frob_norm=100.0,
        )
        for i in range(28)
    ]
    return WeightMap(
        model_path="/fake/model",
        n_layers=28,
        quantized=True,
        quantization_bits=4,
        elapsed_s=3.5,
        analyses=analyses,
    )


def test_weight_map_top_candidates() -> None:
    wm = _make_weight_map()
    top = wm.top_candidates(obstruction="loops", proj_type="down_proj", n=5)
    assert len(top) == 5
    # All are down_proj
    assert all(c.proj_type == "down_proj" for c in top)


def test_weight_map_report_contains_model_path() -> None:
    wm = _make_weight_map()
    report = wm.report(obstruction="loops", n=5)
    assert "/fake/model" in report
    assert "loops" in report.lower() or "LOOPS" in report


def test_weight_map_roundtrip_json() -> None:
    wm = _make_weight_map()
    d = wm.to_json()
    wm2 = WeightMap.from_json(d)
    assert wm2.n_layers == wm.n_layers
    assert wm2.quantized == wm.quantized
    assert len(wm2.analyses) == len(wm.analyses)
    assert abs(wm2.analyses[5].spectral_gap - wm.analyses[5].spectral_gap) < 1e-6


# ── WeightSpaceAnalyzer (mocked) ──────────────────────────────────────────────

def _make_fake_weights(n_layers: int = 4) -> dict:
    """Build a minimal fake weight dict (float16, no quantization)."""
    import mlx.core as mx
    rng = np.random.default_rng(0)
    weights = {}
    for i in range(n_layers):
        for ptype in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{i}.mlp.{ptype}.weight"
            arr = rng.standard_normal((32, 16)).astype(np.float32)
            weights[key] = mx.array(arr)
    return weights


def test_analyzer_scan_fp16(tmp_path: Path) -> None:
    # Write a minimal config.json
    cfg = {"num_hidden_layers": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    # Create a fake .safetensors file (we'll mock mx.load)
    fake_st = tmp_path / "model.safetensors"
    fake_st.touch()

    with patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=_make_fake_weights(4)):
        analyzer = WeightSpaceAnalyzer(tmp_path)
        wm = analyzer.scan(top_k_sv=4, max_rows=32)

    assert wm.n_layers == 4
    assert wm.quantized is False
    assert len(wm.analyses) == 4 * 3  # 4 layers × 3 proj types
    assert all(a.spectral_gap >= 1.0 for a in wm.analyses)


def test_analyzer_scan_no_safetensors(tmp_path: Path) -> None:
    analyzer = WeightSpaceAnalyzer(tmp_path)
    with pytest.raises(FileNotFoundError):
        analyzer.scan()


def test_analyzer_scan_with_progress(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    messages: list[str] = []
    with patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=_make_fake_weights(2)):
        analyzer = WeightSpaceAnalyzer(tmp_path)
        wm = analyzer.scan(top_k_sv=4, max_rows=16, progress_fn=messages.append)

    assert len(messages) > 0
    assert wm.n_layers == 2


# ── find_model_path ───────────────────────────────────────────────────────────

def test_find_model_path_nonexistent() -> None:
    result = find_model_path("this-model-definitely-does-not-exist-xyz123")
    assert result is None


# ── _cmd_locate ───────────────────────────────────────────────────────────────

def test_cmd_locate_no_model_path(tmp_path: Path) -> None:
    args = argparse.Namespace(
        model="nonexistent-model-xyz",
        model_path=None,
        obstruction="loops",
        proj_type=None,
        top=5,
        output_json=None,
    )
    with patch("ollama_forge.wsp.analyzer.find_model_path", return_value=None):
        rc = _cmd_locate(argparse.ArgumentParser(), args)
    assert rc == 2


def test_cmd_locate_with_explicit_path(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()

    args = argparse.Namespace(
        model="my-model",
        model_path=str(tmp_path),
        obstruction="loops",
        proj_type=None,
        top=5,
        output_json=None,
    )
    with patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=_make_fake_weights(2)):
        rc = _cmd_locate(argparse.ArgumentParser(), args)
    assert rc == 0


def test_cmd_locate_saves_json(tmp_path: Path) -> None:
    cfg = {"num_hidden_layers": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").touch()
    out_json = tmp_path / "wmap.json"

    args = argparse.Namespace(
        model="my-model",
        model_path=str(tmp_path),
        obstruction="loops",
        proj_type=None,
        top=3,
        output_json=str(out_json),
    )
    with patch("ollama_forge.wsp.analyzer._load_all_weights", return_value=_make_fake_weights(2)):
        rc = _cmd_locate(argparse.ArgumentParser(), args)
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "analyses" in data
    assert data["n_layers"] == 2
