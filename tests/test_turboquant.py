"""Tests for TurboQuant core algorithms."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ollama_forge.turboquant import (  # noqa: E402
    CompressionStats,
    _apply_structured_rotation,
    _apply_structured_rotation_inverse,
    _generate_structured_rotation_params,
    _get_codebook,
    _next_power_of_2,
    _scalar_dequantize,
    _scalar_quantize,
    dequantize_tensor,
    detect_outlier_channels,
    effective_bits,
    fast_hadamard_inverse,
    fast_hadamard_transform,
    generate_qjl_matrix,
    generate_rotation_matrix,
    pack_indices,
    pack_signs,
    qjl_dequantize,
    qjl_quantize,
    quantize_tensor,
    unpack_indices,
    unpack_signs,
)


class TestRotationMatrix:
    def test_orthogonal(self):
        Q = generate_rotation_matrix(64, device=torch.device("cpu"), seed=42)
        eye = Q @ Q.T
        assert torch.allclose(eye, torch.eye(64), atol=1e-5)

    def test_deterministic_seed(self):
        Q1 = generate_rotation_matrix(32, device=torch.device("cpu"), seed=99)
        Q2 = generate_rotation_matrix(32, device=torch.device("cpu"), seed=99)
        assert torch.allclose(Q1, Q2)

    def test_different_seeds(self):
        Q1 = generate_rotation_matrix(32, device=torch.device("cpu"), seed=1)
        Q2 = generate_rotation_matrix(32, device=torch.device("cpu"), seed=2)
        assert not torch.allclose(Q1, Q2)


class TestCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_size(self, bits):
        cb = _get_codebook(bits, 128, torch.device("cpu"))
        assert cb.shape == (2**bits,)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_sorted(self, bits):
        cb = _get_codebook(bits, 128, torch.device("cpu"))
        assert (cb[1:] >= cb[:-1]).all()

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_symmetric(self, bits):
        cb = _get_codebook(bits, 128, torch.device("cpu"))
        assert torch.allclose(cb, -cb.flip(0), atol=1e-5)


class TestScalarQuantize:
    def test_roundtrip_2bit(self):
        cb = _get_codebook(2, 64, torch.device("cpu"))
        # Values near centroids should roundtrip
        x = cb.clone()
        idx = _scalar_quantize(x, cb)
        recon = _scalar_dequantize(idx, cb)
        assert torch.allclose(recon, x)

    def test_index_range(self):
        cb = _get_codebook(3, 128, torch.device("cpu"))
        x = torch.randn(100) / math.sqrt(128)
        idx = _scalar_quantize(x, cb)
        assert idx.min() >= 0
        assert idx.max() <= 7


class TestPackUnpack:
    @pytest.mark.parametrize("bits", [1, 2, 4])
    def test_roundtrip(self, bits):
        n = 137  # not a multiple of 8
        maxval = (1 << bits) - 1
        indices = torch.randint(0, maxval + 1, (n,), dtype=torch.uint8)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, n)
        assert (unpacked == indices.long()).all()

    def test_3bit_roundtrip(self):
        # 3-bit doesn't divide 8 evenly — packing still handles it
        # (we use 2 values per byte for 3-bit and waste 2 bits)
        # Actually 8//3=2 with waste. Let's verify with small example.
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=torch.uint8)
        packed = pack_indices(indices, 3)
        unpacked = unpack_indices(packed, 3, 10)
        assert (unpacked == indices.long()).all()


class TestSignPacking:
    def test_roundtrip(self):
        n = 200
        signs = torch.where(torch.rand(n) > 0.5, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
        packed = pack_signs(signs)
        unpacked = unpack_signs(packed, n)
        assert (unpacked == signs).all()


class TestQJL:
    def test_unbiased(self):
        """QJL should produce an unbiased inner-product estimator."""
        d = 128
        x = torch.randn(d)
        x = x / x.norm()
        y = torch.randn(d)

        true_ip = (x @ y).item()

        # Average over many random projections → should converge to true IP
        estimates = []
        for seed in range(100):
            S_i = generate_qjl_matrix(d, device=torch.device("cpu"), seed=seed)
            signs = qjl_quantize(x.unsqueeze(0), S_i)
            recon = qjl_dequantize(signs, S_i, 1.0)
            est = (recon.squeeze(0) @ y).item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        # Should be within ~0.3 of true IP with 100 samples
        assert abs(mean_est - true_ip) < 0.5, f"QJL bias: {mean_est:.3f} vs {true_ip:.3f}"


class TestQuantizeTensor:
    def test_basic_roundtrip(self):
        """Quantize → dequantize should preserve approximate structure."""
        W = torch.randn(32, 64)
        qt = quantize_tensor(W, bits=4, rotation_seed=42)
        W_hat = dequantize_tensor(qt)
        # At 4 bits per coordinate, MSE should be small
        mse = (W - W_hat).pow(2).mean().item()
        baseline = W.pow(2).mean().item()
        # Relative error should be reasonable (< 50% at 4-bit)
        assert mse / baseline < 0.5, f"MSE too high: {mse/baseline:.3f}"

    def test_shape_preserved(self):
        W = torch.randn(16, 128)
        qt = quantize_tensor(W, bits=3)
        W_hat = dequantize_tensor(qt)
        assert W_hat.shape == W.shape

    def test_with_outliers(self):
        W = torch.randn(16, 64)
        # Make some columns much larger
        W[:, :4] *= 10.0
        qt = quantize_tensor(W, bits=2, outlier_channels=4, outlier_bits=4)
        W_hat = dequantize_tensor(qt)
        assert W_hat.shape == W.shape
        assert qt.outlier_indices is not None
        assert len(qt.outlier_indices) == 4

    def test_with_qjl(self):
        W = torch.randn(16, 64)
        qt = quantize_tensor(W, bits=3, use_qjl=True)
        W_hat = dequantize_tensor(qt)
        assert W_hat.shape == W.shape
        assert qt.use_qjl
        assert qt.qjl_packed_signs is not None

    def test_higher_bits_lower_error(self):
        """4-bit should be more accurate than 2-bit."""
        W = torch.randn(32, 128)
        qt2 = quantize_tensor(W, bits=2, rotation_seed=42)
        qt4 = quantize_tensor(W, bits=4, rotation_seed=42)
        mse2 = (W - dequantize_tensor(qt2)).pow(2).mean().item()
        mse4 = (W - dequantize_tensor(qt4)).pow(2).mean().item()
        assert mse4 < mse2


class TestOutlierDetection:
    def test_finds_large_columns(self):
        W = torch.randn(32, 64)
        W[:, 5] *= 100  # make column 5 an outlier
        W[:, 20] *= 100
        idx = detect_outlier_channels(W, n_outliers=2)
        assert 5 in idx
        assert 20 in idx


class TestEffectiveBits:
    def test_basic(self):
        assert effective_bits(128, 3) == pytest.approx(3.125, abs=0.01)

    def test_with_outliers(self):
        eff = effective_bits(128, 2, outlier_channels=32, outlier_bits=3)
        expected = (96 * 2 + 32 * 3) / 128 + 16 / 128  # + norm overhead
        assert eff == pytest.approx(expected, abs=0.01)

    def test_with_qjl(self):
        eff = effective_bits(128, 3, use_qjl=True)
        assert eff > 4.0  # 3 + 1 (QJL) + small norm overhead


class TestCompressionStats:
    def test_finalize(self):
        stats = CompressionStats()
        stats.add_layer("w1", (128, 256), 3.0, 128 * 256 * 2, 128 * 256 * 3 // 8)
        stats.add_layer("w2", (256, 128), 3.0, 256 * 128 * 2, 256 * 128 * 3 // 8)
        stats.finalize()
        assert stats.compression_ratio > 1.0
        assert stats.effective_bits_avg == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Fast Hadamard / Structured Rotation tests
# ---------------------------------------------------------------------------

class TestNextPowerOf2:
    def test_exact(self):
        assert _next_power_of_2(64) == 64
        assert _next_power_of_2(1024) == 1024

    def test_non_power(self):
        assert _next_power_of_2(65) == 128
        assert _next_power_of_2(1000) == 1024
        assert _next_power_of_2(1025) == 2048


class TestFastHadamard:
    def test_shape_preserved(self):
        x = torch.randn(4, 64)
        y = fast_hadamard_transform(x)
        assert y.shape == x.shape

    def test_orthonormal(self):
        """Hadamard of identity rows should produce orthonormal columns."""
        d = 32
        eye = torch.eye(d)
        H = fast_hadamard_transform(eye)
        # H^T @ H should be identity (orthonormal)
        prod = H.T @ H
        assert torch.allclose(prod, torch.eye(d), atol=1e-5)

    def test_self_inverse(self):
        """Normalized WHT is its own inverse: H(H(x)) = x."""
        x = torch.randn(8, 32)
        y = fast_hadamard_transform(x)
        x_back = fast_hadamard_inverse(y)
        assert torch.allclose(x_back, x, atol=1e-5)

    def test_norm_preserving(self):
        """Orthogonal transform preserves L2 norm."""
        x = torch.randn(16, 64)
        y = fast_hadamard_transform(x)
        x_norms = x.norm(dim=1)
        y_norms = y.norm(dim=1)
        assert torch.allclose(x_norms, y_norms, atol=1e-4)


class TestStructuredRotation:
    def test_roundtrip(self):
        """Structured rotation → inverse should recover original."""
        d = 64
        d_padded = _next_power_of_2(d)
        x = torch.randn(8, d_padded)
        signs, perm = _generate_structured_rotation_params(d_padded, device=torch.device("cpu"), seed=42)
        y = _apply_structured_rotation(x, signs, perm, d)
        x_back = _apply_structured_rotation_inverse(y, signs, perm, d)
        assert torch.allclose(x_back, x, atol=1e-5)

    def test_deterministic(self):
        d = 32
        signs1, perm1 = _generate_structured_rotation_params(d, device=torch.device("cpu"), seed=99)
        signs2, perm2 = _generate_structured_rotation_params(d, device=torch.device("cpu"), seed=99)
        assert torch.allclose(signs1, signs2)
        assert (perm1 == perm2).all()

    def test_norm_preserving(self):
        """Structured rotation should preserve vector norms."""
        d = 128
        x = torch.randn(4, d)
        signs, perm = _generate_structured_rotation_params(d, device=torch.device("cpu"), seed=42)
        y = _apply_structured_rotation(x, signs, perm, d)
        assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-4)


class TestStructuredRotationInQuantize:
    def test_large_dim_uses_structured(self):
        """Quantize/dequantize with d > 1024 should use structured rotation."""
        from ollama_forge.turboquant import use_structured_rotation
        assert not use_structured_rotation(512)
        assert not use_structured_rotation(1024)
        assert use_structured_rotation(2048)
        assert use_structured_rotation(4096)

    def test_large_dim_roundtrip(self):
        """Quantize → dequantize with structured rotation should work."""
        # Use d=2048 to trigger structured path
        W = torch.randn(4, 2048)
        qt = quantize_tensor(W, bits=3, rotation_seed=42)
        W_hat = dequantize_tensor(qt)
        assert W_hat.shape == W.shape
        # Should have reasonable quality
        mse = (W - W_hat).pow(2).mean().item()
        baseline = W.pow(2).mean().item()
        assert mse / baseline < 0.5


class TestBackendDetection:
    def test_get_turboquant_backend(self):
        from ollama_forge.device import get_turboquant_backend
        # Explicit preference should be returned as-is
        assert get_turboquant_backend("pytorch") == "pytorch"
        assert get_turboquant_backend("mlx") == "mlx"
        assert get_turboquant_backend("triton") == "triton"

    def test_triton_availability_check(self):
        from ollama_forge.turboquant_kernels import is_triton_available
        # Should return bool without crashing
        assert isinstance(is_triton_available(), bool)


# ---------------------------------------------------------------------------
# Class-based API tests (mirrors reference turboquant_plus)
# ---------------------------------------------------------------------------

from ollama_forge.turboquant import (  # noqa: E402
    CompressedKVCache,
    CompressedVector,
    KVCacheCompressor,
    OutlierTurboQuant,
    PolarQuant,
    QJLQuantizer,
    TurboQuant,
    TurboQuantMSE,
)


class TestPolarQuant:
    def test_roundtrip_shape(self):
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(64)
        indices, norms = pq.quantize(x)
        assert indices.shape == (64,)
        x_hat = pq.dequantize(indices, norms)
        assert x_hat.shape == (64,)

    def test_batch_roundtrip(self):
        pq = PolarQuant(d=128, bit_width=3, seed=42)
        x = torch.randn(8, 128)
        indices, norms = pq.quantize(x)
        assert indices.shape == (8, 128)
        assert norms.shape == (8,)
        x_hat = pq.dequantize(indices, norms)
        assert x_hat.shape == (8, 128)

    def test_norm_preservation(self):
        """Original L2 norm should be roughly preserved after dequantize."""
        pq = PolarQuant(d=128, bit_width=4, seed=42)
        x = torch.randn(128) * 3.0  # non-unit norm
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        assert abs(x_hat.norm().item() - x.norm().item()) / x.norm().item() < 0.3

    def test_quantize_and_residual(self):
        pq = PolarQuant(d=64, bit_width=2, seed=42)
        x = torch.randn(64)
        indices, norms, residual = pq.quantize_and_residual(x)
        x_hat = pq.dequantize(indices, norms)
        assert torch.allclose(residual, x - x_hat, atol=1e-5)


class TestQJLQuantizer:
    def test_roundtrip_shape(self):
        qjl = QJLQuantizer(d=64, seed=42)
        r = torch.randn(64)
        signs, norms = qjl.quantize(r)
        assert signs.shape == (64,)
        r_hat = qjl.dequantize(signs, norms)
        assert r_hat.shape == (64,)

    def test_batch(self):
        qjl = QJLQuantizer(d=32, seed=42)
        r = torch.randn(4, 32)
        signs, norms = qjl.quantize(r)
        assert signs.shape == (4, 32)
        assert norms.shape == (4,)
        r_hat = qjl.dequantize(signs, norms)
        assert r_hat.shape == (4, 32)

    def test_signs_are_pm1(self):
        qjl = QJLQuantizer(d=128, seed=42)
        r = torch.randn(128)
        signs, _ = qjl.quantize(r)
        assert ((signs == 1) | (signs == -1)).all()


class TestTurboQuantClass:
    def test_requires_min_bits(self):
        with pytest.raises(ValueError):
            TurboQuant(d=64, bit_width=1)

    def test_roundtrip(self):
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        x = torch.randn(128)
        compressed = tq.quantize(x)
        assert isinstance(compressed, CompressedVector)
        assert compressed.bit_width == 3
        x_hat = tq.dequantize(compressed)
        assert x_hat.shape == (128,)

    def test_batch_roundtrip(self):
        tq = TurboQuant(d=64, bit_width=4, seed=42)
        x = torch.randn(8, 64)
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)
        assert x_hat.shape == (8, 64)

    def test_quality_improves_with_bits(self):
        """Higher bits → lower MSE."""
        x = torch.randn(16, 128)
        tq2 = TurboQuant(d=128, bit_width=2, seed=42)
        tq4 = TurboQuant(d=128, bit_width=4, seed=42)
        mse2 = (x - tq2.dequantize(tq2.quantize(x))).pow(2).mean().item()
        mse4 = (x - tq4.dequantize(tq4.quantize(x))).pow(2).mean().item()
        assert mse4 < mse2

    def test_compression_ratio(self):
        tq = TurboQuant(d=128, bit_width=3)
        ratio = tq.compression_ratio(original_bits=16)
        assert ratio > 4.0  # 16 / (3 + 32/128) ≈ 4.9

    def test_inner_product_correlation(self):
        """Approximate inner products should correlate with true ones."""
        d = 128
        tq = TurboQuant(d=d, bit_width=4, seed=42)
        torch.manual_seed(0)
        true_ips = []
        approx_ips = []
        for _ in range(50):
            x = torch.randn(d)
            y = torch.randn(d)
            true_ips.append((x @ y).item())
            approx_ips.append(
                (tq.dequantize(tq.quantize(x)) @ tq.dequantize(tq.quantize(y))).item()
            )
        # Pearson correlation should be positive (reconstruction is useful)
        t = torch.tensor(true_ips)
        a = torch.tensor(approx_ips)
        corr = ((t - t.mean()) * (a - a.mean())).sum() / (t.std() * a.std() * len(t))
        assert corr > 0.3, f"IP correlation too low: {corr:.3f}"


class TestTurboQuantMSEClass:
    def test_roundtrip(self):
        tqm = TurboQuantMSE(d=128, bit_width=3, seed=42)
        x = torch.randn(128)
        indices, norms = tqm.quantize(x)
        x_hat = tqm.dequantize(indices, norms)
        assert x_hat.shape == (128,)

    def test_batch(self):
        tqm = TurboQuantMSE(d=64, bit_width=4, seed=42)
        x = torch.randn(8, 64)
        indices, norms = tqm.quantize(x)
        x_hat = tqm.dequantize(indices, norms)
        assert x_hat.shape == (8, 64)


class TestKVCacheCompressor:
    def test_compress_decompress(self):
        compressor = KVCacheCompressor(head_dim=32, k_bits=3, v_bits=3)
        num_layers, num_heads, seq_len = 2, 4, 8
        k = torch.randn(num_layers, num_heads, seq_len, 32)
        v = torch.randn(num_layers, num_heads, seq_len, 32)
        compressed = compressor.compress(k, v)
        assert isinstance(compressed, CompressedKVCache)
        assert compressed.num_layers == 2
        assert compressed.num_heads == 4
        assert compressed.seq_len == 8
        k_hat, v_hat = compressor.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_memory_stats(self):
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)
        stats = compressor.memory_stats(seq_len=1024, num_layers=32, num_heads=32)
        # K uses 3 bits + 32-bit norm, V uses 3 bits → ~2.5x compression at 3-bit
        assert stats["compression_ratio"] > 2.0
        assert stats["original_mb"] > stats["compressed_mb"]


class TestOutlierTurboQuant:
    def test_effective_bits_2_5(self):
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        assert abs(oq.effective_bits - 2.5) < 0.1

    def test_effective_bits_3_5(self):
        oq = OutlierTurboQuant(d=128, target_bits=3.5, seed=42)
        assert abs(oq.effective_bits - 3.5) < 0.1

    def test_quantize_shape(self):
        oq = OutlierTurboQuant(d=64, target_bits=2.5, seed=42)
        x = torch.randn(64)
        compressed = oq.quantize(x)
        assert isinstance(compressed, CompressedVector)
        assert compressed.qjl_signs.shape == (64,)

    def test_compression_ratio(self):
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        ratio = oq.compression_ratio(original_bits=16)
        assert ratio > 4.0  # 16 / ~3.25 ≈ 4.9


# ---------------------------------------------------------------------------
# Layer-Adaptive KV Cache tests
# ---------------------------------------------------------------------------

from ollama_forge.turboquant import (  # noqa: E402
    LayerAdaptiveKVCacheCompressor,
    LayerAdaptivePolicy,
    TemporalDecayManager,
)


class TestLayerAdaptivePolicy:
    def test_mode_0_uniform(self):
        """Mode 0: all layers get base_bits."""
        policy = LayerAdaptivePolicy(num_layers=40, mode=0, base_bits=3)
        for i in range(40):
            assert policy.kv_bits(i) == 3
        assert len(policy.protected_layers) == 0

    def test_mode_2_last_n_protected(self):
        """Mode 2: last N layers use protected_bits (0 = no compression)."""
        policy = LayerAdaptivePolicy(
            num_layers=40, mode=2, base_bits=3,
            protected_bits=0, n_protected=8,
        )
        # First 32 layers: compressed at 3 bits
        for i in range(32):
            assert policy.kv_bits(i) == 3
            assert not policy.is_protected(i)
        # Last 8 layers: full precision
        for i in range(32, 40):
            assert policy.kv_bits(i) == 0
            assert policy.is_protected(i)

    def test_mode_7_boundary(self):
        """Mode 7: first 2 + last 2 layers protected."""
        policy = LayerAdaptivePolicy(
            num_layers=10, mode=7, base_bits=3, protected_bits=0,
        )
        assert policy.is_protected(0)
        assert policy.is_protected(1)
        assert not policy.is_protected(2)
        assert not policy.is_protected(7)
        assert policy.is_protected(8)
        assert policy.is_protected(9)
        assert len(policy.protected_layers) == 4

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            LayerAdaptivePolicy(num_layers=10, mode=99)

    def test_effective_compression(self):
        """Protected layers reduce overall compression."""
        # All compressed at 3 bits
        p_uniform = LayerAdaptivePolicy(num_layers=10, mode=0, base_bits=3)
        # Last 2 uncompressed
        p_adaptive = LayerAdaptivePolicy(
            num_layers=10, mode=2, base_bits=3,
            protected_bits=0, n_protected=2,
        )
        # Adaptive should have lower compression ratio (some layers uncompressed)
        assert p_adaptive.effective_compression() < p_uniform.effective_compression()

    def test_small_model(self):
        """Edge case: fewer layers than n_protected."""
        policy = LayerAdaptivePolicy(
            num_layers=4, mode=2, base_bits=3,
            protected_bits=0, n_protected=8,
        )
        # All layers protected since n_protected > num_layers
        for i in range(4):
            assert policy.is_protected(i)


class TestLayerAdaptiveKVCacheCompressor:
    def test_compress_decompress_mode_0(self):
        """Mode 0 should behave identically to KVCacheCompressor."""
        head_dim = 64
        nl, nh, sl = 4, 2, 8
        compressor = LayerAdaptiveKVCacheCompressor(
            head_dim=head_dim, num_layers=nl, base_bits=3, mode=0,
        )
        k = torch.randn(nl, nh, sl, head_dim)
        v = torch.randn(nl, nh, sl, head_dim)

        compressed = compressor.compress(k, v)
        k_hat, v_hat = compressor.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_compress_decompress_mode_2(self):
        """Mode 2: protected layers should be lossless."""
        head_dim = 64
        nl, nh, sl = 6, 2, 8
        compressor = LayerAdaptiveKVCacheCompressor(
            head_dim=head_dim, num_layers=nl, base_bits=3, mode=2,
            n_protected=2,
        )
        k = torch.randn(nl, nh, sl, head_dim)
        v = torch.randn(nl, nh, sl, head_dim)

        compressed = compressor.compress(k, v)
        k_hat, v_hat = compressor.decompress(compressed)
        assert k_hat.shape == k.shape

        # Last 2 layers should be perfectly preserved (raw storage)
        for layer_idx in [4, 5]:
            assert torch.allclose(k[layer_idx], k_hat[layer_idx])
            assert torch.allclose(v[layer_idx], v_hat[layer_idx])

        # First 4 layers should have some quantization error
        for layer_idx in range(4):
            k_err = (k[layer_idx] - k_hat[layer_idx]).abs().mean().item()
            assert k_err > 0  # not lossless

    def test_memory_stats(self):
        compressor = LayerAdaptiveKVCacheCompressor(
            head_dim=128, num_layers=10, base_bits=3, mode=2, n_protected=2,
        )
        stats = compressor.memory_stats(seq_len=1024, num_heads=32)
        assert stats["compression_ratio"] > 1.0
        assert stats["n_protected"] == 2
        assert stats["mode"] == 2


# ---------------------------------------------------------------------------
# Temporal Decay tests
# ---------------------------------------------------------------------------


class TestTemporalDecayManager:
    def test_requires_lower_target(self):
        with pytest.raises(ValueError):
            TemporalDecayManager(d=64, source_bits=3, target_bits=3)

    def test_no_decay_before_interval(self):
        """Decay should only trigger at decay_interval steps."""
        decay = TemporalDecayManager(
            d=64, source_bits=3, target_bits=2,
            decay_interval=4, batch_size=8,
        )
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(32, 64)
        indices, norms = pq.quantize(x)

        # Steps 1-3 should not trigger decay
        for _step in range(3):
            new_idx, new_norms, did_decay = decay.maybe_decay(indices, norms, total_seq_len=32)
            assert not did_decay

    def test_decay_triggers_at_interval(self):
        """Decay should trigger every decay_interval steps."""
        decay = TemporalDecayManager(
            d=64, source_bits=3, target_bits=2,
            decay_interval=4, batch_size=64,
            sink_len=2,
        )
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(32, 64)
        indices, norms = pq.quantize(x)

        # Run 4 steps to trigger first decay
        for _step in range(4):
            new_idx, new_norms, did_decay = decay.maybe_decay(
                indices, norms, total_seq_len=32, recent_window=4,
            )
        assert did_decay
        assert decay.n_decayed > 0

    def test_sinks_exempted(self):
        """Attention sinks (positions 0..sink_len-1) should never be decayed."""
        decay = TemporalDecayManager(
            d=64, source_bits=3, target_bits=2,
            decay_interval=1, batch_size=100,
            sink_len=4,
        )
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(20, 64)
        indices, norms = pq.quantize(x)

        # Run enough steps to decay everything eligible
        for _step in range(10):
            indices, norms, _ = decay.maybe_decay(
                indices, norms, total_seq_len=20, recent_window=2,
            )

        # Sink positions (0-3) should not be in decayed set
        for pos in range(4):
            assert pos not in decay._decayed_positions

    def test_recent_window_protected(self):
        """Tokens in the recent window should not be decayed."""
        decay = TemporalDecayManager(
            d=64, source_bits=3, target_bits=2,
            decay_interval=1, batch_size=100,
            sink_len=0,
        )
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(16, 64)
        indices, norms = pq.quantize(x)

        # recent_window=8 protects last 8 positions
        for _step in range(10):
            indices, norms, _ = decay.maybe_decay(
                indices, norms, total_seq_len=16, recent_window=8,
            )

        # Positions 8-15 should never be decayed
        for pos in range(8, 16):
            assert pos not in decay._decayed_positions

    def test_batch_size_limit(self):
        """Only batch_size positions should be decayed per interval."""
        decay = TemporalDecayManager(
            d=64, source_bits=3, target_bits=2,
            decay_interval=1, batch_size=4,
            sink_len=0,
        )
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(32, 64)
        indices, norms = pq.quantize(x)

        # First decay: should only decay 4 positions
        _, _, did_decay = decay.maybe_decay(
            indices, norms, total_seq_len=32, recent_window=4,
        )
        assert did_decay
        assert decay.n_decayed == 4

    def test_requantize_quality(self):
        """Requantized vectors should still correlate with originals."""
        decay = TemporalDecayManager(
            d=128, source_bits=3, target_bits=2,
            decay_interval=1, batch_size=100,
            sink_len=0,
        )
        pq_source = PolarQuant(d=128, bit_width=3, seed=42)
        x = torch.randn(16, 128)
        indices, norms = pq_source.quantize(x)

        # Get source reconstruction for comparison
        x_source = pq_source.dequantize(indices, norms)

        # Decay all positions
        new_idx, new_norms, _ = decay.maybe_decay(
            indices, norms, total_seq_len=16, recent_window=0,
        )

        # Requantized vectors should still be in the right ballpark
        # Use the source PQ to dequant (the norms carry the scale info)
        x_decayed = decay._target_pq.dequantize(new_idx, new_norms)

        # Check cosine similarity between source and decayed
        cos_sims = torch.nn.functional.cosine_similarity(x_source, x_decayed, dim=1)
        assert cos_sims.mean().item() > 0.5  # reasonable for 3→2 bit

    def test_memory_savings_ratio(self):
        """Memory savings ratio should reflect the bit reduction."""
        decay = TemporalDecayManager(
            d=128, source_bits=3, target_bits=2,
            decay_interval=1, batch_size=100,
            sink_len=0,
        )
        pq = PolarQuant(d=128, bit_width=3, seed=42)
        x = torch.randn(100, 128)
        indices, norms = pq.quantize(x)

        # Decay everything
        decay.maybe_decay(indices, norms, total_seq_len=100, recent_window=0)

        ratio = decay.memory_savings_ratio(100)
        # 2/3 = 0.667 — all positions decayed from 3→2 bits
        assert ratio < 0.75
        assert ratio > 0.5

    def test_reset_clears_state(self):
        decay = TemporalDecayManager(
            d=64, source_bits=3, target_bits=2,
            decay_interval=1, batch_size=100,
        )
        pq = PolarQuant(d=64, bit_width=3, seed=42)
        x = torch.randn(16, 64)
        indices, norms = pq.quantize(x)

        decay.maybe_decay(indices, norms, total_seq_len=16, recent_window=0)
        assert decay.n_decayed > 0

        decay.reset()
        assert decay.n_decayed == 0
        assert decay._step_count == 0
