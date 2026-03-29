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
