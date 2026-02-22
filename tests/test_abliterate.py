"""Unit tests for abliterate module: _strength_kernel_scale and get_D_for_layer."""

import math

import pytest

from ollama_forge.abliterate import _strength_kernel_scale, get_D_for_layer


class TestStrengthKernelScale:
    """Tests for _strength_kernel_scale."""

    def test_constant_returns_one(self) -> None:
        assert _strength_kernel_scale(0, 10, "constant", 0.5, 0.4) == 1.0
        assert _strength_kernel_scale(5, 10, "constant", 0.0, 1.0) == 1.0

    def test_zero_layers_returns_one(self) -> None:
        assert _strength_kernel_scale(0, 0, "linear_peak", 0.5, 0.4) == 1.0

    def test_linear_peak_center(self) -> None:
        # center_frac=0.5, 10 layers -> layer 4 (0.45) and 5 (0.55) are near center
        n = 10
        center = 0.5
        width = 0.4
        # layer 4: x = 0.45, dist = 0.05, half_width = 0.2 -> inside, scale = 1 - 0.05/0.2 = 0.75
        s = _strength_kernel_scale(4, n, "linear_peak", center, width)
        assert s == pytest.approx(0.75)
        # layer 5: x = 0.55, dist = 0.05 -> same
        assert _strength_kernel_scale(5, n, "linear_peak", center, width) == pytest.approx(0.75)

    def test_linear_peak_at_center(self) -> None:
        # Exactly at center: layer such that (i+0.5)/n = 0.5 -> i = 4.5, so layer 4 or 5
        s4 = _strength_kernel_scale(4, 10, "linear_peak", 0.5, 0.4)
        s5 = _strength_kernel_scale(5, 10, "linear_peak", 0.5, 0.4)
        assert s4 == pytest.approx(1.0 - abs(0.45 - 0.5) / 0.2)
        assert s5 == pytest.approx(1.0 - abs(0.55 - 0.5) / 0.2)

    def test_linear_peak_far_zero(self) -> None:
        # Layer 0 with center 0.5, width 0.4: x=0.05, dist=0.45 >= half_width 0.2 -> 0
        assert _strength_kernel_scale(0, 10, "linear_peak", 0.5, 0.4) == 0.0
        assert _strength_kernel_scale(9, 10, "linear_peak", 0.5, 0.4) == 0.0

    def test_gaussian_peak_at_center(self) -> None:
        # x = center -> exp(0) = 1
        n = 10
        center = 0.5
        # layer 4: (4+0.5)/10 = 0.45
        s = _strength_kernel_scale(4, n, "gaussian", center, 0.4)
        expected = math.exp(-((0.45 - 0.5) ** 2) / (2 * 0.4**2))
        assert s == pytest.approx(expected)

    def test_gaussian_center_layer(self) -> None:
        # For 2 layers, center 0.5: layer 0 -> 0.25, layer 1 -> 0.75. Middle is 0.5 -> layer 1 (0.75 closer?)
        # Actually (1+0.5)/2 = 0.75. So layer 1 gives x=0.75, dist=0.25.
        s = _strength_kernel_scale(1, 2, "gaussian", 0.5, 0.4)
        assert 0 < s <= 1.0
        assert s == pytest.approx(math.exp(-(0.25**2) / (2 * 0.16)))

    def test_unknown_kernel_returns_one(self) -> None:
        assert _strength_kernel_scale(0, 10, "unknown", 0.5, 0.4) == 1.0


class TestGetDForLayer:
    """Tests for get_D_for_layer (requires torch)."""

    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    def test_single_d_when_not_per_layer(self) -> None:
        import torch

        single_d = torch.randn(8, 1)
        out = get_D_for_layer(0, False, None, single_d, None)
        assert out is single_d
        out = get_D_for_layer(3, False, None, single_d, 1)
        assert out is single_d

    def test_per_layer_direction_index_none(self) -> None:
        import torch

        # (2 layers, hidden=4)
        directions_tensor = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=torch.float32)
        single_d = torch.zeros(4, 1)
        # layer_idx 0 -> idx 0, layer_idx 1 -> idx 1, layer_idx 5 -> idx 1 (capped)
        d0 = get_D_for_layer(0, True, directions_tensor, single_d, None)
        assert d0.shape == (4, 1)
        assert d0[0, 0] == pytest.approx(1.0)
        d1 = get_D_for_layer(1, True, directions_tensor, single_d, None)
        assert d1[1, 0] == pytest.approx(1.0)
        d5 = get_D_for_layer(5, True, directions_tensor, single_d, None)
        assert d5[1, 0] == pytest.approx(1.0)

    def test_per_layer_direction_index_int(self) -> None:
        import torch

        directions_tensor = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float32)
        single_d = torch.zeros(3, 1)
        # direction_index=1 -> always second row
        d = get_D_for_layer(0, True, directions_tensor, single_d, 1)
        assert d[1, 0] == pytest.approx(1.0)
        d = get_D_for_layer(2, True, directions_tensor, single_d, 1)
        assert d[1, 0] == pytest.approx(1.0)

    def test_per_layer_direction_index_int_clamped(self) -> None:
        import torch

        directions_tensor = torch.tensor([[1.0, 0], [0, 1.0]], dtype=torch.float32)
        single_d = torch.zeros(2, 1)
        # direction_index=10 -> clamped to 1
        d = get_D_for_layer(0, True, directions_tensor, single_d, 10)
        assert d[1, 0] == pytest.approx(1.0)
        d = get_D_for_layer(0, True, directions_tensor, single_d, -1)
        assert d[0, 0] == pytest.approx(1.0)

    def test_per_layer_direction_index_float_blend(self) -> None:
        import torch

        # Two layers: [1,0] and [0,1]. direction_index=0.5 -> 50% blend -> normalized [1,1]/sqrt(2)
        directions_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        single_d = torch.zeros(2, 1)
        d = get_D_for_layer(0, True, directions_tensor, single_d, 0.5)
        assert d.shape == (2, 1)
        norm = (d ** 2).sum().sqrt().item()
        assert norm == pytest.approx(1.0, abs=1e-5)
        # Blend of [1,0] and [0,1] with alpha=0.5 -> [0.5, 0.5], normalized
        assert d[0, 0] == pytest.approx(1.0 / (2**0.5), abs=1e-5)
        assert d[1, 0] == pytest.approx(1.0 / (2**0.5), abs=1e-5)
