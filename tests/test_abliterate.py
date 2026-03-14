"""Unit tests for abliterate module: _strength_kernel_scale and get_D_for_layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ollama_forge.abliterate import _is_multimodal_model_id, _strength_kernel_scale, get_D_for_layer, get_layers


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


# ---------------------------------------------------------------------------
# Tiny fake model/tokenizer for apply_refusal_dir_and_save tests
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")  # skip entire module if torch missing

HIDDEN = 8
N_LAYERS = 4


class _FakeLinear:
    def __init__(self, out_f: int, in_f: int) -> None:
        self.weight = torch.randn(out_f, in_f)


class _FakeAttn:
    def __init__(self, h: int) -> None:
        self.q_proj = _FakeLinear(h, h)
        self.k_proj = _FakeLinear(h, h)
        self.v_proj = _FakeLinear(h, h)
        self.o_proj = _FakeLinear(h, h)


class _FakeMLP:
    def __init__(self, h: int) -> None:
        self.gate_proj = _FakeLinear(h * 2, h)
        self.up_proj   = _FakeLinear(h * 2, h)
        self.down_proj = _FakeLinear(h, h * 2)


class _FakeLayer:
    def __init__(self, h: int) -> None:
        self.self_attn = _FakeAttn(h)
        self.mlp       = _FakeMLP(h)


class _FakeModelInner:
    def __init__(self, h: int, n: int) -> None:
        self.layers = [_FakeLayer(h) for _ in range(n)]


class _FakeModel:
    def __init__(self, hidden: int = HIDDEN, n_layers: int = N_LAYERS) -> None:
        self.model  = _FakeModelInner(hidden, n_layers)
        self.device = torch.device("cpu")
        self._h     = hidden
        self._n     = n_layers
        self.config = MagicMock()
        self.config.hidden_size = hidden

    def named_parameters(self):
        for i, layer in enumerate(self.model.layers):
            pairs = [
                (f"model.layers.{i}.self_attn.q_proj.weight", layer.self_attn.q_proj.weight),
                (f"model.layers.{i}.self_attn.k_proj.weight", layer.self_attn.k_proj.weight),
                (f"model.layers.{i}.self_attn.v_proj.weight", layer.self_attn.v_proj.weight),
                (f"model.layers.{i}.self_attn.o_proj.weight", layer.self_attn.o_proj.weight),
                (f"model.layers.{i}.mlp.gate_proj.weight",    layer.mlp.gate_proj.weight),
                (f"model.layers.{i}.mlp.up_proj.weight",      layer.mlp.up_proj.weight),
                (f"model.layers.{i}.mlp.down_proj.weight",    layer.mlp.down_proj.weight),
            ]
            yield from pairs

    def save_pretrained(self, save_directory: str, **_kw: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        cfg = {"model_type": "fake", "hidden_size": self._h, "num_hidden_layers": self._n}
        (path / "config.json").write_text(json.dumps(cfg))


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.eos_token_id  = 2

    def __call__(self, text: str, *, return_tensors: str = "pt", **_kw: Any):
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, conversation: Any, *, return_tensors: str = "pt", **_kw: Any):
        return torch.tensor([[1, 2, 3]])

    def save_pretrained(self, save_directory: str, **_kw: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "FakeTokenizer"}))

    def decode(self, ids: Any, **_kw: Any) -> str:
        return "fake output"


def _make_direction_pt(tmp: Path, hidden: int = HIDDEN) -> Path:
    d = torch.randn(hidden, 1)
    d = d / d.norm()
    pt = tmp / "refusal_dir.pt"
    torch.save(d, str(pt))
    return pt


def _apply_fake(
    tmp: Path,
    fake_model: _FakeModel,
    *,
    skip_begin: int = 0,
    skip_end: int = 0,
    norm_preserving: bool = False,
    strength: float = 1.0,
    output_only: bool = False,
) -> None:
    from ollama_forge.abliterate import apply_refusal_dir_and_save

    pt = _make_direction_pt(tmp)
    out = tmp / "checkpoint"
    with (
        patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=fake_model),
        patch("transformers.AutoTokenizer") as mock_tok,
    ):
        mock_tok.from_pretrained.return_value = _FakeTokenizer()
        apply_refusal_dir_and_save(
            "fake/model",
            pt,
            out,
            verify=False,
            skip_begin_layers=skip_begin,
            skip_end_layers=skip_end,
            norm_preserving=norm_preserving,
            strength=strength,
            output_only=output_only,
        )


# ---------------------------------------------------------------------------
# Tests for apply_refusal_dir_and_save
# ---------------------------------------------------------------------------


class TestApplyRefusalDirAndSaveUnit:
    """Unit tests for apply_refusal_dir_and_save weight modification behavior."""

    def test_inner_layers_are_modified(self, tmp_path: Path) -> None:
        """A layer in the ablated range should have its weights changed."""
        model = _FakeModel()
        before = model.model.layers[1].self_attn.q_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        after = model.model.layers[1].self_attn.q_proj.weight
        assert not torch.allclose(before, after), "q_proj should be modified by ablation"

    def test_skip_begin_layers_not_modified(self, tmp_path: Path) -> None:
        """Layer 0 must be untouched when skip_begin_layers=1."""
        model = _FakeModel()
        q0_before = model.model.layers[0].self_attn.q_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=1, skip_end=0)
        q0_after = model.model.layers[0].self_attn.q_proj.weight
        assert torch.allclose(q0_before, q0_after), "Layer 0 should be skipped with skip_begin_layers=1"

    def test_skip_end_layers_not_modified(self, tmp_path: Path) -> None:
        """Last layer must be untouched when skip_end_layers=1."""
        model = _FakeModel()
        last = N_LAYERS - 1
        q_last_before = model.model.layers[last].self_attn.q_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=1)
        q_last_after = model.model.layers[last].self_attn.q_proj.weight
        assert torch.allclose(q_last_before, q_last_after), f"Layer {last} should be skipped with skip_end_layers=1"

    def test_zero_layer_ablation_emits_warning(self, tmp_path: Path) -> None:
        """When skip_begin + skip_end >= n_layers, a warning should be issued via log."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModel()
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint_zero"

        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("ollama_forge.abliterate.log") as mock_log,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=N_LAYERS - 1,  # start_idx = n-1; skip_end=1 -> end_idx = n-1
                skip_end_layers=1,
                norm_preserving=False,
            )
        assert mock_log.warning.called, "log.warning should be called when zero layers are ablated"
        warning_args = " ".join(str(a) for call in mock_log.warning.call_args_list for a in call.args)
        assert "zero layers" in warning_args.lower() or "will not be modified" in warning_args.lower()

    def test_norm_preserving_maintains_frobenius_norm(self, tmp_path: Path) -> None:
        """With norm_preserving=True, the Frobenius norm of each modified weight should be unchanged."""
        model = _FakeModel()
        layer_idx = 1
        q_before_norm = model.model.layers[layer_idx].self_attn.q_proj.weight.norm().item()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0, norm_preserving=True, strength=0.8)
        q_after = model.model.layers[layer_idx].self_attn.q_proj.weight
        assert q_after.norm().item() == pytest.approx(q_before_norm, rel=1e-3), (
            "norm_preserving=True should keep Frobenius norm unchanged"
        )

    def test_hidden_size_mismatch_raises(self, tmp_path: Path) -> None:
        """Applying a direction with wrong hidden size should raise ValueError."""
        wrong_hidden = HIDDEN + 4
        d = torch.randn(wrong_hidden, 1)
        d = d / d.norm()
        pt = tmp_path / "wrong_dir.pt"
        torch.save(d, str(pt))
        out = tmp_path / "checkpoint_bad"

        model = _FakeModel(hidden=HIDDEN)
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
            pytest.raises(ValueError, match="hidden_size"),
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            from ollama_forge.abliterate import apply_refusal_dir_and_save
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=0,
                norm_preserving=False,
            )

    def test_invalid_strength_raises(self, tmp_path: Path) -> None:
        """strength <= 0 should raise ValueError immediately."""
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint_bad_strength"
        with pytest.raises(ValueError, match="strength"):
            from ollama_forge.abliterate import apply_refusal_dir_and_save
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                strength=0.0,
            )

    def test_o_proj_is_modified(self, tmp_path: Path) -> None:
        """o_proj (output projection) should be modified by ablation (left-multiply)."""
        model = _FakeModel()
        o_before = model.model.layers[1].self_attn.o_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        o_after = model.model.layers[1].self_attn.o_proj.weight
        assert not torch.allclose(o_before, o_after), "o_proj should be modified by ablation (left-multiply)"

    def test_mlp_weights_modified(self, tmp_path: Path) -> None:
        """gate_proj and up_proj (input-side MLP) should be modified."""
        model = _FakeModel()
        gate_before = model.model.layers[1].mlp.gate_proj.weight.clone()
        down_before  = model.model.layers[1].mlp.down_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        gate_after = model.model.layers[1].mlp.gate_proj.weight
        down_after  = model.model.layers[1].mlp.down_proj.weight
        assert not torch.allclose(gate_before, gate_after), "gate_proj should be modified"
        assert not torch.allclose(down_before, down_after), "down_proj should be modified (left-multiply)"


class _FakeLinearAttn:
    """Fake linear attention module (Mamba / GatedDeltaNet style)."""

    def __init__(self, h: int) -> None:
        self.in_proj_qkv = _FakeLinear(h * 6, h)
        self.in_proj_z = _FakeLinear(h * 2, h)
        self.in_proj_a = _FakeLinear(16, h)
        self.in_proj_b = _FakeLinear(16, h)
        self.out_proj = _FakeLinear(h, h * 2)


class _FakeLinearAttnLayer:
    """Layer with linear_attn instead of self_attn (e.g. Qwen3.5 hybrid layers)."""

    def __init__(self, h: int) -> None:
        self.linear_attn = _FakeLinearAttn(h)
        self.mlp = _FakeMLP(h)


class _FakeHybridModelInner:
    """Mix of standard and linear-attention layers (like Qwen3.5)."""

    def __init__(self, h: int, n: int) -> None:
        # Every 4th layer is standard attention, rest are linear attention
        self.layers = []
        for i in range(n):
            if i % 4 == 3:
                self.layers.append(_FakeLayer(h))
            else:
                self.layers.append(_FakeLinearAttnLayer(h))


class _FakeHybridModel:
    def __init__(self, hidden: int = HIDDEN, n_layers: int = 4) -> None:
        self.model = _FakeHybridModelInner(hidden, n_layers)
        self.device = torch.device("cpu")
        self._h = hidden
        self._n = n_layers
        self.config = MagicMock()
        self.config.hidden_size = hidden

    def named_parameters(self):
        for i, layer in enumerate(self.model.layers):
            mlp = layer.mlp
            yield (f"model.layers.{i}.mlp.gate_proj.weight", mlp.gate_proj.weight)
            yield (f"model.layers.{i}.mlp.down_proj.weight", mlp.down_proj.weight)
            attn = getattr(layer, "self_attn", None)
            if attn:
                yield (f"model.layers.{i}.self_attn.q_proj.weight", attn.q_proj.weight)
                yield (f"model.layers.{i}.self_attn.o_proj.weight", attn.o_proj.weight)
            lin = getattr(layer, "linear_attn", None)
            if lin:
                yield (f"model.layers.{i}.linear_attn.in_proj_qkv.weight", lin.in_proj_qkv.weight)
                yield (f"model.layers.{i}.linear_attn.out_proj.weight", lin.out_proj.weight)

    def save_pretrained(self, save_directory: str, **_kw: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        cfg = {"model_type": "fake", "hidden_size": self._h, "num_hidden_layers": self._n}
        (path / "config.json").write_text(json.dumps(cfg))


class TestLinearAttnAblation:
    """Tests for linear attention (GatedDeltaNet/Mamba) layer ablation."""

    def test_linear_attn_input_proj_modified(self, tmp_path: Path) -> None:
        """in_proj_qkv (input-side linear attn) should be modified by ablation."""
        model = _FakeHybridModel()
        before = model.model.layers[0].linear_attn.in_proj_qkv.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        after = model.model.layers[0].linear_attn.in_proj_qkv.weight
        assert not torch.allclose(before, after), "in_proj_qkv should be modified (right-multiply)"

    def test_linear_attn_out_proj_modified(self, tmp_path: Path) -> None:
        """out_proj (output-side linear attn) should be modified by ablation."""
        model = _FakeHybridModel()
        before = model.model.layers[0].linear_attn.out_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        after = model.model.layers[0].linear_attn.out_proj.weight
        assert not torch.allclose(before, after), "out_proj should be modified (left-multiply)"

    def test_standard_attn_still_ablated_in_hybrid(self, tmp_path: Path) -> None:
        """Standard self_attn layers in hybrid model should also be ablated."""
        model = _FakeHybridModel()
        # Layer 3 is the standard attention layer (i % 4 == 3)
        before = model.model.layers[3].self_attn.q_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        after = model.model.layers[3].self_attn.q_proj.weight
        assert not torch.allclose(before, after), "q_proj in standard layer should be modified"

    def test_linear_attn_mlp_also_ablated(self, tmp_path: Path) -> None:
        """MLP in linear-attention layers should also be ablated."""
        model = _FakeHybridModel()
        before = model.model.layers[0].mlp.gate_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        after = model.model.layers[0].mlp.gate_proj.weight
        assert not torch.allclose(before, after), "MLP gate_proj in linear_attn layer should be modified"


class TestOutputOnlyAblation:
    """Tests for output_only=True: only output projections modified, input projections untouched."""

    def test_output_only_skips_input_projections(self, tmp_path: Path) -> None:
        """With output_only=True, q_proj and gate_proj (input-side) should NOT be modified."""
        model = _FakeModel()
        q_before = model.model.layers[1].self_attn.q_proj.weight.clone()
        gate_before = model.model.layers[1].mlp.gate_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0, output_only=True)
        assert torch.allclose(q_before, model.model.layers[1].self_attn.q_proj.weight), (
            "q_proj should NOT be modified with output_only=True"
        )
        assert torch.allclose(gate_before, model.model.layers[1].mlp.gate_proj.weight), (
            "gate_proj should NOT be modified with output_only=True"
        )

    def test_output_only_modifies_output_projections(self, tmp_path: Path) -> None:
        """With output_only=True, o_proj and down_proj (output-side) should still be modified."""
        model = _FakeModel()
        o_before = model.model.layers[1].self_attn.o_proj.weight.clone()
        down_before = model.model.layers[1].mlp.down_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0, output_only=True)
        assert not torch.allclose(o_before, model.model.layers[1].self_attn.o_proj.weight), (
            "o_proj should be modified with output_only=True"
        )
        assert not torch.allclose(down_before, model.model.layers[1].mlp.down_proj.weight), (
            "down_proj should be modified with output_only=True"
        )

    def test_output_only_hybrid_skips_linear_attn_input(self, tmp_path: Path) -> None:
        """With output_only=True on hybrid model, linear_attn input projs should NOT be modified."""
        model = _FakeHybridModel()
        in_before = model.model.layers[0].linear_attn.in_proj_qkv.weight.clone()
        out_before = model.model.layers[0].linear_attn.out_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0, output_only=True)
        assert torch.allclose(in_before, model.model.layers[0].linear_attn.in_proj_qkv.weight), (
            "in_proj_qkv should NOT be modified with output_only=True"
        )
        assert not torch.allclose(out_before, model.model.layers[0].linear_attn.out_proj.weight), (
            "out_proj should be modified with output_only=True"
        )


class _FakeLinearWithBias:
    """Linear layer with both weight and bias for bias projection tests."""

    def __init__(self, out_f: int, in_f: int) -> None:
        self.weight = torch.randn(out_f, in_f)
        self.bias = torch.randn(out_f)


class _FakeAttnWithBias:
    def __init__(self, h: int) -> None:
        self.q_proj = _FakeLinearWithBias(h, h)
        self.k_proj = _FakeLinearWithBias(h, h)
        self.v_proj = _FakeLinearWithBias(h, h)
        self.o_proj = _FakeLinearWithBias(h, h)


class _FakeMLPWithBias:
    def __init__(self, h: int) -> None:
        self.gate_proj = _FakeLinearWithBias(h * 2, h)
        self.up_proj = _FakeLinearWithBias(h * 2, h)
        self.down_proj = _FakeLinearWithBias(h, h * 2)


class _FakeLayerWithBias:
    def __init__(self, h: int) -> None:
        self.self_attn = _FakeAttnWithBias(h)
        self.mlp = _FakeMLPWithBias(h)


class _FakeModelWithBias(_FakeModel):
    def __init__(self, hidden: int = HIDDEN, n_layers: int = N_LAYERS) -> None:
        super().__init__(hidden, n_layers)
        self.model.layers = [_FakeLayerWithBias(hidden) for _ in range(n_layers)]


class TestBiasProjection:
    """Tests for project_bias=True: bias vectors are also projected."""

    def test_bias_is_projected_by_default(self, tmp_path: Path) -> None:
        """With project_bias=True (default), bias vectors should be modified."""
        model = _FakeModelWithBias()
        bias_before = model.model.layers[1].self_attn.o_proj.bias.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        bias_after = model.model.layers[1].self_attn.o_proj.bias
        assert not torch.allclose(bias_before, bias_after), (
            "bias should be modified with project_bias=True (default)"
        )

    def test_no_project_bias_leaves_bias_unchanged(self, tmp_path: Path) -> None:
        """With project_bias=False, bias vectors should NOT be modified."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModelWithBias()
        bias_before = model.model.layers[1].self_attn.o_proj.bias.clone()
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint"
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model", pt, out, verify=False,
                skip_begin_layers=0, skip_end_layers=0,
                project_bias=False,
            )
        bias_after = model.model.layers[1].self_attn.o_proj.bias
        assert torch.allclose(bias_before, bias_after), (
            "bias should NOT be modified with project_bias=False"
        )

    def test_mlp_bias_is_projected(self, tmp_path: Path) -> None:
        """MLP bias vectors should also be projected."""
        model = _FakeModelWithBias()
        bias_before = model.model.layers[1].mlp.down_proj.bias.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        bias_after = model.model.layers[1].mlp.down_proj.bias
        assert not torch.allclose(bias_before, bias_after), (
            "MLP down_proj bias should be modified"
        )

    def test_no_bias_model_works_fine(self, tmp_path: Path) -> None:
        """Models without bias (standard case) should still work."""
        model = _FakeModel()
        assert not hasattr(model.model.layers[0].self_attn.q_proj, "bias")
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        # No error = success


class TestSparseSurgery:
    """Tests for sparse_surgery=True: only top-k rows are modified."""

    def test_sparse_surgery_preserves_some_rows(self, tmp_path: Path) -> None:
        """With sparse_surgery, some rows of o_proj (left-multiply) should be unchanged."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModel()
        w_before = model.model.layers[1].self_attn.o_proj.weight.clone()
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint"
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model", pt, out, verify=False,
                skip_begin_layers=0, skip_end_layers=0,
                norm_preserving=False,
                sparse_surgery=True, surgery_top_k=0.5,
            )
        w_after = model.model.layers[1].self_attn.o_proj.weight
        row_changed = (w_before != w_after).any(dim=1).sum().item()
        # At most ~50% of rows should be modified (norm_preserving off so no global rescale)
        assert row_changed <= int(HIDDEN * 0.5) + 1, (
            f"Sparse surgery (top_k=0.5) should modify at most ~50% rows, got {row_changed}/{HIDDEN}"
        )
        # But at least 1 row should be modified
        assert row_changed >= 1, "At least one row should be modified"

    def test_sparse_surgery_still_modifies_weights(self, tmp_path: Path) -> None:
        """Sparse surgery should still modify at least some weights."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModel()
        w_before = model.model.layers[1].self_attn.o_proj.weight.clone()
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint"
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model", pt, out, verify=False,
                skip_begin_layers=0, skip_end_layers=0,
                sparse_surgery=True, surgery_top_k=0.3,
            )
        w_after = model.model.layers[1].self_attn.o_proj.weight
        assert not torch.allclose(w_before, w_after), "Some weights should be modified"


class TestApplyRefusalDirPerLayer:
    """Tests for per-layer direction .pt format."""

    def test_per_layer_pt_is_accepted(self, tmp_path: Path) -> None:
        """A dict .pt with per_layer=True should be accepted without errors."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        directions = torch.randn(N_LAYERS, HIDDEN)
        for i in range(N_LAYERS):
            directions[i] = directions[i] / directions[i].norm()
        pt = tmp_path / "per_layer.pt"
        torch.save({"per_layer": True, "directions": directions}, str(pt))
        out = tmp_path / "checkpoint"

        model = _FakeModel()
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=0,
                norm_preserving=False,
            )
        assert (out / "config.json").is_file(), "Checkpoint should be saved with per-layer directions"

    def test_per_layer_modifies_weights(self, tmp_path: Path) -> None:
        """Per-layer directions should still modify model weights."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        directions = torch.randn(N_LAYERS, HIDDEN)
        for i in range(N_LAYERS):
            directions[i] = directions[i] / directions[i].norm()
        pt = tmp_path / "per_layer.pt"
        torch.save({"per_layer": True, "directions": directions}, str(pt))
        out = tmp_path / "checkpoint"

        model = _FakeModel()
        q_before = model.model.layers[1].self_attn.q_proj.weight.clone()
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=0,
                norm_preserving=False,
            )
        q_after = model.model.layers[1].self_attn.q_proj.weight
        assert not torch.allclose(q_before, q_after), "Per-layer ablation should modify weights"


# ---------------------------------------------------------------------------
# Edge-case: 1-layer model and clamped skip values
# ---------------------------------------------------------------------------


class TestLayerSkipEdgeCases:
    """Edge-case tests for layer-skip clamping and 1-layer models."""

    def test_single_layer_model_is_modified(self, tmp_path: Path) -> None:
        """A 1-layer model with no skips should still have its layer ablated."""
        model = _FakeModel(hidden=HIDDEN, n_layers=1)
        q_before = model.model.layers[0].self_attn.q_proj.weight.clone()
        _apply_fake(tmp_path, model, skip_begin=0, skip_end=0)
        q_after = model.model.layers[0].self_attn.q_proj.weight
        assert not torch.allclose(q_before, q_after), "Single layer should be ablated"

    def test_single_layer_skip_end_triggers_zero_layer_warning(self, tmp_path: Path) -> None:
        """1-layer model + skip_end=1 → end_idx=max(0,0)=0 == start_idx → warning."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModel(hidden=HIDDEN, n_layers=1)
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "ckpt_1layer"
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("ollama_forge.abliterate.log") as mock_log,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=1,
                norm_preserving=False,
            )
        assert mock_log.warning.called, "Warning expected when skip_end >= n_layers on 1-layer model"

    def test_skip_begin_clamped_to_last_layer(self, tmp_path: Path) -> None:
        """skip_begin_layers=100 is clamped to n_layers-1; only the last layer is ablated."""
        model = _FakeModel(hidden=HIDDEN, n_layers=N_LAYERS)
        # Capture state of all layers before
        q_before = [layer.self_attn.q_proj.weight.clone() for layer in model.model.layers]
        _apply_fake(tmp_path, model, skip_begin=100, skip_end=0)
        # All layers except the last should be unchanged
        for i in range(N_LAYERS - 1):
            assert torch.allclose(q_before[i], model.model.layers[i].self_attn.q_proj.weight), (
                f"Layer {i} should be untouched when skip_begin=100"
            )
        # The last layer should be modified (start_idx=n-1, end_idx=n)
        assert not torch.allclose(
            q_before[N_LAYERS - 1], model.model.layers[N_LAYERS - 1].self_attn.q_proj.weight
        ), "Last layer should be ablated when skip_begin is clamped to n_layers-1"

    def test_skip_end_clamped_triggers_warning(self, tmp_path: Path) -> None:
        """skip_end_layers=100 makes end_idx=max(0, n-100)=0 == start_idx=0 → warning."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModel(hidden=HIDDEN, n_layers=N_LAYERS)
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "ckpt_skipend"
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("ollama_forge.abliterate.log") as mock_log,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=100,
                norm_preserving=False,
            )
        assert mock_log.warning.called, "Warning expected when skip_end_layers > n_layers"

    def test_skip_begin_plus_skip_end_equals_n_layers_warns(self, tmp_path: Path) -> None:
        """When skip_begin + skip_end == n_layers exactly, zero layers are ablated → warning."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        model = _FakeModel(hidden=HIDDEN, n_layers=N_LAYERS)
        pt = _make_direction_pt(tmp_path)
        out = tmp_path / "ckpt_exact"
        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=model),
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("ollama_forge.abliterate.log") as mock_log,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            # skip_begin=2, skip_end=2, n_layers=4:
            # start_idx=min(2,3)=2, end_idx=max(2,4-2)=2 → 2>=2 → warning
            apply_refusal_dir_and_save(
                "fake/model",
                pt,
                out,
                verify=False,
                skip_begin_layers=N_LAYERS // 2,
                skip_end_layers=N_LAYERS // 2,
                norm_preserving=False,
            )
        assert mock_log.warning.called, (
            "Warning expected when skip_begin + skip_end == n_layers"
        )


# ---------------------------------------------------------------------------
# _is_multimodal_model_id
# ---------------------------------------------------------------------------


class TestIsMultimodalModelId:
    """Tests for _is_multimodal_model_id."""

    def test_local_multimodal_dir(self, tmp_path: Path) -> None:
        """Returns True for local dir with vision_config in config.json."""
        config = {"model_type": "qwen3_5", "vision_config": {"hidden_size": 768}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_multimodal_model_id(str(tmp_path)) is True

    def test_local_text_only_dir(self, tmp_path: Path) -> None:
        """Returns False for local dir without vision indicators."""
        config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_multimodal_model_id(str(tmp_path)) is False

    def test_hf_repo_id_uses_autoconfig(self) -> None:
        """HF repo ID uses AutoConfig to check for multimodal."""
        mock_cfg = MagicMock()
        mock_cfg.vision_config = {"hidden_size": 768}
        mock_cfg.image_token_id = None
        mock_cfg.visual = None
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = mock_cfg
            assert _is_multimodal_model_id("Qwen/Qwen3.5-0.8B") is True

    def test_hf_repo_id_text_only(self) -> None:
        """HF repo ID for text-only model returns False."""
        mock_cfg = MagicMock(spec=[])  # no vision_config, image_token_id, visual attributes
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = mock_cfg
            assert _is_multimodal_model_id("meta-llama/Llama-3-8B") is False

    def test_nonexistent_path_returns_false(self) -> None:
        """Non-existent path returns False."""
        assert _is_multimodal_model_id("/nonexistent/path") is False


class TestLoadModelMultimodal:
    """Tests for model loading in _load_model_with_gguf_version_workaround."""

    def test_multimodal_uses_causal_lm(self, tmp_path: Path) -> None:
        """Multimodal models fall back to AutoModelForCausalLM when multimodal classes are unavailable."""
        from ollama_forge.abliterate import _load_model_with_gguf_version_workaround

        config = {"model_type": "qwen3_5", "vision_config": {"hidden_size": 768}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        mock_text = MagicMock()
        mock_text.generate = MagicMock()
        mock_text.lm_head = MagicMock()
        mock_model = MagicMock()
        mock_model.language_model = mock_text
        with patch("transformers.AutoModelForCausalLM") as mock_causal:
            mock_causal.from_pretrained.return_value = mock_model
            result = _load_model_with_gguf_version_workaround(str(tmp_path), {"trust_remote_code": True})
        assert result is mock_text
        mock_causal.from_pretrained.assert_called_once()

    def test_text_only_uses_causal_lm(self, tmp_path: Path) -> None:
        """Text-only model uses AutoModelForCausalLM directly."""
        from ollama_forge.abliterate import _load_model_with_gguf_version_workaround

        config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
        (tmp_path / "config.json").write_text(json.dumps(config))

        mock_model = MagicMock()
        with patch("transformers.AutoModelForCausalLM") as mock_causal:
            mock_causal.from_pretrained.return_value = mock_model
            result = _load_model_with_gguf_version_workaround(str(tmp_path), {"trust_remote_code": True})
        assert result is mock_model
        mock_causal.from_pretrained.assert_called_once()


class TestGetLayers:
    """Tests for get_layers() with various model structures."""

    def test_standard_causal_lm(self) -> None:
        """model.model.layers — standard CausalLM (Llama, Qwen, Gemma)."""
        layers = MagicMock()
        model = MagicMock(spec=[])
        model.model = MagicMock(spec=[])
        model.model.layers = layers
        assert get_layers(model) is layers

    def test_nested_language_model(self) -> None:
        """model.model.language_model.layers — nested multimodal wrapper."""
        layers = MagicMock()
        model = MagicMock(spec=[])
        model.model = MagicMock(spec=[])
        model.model.language_model = MagicMock(spec=[])
        model.model.language_model.layers = layers
        # Ensure model.model doesn't have .layers directly
        del model.model.layers
        assert get_layers(model) is layers

    def test_automodel_multimodal_with_inner_model(self) -> None:
        """model.language_model.model.layers — AutoModel multimodal with inner .model."""
        layers = MagicMock()
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model.model = MagicMock(spec=[])
        model.language_model.model.layers = layers
        assert get_layers(model) is layers

    def test_automodel_multimodal_direct(self) -> None:
        """model.language_model.layers — AutoModel multimodal direct (Qwen3.5)."""
        layers = MagicMock()
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model.layers = layers
        assert get_layers(model) is layers

    def test_gpt2_falcon(self) -> None:
        """model.transformer.h — Falcon / GPT-2 / GPT-NeoX."""
        layers = MagicMock()
        model = MagicMock(spec=[])
        model.transformer = MagicMock(spec=[])
        model.transformer.h = layers
        assert get_layers(model) is layers

    def test_unknown_raises(self) -> None:
        """Unrecognised model structure raises AttributeError."""
        model = MagicMock(spec=[])
        with pytest.raises(AttributeError, match="Could not find layers"):
            get_layers(model)
