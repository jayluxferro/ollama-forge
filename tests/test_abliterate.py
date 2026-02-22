"""Unit tests for abliterate module: _strength_kernel_scale and get_D_for_layer."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
        """strength outside (0, 1] should raise ValueError immediately."""
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
