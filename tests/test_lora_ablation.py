"""Tests for LoRA-based reversible ablation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from ollama_forge.lora_ablation import compute_lora_adapters, save_lora_adapter  # noqa: E402

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
        self.up_proj = _FakeLinear(h * 2, h)
        self.down_proj = _FakeLinear(h, h * 2)


class _FakeLayer:
    def __init__(self, h: int) -> None:
        self.self_attn = _FakeAttn(h)
        self.mlp = _FakeMLP(h)


class _FakeModelInner:
    def __init__(self, h: int, n: int) -> None:
        self.layers = [_FakeLayer(h) for _ in range(n)]


class _FakeModel:
    def __init__(self) -> None:
        self.model = _FakeModelInner(HIDDEN, N_LAYERS)
        self.config = MagicMock()
        self.config._name_or_path = "fake/model"
        self.config.hidden_size = HIDDEN


class TestComputeLoRAAdapters:
    def test_produces_adapters(self) -> None:
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        bundle = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0)
        assert len(bundle.adapters) > 0
        assert bundle.rank == 1
        assert bundle.strength == 1.0

    def test_output_only_has_fewer_adapters(self) -> None:
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        full = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0, output_only=False)
        out_only = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0, output_only=True)
        assert len(out_only.adapters) < len(full.adapters)

    def test_skip_layers_reduces_adapters(self) -> None:
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        all_layers = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0)
        skipped = compute_lora_adapters(model, d, skip_begin_layers=1, skip_end_layers=1)
        assert len(skipped.adapters) < len(all_layers.adapters)

    def test_adapter_shapes_are_correct(self) -> None:
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        bundle = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0)
        for adapter in bundle.adapters:
            assert adapter.lora_A.shape[0] == 1  # rank=1
            assert adapter.lora_B.shape[1] == 1  # rank=1

    def test_lora_equivalent_to_projection(self) -> None:
        """W + B@A should equal W @ (I - s*D@D^T) for right-multiply (input proj)."""
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        s = 0.8

        # Get original weight
        w = model.model.layers[1].self_attn.q_proj.weight.data.float().clone()

        bundle = compute_lora_adapters(model, d, strength=s, skip_begin_layers=0, skip_end_layers=0)
        # Find q_proj adapter for layer 1
        q_adapter = None
        for a in bundle.adapters:
            if "layers.1.self_attn.q_proj" in a.target_module:
                q_adapter = a
                break
        assert q_adapter is not None

        # LoRA result: W + B @ A
        lora_result = w + q_adapter.lora_B.float() @ q_adapter.lora_A.float()

        # Direct projection: W @ (I - s * D @ D^T)
        D = d.unsqueeze(1)
        I_minus = torch.eye(HIDDEN) - s * (D @ D.T)
        direct_result = w @ I_minus

        assert torch.allclose(lora_result, direct_result, atol=1e-5), (
            "LoRA adapter should be mathematically equivalent to direct projection"
        )


class TestSaveLoRAAdapter:
    def test_saves_files(self, tmp_path: Path) -> None:
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        bundle = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0)
        out = save_lora_adapter(bundle, tmp_path / "lora")
        assert (out / "adapter_model.bin").is_file()
        assert (out / "adapter_config.json").is_file()
        config = json.loads((out / "adapter_config.json").read_text())
        assert config["peft_type"] == "LORA"
        assert config["r"] == 1

    def test_state_dict_has_all_adapters(self, tmp_path: Path) -> None:
        model = _FakeModel()
        d = torch.randn(HIDDEN)
        d = d / d.norm()
        bundle = compute_lora_adapters(model, d, skip_begin_layers=0, skip_end_layers=0)
        out = save_lora_adapter(bundle, tmp_path / "lora")
        state_dict = torch.load(str(out / "adapter_model.bin"), map_location="cpu", weights_only=True)
        # Each adapter produces 2 keys (lora_A.weight, lora_B.weight)
        assert len(state_dict) == len(bundle.adapters) * 2
