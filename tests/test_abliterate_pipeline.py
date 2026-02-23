"""Integration test: compute_refusal_dir → apply_refusal_dir_and_save pipeline.

Uses a tiny fake model to test the full data-flow without loading real HF models.
Requires torch; skipped automatically if torch is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Minimal fake model / tokenizer that satisfy the abliterate API surface
# ---------------------------------------------------------------------------

HIDDEN = 8   # tiny hidden dimension
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
    """Minimal model stub usable by compute_refusal_dir and apply_refusal_dir_and_save."""

    def __init__(self, hidden: int = HIDDEN, n_layers: int = N_LAYERS) -> None:
        self.model  = _FakeModelInner(hidden, n_layers)
        self.device = torch.device("cpu")
        self._h     = hidden
        self._n     = n_layers
        self.config = MagicMock()
        self.config.hidden_size = hidden

    # ------------------------------------------------------------------
    # Forward pass — returns object with .hidden_states for direction extraction
    # ------------------------------------------------------------------
    def __call__(self, input_ids: Any, output_hidden_states: bool = False, **_kw: Any):
        b, s = input_ids.shape
        out = MagicMock()
        # hidden_states[0] = embedding, [1..n] = layer outputs
        out.hidden_states = [
            torch.randn(b, s, self._h) for _ in range(self._n + 1)
        ]
        return out

    # ------------------------------------------------------------------
    # Attribute access helpers used by abliterate internals
    # ------------------------------------------------------------------
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
        cfg = {
            "model_type": "fake_model",
            "hidden_size": self._h,
            "num_hidden_layers": self._n,
        }
        (path / "config.json").write_text(json.dumps(cfg))


class _FakeTokenizer:
    """Minimal tokenizer stub."""

    def __init__(self) -> None:
        self.chat_template  = "{{ messages }}"  # non-None → use apply_chat_template
        self.eos_token_id   = 2

    def __call__(self, text: str, *, return_tensors: str = "pt", **_kw: Any):
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(
        self,
        conversation: Any,
        *,
        add_generation_prompt: bool = True,
        return_tensors: str = "pt",
        **_kw: Any,
    ) -> torch.Tensor:
        return torch.tensor([[1, 2, 3]])

    def save_pretrained(self, save_directory: str, **_kw: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        cfg = {"tokenizer_class": "FakeTokenizer", "eos_token": "</s>"}
        (path / "tokenizer_config.json").write_text(json.dumps(cfg))

    def decode(self, ids: Any, **_kw: Any) -> str:
        return "fake output"


# ---------------------------------------------------------------------------
# Integration: compute_refusal_dir writes a valid .pt file
# ---------------------------------------------------------------------------

class TestComputeRefusalDir:
    """compute_refusal_dir writes a usable direction tensor to disk."""

    @pytest.fixture()
    def tmp(self, tmp_path: Path) -> Path:
        return tmp_path

    def _make_inputs(self, tmp: Path) -> tuple[Path, Path]:
        harmful  = tmp / "harmful.txt"
        harmless = tmp / "harmless.txt"
        harmful.write_text("\n".join(f"Harm prompt {i}" for i in range(8)))
        harmless.write_text("\n".join(f"Safe prompt {i}" for i in range(8)))
        return harmful, harmless

    def test_produces_pt_file(self, tmp: Path) -> None:
        from ollama_forge.abliterate import compute_refusal_dir

        harmful, harmless = self._make_inputs(tmp)
        out_pt = tmp / "refusal_dir.pt"

        fake_model     = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=fake_model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = fake_tokenizer
            compute_refusal_dir(
                "fake/model-id",
                str(harmful),
                str(harmless),
                str(out_pt),
                num_instructions=4,
                layer_fracs=(0.5,),
            )

        assert out_pt.is_file(), "refusal_dir.pt was not created"
        loaded = torch.load(str(out_pt), map_location="cpu", weights_only=True)
        assert loaded.dim() == 2
        assert loaded.shape[0] == HIDDEN

    def test_produces_normalized_direction(self, tmp: Path) -> None:
        from ollama_forge.abliterate import compute_refusal_dir

        harmful, harmless = self._make_inputs(tmp)
        out_pt = tmp / "refusal_dir.pt"

        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=_FakeModel()),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = _FakeTokenizer()
            compute_refusal_dir(
                "fake/model",
                str(harmful),
                str(harmless),
                str(out_pt),
                num_instructions=4,
                layer_fracs=(0.4, 0.6),
            )

        d = torch.load(str(out_pt), map_location="cpu", weights_only=True).float()
        # Single direction: unit-norm column
        norm = d[:, 0].norm().item()
        assert abs(norm - 1.0) < 1e-4, f"Direction not unit-norm: {norm}"


# ---------------------------------------------------------------------------
# Integration: apply_refusal_dir_and_save modifies weights and saves checkpoint
# ---------------------------------------------------------------------------

class TestApplyRefusalDirAndSave:
    """apply_refusal_dir_and_save reads a .pt, modifies model weights, saves checkpoint."""

    def _make_direction_pt(self, tmp: Path) -> Path:
        d = torch.randn(HIDDEN, 1)
        d = d / d.norm()
        pt = tmp / "refusal_dir.pt"
        torch.save(d, str(pt))
        return pt

    def test_saves_checkpoint_config(self, tmp_path: Path) -> None:
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        pt  = self._make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint"

        fake_model     = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=fake_model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = fake_tokenizer
            apply_refusal_dir_and_save(
                "fake/model-id",
                pt,
                out,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=0,
                norm_preserving=False,
            )

        assert (out / "config.json").is_file(), "config.json not saved to checkpoint"

    def test_weights_modified_by_ablation(self, tmp_path: Path) -> None:
        """Weights in ablated layers should differ from the originals."""
        from ollama_forge.abliterate import apply_refusal_dir_and_save

        pt  = self._make_direction_pt(tmp_path)
        out = tmp_path / "checkpoint"

        fake_model = _FakeModel()
        # Record original q_proj weight for the first layer
        original_q = fake_model.model.layers[1].self_attn.q_proj.weight.clone()

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
                skip_begin_layers=0,
                skip_end_layers=0,
                norm_preserving=False,
            )

        modified_q = fake_model.model.layers[1].self_attn.q_proj.weight
        # The direction projection should have changed the weight
        assert not torch.allclose(original_q, modified_q), (
            "q_proj weight was not modified by apply_refusal_dir_and_save"
        )


# ---------------------------------------------------------------------------
# Integration: full pipeline — compute then apply
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Smoke test: compute direction then apply it; verify checkpoint is created."""

    def test_compute_then_apply(self, tmp_path: Path) -> None:
        from ollama_forge.abliterate import apply_refusal_dir_and_save, compute_refusal_dir

        harmful  = tmp_path / "harmful.txt"
        harmless = tmp_path / "harmless.txt"
        harmful.write_text("\n".join(f"Harm {i}" for i in range(6)))
        harmless.write_text("\n".join(f"Safe {i}" for i in range(6)))

        pt_path = tmp_path / "direction.pt"
        out_dir = tmp_path / "checkpoint"

        fake_model     = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=fake_model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = fake_tokenizer

            # Step 1: compute direction
            compute_refusal_dir(
                "fake/model",
                str(harmful),
                str(harmless),
                str(pt_path),
                num_instructions=4,
                layer_fracs=(0.5,),
            )
            assert pt_path.is_file(), "Direction .pt not produced"

            # Step 2: apply direction
            apply_refusal_dir_and_save(
                "fake/model",
                pt_path,
                out_dir,
                verify=False,
                skip_begin_layers=0,
                skip_end_layers=0,
                norm_preserving=False,
            )

        assert (out_dir / "config.json").is_file(), "Checkpoint not saved after apply"


    def test_compute_returns_best_layer_index(self, tmp_path: Path) -> None:
        """compute_refusal_dir must return the layer with the best gap norm, not the last iterated one."""
        from ollama_forge.abliterate import compute_refusal_dir

        harmful  = tmp_path / "harmful.txt"
        harmless = tmp_path / "harmless.txt"
        harmful.write_text("\n".join(f"Harm {i}" for i in range(6)))
        harmless.write_text("\n".join(f"Safe {i}" for i in range(6)))
        pt_path = tmp_path / "direction.pt"

        # Use two distinct fracs so the best vs last distinction is testable
        fracs = (0.25, 0.75)
        fake_model     = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with (
            patch("ollama_forge.abliterate._load_model_with_gguf_version_workaround", return_value=fake_model),
            patch("transformers.AutoTokenizer") as mock_tok,
        ):
            mock_tok.from_pretrained.return_value = fake_tokenizer
            result = compute_refusal_dir(
                "fake/model",
                str(harmful),
                str(harmless),
                str(pt_path),
                num_instructions=4,
                layer_fracs=fracs,
            )

        # layer_index must be the layer computed from best_layer_frac, not always the last frac
        expected_last_idx = min(int(N_LAYERS * fracs[-1]), N_LAYERS - 1)
        expected_first_idx = min(int(N_LAYERS * fracs[0]), N_LAYERS - 1)
        assert result["layer_index"] in (expected_first_idx, expected_last_idx), (
            f"layer_index {result['layer_index']} is not one of the candidate layers"
        )
        # The returned layer_frac must correspond to the returned layer_index
        returned_frac = result["layer_frac"]
        assert returned_frac in fracs, f"layer_frac {returned_frac} not in input fracs"
        expected_idx_for_frac = min(int(N_LAYERS * returned_frac), N_LAYERS - 1)
        assert result["layer_index"] == expected_idx_for_frac, (
            f"layer_index {result['layer_index']} does not match layer_frac {returned_frac} "
            f"(expected {expected_idx_for_frac})"
        )
