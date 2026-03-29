"""Tests for TurboQuant quantization pipeline (save/load .tqf format)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from ollama_forge.turboquant_pipeline import (  # noqa: E402
    TurboQuantConfig,
    _load_hf_config,
    _parse_dtype,
    _should_quantize,
    copy_tokenizer,
    load_tqf,
    quantize_model,
)

# ---------------------------------------------------------------------------
# Fixtures: tiny fake HF checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_hf_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal fake HF safetensors checkpoint."""
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()

    # config.json
    config = {
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    # Create weight tensors
    H, INTER = 32, 64
    tensors = {
        "model.embed_tokens.weight": torch.randn(128, H),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(H, H),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(H, H),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(H, H),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(INTER, H),
        "model.layers.0.mlp.up_proj.weight": torch.randn(INTER, H),
        "model.layers.0.mlp.down_proj.weight": torch.randn(H, INTER),
        "model.layers.0.input_layernorm.weight": torch.ones(H),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(H),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(H, H),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(H, H),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(H, H),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(H, H),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(INTER, H),
        "model.layers.1.mlp.up_proj.weight": torch.randn(INTER, H),
        "model.layers.1.mlp.down_proj.weight": torch.randn(H, INTER),
        "model.layers.1.input_layernorm.weight": torch.ones(H),
        "model.layers.1.post_attention_layernorm.weight": torch.ones(H),
        "model.norm.weight": torch.ones(H),
        "lm_head.weight": torch.randn(128, H),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))

    return model_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShouldQuantize:
    def test_skip_norms(self):
        t = torch.randn(32, 64)
        assert not _should_quantize("model.layers.0.input_layernorm.weight", t)
        assert not _should_quantize("model.norm.weight", t)

    def test_skip_1d(self):
        t = torch.randn(64)
        assert not _should_quantize("model.layers.0.self_attn.q_proj.weight", t)

    def test_skip_small(self):
        t = torch.randn(4, 4)  # 16 elements < 1024 threshold
        assert not _should_quantize("some.weight", t)

    def test_quantize_large_2d(self):
        t = torch.randn(64, 64)  # 4096 elements > 1024
        assert _should_quantize("model.layers.0.self_attn.q_proj.weight", t)


class TestTurboQuantConfig:
    def test_bits_for_embed(self):
        cfg = TurboQuantConfig(bits=3, embed_bits=4)
        assert cfg.bits_for("model.embed_tokens.weight") == 4

    def test_bits_for_attn(self):
        cfg = TurboQuantConfig(bits=3, attn_bits=2)
        assert cfg.bits_for("model.layers.0.self_attn.q_proj.weight") == 2

    def test_bits_for_ffn(self):
        cfg = TurboQuantConfig(bits=3, ffn_bits=2)
        assert cfg.bits_for("model.layers.0.mlp.gate_proj.weight") == 2

    def test_bits_for_default(self):
        cfg = TurboQuantConfig(bits=3)
        assert cfg.bits_for("model.layers.0.self_attn.q_proj.weight") == 3


class TestLoadHFConfig:
    def test_loads(self, fake_hf_checkpoint):
        cfg = _load_hf_config(fake_hf_checkpoint)
        assert cfg["model_type"] == "llama"
        assert cfg["hidden_size"] == 32

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_hf_config(tmp_path)


class TestParseDtype:
    def test_float16(self):
        assert _parse_dtype("torch.float16") == torch.float16

    def test_bfloat16(self):
        assert _parse_dtype("torch.bfloat16") == torch.bfloat16

    def test_unknown(self):
        assert _parse_dtype("unknown") == torch.float16


class TestQuantizeModelRoundtrip:
    def test_quantize_and_load(self, fake_hf_checkpoint, tmp_path):
        """Full pipeline: package HF model metadata → save .tqf → load back."""
        output = tmp_path / "test.tqf"
        config = TurboQuantConfig(bits=2, outlier_channels=0, embed_bits=2)

        result = quantize_model(
            fake_hf_checkpoint,
            output,
            config,
            device="cpu",
            source_model="fake/repo",
        )

        # Check stats
        assert result.stats.original_params > 0
        assert result.stats.compression_ratio > 0

        # Check files exist
        assert (output / "metadata.json").exists()
        assert (output / "config.json").exists()

        # Load back
        loaded = load_tqf(output)
        assert loaded.config["model_type"] == "llama"
        assert loaded.layers == {}
        assert loaded.unquantized == {}
        assert loaded.source_model == "fake/repo"
        assert loaded.resolved_model_path == str(fake_hf_checkpoint)

    def test_metadata_content(self, fake_hf_checkpoint, tmp_path):
        """Metadata.json should have correct structure."""
        output = tmp_path / "test2.tqf"
        config = TurboQuantConfig(bits=3)
        quantize_model(fake_hf_checkpoint, output, config, device="cpu")

        meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert meta["format"] == "turboquant"
        assert meta["version"] == 2
        assert meta["implementation"] == "hf-kv-cache"
        assert "quant_config" in meta
        assert "quantized_layers" in meta
        assert "stats" in meta
        assert meta["quant_config"]["bits"] == 3
        assert meta["quantized_layers"] == {}


class TestCopyTokenizer:
    def test_copies_files(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        (src / "tokenizer.json").write_text("{}", encoding="utf-8")
        (src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (src / "unrelated.txt").write_text("x", encoding="utf-8")

        copy_tokenizer(src, dst)

        assert (dst / "tokenizer.json").exists()
        assert (dst / "tokenizer_config.json").exists()
        assert not (dst / "unrelated.txt").exists()
