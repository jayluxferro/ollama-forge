"""Tests for TurboQuant inference engine."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ollama_forge.turboquant import quantize_tensor  # noqa: E402
from ollama_forge.turboquant_engine import (  # noqa: E402
    GenerationConfig,
    KVCache,
    Qwen35TurboQuantCache,
    RoPECache,
    TQWeight,
    TurboQuantHFModel,
    TurboQuantTransformer,
    _parse_model_config,
    _sample_token,
    apply_rope,
    generate,
    load_model,
)

# ---------------------------------------------------------------------------
# ModelConfig parsing
# ---------------------------------------------------------------------------

class TestParseModelConfig:
    def test_llama_defaults(self):
        hf = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }
        cfg = _parse_model_config(hf)
        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 4096
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128  # 4096 / 32

    def test_gqa_head_dim(self):
        hf = {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
        }
        cfg = _parse_model_config(hf)
        assert cfg.head_dim == 128
        assert cfg.num_key_value_heads == 4

    def test_defaults_for_missing(self):
        cfg = _parse_model_config({})
        assert cfg.vocab_size == 32000
        assert cfg.rms_norm_eps == 1e-6


# ---------------------------------------------------------------------------
# TQWeight wrapper
# ---------------------------------------------------------------------------

class TestTQWeight:
    def test_raw_tensor(self):
        raw = torch.randn(16, 32)
        w = TQWeight(raw=raw)
        assert w.shape == (16, 32)
        result = w.get(torch.device("cpu"))
        assert torch.allclose(result, raw)

    def test_quantized_tensor(self):
        W = torch.randn(16, 64)
        qt = quantize_tensor(W, bits=4)
        w = TQWeight(qt=qt)
        assert w.shape == (16, 64)
        result = w.get(torch.device("cpu"))
        assert result.shape == (16, 64)

    def test_cache_reuse(self):
        raw = torch.randn(8, 8)
        w = TQWeight(raw=raw)
        r1 = w.get(torch.device("cpu"))
        r2 = w.get(torch.device("cpu"))
        assert r1 is r2  # same object, cached

    def test_clear_cache(self):
        raw = torch.randn(8, 8)
        w = TQWeight(raw=raw)
        w.get(torch.device("cpu"))
        w.clear_cache()
        assert w._cached is None


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_shape(self):
        rope = RoPECache(64, 128, theta=10000.0)
        cos, sin = rope.get(10)
        assert cos.shape == (1, 1, 10, 64)
        assert sin.shape == (1, 1, 10, 64)

    def test_offset(self):
        rope = RoPECache(64, 128)
        cos0, sin0 = rope.get(1, offset=0)
        cos5, sin5 = rope.get(1, offset=5)
        # Different positions should have different values
        assert not torch.allclose(cos0, cos5)

    def test_apply_rope_shape(self):
        rope = RoPECache(64, 128)
        cos, sin = rope.get(8)
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 8, 64)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_fp16_cache(self):
        cache = KVCache(max_len=32, n_heads=4, head_dim=16,
                        device=torch.device("cpu"), kv_bits=0)
        k = torch.randn(1, 4, 5, 16, dtype=torch.float16)
        v = torch.randn(1, 4, 5, 16, dtype=torch.float16)
        cache.append(k, v)
        assert cache.len == 5

        k_out, v_out = cache.get_kv()
        assert k_out.shape == (1, 4, 5, 16)
        assert torch.allclose(k_out, k)

    def test_fp16_cache_accumulate(self):
        cache = KVCache(max_len=32, n_heads=2, head_dim=8,
                        device=torch.device("cpu"), kv_bits=0)
        k1 = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        v1 = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        cache.append(k1, v1)

        k2 = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        v2 = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        cache.append(k2, v2)

        assert cache.len == 4
        k_out, v_out = cache.get_kv()
        assert k_out.shape == (1, 2, 4, 8)

    def test_quantized_cache(self):
        cache = KVCache(max_len=32, n_heads=2, head_dim=16,
                        device=torch.device("cpu"), kv_bits=3)
        k = torch.randn(1, 2, 3, 16, dtype=torch.float16)
        v = torch.randn(1, 2, 3, 16, dtype=torch.float16)
        cache.append(k, v)
        assert cache.len == 3

        k_out, v_out = cache.get_kv()
        assert k_out.shape == (1, 2, 3, 16)
        # Quantized cache should be approximate but reasonable
        # Just check shapes and non-zero
        assert k_out.abs().sum() > 0

    def test_quantized_cache_empty(self):
        cache = KVCache(max_len=16, n_heads=2, head_dim=8,
                        device=torch.device("cpu"), kv_bits=3)
        k, v = cache.get_kv()
        assert k.shape[2] == 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_greedy(self):
        logits = torch.zeros(100)
        logits[42] = 10.0
        cfg = GenerationConfig(temperature=0.0)
        token = _sample_token(logits, cfg, [])
        assert token == 42

    def test_top_k(self):
        logits = torch.zeros(100)
        logits[10] = 5.0
        logits[20] = 4.0
        logits[30] = 3.0
        cfg = GenerationConfig(temperature=0.5, top_k=2, top_p=1.0)
        # With top_k=2, only tokens 10 and 20 should be sampled
        tokens = set()
        for _ in range(50):
            tokens.add(_sample_token(logits.clone(), cfg, []))
        assert tokens.issubset({10, 20})

    def test_repetition_penalty(self):
        logits = torch.zeros(100)
        logits[5] = 3.0
        logits[10] = 3.0
        # Penalize token 5
        cfg = GenerationConfig(temperature=0.01, repetition_penalty=100.0, top_k=0, top_p=1.0)
        token = _sample_token(logits.clone(), cfg, [5, 5, 5])
        assert token == 10  # should avoid 5


class TestHFBackedRuntime:
    def test_load_model_prefers_hf_runtime(self, tmp_path, monkeypatch):
        tqf_dir = tmp_path / "model.tqf"
        tqf_dir.mkdir()
        meta = {
            "format": "turboquant",
            "version": 2,
            "implementation": "hf-kv-cache",
            "model_config": {"model_type": "llama"},
            "source_model": "fake/repo",
            "resolved_model_path": str(tmp_path / "source-model"),
            "quant_config": {
                "bits": 3,
                "outlier_channels": 32,
                "outlier_bits": 4,
                "use_qjl": False,
                "embed_bits": 4,
                "kv_bits": 4,
                "rotation_seed": 42,
                "qjl_seed": 137,
                "attn_bits": None,
                "ffn_bits": None,
            },
            "quantized_layers": {},
            "unquantized_layers": [],
            "stats": {},
        }
        (tqf_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        source_dir = tmp_path / "source-model"
        source_dir.mkdir()
        (source_dir / "config.json").write_text("{}", encoding="utf-8")

        class FakeTokenizer:
            eos_token_id = 2

        class FakeHFModel:
            def __init__(self):
                self.emb = torch.nn.Embedding(8, 4)

            def get_input_embeddings(self):
                return self.emb

            def parameters(self):
                return iter([self.emb.weight])

        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: FakeTokenizer()),
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: FakeHFModel()),
        )

        model, tokenizer = load_model(tqf_dir, device="cpu", dtype="float32")
        assert isinstance(model, TurboQuantHFModel)
        assert model.kv_bits == 4
        assert tokenizer.eos_token_id == 2

    def test_load_model_without_accelerate_avoids_device_map(self, tmp_path, monkeypatch):
        tqf_dir = tmp_path / "model.tqf"
        tqf_dir.mkdir()
        meta = {
            "format": "turboquant",
            "version": 2,
            "implementation": "hf-kv-cache",
            "model_config": {"model_type": "llama"},
            "source_model": "fake/repo",
            "resolved_model_path": str(tmp_path / "source-model"),
            "quant_config": {
                "bits": 3,
                "outlier_channels": 32,
                "outlier_bits": 4,
                "use_qjl": False,
                "embed_bits": 4,
                "kv_bits": 4,
                "rotation_seed": 42,
                "qjl_seed": 137,
                "attn_bits": None,
                "ffn_bits": None,
            },
            "quantized_layers": {},
            "unquantized_layers": [],
            "stats": {},
        }
        (tqf_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        source_dir = tmp_path / "source-model"
        source_dir.mkdir()
        (source_dir / "config.json").write_text("{}", encoding="utf-8")

        class FakeTokenizer:
            eos_token_id = 2

        class FakeHFModel:
            def __init__(self):
                self.emb = torch.nn.Embedding(8, 4)
                self.to_calls = []

            def get_input_embeddings(self):
                return self.emb

            def parameters(self):
                return iter([self.emb.weight])

            def to(self, *args, **kwargs):
                self.to_calls.append((args, kwargs))
                return self

        fake_model = FakeHFModel()
        seen_kwargs = {}

        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: FakeTokenizer()),
        )

        def _fake_from_pretrained(*args, **kwargs):
            seen_kwargs.update(kwargs)
            return fake_model

        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            staticmethod(_fake_from_pretrained),
        )
        monkeypatch.setattr(
            "ollama_forge.turboquant_engine._is_accelerate_available",
            lambda: False,
        )

        model, _tokenizer = load_model(tqf_dir, device="cpu", dtype="float32")
        assert isinstance(model, TurboQuantHFModel)
        assert "device_map" not in seen_kwargs
        assert seen_kwargs["dtype"] == torch.float32
        assert fake_model.to_calls

    def test_generate_uses_hf_cache_runtime(self):
        class FakeEmb:
            weight = torch.zeros(4, 4)

        class FakeOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        class FakeHFModel:
            def __init__(self):
                self.calls = []
                self._emb = FakeEmb()

            def eval(self):
                return self

            def get_input_embeddings(self):
                return self._emb

            def __call__(self, input_ids, use_cache=True, past_key_values=None):
                self.calls.append({"shape": tuple(input_ids.shape), "has_past": past_key_values is not None})
                logits = torch.full((1, input_ids.shape[1], 10), -100.0)
                if len(self.calls) == 1:
                    logits[0, -1, 3] = 10.0
                else:
                    logits[0, -1, 9] = 10.0
                return FakeOutput(logits, past_key_values or "cache")

        model = TurboQuantHFModel(hf_model=FakeHFModel(), device=torch.device("cpu"), kv_bits=4)
        cfg = GenerationConfig(max_new_tokens=2, temperature=0.0, stop_tokens=[9])

        tokens = list(generate(model, [1, 2], cfg, tokenizer=None))
        assert tokens == [3]
        assert model.hf_model.calls[0]["has_past"] is True
        assert model.hf_model.calls[1]["has_past"] is True


class TestQwen35TurboQuantCache:
    def test_hybrid_interface(self):
        class FakeConfig:
            num_hidden_layers = 3
            layer_types = ["linear_attention", "full_attention", "linear_attention"]

        cache = Qwen35TurboQuantCache(FakeConfig(), bits=3, residual_len=2)
        assert len(cache) == 3
        assert cache.has_previous_state is False

        k = torch.randn(1, 2, 3, 8)
        v = torch.randn(1, 2, 3, 8)
        k_out, v_out = cache.update(k, v, layer_idx=1)

        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
        assert cache.get_seq_length() == 3
        assert cache.get_mask_sizes(torch.arange(2), 1) == (5, 0)

        cache.conv_states[2] = torch.randn(1, 4, 4)
        assert cache.has_previous_state is True


# ---------------------------------------------------------------------------
# Weight name resolution — tests with real model naming conventions
# ---------------------------------------------------------------------------

class _FakeTransformer(TurboQuantTransformer):
    """Minimal subclass to test weight name resolution without loading a model."""

    def __init__(self, weight_names: list[str]):
        # Skip real __init__; just set up _weights dict with dummy entries
        self._weights = {n: TQWeight(raw=torch.zeros(1)) for n in weight_names}


# BERT / encoder-style weight names (from all-MiniLM-L6-v2)
_BERT_WEIGHTS = [
    "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias",
    "embeddings.word_embeddings.weight", "embeddings.position_embeddings.weight",
    "encoder.layer.0.attention.self.query.weight", "encoder.layer.0.attention.self.query.bias",
    "encoder.layer.0.attention.self.key.weight", "encoder.layer.0.attention.self.key.bias",
    "encoder.layer.0.attention.self.value.weight", "encoder.layer.0.attention.self.value.bias",
    "encoder.layer.0.attention.output.dense.weight", "encoder.layer.0.attention.output.dense.bias",
    "encoder.layer.0.attention.output.LayerNorm.weight", "encoder.layer.0.attention.output.LayerNorm.bias",
    "encoder.layer.0.intermediate.dense.weight", "encoder.layer.0.intermediate.dense.bias",
    "encoder.layer.0.output.dense.weight", "encoder.layer.0.output.dense.bias",
    "encoder.layer.0.output.LayerNorm.weight", "encoder.layer.0.output.LayerNorm.bias",
]

# LLaMA / decoder-style weight names
_LLAMA_WEIGHTS = [
    "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


class TestBertWeightResolution:
    """Verify weight name lookup works for BERT-style encoder models."""

    def test_find_q_proj(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight(0, "q_proj", "query", "self.query")
        assert "query" in name and "layer.0" in name

    def test_find_k_proj(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight(0, "k_proj", "key", "self.key")
        assert "key" in name and "layer.0" in name

    def test_find_v_proj(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight(0, "v_proj", "value", "self.value")
        assert "value" in name and "layer.0" in name

    def test_find_attn_output(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight(0, "o_proj", "out_proj", "attention.output.dense")
        assert "attention.output.dense" in name

    def test_find_ffn_fc1(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight(0, "fc1", "intermediate.dense")
        assert "intermediate.dense" in name

    def test_find_ffn_fc2_excludes_attention(self):
        """FFN output.dense must NOT match attention.output.dense."""
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight(0, "fc2", "output.dense", exclude="attention")
        assert name == "encoder.layer.0.output.dense.weight"
        assert "attention" not in name

    def test_find_attn_norm(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_norm(0, "input_layernorm", "attention.output.LayerNorm")
        assert "LayerNorm" in name and "attention" in name

    def test_find_ffn_norm(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_norm(0, "post_attention_layernorm", "output.LayerNorm", exclude="attention")
        assert "output.LayerNorm" in name
        # Should NOT be the attention one
        assert name == "encoder.layer.0.output.LayerNorm.weight"

    def test_find_embed(self):
        t = _FakeTransformer(_BERT_WEIGHTS)
        name = t._find_weight_global("embed_tokens", "wte", "word_embeddings")
        assert "word_embeddings" in name


class TestLlamaWeightResolution:
    """Verify weight name lookup still works for LLaMA-style decoder models."""

    def test_find_q_proj(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "q_proj" in t._find_weight(0, "q_proj", "query", "self.query")

    def test_find_attn_output(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "o_proj" in t._find_weight(0, "o_proj", "out_proj", "attention.output.dense")

    def test_find_gate_proj(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "gate_proj" in t._find_weight(0, "gate_proj", "w1")

    def test_find_input_norm(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "input_layernorm" in t._find_norm(0, "input_layernorm", "attention.output.LayerNorm")

    def test_find_final_norm(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "model.norm" in t._find_norm(None, "norm", "ln_f", "final_layer_norm", "embeddings.LayerNorm")

    def test_find_embed(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "embed_tokens" in t._find_weight_global("embed_tokens", "wte", "word_embeddings")

    def test_find_lm_head(self):
        t = _FakeTransformer(_LLAMA_WEIGHTS)
        assert "lm_head" in t._find_weight_global("lm_head")
