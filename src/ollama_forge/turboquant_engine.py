"""TurboQuant inference engine.

The working runtime follows the reference implementation: load a normal
Transformers causal LM and compress its KV cache online during generation.
The older custom transformer path remains as a compatibility fallback for
legacy `.tqf` artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer

    _TRANSFORMERS_CACHE_AVAILABLE = True
except ImportError:
    DynamicCache = object  # type: ignore[assignment]
    DynamicLayer = object  # type: ignore[assignment]
    _TRANSFORMERS_CACHE_AVAILABLE = False

from ollama_forge.turboquant import (
    QuantizedTensor,
    _get_codebook,
    _scalar_dequantize,
    _scalar_quantize,
    dequantize_tensor,
    generate_rotation_matrix,
)
from ollama_forge.turboquant_pipeline import TurboQuantModel, load_tqf


@dataclass
class TurboQuantHFModel:
    """Wrapper for a standard Transformers model using TurboQuant KV cache."""

    hf_model: Any
    device: torch.device
    kv_bits: int

# ---------------------------------------------------------------------------
# Config parsing from HF config.json
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Unified model config extracted from HF config.json."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32       # for GQA; same as num_attention_heads if MHA
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
    # Architecture details
    hidden_act: str = "silu"
    norm_type: str = "rmsnorm"          # "rmsnorm" or "layernorm"
    # Derived
    model_type: str = "llama"


def _parse_model_config(hf_cfg: dict[str, Any]) -> ModelConfig:
    """Extract ModelConfig from HF config.json dict."""
    n_heads = hf_cfg.get("num_attention_heads", 32)
    n_kv = hf_cfg.get("num_key_value_heads", n_heads)
    hidden = hf_cfg.get("hidden_size", 4096)
    head_dim = hf_cfg.get("head_dim", hidden // n_heads)

    return ModelConfig(
        vocab_size=hf_cfg.get("vocab_size", 32000),
        hidden_size=hidden,
        intermediate_size=hf_cfg.get("intermediate_size", 11008),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 32),
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        max_position_embeddings=hf_cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=hf_cfg.get("rms_norm_eps", hf_cfg.get("layer_norm_epsilon", 1e-6)),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        rope_scaling=hf_cfg.get("rope_scaling"),
        tie_word_embeddings=hf_cfg.get("tie_word_embeddings", False),
        hidden_act=hf_cfg.get("hidden_act", hf_cfg.get("hidden_activation", "silu")),
        model_type=hf_cfg.get("model_type", "llama"),
    )


# ---------------------------------------------------------------------------
# Weight wrapper — dequantizes on the fly or uses fp16 directly
# ---------------------------------------------------------------------------

class TQWeight:
    """Wraps a QuantizedTensor or raw tensor for lazy dequantization."""

    def __init__(self, qt: QuantizedTensor | None = None, raw: torch.Tensor | None = None):
        self._qt = qt
        self._raw = raw
        self._cached: torch.Tensor | None = None

    def get(self, device: torch.device) -> torch.Tensor:
        """Return the dequantized weight on *device*.  Caches the result."""
        if self._cached is not None and self._cached.device == device:
            return self._cached
        if self._qt is not None:
            self._cached = dequantize_tensor(self._qt, device=device)
        elif self._raw is not None:
            self._cached = self._raw.to(device)
        else:
            raise RuntimeError("TQWeight has no data")
        return self._cached

    def clear_cache(self):
        self._cached = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._qt is not None:
            return self._qt.shape
        return tuple(self._raw.shape)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class RoPECache:
    """Precomputed RoPE sin/cos tables — auto-extends when position exceeds cache."""

    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0,
                 device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        if device is None:
            device = torch.device("cpu")
        self._head_dim = head_dim
        self._theta = theta
        self._device = device
        self._dtype = dtype
        self._build(max_len)

    def _build(self, length: int):
        inv_freq = 1.0 / (self._theta ** (torch.arange(0, self._head_dim, 2,
                           device=self._device, dtype=self._dtype) / self._head_dim))
        t = torch.arange(length, device=self._device, dtype=self._dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos = emb.cos().unsqueeze(0).unsqueeze(0)
        self.sin = emb.sin().unsqueeze(0).unsqueeze(0)
        self._len = length

    def get(self, seq_len: int, offset: int = 0):
        needed = offset + seq_len
        if needed > self._len:
            self._build(needed * 2)
        return (
            self.cos[:, :, offset:offset + seq_len, :],
            self.sin[:, :, offset:offset + seq_len, :],
        )


# ---------------------------------------------------------------------------
# TurboQuant KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Key-value cache with optional TurboQuant compression.

    When kv_bits > 0, K and V vectors are quantized online as they're
    appended, reducing memory for long contexts.  All operations are
    batched and stay on-device (no CPU round-trips).
    """

    def __init__(self, max_len: int, n_heads: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype = torch.float16,
                 kv_bits: int = 0):
        self.max_len = max_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.kv_bits = kv_bits
        self.len = 0

        if kv_bits > 0:
            # Quantized storage: batched indices + norms, all on-device
            self._k_indices: list[torch.Tensor] = []   # each: (nh*sl, hd)
            self._v_indices: list[torch.Tensor] = []
            self._k_norms: list[torch.Tensor] = []     # each: (nh*sl,)
            self._v_norms: list[torch.Tensor] = []
            self._token_counts: list[int] = []          # track shapes
            self._rotation_seed = 42
            self._codebook = _get_codebook(kv_bits, head_dim, device)
            # Pre-generate rotation matrix once
            self._rotation = generate_rotation_matrix(
                head_dim, device=device, seed=self._rotation_seed
            )
        else:
            # Full precision cache
            self.k = torch.zeros(1, n_heads, max_len, head_dim, device=device, dtype=dtype)
            self.v = torch.zeros(1, n_heads, max_len, head_dim, device=device, dtype=dtype)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new K, V tensors of shape (1, n_heads, seq_len, head_dim)."""
        seq_len = k_new.shape[2]

        if self.kv_bits > 0:
            self._append_quantized(k_new, v_new)
        else:
            end = self.len + seq_len
            self.k[:, :, self.len:end, :] = k_new
            self.v[:, :, self.len:end, :] = v_new

        self.len += seq_len

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full K, V tensors up to current length."""
        if self.kv_bits > 0:
            return self._get_quantized_kv()
        return self.k[:, :, :self.len, :], self.v[:, :, :self.len, :]

    def _append_quantized(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Quantize and store K, V using batched TurboQuant (all on-device)."""
        # k_new: (1, n_heads, seq_len, head_dim)
        _, nh, sl, hd = k_new.shape

        # Reshape to (nh*sl, hd) for batched processing
        k_flat = k_new.squeeze(0).transpose(0, 1).reshape(nh * sl, hd).float()
        v_flat = v_new.squeeze(0).transpose(0, 1).reshape(nh * sl, hd).float()

        # Batched norm + normalize + rotate + quantize
        k_norms = k_flat.norm(dim=1)
        v_norms = v_flat.norm(dim=1)
        k_normed = k_flat / (k_norms.unsqueeze(1) + 1e-10)
        v_normed = v_flat / (v_norms.unsqueeze(1) + 1e-10)

        k_rot = k_normed @ self._rotation.T
        v_rot = v_normed @ self._rotation.T

        k_idx = _scalar_quantize(k_rot, self._codebook)
        v_idx = _scalar_quantize(v_rot, self._codebook)

        # Store on-device (no .cpu())
        self._k_indices.append(k_idx)
        self._v_indices.append(v_idx)
        self._k_norms.append(k_norms)
        self._v_norms.append(v_norms)
        self._token_counts.append(sl)

    def _get_quantized_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched dequantize all cached K, V."""
        if not self._k_indices:
            hd = self.head_dim
            empty = torch.zeros(1, self.n_heads, 0, hd, device=self.device, dtype=self.dtype)
            return empty, empty

        # Concatenate all cached segments
        k_idx_all = torch.cat(self._k_indices, dim=0)   # (total_nh_tokens, hd)
        v_idx_all = torch.cat(self._v_indices, dim=0)
        k_norms_all = torch.cat(self._k_norms, dim=0)   # (total_nh_tokens,)
        v_norms_all = torch.cat(self._v_norms, dim=0)

        # Batched dequantize: codebook lookup + inverse rotation + scale
        k_deq = _scalar_dequantize(k_idx_all, self._codebook) @ self._rotation
        v_deq = _scalar_dequantize(v_idx_all, self._codebook) @ self._rotation

        k_deq = (k_deq * k_norms_all.unsqueeze(1)).to(self.dtype)
        v_deq = (v_deq * v_norms_all.unsqueeze(1)).to(self.dtype)

        # Reshape back: (total_tokens * n_heads, hd) → (1, n_heads, total_tokens, hd)
        total_tokens = sum(self._token_counts)
        nh = self.n_heads
        k = k_deq.view(total_tokens, nh, self.head_dim).transpose(0, 1).unsqueeze(0)
        v = v_deq.view(total_tokens, nh, self.head_dim).transpose(0, 1).unsqueeze(0)
        return k, v


class TurboQuantLayer(DynamicLayer):
    """Transformers DynamicCache layer backed by TurboQuant vector quantization."""

    def __init__(self, bits: int = 4, residual_len: int = 128):
        if not _TRANSFORMERS_CACHE_AVAILABLE:
            raise ImportError("transformers cache_utils is required for TurboQuant KV cache support")
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._key_indices: torch.Tensor | None = None
        self._key_norms: torch.Tensor | None = None
        self._value_indices: torch.Tensor | None = None
        self._value_norms: torch.Tensor | None = None
        self._residual_keys: torch.Tensor | None = None
        self._residual_values: torch.Tensor | None = None
        self._total_len = 0
        self._head_dim: int | None = None
        self._rotation: torch.Tensor | None = None
        self._codebook: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self._head_dim = key_states.shape[-1]
        self._rotation = generate_rotation_matrix(self._head_dim, device=self.device, seed=42)
        self._codebook = _get_codebook(self.bits, self._head_dim, self.device)
        self._key_indices = torch.empty(0, dtype=torch.uint8, device=self.device)
        self._key_norms = torch.empty(0, dtype=torch.float32, device=self.device)
        self._value_indices = torch.empty(0, dtype=torch.uint8, device=self.device)
        self._value_norms = torch.empty(0, dtype=torch.float32, device=self.device)
        self._residual_keys = torch.empty(0, dtype=self.dtype, device=self.device)
        self._residual_values = torch.empty(0, dtype=self.dtype, device=self.device)
        self.keys = torch.empty(0, dtype=self.dtype, device=self.device)
        self.values = torch.empty(0, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not getattr(self, "is_initialized", False):
            self.lazy_initialization(key_states, value_states)

        assert self._residual_keys is not None
        assert self._residual_values is not None
        assert self._head_dim is not None
        assert self._key_indices is not None
        assert self._key_norms is not None
        assert self._value_indices is not None
        assert self._value_norms is not None

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_quantize_k = self._residual_keys[..., :overflow, :]
            to_quantize_v = self._residual_values[..., :overflow, :]

            k_flat = to_quantize_k.reshape(-1, self._head_dim)
            v_flat = to_quantize_v.reshape(-1, self._head_dim)
            k_idx, k_norms = self._quantize_vectors(k_flat)
            v_idx, v_norms = self._quantize_vectors(v_flat)

            k_idx = k_idx.reshape(to_quantize_k.shape)
            v_idx = v_idx.reshape(to_quantize_v.shape)
            k_norms = k_norms.reshape(to_quantize_k.shape[:-1] + (1,))
            v_norms = v_norms.reshape(to_quantize_v.shape[:-1] + (1,))

            self._key_indices = torch.cat([self._key_indices, k_idx], dim=-2) if self._key_indices.numel() else k_idx
            self._key_norms = torch.cat([self._key_norms, k_norms], dim=-2) if self._key_norms.numel() else k_norms
            self._value_indices = torch.cat([self._value_indices, v_idx], dim=-2) if self._value_indices.numel() else v_idx
            self._value_norms = torch.cat([self._value_norms, v_norms], dim=-2) if self._value_norms.numel() else v_norms

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        if self._key_indices.numel():
            k_deq = self._dequantize_vectors(self._key_indices, self._key_norms)
            v_deq = self._dequantize_vectors(self._value_indices, self._value_norms)
            self.keys = torch.cat([k_deq.to(self.dtype), self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq.to(self.dtype), self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def _quantize_vectors(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norms = torch.norm(vectors.float(), dim=-1, keepdim=True)
        vectors_unit = vectors.float() / (norms + 1e-10)
        rotated = vectors_unit @ self._rotation.T
        indices = _scalar_quantize(rotated, self._codebook).to(torch.uint8)
        return indices, norms

    def _dequantize_vectors(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        dequant = _scalar_dequantize(indices.long(), self._codebook)
        return (dequant @ self._rotation) * norms

    def get_seq_length(self) -> int:
        return self._total_len


class TurboQuantCache(DynamicCache):
    """Drop-in Hugging Face cache that compresses older KV states."""

    def __init__(self, bits: int = 4, residual_len: int = 128, **kwargs):
        if not _TRANSFORMERS_CACHE_AVAILABLE:
            raise ImportError("transformers cache_utils is required for TurboQuant KV cache support")
        super().__init__(**kwargs)
        self.bits = bits
        self.residual_len = residual_len
        self.layer_class_to_replicate = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(TurboQuantLayer(bits=self.bits, residual_len=self.residual_len))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class Qwen35TurboQuantCache:
    """Qwen3.5-compatible hybrid cache.

    Qwen3.5 mixes full-attention layers with linear-attention layers that keep
    convolution/recurrent state. This cache preserves that interface while
    applying TurboQuant only to the full-attention KV tensors.
    """

    is_compileable = False

    def __init__(self, config: Any, bits: int = 4, residual_len: int = 128):
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [
            i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
        ]
        linear_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else None

        self.bits = bits
        self.residual_len = residual_len
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        self._kv_layers: dict[int, TurboQuantLayer] = {}

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._kv_layers:
            self._kv_layers[layer_idx] = TurboQuantLayer(bits=self.bits, residual_len=self.residual_len)
        keys, values = self._kv_layers[layer_idx].update(key_states, value_states, cache_kwargs)
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values
        return keys, values

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx, kv_layer in self._kv_layers.items():
            kv_layer.reorder_cache(beam_idx)
            self.key_cache[layer_idx] = kv_layer.keys
            self.value_cache[layer_idx] = kv_layer.values

        for layer_idx in range(len(self.layer_types)):
            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                index = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, index)
            if self.recurrent_states[layer_idx] is not None:
                device = self.recurrent_states[layer_idx].device
                index = beam_idx.to(device)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, index)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if not self.transformer_layers:
            return 0
        if layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        layer = self._kv_layers.get(layer_idx)
        return 0 if layer is None else layer.get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        if self.last_linear_layer is None:
            return False
        return self.conv_states[self.last_linear_layer] is not None


# ---------------------------------------------------------------------------
# TurboQuant Transformer
# ---------------------------------------------------------------------------

class TurboQuantTransformer:
    """Full transformer model backed by TurboQuant-compressed weights.

    Not an nn.Module — weights are stored as QuantizedTensor/raw and
    dequantized on-the-fly during the forward pass.
    """

    def __init__(self, tq_model: TurboQuantModel, device: torch.device,
                 dtype: torch.dtype = torch.float16, kv_bits: int | None = None):
        self.device = device
        self.dtype = dtype
        self.cfg = _parse_model_config(tq_model.config)
        self.quant_cfg = tq_model.quant_config

        # Build weight lookup
        self._weights: dict[str, TQWeight] = {}
        for name, qt in tq_model.layers.items():
            self._weights[name] = TQWeight(qt=qt)
        for name, raw in tq_model.unquantized.items():
            self._weights[name] = TQWeight(raw=raw)

        # Build norms
        self._norms: dict[str, torch.Tensor] = {}
        for name, w in self._weights.items():
            if "norm" in name.lower() and w._raw is not None:
                self._norms[name] = w._raw.to(device=device, dtype=torch.float32)

        # RoPE
        self.rope = RoPECache(
            self.cfg.head_dim,
            self.cfg.max_position_embeddings,
            self.cfg.rope_theta,
            device=device,
        )

        # KV caches — default to unquantized (kv_bits=0) for best quality
        effective_kv_bits = kv_bits if kv_bits is not None else 0
        self.kv_caches: list[KVCache] = []
        for _ in range(self.cfg.num_hidden_layers):
            self.kv_caches.append(KVCache(
                max_len=self.cfg.max_position_embeddings,
                n_heads=self.cfg.num_key_value_heads,
                head_dim=self.cfg.head_dim,
                device=device, dtype=dtype,
                kv_bits=effective_kv_bits,
            ))

        # torch.compile acceleration (CUDA only — MPS support is limited)
        self._compiled = False
        if device.type == "cuda":
            try:
                self._compiled_forward = torch.compile(
                    self._transformer_block, mode="reduce-overhead", fullgraph=False,
                )
                self._compiled = True
            except Exception:
                pass

    def _w(self, name: str) -> torch.Tensor:
        """Get dequantized weight by name, cast to model dtype."""
        return self._weights[name].get(self.device).to(self.dtype)

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Apply RMSNorm or LayerNorm with the named weight."""
        w = self._norms.get(name)
        if w is None:
            w = self._w(name).float()
        # Check for LayerNorm bias (BERT uses LayerNorm with bias)
        bias_name = name.replace(".weight", ".bias")
        bias = self._norms.get(bias_name)
        if bias is None and bias_name in self._weights:
            bias = self._weights[bias_name].get(self.device).float()
        if bias is not None:
            # Full LayerNorm
            return F.layer_norm(x.float(), (x.shape[-1],), weight=w, bias=bias, eps=self.cfg.rms_norm_eps).to(x.dtype)
        # RMSNorm (no bias)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.cfg.rms_norm_eps).rsqrt()
        return (x.float() * norm * w).to(x.dtype)

    # Layer patterns: matches both decoder ("layers.0.") and encoder ("layer.0.") naming
    _LAYER_PATTERNS = ("layers.{idx}.", "layer.{idx}.")

    def _matches_layer(self, name: str, layer_idx: int) -> bool:
        return any(p.format(idx=layer_idx) in name for p in self._LAYER_PATTERNS)

    def _is_layer_weight(self, name: str) -> bool:
        return "layers." in name or "layer." in name

    def _find_weight(self, layer_idx: int, *suffixes: str, exclude: str | None = None) -> str:
        """Find weight name matching layer index and one of the suffixes."""
        for name in self._weights:
            if self._matches_layer(name, layer_idx):
                if exclude and exclude in name:
                    continue
                for sfx in suffixes:
                    if name.endswith(sfx) or name.endswith(f"{sfx}.weight"):
                        return name
        raise KeyError(f"No weight found for layer {layer_idx} with suffixes {suffixes}")

    def _find_norm(self, layer_idx: int | None, *suffixes: str, exclude: str | None = None) -> str:
        """Find norm weight name."""
        for name in self._weights:
            if exclude and exclude in name:
                continue
            for sfx in suffixes:
                if layer_idx is not None and self._matches_layer(name, layer_idx) and sfx in name:
                    return name
                if layer_idx is None and sfx in name and not self._is_layer_weight(name):
                    return name
        raise KeyError(f"No norm found for layer_idx={layer_idx} suffixes={suffixes}")

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding lookup."""
        embed_w = self._w(self._find_weight_global("embed_tokens", "wte", "word_embeddings"))
        return F.embedding(token_ids, embed_w)

    def _find_weight_global(self, *suffixes: str) -> str:
        """Find a global weight (not layer-specific)."""
        for name in self._weights:
            if not self._is_layer_weight(name):
                for sfx in suffixes:
                    if sfx in name:
                        return name
        raise KeyError(f"No global weight found for suffixes {suffixes}")

    def forward(self, token_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass through the full transformer.

        Args:
            token_ids: (1, seq_len) input token IDs.
            start_pos: position offset for KV cache (0 for prompt, >0 for generation).

        Returns:
            logits: (1, seq_len, vocab_size).
        """
        h = self.embed(token_ids)  # (1, seq_len, hidden)
        seq_len = h.shape[1]
        cos, sin = self.rope.get(seq_len, offset=start_pos)
        cos = cos.to(device=self.device, dtype=self.dtype)
        sin = sin.to(device=self.device, dtype=self.dtype)

        block_fn = self._compiled_forward if self._compiled else self._transformer_block
        for i in range(self.cfg.num_hidden_layers):
            h = block_fn(h, i, cos, sin, start_pos)

        # Final norm
        h = self._norm(h, self._find_norm(None, "norm", "ln_f", "final_layer_norm", "embeddings.LayerNorm"))

        # LM head
        try:
            lm_head_name = self._find_weight_global("lm_head")
            logits = h @ self._w(lm_head_name).T
        except KeyError:
            # Tied embeddings
            embed_name = self._find_weight_global("embed_tokens", "wte", "word_embeddings")
            logits = h @ self._w(embed_name).T

        return logits

    def _transformer_block(self, h: torch.Tensor, layer_idx: int,
                           cos: torch.Tensor, sin: torch.Tensor,
                           start_pos: int) -> torch.Tensor:
        """Single transformer layer: attention + FFN with residual connections."""
        # Pre-attention norm (BERT: attention.output.LayerNorm)
        norm_name = self._find_norm(layer_idx,
                                     "input_layernorm", "attention_norm", "ln_1",
                                     "attention.output.LayerNorm")
        h_normed = self._norm(h, norm_name)

        # Self-attention
        attn_out = self._attention(h_normed, layer_idx, cos, sin, start_pos)
        h = h + attn_out

        # Pre-FFN norm (BERT: output.LayerNorm — exclude attention.output.LayerNorm)
        ffn_norm_name = self._find_norm(layer_idx,
                                         "post_attention_layernorm", "ffn_norm", "ln_2",
                                         "output.LayerNorm", exclude="attention.output")
        h_normed = self._norm(h, ffn_norm_name)

        # FFN
        ffn_out = self._ffn(h_normed, layer_idx)
        h = h + ffn_out

        return h

    def _attention(self, x: torch.Tensor, layer_idx: int,
                   cos: torch.Tensor, sin: torch.Tensor,
                   start_pos: int) -> torch.Tensor:
        """Multi-head (grouped) attention with RoPE and KV cache."""
        bsz, seq_len, _ = x.shape
        cfg = self.cfg

        # Project Q, K, V (BERT: attention.self.query / key / value)
        q_name = self._find_weight(layer_idx, "q_proj", "query", "self.query")
        k_name = self._find_weight(layer_idx, "k_proj", "key", "self.key")
        v_name = self._find_weight(layer_idx, "v_proj", "value", "self.value")

        q = x @ self._w(q_name).T
        k = x @ self._w(k_name).T
        v = x @ self._w(v_name).T

        # Add biases if present (e.g. Qwen2.5)
        for wn in (q_name, k_name, v_name):
            bn = wn.replace(".weight", ".bias")
            if bn in self._weights:
                b = self._w(bn)
                if wn is q_name:
                    q = q + b
                elif wn is k_name:
                    k = k + b
                else:
                    v = v + b

        # Reshape for multi-head
        q = q.view(bsz, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        # RoPE
        q, k = apply_rope(q, k, cos, sin)

        # KV cache
        cache = self.kv_caches[layer_idx]
        cache.append(k, v)
        k, v = cache.get_kv()

        # GQA: repeat KV heads if needed
        if cfg.num_key_value_heads < cfg.num_attention_heads:
            n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=(seq_len > 1),
        )

        # Reshape and project output (BERT: attention.output.dense)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return attn @ self._w(self._find_weight(layer_idx, "o_proj", "out_proj", "attention.output.dense")).T

    def _ffn(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """SiLU-gated FFN (LLaMA-style) or standard 2-layer FFN (BERT-style)."""
        try:
            gate_w = self._w(self._find_weight(layer_idx, "gate_proj", "w1"))
            up_w = self._w(self._find_weight(layer_idx, "up_proj", "w3"))
            down_w = self._w(self._find_weight(layer_idx, "down_proj", "w2"))
            return (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T
        except KeyError:
            # Standard 2-layer FFN (BERT: intermediate.dense + output.dense)
            fc1_w = self._w(self._find_weight(layer_idx, "fc1", "dense_h_to_4h", "c_fc", "intermediate.dense"))
            # exclude="attention.output" prevents matching attention.output.dense
            fc2_w = self._w(self._find_weight(layer_idx, "fc2", "dense_4h_to_h", "c_proj", "output.dense",
                                               exclude="attention.output"))
            return F.gelu(x @ fc1_w.T) @ fc2_w.T

    def reset_caches(self):
        """Clear KV caches (for new conversation)."""
        for cache in self.kv_caches:
            cache.len = 0
            if cache.kv_bits > 0:
                cache._k_indices.clear()
                cache._v_indices.clear()
                cache._k_norms.clear()
                cache._v_norms.clear()
                cache._token_counts.clear()

    def clear_weight_caches(self):
        """Free dequantized weight caches to save memory."""
        for w in self._weights.values():
            w.clear_cache()


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_tokens: list[int] | None = None


def _sample_token(logits: torch.Tensor, config: GenerationConfig,
                  generated: list[int]) -> int:
    """Sample a single token from logits with temperature, top-k, top-p."""
    logits = logits.float()

    # Repetition penalty
    if config.repetition_penalty != 1.0 and generated:
        for token_id in set(generated[-64:]):
            if logits[token_id] > 0:
                logits[token_id] /= config.repetition_penalty
            else:
                logits[token_id] *= config.repetition_penalty

    if config.temperature <= 0 or config.temperature < 1e-6:
        return logits.argmax().item()

    logits = logits / config.temperature

    # Top-k
    if config.top_k > 0:
        top_k = min(config.top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus)
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate(
    model: TurboQuantTransformer | TurboQuantHFModel,
    token_ids: list[int],
    config: GenerationConfig | None = None,
    tokenizer: Any = None,
) -> Generator[int, None, None]:
    """Stream-generate tokens from a TurboQuant model.

    Yields one token ID at a time.
    """
    if config is None:
        config = GenerationConfig()

    if isinstance(model, TurboQuantHFModel):
        yield from _generate_hf(model, token_ids, config, tokenizer)
        return

    device = model.device
    model.reset_caches()

    # Normalize token_ids to a plain list of ints
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(t) for t in token_ids]

    # Prefill: process full prompt
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model.forward(input_ids, start_pos=0)
    next_logits = logits[0, -1, :]

    generated: list[int] = list(token_ids)
    stop_tokens = set(config.stop_tokens or [])
    if tokenizer is not None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_tokens.add(eos)

    for _ in range(config.max_new_tokens):
        token = _sample_token(next_logits, config, generated)
        if token in stop_tokens:
            break
        generated.append(token)
        yield token

        # Decode step: single token
        input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        logits = model.forward(input_ids, start_pos=len(generated) - 1)
        next_logits = logits[0, -1, :]


def _normalize_token_ids(token_ids: Any) -> list[int]:
    """Normalize token IDs to a plain flat Python list."""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(t) for t in token_ids]


def _model_input_device(model: Any) -> torch.device:
    """Best-effort device for feeding token IDs into a HF model."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _is_accelerate_available() -> bool:
    """Whether accelerate is importable for device_map-based loading."""
    try:
        import accelerate  # noqa: F401

        return True
    except ImportError:
        return False


def _generate_hf(
    model: TurboQuantHFModel,
    token_ids: list[int],
    config: GenerationConfig,
    tokenizer: Any = None,
) -> Generator[int, None, None]:
    """Generate using a standard HF causal LM with a TurboQuant cache."""
    hf_model = model.hf_model
    hf_model.eval()

    prompt_ids = _normalize_token_ids(token_ids)
    input_device = _model_input_device(hf_model)
    cache = None
    if model.kv_bits > 0:
        hf_config = getattr(hf_model, "config", None)
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types is None:
            layer_types = getattr(getattr(hf_config, "text_config", None), "layer_types", None)
        if layer_types and "linear_attention" in layer_types:
            hybrid_config = getattr(hf_config, "text_config", None) or hf_config
            cache = Qwen35TurboQuantCache(hybrid_config, bits=model.kv_bits)
        else:
            cache = TurboQuantCache(bits=model.kv_bits)

    stop_tokens = set(config.stop_tokens or [])
    if tokenizer is not None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_tokens.add(eos)

    generated: list[int] = list(prompt_ids)
    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=input_device)
        if cache is None:
            outputs = hf_model(input_ids=input_ids, use_cache=True)
        else:
            outputs = hf_model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        past = outputs.past_key_values
        next_logits = outputs.logits[0, -1, :]

        for _ in range(config.max_new_tokens):
            token = _sample_token(next_logits, config, generated)
            if token in stop_tokens:
                break
            generated.append(token)
            yield token

            input_ids = torch.tensor([[token]], dtype=torch.long, device=input_device)
            outputs = hf_model(input_ids=input_ids, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_logits = outputs.logits[0, -1, :]


# ---------------------------------------------------------------------------
# Load model + tokenizer convenience
# ---------------------------------------------------------------------------

def load_model(
    tqf_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float16",
) -> tuple[TurboQuantTransformer | TurboQuantHFModel, Any]:
    """Load a TurboQuant model and its tokenizer.

    Returns:
        (model, tokenizer) — tokenizer may be None if not found.
    """
    from ollama_forge.device import get_device, get_device_map

    runtime_device = "mps" if device == "mlx" else device
    dev = torch.device(get_device(runtime_device))
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    if dev.type == "cpu" and dt == torch.float16:
        dt = torch.float32

    tq_model = load_tqf(tqf_path)
    tqf_dir = Path(tqf_path)
    source_path = tq_model.resolved_model_path
    if source_path and not Path(source_path).is_dir():
        source_path = None

    # Prefer the working HF-backed runtime for new TurboQuant packages.
    if tq_model.source_model or source_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_source = source_path or tq_model.source_model
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tqf_dir), trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dt,
        }

        # `device_map` requires accelerate. Fall back to a normal single-device
        # load when accelerate is not installed.
        if _is_accelerate_available():
            load_kwargs["device_map"] = get_device_map(runtime_device)

        hf_model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        if "device_map" not in load_kwargs:
            hf_model = hf_model.to(device=dev, dtype=dt)

        return TurboQuantHFModel(
            hf_model=hf_model,
            device=_model_input_device(hf_model),
            kv_bits=max(int(tq_model.quant_config.kv_bits or 0), 0),
        ), tokenizer

    transformer = TurboQuantTransformer(tq_model, device=dev, dtype=dt)

    tokenizer = None
    tok_json = tqf_dir / "tokenizer.json"
    tok_config = tqf_dir / "tokenizer_config.json"
    if tok_json.exists() or tok_config.exists():
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(tqf_dir), trust_remote_code=True)
        except Exception:
            pass

    return transformer, tokenizer
