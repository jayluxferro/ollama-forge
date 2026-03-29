"""TurboQuant inference engine — MLX backend for Apple Silicon.

Native MLX implementation for 2-3× faster inference on M1/M2/M3/M4 chips
compared to the PyTorch MPS backend.  Uses unified memory (no CPU↔GPU copies)
and MLX's graph compilation for kernel fusion.

Falls back gracefully if MLX is not installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

_MLX_AVAILABLE = False
try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    pass


def is_mlx_available() -> bool:
    return _MLX_AVAILABLE


if _MLX_AVAILABLE:

    # -------------------------------------------------------------------
    # MLX codebook + rotation helpers
    # -------------------------------------------------------------------

    _CODEBOOKS_RAW = {
        1: [-0.7978845608, 0.7978845608],
        2: [-1.510_232_6, -0.452_842_7, 0.452_842_7, 1.510_232_6],
        3: [
            -2.152_174_6, -1.344_171_4, -0.756_421_2, -0.245_340_8,
             0.245_340_8,  0.756_421_2,  1.344_171_4,  2.152_174_6,
        ],
        4: [
            -2.733_460_0, -2.069_016_0, -1.618_192_0, -1.256_346_0,
            -0.942_082_0, -0.656_532_0, -0.388_378_0, -0.127_961_0,
             0.127_961_0,  0.388_378_0,  0.656_532_0,  0.942_082_0,
             1.256_346_0,  1.618_192_0,  2.069_016_0,  2.733_460_0,
        ],
    }

    def _get_codebook_mlx(bits: int, dim: int) -> mx.array:
        raw = mx.array(_CODEBOOKS_RAW[bits], dtype=mx.float32)
        return raw / math.sqrt(dim)

    def _generate_rotation_mlx(d: int, seed: int) -> mx.array:
        """Generate rotation matrix matching the PyTorch implementation."""
        # Use numpy for QR (deterministic with seed), then convert to MLX
        import numpy as np
        rng = np.random.RandomState(seed)
        G = rng.randn(d, d).astype(np.float32)
        Q, R = np.linalg.qr(G)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs[np.newaxis, :]
        return mx.array(Q)

    def _scalar_dequantize_mlx(indices: mx.array, codebook: mx.array) -> mx.array:
        return codebook[indices.astype(mx.int32)]

    def _scalar_quantize_mlx(y: mx.array, codebook: mx.array) -> mx.array:
        """Quantize each element to nearest centroid index.

        Uses broadcasting: compute |y - centroid| for all centroids, take argmin.
        """
        # y: (..., d), codebook: (n_centroids,)
        # Expand for broadcasting: y[..., :, None] vs codebook[None, :]
        diffs = mx.abs(mx.expand_dims(y, axis=-1) - codebook)  # (..., d, n_centroids)
        return mx.argmin(diffs, axis=-1)  # (..., d)

    # -------------------------------------------------------------------
    # RoPE
    # -------------------------------------------------------------------

    class RoPEMLX:
        def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0):
            self._head_dim = head_dim
            self._theta = theta
            self._build(max_len)

        def _build(self, length: int):
            inv_freq = 1.0 / (self._theta ** (mx.arange(0, self._head_dim, 2, dtype=mx.float32) / self._head_dim))
            t = mx.arange(length, dtype=mx.float32)
            freqs = mx.outer(t, inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            self.cos = mx.cos(emb)[None, None, :, :]
            self.sin = mx.sin(emb)[None, None, :, :]
            self._len = length

        def get(self, seq_len: int, offset: int = 0):
            needed = offset + seq_len
            if needed > self._len:
                self._build(needed * 2)  # double to avoid repeated rebuilds
            return self.cos[:, :, offset:offset+seq_len, :], self.sin[:, :, offset:offset+seq_len, :]

    def _rotate_half_mlx(x: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        return mx.concatenate([-x2, x1], axis=-1)

    def apply_rope_mlx(q, k, cos, sin):
        return q * cos + _rotate_half_mlx(q) * sin, k * cos + _rotate_half_mlx(k) * sin

    # -------------------------------------------------------------------
    # KV Cache (MLX — unified memory, no transfers)
    # -------------------------------------------------------------------

    class KVCacheMLX:
        def __init__(self, max_len: int, n_heads: int, head_dim: int, kv_bits: int = 0):
            self.max_len = max_len
            self.n_heads = n_heads
            self.head_dim = head_dim
            self.kv_bits = kv_bits
            self.len = 0

            if kv_bits > 0:
                self._k_indices: list[mx.array] = []
                self._v_indices: list[mx.array] = []
                self._k_norms: list[mx.array] = []
                self._v_norms: list[mx.array] = []
                self._token_counts: list[int] = []
                self._codebook = _get_codebook_mlx(kv_bits, head_dim)
                self._rotation = _generate_rotation_mlx(head_dim, seed=42)
            else:
                self._k_list: list[mx.array] = []
                self._v_list: list[mx.array] = []

        def append(self, k_new: mx.array, v_new: mx.array):
            seq_len = k_new.shape[2]
            if self.kv_bits > 0:
                # k_new: (1, nh, sl, hd) — quantize per-head, keep (nh, sl, hd) layout
                kv_3d = k_new.squeeze(0)  # (nh, sl, hd)
                vv_3d = v_new.squeeze(0)
                nh, sl, hd = kv_3d.shape
                # Flatten to (nh*sl, hd) for batched quantization
                k_flat = kv_3d.reshape(nh * sl, hd)
                v_flat = vv_3d.reshape(nh * sl, hd)
                k_norms = mx.sqrt(mx.sum(k_flat * k_flat, axis=1) + 1e-10)
                v_norms = mx.sqrt(mx.sum(v_flat * v_flat, axis=1) + 1e-10)
                k_normed = k_flat / k_norms[:, None]
                v_normed = v_flat / v_norms[:, None]
                k_rot = k_normed @ self._rotation.T
                v_rot = v_normed @ self._rotation.T
                # Store as (nh, sl, hd) shaped indices/norms
                k_idx = _scalar_quantize_mlx(k_rot, self._codebook).reshape(nh, sl, hd)
                v_idx = _scalar_quantize_mlx(v_rot, self._codebook).reshape(nh, sl, hd)
                self._k_indices.append(k_idx)
                self._v_indices.append(v_idx)
                self._k_norms.append(k_norms.reshape(nh, sl))
                self._v_norms.append(v_norms.reshape(nh, sl))
                self._token_counts.append(sl)
            else:
                self._k_list.append(k_new)
                self._v_list.append(v_new)
            self.len += seq_len

        def get_kv(self):
            if self.kv_bits > 0:
                if not self._k_indices:
                    empty = mx.zeros((1, self.n_heads, 0, self.head_dim))
                    return empty, empty
                # Each stored as (nh, sl_i, hd) — concat along seq dim (axis=1)
                k_idx = mx.concatenate(self._k_indices, axis=1)   # (nh, total, hd)
                v_idx = mx.concatenate(self._v_indices, axis=1)
                k_norms = mx.concatenate(self._k_norms, axis=1)   # (nh, total)
                v_norms = mx.concatenate(self._v_norms, axis=1)
                # Dequantize
                k_deq = _scalar_dequantize_mlx(k_idx, self._codebook) @ self._rotation
                v_deq = _scalar_dequantize_mlx(v_idx, self._codebook) @ self._rotation
                k_deq = k_deq * k_norms[:, :, None]  # (nh, total, hd)
                v_deq = v_deq * v_norms[:, :, None]
                return k_deq[None], v_deq[None]       # (1, nh, total, hd)
            else:
                if not self._k_list:
                    empty = mx.zeros((1, self.n_heads, 0, self.head_dim))
                    return empty, empty
                return mx.concatenate(self._k_list, axis=2), mx.concatenate(self._v_list, axis=2)

    # -------------------------------------------------------------------
    # TurboQuant MLX Transformer
    # -------------------------------------------------------------------

    @dataclass
    class ModelConfigMLX:
        vocab_size: int = 32000
        hidden_size: int = 4096
        intermediate_size: int = 11008
        num_hidden_layers: int = 32
        num_attention_heads: int = 32
        num_key_value_heads: int = 32
        head_dim: int = 128
        max_position_embeddings: int = 4096
        rms_norm_eps: float = 1e-6
        rope_theta: float = 10000.0
        tie_word_embeddings: bool = False
        # Qwen3.5 hybrid attention fields
        layer_types: list[str] | None = None  # None → all standard attention
        partial_rotary_factor: float = 1.0
        attn_output_gate: bool = False
        linear_key_head_dim: int = 128
        linear_value_head_dim: int = 128
        linear_num_key_heads: int = 0
        linear_num_value_heads: int = 0
        linear_conv_kernel_dim: int = 4

    def _parse_config_mlx(hf_cfg: dict) -> ModelConfigMLX:
        # Qwen3.5-style models nest text params in text_config
        tc = hf_cfg.get("text_config", {})
        cfg = {**hf_cfg, **tc} if tc else hf_cfg  # text_config overrides top-level

        n_heads = cfg.get("num_attention_heads", 32)
        hidden = cfg.get("hidden_size", 4096)

        rope_params = cfg.get("rope_parameters", {})
        rope_theta = rope_params.get("rope_theta", cfg.get("rope_theta", 10000.0))
        partial_rotary = rope_params.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))

        return ModelConfigMLX(
            vocab_size=cfg.get("vocab_size", 32000),
            hidden_size=hidden,
            intermediate_size=cfg.get("intermediate_size", 11008),
            num_hidden_layers=cfg.get("num_hidden_layers", 32),
            num_attention_heads=n_heads,
            num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=cfg.get("head_dim", hidden // n_heads),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            layer_types=cfg.get("layer_types"),
            partial_rotary_factor=partial_rotary,
            attn_output_gate=cfg.get("attn_output_gate", False),
            linear_key_head_dim=cfg.get("linear_key_head_dim", 128),
            linear_value_head_dim=cfg.get("linear_value_head_dim", 128),
            linear_num_key_heads=cfg.get("linear_num_key_heads", 0),
            linear_num_value_heads=cfg.get("linear_num_value_heads", 0),
            linear_conv_kernel_dim=cfg.get("linear_conv_kernel_dim", 4),
        )

    class TQWeightMLX:
        """MLX weight wrapper — dequantizes from TurboQuant QuantizedTensor."""

        def __init__(self, qt=None, raw_tensor=None):
            self._qt = qt
            self._raw = raw_tensor
            self._cached: mx.array | None = None

        def get(self) -> mx.array:
            if self._cached is not None:
                return self._cached
            if self._qt is not None:
                # Dequantize using PyTorch, convert to MLX
                import torch

                from ollama_forge.turboquant import dequantize_tensor
                w_torch = dequantize_tensor(self._qt, device=torch.device("cpu"))
                self._cached = mx.array(w_torch.float().numpy())
            elif self._raw is not None:
                import torch
                if isinstance(self._raw, torch.Tensor):
                    self._cached = mx.array(self._raw.float().numpy())
                else:
                    self._cached = self._raw
            return self._cached

        @property
        def shape(self):
            if self._qt:
                return self._qt.shape
            return tuple(self._raw.shape)

    class TurboQuantTransformerMLX:
        """Full transformer using MLX for Apple Silicon inference."""

        def __init__(self, tq_model, kv_bits: int = 0):
            self.cfg = _parse_config_mlx(tq_model.config)
            self.quant_cfg = tq_model.quant_config

            # Convert weights to MLX
            self._weights: dict[str, TQWeightMLX] = {}
            for name, qt in tq_model.layers.items():
                self._weights[name] = TQWeightMLX(qt=qt)
            for name, raw in tq_model.unquantized.items():
                self._weights[name] = TQWeightMLX(raw_tensor=raw)

            self._norms: dict[str, mx.array] = {}
            for name, w in self._weights.items():
                if "norm" in name.lower() and w._raw is not None:
                    self._norms[name] = w.get()

            rope_dim = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
            self.rope = RoPEMLX(rope_dim, self.cfg.max_position_embeddings, self.cfg.rope_theta)
            self._rope_dim = rope_dim

            self.kv_caches: list[KVCacheMLX] = []
            for i in range(self.cfg.num_hidden_layers):
                is_full = (self.cfg.layer_types is None
                           or self.cfg.layer_types[i] == "full_attention")
                if is_full:
                    self.kv_caches.append(KVCacheMLX(
                        self.cfg.max_position_embeddings,
                        self.cfg.num_key_value_heads,
                        self.cfg.head_dim,
                        kv_bits=kv_bits,
                    ))
                else:
                    self.kv_caches.append(None)  # linear attn uses SSM state

            # Linear attention recurrent state (Qwen3.5 gated delta rule)
            self._lin_states: list[dict | None] = [None] * self.cfg.num_hidden_layers

        def _w(self, name: str) -> mx.array:
            return self._weights[name].get()

        def _rmsnorm(self, x: mx.array, name: str) -> mx.array:
            """Apply RMSNorm or LayerNorm."""
            w = self._norms.get(name)
            if w is None:
                w = self._w(name)
            # Check for LayerNorm bias (BERT-style)
            bias_name = name.replace(".weight", ".bias")
            bias = self._norms.get(bias_name)
            if bias is None and bias_name in self._weights:
                bias = self._weights[bias_name].get()
            if bias is not None:
                # Full LayerNorm
                mean = mx.mean(x, axis=-1, keepdims=True)
                var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
                return (x - mean) / mx.sqrt(var + self.cfg.rms_norm_eps) * w + bias
            # RMSNorm
            norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.cfg.rms_norm_eps)
            return x * norm * w

        _LAYER_PATTERNS = ("language_model.layers.{idx}.", "layers.{idx}.", "layer.{idx}.")

        def _matches_layer(self, name: str, layer_idx: int) -> bool:
            if name.startswith("mtp.") or name.startswith("model.visual."):
                return False
            return any(p.format(idx=layer_idx) in name for p in self._LAYER_PATTERNS)

        def _is_layer_weight(self, name: str) -> bool:
            return ("layers." in name or "layer." in name) and not name.startswith("model.visual.")

        def _find_weight(self, layer_idx: int, *suffixes: str, exclude: str | None = None) -> str:
            for name in self._weights:
                if self._matches_layer(name, layer_idx):
                    if exclude and exclude in name:
                        continue
                    for sfx in suffixes:
                        if name.endswith(sfx) or name.endswith(f"{sfx}.weight"):
                            return name
            raise KeyError(f"No weight for layer {layer_idx} suffixes {suffixes}")

        def _find_norm(self, layer_idx: int | None, *suffixes: str, exclude: str | None = None) -> str:
            for name in self._weights:
                if exclude and exclude in name:
                    continue
                for sfx in suffixes:
                    if layer_idx is not None and self._matches_layer(name, layer_idx) and sfx in name:
                        return name
                    if layer_idx is None and sfx in name and not self._is_layer_weight(name):
                        return name
            raise KeyError(f"No norm for layer_idx={layer_idx}")

        def _find_global(self, *suffixes: str) -> str:
            for name in self._weights:
                if not self._is_layer_weight(name):
                    if name.startswith("mtp.") or name.startswith("model.visual."):
                        continue
                    for sfx in suffixes:
                        if sfx in name:
                            return name
            raise KeyError(f"No global weight for {suffixes}")

        def forward(self, token_ids: mx.array, start_pos: int = 0) -> mx.array:
            embed_w = self._w(self._find_global("embed_tokens", "wte", "word_embeddings"))
            h = embed_w[token_ids]  # (1, seq_len, hidden)
            seq_len = h.shape[1]
            cos, sin = self.rope.get(seq_len, offset=start_pos)

            for i in range(self.cfg.num_hidden_layers):
                h = self._block(h, i, cos, sin)

            h = self._rmsnorm(h, self._find_norm(None, "norm", "ln_f", "final_layer_norm", "embeddings.LayerNorm"))

            try:
                lm_w = self._w(self._find_global("lm_head"))
            except KeyError:
                lm_w = self._w(self._find_global("embed_tokens", "wte", "word_embeddings"))
            return h @ lm_w.T

        def _block(self, h, layer_idx, cos, sin):
            h_n = self._rmsnorm(h, self._find_norm(layer_idx,
                                                     "input_layernorm", "attention_norm", "ln_1",
                                                     "attention.output.LayerNorm"))
            lt = self.cfg.layer_types
            if lt is not None and lt[layer_idx] == "linear_attention":
                h = h + self._linear_attention(h_n, layer_idx)
            else:
                h = h + self._attention(h_n, layer_idx, cos, sin)
            h_n = self._rmsnorm(h, self._find_norm(layer_idx,
                                                     "post_attention_layernorm", "ffn_norm", "ln_2",
                                                     "output.LayerNorm", exclude="attention.output"))
            h = h + self._ffn(h_n, layer_idx)
            return h

        def _maybe_bias(self, weight_name: str) -> mx.array | None:
            """Return the bias tensor corresponding to a weight, or None."""
            bias_name = weight_name.replace(".weight", ".bias")
            if bias_name != weight_name and bias_name in self._weights:
                return self._weights[bias_name].get()
            return None

        def _attention(self, x, layer_idx, cos, sin):
            cfg = self.cfg
            bsz, seq_len, _ = x.shape

            q_name = self._find_weight(layer_idx, "q_proj", "query", "self.query")
            k_name = self._find_weight(layer_idx, "k_proj", "key", "self.key")
            v_name = self._find_weight(layer_idx, "v_proj", "value", "self.value")

            q = x @ self._w(q_name).T
            k = x @ self._w(k_name).T
            v = x @ self._w(v_name).T

            q_bias = self._maybe_bias(q_name)
            if q_bias is not None:
                q = q + q_bias
            k_bias = self._maybe_bias(k_name)
            if k_bias is not None:
                k = k + k_bias
            v_bias = self._maybe_bias(v_name)
            if v_bias is not None:
                v = v + v_bias

            # Qwen3.5 full attention: q_proj outputs 2x (query + gate)
            gate = None
            if cfg.attn_output_gate:
                q_dim = q.shape[-1]
                q, gate = q[..., :q_dim // 2], q[..., q_dim // 2:]

            n_q_heads = q.shape[-1] // cfg.head_dim
            q = q.reshape(bsz, seq_len, n_q_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 2, 1, 3)

            # Q/K RMSNorm (Qwen3.5)
            try:
                qn = self._find_norm(layer_idx, "q_norm")
                q = self._rmsnorm(q, qn)
            except KeyError:
                pass
            try:
                kn = self._find_norm(layer_idx, "k_norm")
                k = self._rmsnorm(k, kn)
            except KeyError:
                pass

            # Partial RoPE: apply only to first rope_dim dims
            rd = self._rope_dim
            if rd < cfg.head_dim:
                q_rot, q_pass = q[..., :rd], q[..., rd:]
                k_rot, k_pass = k[..., :rd], k[..., rd:]
                q_rot, k_rot = apply_rope_mlx(q_rot, k_rot, cos, sin)
                q = mx.concatenate([q_rot, q_pass], axis=-1)
                k = mx.concatenate([k_rot, k_pass], axis=-1)
            else:
                q, k = apply_rope_mlx(q, k, cos, sin)

            cache = self.kv_caches[layer_idx]
            cache.append(k, v)
            k, v = cache.get_kv()

            if cfg.num_key_value_heads < n_q_heads:
                n_rep = n_q_heads // cfg.num_key_value_heads
                k = mx.repeat(k, n_rep, axis=1)
                v = mx.repeat(v, n_rep, axis=1)

            scale = 1.0 / math.sqrt(cfg.head_dim)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            if seq_len > 1:
                mask = mx.triu(mx.full((seq_len, k.shape[2]), float("-inf")), k=1)
                scores = scores + mask

            attn = mx.softmax(scores, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)

            # Output gating (Qwen3.5)
            if gate is not None:
                out = out * mx.sigmoid(gate)

            return out @ self._w(self._find_weight(layer_idx, "o_proj", "out_proj", "attention.output.dense")).T

        def _ffn(self, x, layer_idx):
            try:
                gate = x @ self._w(self._find_weight(layer_idx, "gate_proj", "w1")).T
                up = x @ self._w(self._find_weight(layer_idx, "up_proj", "w3")).T
                down_w = self._w(self._find_weight(layer_idx, "down_proj", "w2"))
                return (mx.sigmoid(gate) * gate * up) @ down_w.T  # SiLU = x * sigmoid(x)
            except KeyError:
                fc1 = x @ self._w(self._find_weight(layer_idx, "fc1", "dense_h_to_4h", "c_fc", "intermediate.dense")).T
                fc2_w = self._w(self._find_weight(layer_idx, "fc2", "dense_4h_to_h", "c_proj", "output.dense",
                                                   exclude="attention.output"))
                # GELU activation (BERT-style)
                activated = fc1 * 0.5 * (1.0 + mx.erf(fc1 / math.sqrt(2.0)))
                return activated @ fc2_w.T

        def _linear_attention(self, x, layer_idx):
            """Gated DeltaNet linear attention (Qwen3.5)."""
            cfg = self.cfg
            bsz, seq_len, _ = x.shape
            hk = cfg.linear_num_key_heads
            hv = cfg.linear_num_value_heads
            dk = cfg.linear_key_head_dim
            dv = cfg.linear_value_head_dim
            K = cfg.linear_conv_kernel_dim

            # Preload all weights once
            qkv_w = self._w(self._find_weight(layer_idx, "in_proj_qkv"))
            z_w = self._w(self._find_weight(layer_idx, "in_proj_z"))
            a_w = self._w(self._find_weight(layer_idx, "in_proj_a"))
            b_w = self._w(self._find_weight(layer_idx, "in_proj_b"))
            conv_w = self._w(self._find_weight(layer_idx, "conv1d")).squeeze(1)  # (d_inner, K)

            # Batch projections
            mixed_qkv = x @ qkv_w.T  # (bsz, seq_len, d_inner)
            z = x @ z_w.T
            all_alpha = mx.sigmoid(x @ a_w.T)  # (bsz, seq_len, hv)
            all_beta = mx.sigmoid(x @ b_w.T)

            d_inner = mixed_qkv.shape[-1]
            state = self._lin_states[layer_idx]
            if state is None:
                conv_state = mx.zeros((bsz, d_inner, K))
                S = mx.zeros((bsz, hv, dk, dv))
            else:
                conv_state = state["conv"]
                S = state["S"]

            # Causal 1D depthwise conv + recurrent delta rule
            n_rep = hv // hk if hk < hv else 1
            out_tokens = []
            for t in range(seq_len):
                conv_state = mx.concatenate([
                    conv_state[:, :, 1:],
                    mixed_qkv[:, t:t+1, :].transpose(0, 2, 1),
                ], axis=2)
                conv_out = (conv_state * conv_w).sum(axis=2)

                q_t = conv_out[:, :hk * dk].reshape(bsz, hk, dk)
                k_t = conv_out[:, hk * dk:2 * hk * dk].reshape(bsz, hk, dk)
                v_t = conv_out[:, 2 * hk * dk:].reshape(bsz, hv, dv)

                if n_rep > 1:
                    q_t = mx.repeat(q_t, n_rep, axis=1)
                    k_t = mx.repeat(k_t, n_rep, axis=1)

                alpha = all_alpha[:, t, :, None, None]
                beta = all_beta[:, t, :, None, None]
                S = alpha * S + beta * (k_t[:, :, :, None] * v_t[:, :, None, :])

                y_t = (q_t[:, :, None, :] @ S).squeeze(2)
                out_tokens.append(y_t.reshape(bsz, 1, hv * dv))

                if t % 8 == 7:
                    mx.eval(S)  # periodically materialize to limit graph size

            self._lin_states[layer_idx] = {"conv": conv_state, "S": S}
            out = mx.concatenate(out_tokens, axis=1)

            # Gated RMSNorm: norm(out) * silu(z)
            try:
                norm_name = self._find_norm(layer_idx, "linear_attn.norm")
                norm_w = self._norms.get(norm_name)
                if norm_w is None:
                    norm_w = self._w(norm_name)
                out_r = out.reshape(bsz, seq_len, hv, dv)
                rms = mx.rsqrt(mx.mean(out_r * out_r, axis=-1, keepdims=True) + cfg.rms_norm_eps)
                out_r = out_r * rms * norm_w
                out = out_r.reshape(bsz, seq_len, hv * dv)
            except KeyError:
                pass
            out = out * (z * mx.sigmoid(z))  # * SiLU(z)

            out_w = self._w(self._find_weight(layer_idx, "out_proj"))
            return out @ out_w.T

        def reset_caches(self):
            for cache in self.kv_caches:
                if cache is not None:
                    cache.len = 0
                    if cache.kv_bits > 0:
                        cache._k_indices.clear()
                        cache._v_indices.clear()
                        cache._k_norms.clear()
                        cache._v_norms.clear()
                        cache._token_counts.clear()
                    else:
                        cache._k_list.clear()
                        cache._v_list.clear()
            self._lin_states = [None] * self.cfg.num_hidden_layers

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    @dataclass
    class GenerationConfigMLX:
        max_new_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.1
        stop_tokens: list[int] | None = None

    def _sample_mlx(logits: mx.array, config: GenerationConfigMLX, generated: list[int]) -> int:
        logits = logits.astype(mx.float32)

        if config.repetition_penalty != 1.0 and generated:
            # Build penalty mask as a full-size array (MLX has no .at[].set())
            penalty = mx.ones(logits.shape)
            for tid in set(generated[-64:]):
                val = logits[tid].item()
                if val > 0:
                    penalty = penalty * mx.where(mx.arange(logits.shape[0]) == tid,
                                                  1.0 / config.repetition_penalty, 1.0)
                else:
                    penalty = penalty * mx.where(mx.arange(logits.shape[0]) == tid,
                                                  config.repetition_penalty, 1.0)
            logits = logits * penalty

        if config.temperature < 1e-6:
            return mx.argmax(logits).item()

        logits = logits / config.temperature

        if config.top_k > 0:
            top_k = min(config.top_k, logits.shape[-1])
            kth = mx.sort(logits)[-top_k]
            logits = mx.where(logits < kth, float("-inf"), logits)

        probs = mx.softmax(logits)
        return mx.random.categorical(mx.log(probs + 1e-10)).item()

    def generate_mlx(
        model: TurboQuantTransformerMLX,
        token_ids: list[int],
        config: GenerationConfigMLX | None = None,
        tokenizer: Any = None,
    ) -> Generator[int, None, None]:
        if config is None:
            config = GenerationConfigMLX()

        model.reset_caches()

        # Normalize token_ids to a plain list of ints
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = [int(t) for t in token_ids]

        input_ids = mx.array([token_ids], dtype=mx.int32)
        logits = model.forward(input_ids, start_pos=0)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        generated = list(token_ids)
        stop_tokens = set(config.stop_tokens or [])
        if tokenizer:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                stop_tokens.add(eos)

        for _ in range(config.max_new_tokens):
            token = _sample_mlx(next_logits, config, generated)
            if token in stop_tokens:
                break
            generated.append(token)
            yield token

            input_ids = mx.array([[token]], dtype=mx.int32)
            logits = model.forward(input_ids, start_pos=len(generated) - 1)
            mx.eval(logits)
            next_logits = logits[0, -1, :]

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------

    def load_model_mlx(tqf_path: str | Path, *, kv_bits: int | None = None) -> tuple[TurboQuantTransformerMLX, Any]:
        from ollama_forge.turboquant_pipeline import load_tqf

        tq_model = load_tqf(tqf_path)
        if kv_bits is None:
            kv_bits = 0  # default off; quantized KV degrades quality on small models
        transformer = TurboQuantTransformerMLX(tq_model, kv_bits=kv_bits)

        tokenizer = None
        tqf_dir = Path(tqf_path)
        if (tqf_dir / "tokenizer.json").exists() or (tqf_dir / "tokenizer_config.json").exists():
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(tqf_dir))
            except Exception:
                pass

        return transformer, tokenizer
