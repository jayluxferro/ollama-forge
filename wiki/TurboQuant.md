# TurboQuant

Extreme model compression using near-optimal vector quantization. Produces fast inference models at 2-4 bits per weight — no llama.cpp required.

Based on [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., Google Research, 2025).

---

## Quick Start

```bash
# Install dependencies
uv sync

# Quantize a Hugging Face model to 3-bit
ollama-forge turboquant quantize meta-llama/Llama-3.1-8B-Instruct --bits 3

# Chat with the quantized model
ollama-forge turboquant chat Llama-3.1-8B-Instruct.tqf

# Or serve it via OpenAI-compatible API
ollama-forge turboquant serve Llama-3.1-8B-Instruct.tqf --port 8811
curl http://localhost:8811/v1/chat/completions \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## How It Works

TurboQuant uses a two-stage algorithm to compress high-dimensional weight vectors:

### Stage 1: MSE-Optimal Quantization (Algorithm 1)

1. **Random rotation**: Multiply weight rows by a random orthogonal matrix (QR decomposition of Gaussian). This decorrelates coordinates.
2. **Beta distribution**: After rotation, each coordinate follows a known Beta distribution (converges to Gaussian in high dimensions). This is key — the distribution is known *without* looking at the data.
3. **Optimal scalar quantizer**: Apply precomputed Lloyd-Max centroids for this distribution to each coordinate independently. For b bits, we get 2^b centroids that minimize MSE.
4. **Pack and store**: Store b-bit centroid indices plus per-row norms.

### Stage 2: QJL Residual Correction (Algorithm 2, optional)

MSE-optimal quantizers introduce bias in inner-product estimation. To correct this:

1. Compute the residual: `r = x - DeQuant(Quant(x))`
2. Apply 1-bit Quantized Johnson-Lindenstrauss: `sign(S · r)` where S is a random Gaussian matrix
3. This yields an **unbiased** inner-product estimator at the cost of 1 extra bit per coordinate

### Outlier Channel Handling

Per the paper's Section 4.3, channels with unusually high magnitude ("outliers") are quantized at higher precision. Default: 32 outlier channels at 4-bit, remaining at target bits. This gives mixed precision, e.g. 2.5 effective bits with `--bits 2 --outlier-bits 3`.

---

## Commands

### `turboquant quantize`

Quantize a HF model to `.tqf` format.

```bash
ollama-forge turboquant quantize <model> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bits` | 3 | Bits per weight (1, 2, 3, or 4) |
| `--outlier-channels` | 32 | Number of outlier channels |
| `--outlier-bits` | 4 | Bits for outlier channels |
| `--embed-bits` | 4 | Bits for embedding layer |
| `--kv-bits` | 3 | Bits for KV cache at inference |
| `--qjl` | off | Enable QJL residual correction |
| `--device` | auto | Device: auto, cuda, mps, cpu |
| `-o, --output` | `<model>.tqf` | Output directory |

`<model>` can be a HF repo ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`) or a local path to a safetensors checkpoint.

### `turboquant serve`

Serve a `.tqf` model via OpenAI-compatible API.

```bash
ollama-forge turboquant serve <model.tqf> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8811 | Bind port |
| `--device` | auto | Compute device |
| `--dtype` | float16 | Compute dtype (float16, bfloat16, float32) |
| `--name` | directory name | Model name for `/v1/models` |

**API endpoints:**
- `POST /v1/chat/completions` — Chat completion (streaming supported)
- `POST /v1/completions` — Text completion
- `GET /v1/models` — List loaded models
- `GET /health` — Health check

### `turboquant chat`

Interactive chat with a `.tqf` model (no server needed).

```bash
ollama-forge turboquant chat <model.tqf> [--temperature 0.7] [--system "You are helpful."]
```

### `turboquant info`

Show compression statistics for a `.tqf` model.

```bash
ollama-forge turboquant info <model.tqf> [--json]
```

---

## .tqf Format

TurboQuant models are stored as a directory:

```
model.tqf/
  metadata.json            # Model config, quant params, compression stats
  unquantized.safetensors  # Norms, biases, layer norms (kept in fp16)
  quantized_0000.safetensors  # Packed indices, codebooks, rotation seeds
  quantized_0001.safetensors  # (split into chunks if >500MB)
  tokenizer.json           # Tokenizer files (copied from HF checkpoint)
  tokenizer_config.json
```

**metadata.json** contains:
- `format`: always `"turboquant"`
- `version`: format version (currently 1)
- `model_config`: original HF config.json
- `quant_config`: quantization parameters used
- `quantized_layers`: per-layer metadata (shape, bits, seeds)
- `stats`: compression statistics

---

## Performance Backends

TurboQuant auto-selects the fastest backend for your hardware:

| Backend | Hardware | How | Speedup |
|---------|----------|-----|---------|
| **MLX** | Apple Silicon (M1-M4) | Native MLX arrays, unified memory, no data transfers | ~2-3× vs MPS |
| **Triton** | NVIDIA GPUs | JIT-compiled fused CUDA kernels for dequant+matmul | ~2-5× vs PyTorch |
| **PyTorch** | Any (CPU/MPS/CUDA) | Pure PyTorch with `torch.compile` on CUDA | Baseline |

Install the right extras for your hardware:

```bash
uv sync
```

Backend selection is automatic (`auto`), or force with `--device`:
```bash
ollama-forge turboquant serve model.tqf --device mlx     # force MLX
ollama-forge turboquant serve model.tqf --device cuda    # force CUDA+Triton
ollama-forge turboquant serve model.tqf --device cpu     # force CPU
```

### Optimizations

- **Structured rotation (Hadamard)**: For weight dimensions > 1024, uses Walsh-Hadamard transform instead of full matrix rotation — O(d log d) vs O(d²)
- **Batched KV cache**: All KV cache quantize/dequantize operations are batched and stay on-device (no CPU round-trips)
- **torch.compile**: On CUDA, attention and FFN blocks are compiled with `reduce-overhead` mode for kernel fusion
- **Weight caching**: Dequantized weights are cached after first access

---

## KV Cache Compression

At inference time, TurboQuant compresses the KV cache using the same algorithm. This is configured via `--kv-bits` during quantization (stored in metadata) and applied automatically during serving.

### Asymmetric K/V Compression

K and V caches serve different roles in attention and benefit from different compression strategies:

- **K cache** uses `TurboQuant` (Algorithm 2: PolarQuant + QJL) — preserves inner products for accurate attention score computation (`Q @ K^T`)
- **V cache** uses `TurboQuantMSE` (Algorithm 1: PolarQuant only) — preserves MSE for accurate value reconstruction (`attn_weights @ V`)

This asymmetric approach is applied automatically. K precision matters more than V precision because softmax amplifies small errors in `Q*K` scores exponentially, while V errors scale linearly.

### Layer-Adaptive KV Cache

Not all transformer layers need the same KV cache precision. Validated findings show the last ~20% of layers account for nearly all of turbo's quality loss. Layer-adaptive mode protects those critical layers.

**Modes** (set via `TURBO_LAYER_ADAPTIVE` environment variable):

| Mode | Strategy | Use case |
|------|----------|----------|
| `0` | Uniform — all layers at `kv_bits` | Default, simplest |
| `2` | Last N layers at full precision, rest compressed | Best quality/compression tradeoff |
| `7` | Boundary V — first 2 + last 2 layers at full precision | Minimal overhead boundary protection |

```bash
# Mode 2: protect last 8 layers at full precision, rest at 3-bit
TURBO_LAYER_ADAPTIVE=2 ollama-forge turboquant serve model.tqf

# Mode 7: boundary protection (first 2 + last 2 layers)
TURBO_LAYER_ADAPTIVE=7 ollama-forge turboquant serve model.tqf
```

**Validated results (Mode 2, q8_0 last 8 of 40 layers):**

| Metric | Uniform turbo3 | Mode 2 | vs q8_0 baseline |
|--------|---------------|--------|-----------------|
| PPL (8-chunk) | 6.211 (+1.6%) | 6.120 (+0.14%) | ~100% quality recovery |
| PPL (32-chunk) | 5.471 (+1.0%) | 5.435 (+0.37%) | ~60% quality recovery |
| Effective compression | 4.6x | ~3.5x | Trades 25% compression for ~100% quality |

### Temporal Decay

Old KV cache tokens are progressively requantized to fewer bits, saving memory while keeping recent tokens at full precision. This is especially effective at long contexts where most tokens are "old."

**How it works:**
1. Every 64 decode steps, identify old tokens (outside the recent window)
2. Requantize them from source bits to target bits (e.g., 3-bit → 2-bit)
3. Process in batches of 64 to eliminate GPU transfer overhead
4. Exempt attention sinks (positions 0-3) — these are critical for attention stability
5. Respect layer-adaptive policy — protected layers don't decay

```bash
# Temporal decay is enabled automatically when layer-adaptive mode is active.
# It can also be configured programmatically:
```

```python
from ollama_forge.turboquant import TemporalDecayManager

decay = TemporalDecayManager(
    d=128,              # head dimension
    source_bits=3,      # current precision
    target_bits=2,      # decay target
    decay_interval=64,  # steps between decay passes
    batch_size=64,      # positions per pass
    sink_len=4,         # attention sinks to exempt
)
```

**Memory savings by context length:**

| Context | Without decay | With decay | Savings |
|---------|--------------|------------|---------|
| 32K | 17.5 MB | 12.3 MB | ~30% |
| 64K | 35.0 MB | 23.1 MB | ~34% |
| 128K | 70.0 MB | 46.2 MB | ~34% |
| 256K | 140.0 MB | 92.4 MB | ~34% |

Savings increase with context because the fraction of "old" tokens grows.

**Quality:** Validated cosine similarity > 0.94 on real Qwen3 KV tensors. NIAH retrieval preserved (same scores with and without decay).

### Summary of KV Cache Features

| Feature | What it does | Quality impact |
|---------|-------------|----------------|
| Asymmetric K/V | Different algorithms for K (inner product) vs V (MSE) | Automatic, no user action |
| Layer-Adaptive | Critical layers keep full precision | PPL +0.14% vs +1.6% uniform |
| Temporal Decay | Old tokens progressively compressed | 30-34% memory savings at zero decode cost |

With 3-bit KV cache:
- ~5x reduction in KV cache memory
- Layer-adaptive brings quality within 0.14% of uncompressed
- Temporal decay adds 30-34% additional savings at long contexts
- Negligible quality impact (validated NIAH scores at 4x compression)

---

## Comparison with GGUF Quantization

| Feature | GGUF (llama.cpp) | TurboQuant |
|---------|-------------------|------------|
| **Dependencies** | llama.cpp binary | Pure Python/PyTorch |
| **Quantization** | Block-wise k-quant | Rotation + optimal scalar |
| **Bit widths** | Q2_K to Q8_0 | 1-4 bits, mixed precision |
| **KV cache** | Separate config | Integrated, online |
| **Theoretical** | Empirical | Near-optimal (≈2.7× Shannon bound) |
| **Calibration** | Some types need data | Data-oblivious (online) |
| **Speed** | C/C++ kernels | PyTorch (GPU-accelerated) |
| **Serving** | Ollama/llama-server | Built-in OpenAI-compatible API |

**When to use TurboQuant:**
- You want extreme compression (2-3 bits) with theoretical quality guarantees
- You don't have llama.cpp set up
- You need online KV cache compression for long contexts
- You want a self-contained Python solution

**When to use GGUF:**
- Maximum inference speed (optimized C/C++ kernels)
- Integration with Ollama ecosystem
- Wider model/quantization type support
- CPU inference

---

## Python API

TurboQuant's core classes can be used directly for custom compression workflows.

### Core Quantizers

```python
import torch
from ollama_forge.turboquant import (
    PolarQuant,       # MSE-optimal scalar quantizer (Algorithm 1)
    QJLQuantizer,     # 1-bit Johnson-Lindenstrauss for residuals
    TurboQuant,       # Full Algorithm 2: PolarQuant(b-1) + QJL(1)
    TurboQuantMSE,    # MSE-only, no QJL (for V cache)
)

# K cache: inner product preservation
k_quantizer = TurboQuant(d=128, bit_width=3, seed=42)
compressed = k_quantizer.quantize(k_vectors)  # (batch, 128)
k_hat = k_quantizer.dequantize(compressed)

# V cache: MSE preservation
v_quantizer = TurboQuantMSE(d=128, bit_width=3, seed=542)
indices, norms = v_quantizer.quantize(v_vectors)
v_hat = v_quantizer.dequantize(indices, norms)
```

### KV Cache Compression

```python
from ollama_forge.turboquant import KVCacheCompressor

compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)

# Compress: (num_layers, num_heads, seq_len, head_dim)
compressed = compressor.compress(k_cache, v_cache)
k_hat, v_hat = compressor.decompress(compressed)

# Memory stats
stats = compressor.memory_stats(seq_len=4096, num_layers=32, num_heads=32)
print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
```

### Layer-Adaptive Compression

```python
from ollama_forge.turboquant import (
    LayerAdaptivePolicy,
    LayerAdaptiveKVCacheCompressor,
)

# Mode 2: protect last 8 layers
policy = LayerAdaptivePolicy(
    num_layers=40, mode=2, base_bits=3,
    protected_bits=0, n_protected=8,
)

# Per-layer bit-width query
for layer in range(40):
    bits = policy.kv_bits(layer)
    # layers 0-31: 3 bits, layers 32-39: 0 (uncompressed)

# Full compressor with layer-adaptive policy
compressor = LayerAdaptiveKVCacheCompressor(
    head_dim=128, num_layers=40, base_bits=3,
    mode=2, n_protected=8,
)
compressed = compressor.compress(k_cache, v_cache)
k_hat, v_hat = compressor.decompress(compressed)
# Protected layers are lossless; compressed layers use turbo
```

### Temporal Decay

```python
from ollama_forge.turboquant import TemporalDecayManager, PolarQuant

decay = TemporalDecayManager(
    d=128,
    source_bits=3,       # current precision
    target_bits=2,       # decay target
    decay_interval=64,   # every 64 decode steps
    batch_size=64,       # positions per pass
    sink_len=4,          # exempt attention sinks
)

# During generation, call every decode step:
new_indices, new_norms, did_decay = decay.maybe_decay(
    v_indices, v_norms,
    total_seq_len=current_seq_len,
    recent_window=128,  # keep last 128 tokens at full precision
)

# Track progress
print(f"Decayed: {decay.n_decayed} positions")
print(f"Memory savings: {1 - decay.memory_savings_ratio(total_positions):.0%}")

# Reset for new sequence
decay.reset()
```

### Outlier-Aware Quantization

```python
from ollama_forge.turboquant import OutlierTurboQuant

# Fractional bit-widths via outlier channel strategy
oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
compressed = oq.quantize(x)
print(f"Effective bits: {oq.effective_bits:.1f}")
print(f"Compression ratio: {oq.compression_ratio():.1f}x")
```

---

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `TURBO_LAYER_ADAPTIVE` | `0`, `2`, `7` | Layer-adaptive mode (0=uniform, 2=protect last N, 7=boundary) |
