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

At inference time, TurboQuant can also compress the KV cache using the same algorithm. This is configured via `--kv-bits` during quantization (stored in metadata) and applied automatically during serving.

With 3-bit KV cache:
- ~5× reduction in KV cache memory
- Enables much longer context windows on the same hardware
- Negligible quality impact (per paper: identical Needle-in-a-Haystack scores at 4× compression)

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
