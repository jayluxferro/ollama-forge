# VLM (Vision Language Models)

Multimodal (image + audio + video + text) inference, conversion, and fine-tuning on Apple Silicon via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) >=0.4.4. Supports 55+ model architectures including Qwen2-VL, Gemma 4, LLaVA, Phi-3/4 Vision, Pixtral, and more.

> **Requires Apple Silicon** (M1/M2/M3/M4). mlx-vlm is automatically installed on macOS arm64. On other platforms the `vlm` commands print an install hint and exit gracefully.

---

## What's new in 0.4.4

- **TurboQuant KV cache** — 76% memory reduction via random rotation + codebook quantization. Use `--kv-bits 3.5 --kv-quant-scheme turboquant` on generate, chat, or serve.
- **VisionFeatureCache** — 11x+ speedup in multi-turn conversations by caching encoded image features. Use `--vision-cache-size 20` with `vlm chat`.
- **Video generation** — New `vlm video-generate` command for video understanding.
- **ORPO training** — Preference-based fine-tuning with `--train-mode orpo`.
- **Quantization modes** — `mxfp4`, `nvfp4`, `mxfp8` quantization in addition to `affine`.
- **Activation quantization** — `--quantize-activations` for mxfp8 models.
- **Thinking/reasoning** — Full control with `--enable-thinking`, `--thinking-budget`, `--thinking-start-token`, `--thinking-end-token`.

---

## Quick start

```bash
# Generate text from an image
ollama-forge vlm generate \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --prompt "Describe this image in detail" \
  --image photo.jpg

# Interactive multimodal chat (with vision cache for fast follow-ups)
ollama-forge vlm chat \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --vision-cache-size 20

# Start an OpenAI-compatible server with TurboQuant KV cache
ollama-forge vlm serve \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --port 11434 \
  --kv-bits 3.5 --kv-quant-scheme turboquant

# Describe a video
ollama-forge vlm video-generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --video clip.mp4 \
  --prompt "What happens in this video?"
```

---

## Commands

| Command | Description |
|---------|-------------|
| `vlm generate` | One-shot generation from text + images/audio |
| `vlm chat` | Interactive multimodal chat with VisionFeatureCache support |
| `vlm chat-ui` | Gradio web UI with image drag-and-drop |
| `vlm serve` | OpenAI-compatible HTTP server (`/v1/chat/completions` + `/v1/responses`) |
| `vlm video-generate` | Generate text from video input |
| `vlm convert` | Convert a HuggingFace VLM to MLX format |
| `vlm quantize` | Quantize a HuggingFace VLM to MLX (convenience wrapper) |
| `vlm finetune` | Fine-tune a VLM with LoRA/QLoRA/full/ORPO via mlx-vlm |

---

## Generate

Generate text from multimodal input (images, audio, text).

```bash
ollama-forge vlm generate \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --prompt "What objects are in this image?" \
  --image photo1.jpg --image photo2.jpg \
  --max-tokens 512 \
  --temperature 0.0 \
  --verbose
```

**With TurboQuant KV cache (76% memory savings):**

```bash
ollama-forge vlm generate \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --prompt "Describe this scene" \
  --image large_image.jpg \
  --kv-bits 3.5 --kv-quant-scheme turboquant \
  --max-tokens 1024
```

**With thinking/reasoning mode:**

```bash
ollama-forge vlm generate \
  --model mlx-community/QwQ-32B-4bit \
  --prompt "Solve this math problem in the image" \
  --image problem.png \
  --enable-thinking --thinking-budget 4096
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--prompt` | *(required)* | Text prompt |
| `--image` | | Image path or URL (repeatable) |
| `--audio` | | Audio file path (repeatable) |
| `--resize-shape` | | Resize images to this shape (e.g. `224 224`) |
| `--system` | | System message |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.0 | Sampling temperature |
| `--top-p` | | Nucleus sampling threshold |
| **KV cache** | | |
| `--kv-bits` | | KV cache quantization bits (float, e.g. `4`, `3.5`) |
| `--kv-quant-scheme` | | `uniform` or `turboquant` (fractional bits auto-select turboquant) |
| `--kv-group-size` | | Group size for uniform KV quantization |
| `--quantized-kv-start` | | Start token index for quantized KV cache |
| `--max-kv-size` | | Maximum KV cache size in tokens |
| `--prefill-step-size` | | Tokens per prefill step (lower = less peak memory) |
| **Thinking** | | |
| `--enable-thinking` | false | Enable thinking/reasoning mode |
| `--thinking-budget` | | Max thinking tokens |
| `--thinking-start-token` | `<think>` | Token marking start of thinking block |
| `--thinking-end-token` | `</think>` | Token marking end of thinking block |
| **Model** | | |
| `--adapter-path` | | LoRA adapter path |
| `--trust-remote-code` | false | Trust remote code when loading from HF Hub |
| `--quantize-activations` | false | Enable activation quantization for mxfp8 models |
| `--verbose` | false | Show throughput and memory stats |

---

## Chat

Interactive multimodal chat session. Attach images and audio during conversation.

```bash
ollama-forge vlm chat \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --system "You are a helpful visual assistant." \
  --temperature 0.7 \
  --vision-cache-size 20
```

**In-session commands:**
- `/image <path>` — attach an image to the next message
- `/audio <path>` — attach audio to the next message
- `/clear` — reset history, attachments, and vision cache
- `/quit` or `/exit` — end the session

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--max-tokens` | 512 | Maximum tokens per response |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | | Nucleus sampling threshold |
| `--system` | | System prompt |
| **KV cache** | | |
| `--kv-bits` | | KV cache quantization bits (float, e.g. `3.5`) |
| `--kv-quant-scheme` | | `uniform` or `turboquant` |
| `--kv-group-size` | | Group size for uniform KV quantization |
| `--quantized-kv-start` | | Start token index for quantized KV cache |
| `--max-kv-size` | | Maximum KV cache size in tokens |
| `--prefill-step-size` | | Tokens per prefill step |
| **Thinking** | | |
| `--enable-thinking` | false | Enable thinking/reasoning mode |
| `--thinking-budget` | | Max thinking tokens |
| **Vision cache** | | |
| `--vision-cache-size` | 0 (disabled) | Max cached image features (e.g. `20` for 11x+ speedup) |
| **Model** | | |
| `--adapter-path` | | LoRA adapter path |
| `--trust-remote-code` | false | Trust remote code from HF Hub |

### VisionFeatureCache

When `--vision-cache-size` is set to a value > 0, the chat session caches encoded image features in an LRU cache. This avoids re-encoding the same image through the vision encoder on every turn, giving **11x+ speedup** in multi-turn conversations that reference the same images.

The cache is keyed by file path (or content hash for in-memory images). Use `/clear` in the chat session to flush the cache along with conversation history.

---

## Chat UI

Launch a Gradio web interface for visual VLM chat. Supports image drag-and-drop, adjustable generation settings (temperature, top-p, max tokens), system prompts, and dark/light theme toggle.

```bash
ollama-forge vlm chat-ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

The browser opens automatically. Press Ctrl+C to stop the server.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |

---

## Serve

Start an OpenAI-compatible HTTP server for VLM inference. Exposes `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/health`, and `/unload` endpoints.

```bash
ollama-forge vlm serve \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 11434
```

**With TurboQuant KV cache for longer contexts:**

```bash
ollama-forge vlm serve \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --kv-bits 3.5 --kv-quant-scheme turboquant \
  --prefill-step-size 512 \
  --port 11434
```

Compatible with any OpenAI client library. Dynamic model loading is supported via the `/v1/chat/completions` endpoint — send a different model name in the request body.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 11434 | Port |
| **KV cache** | | |
| `--kv-bits` | | KV cache quantization bits (float, e.g. `3.5`) |
| `--kv-quant-scheme` | | `uniform` or `turboquant` |
| `--kv-group-size` | | Group size for uniform KV quantization |
| `--quantized-kv-start` | | Start token index for quantized KV cache |
| `--max-kv-size` | | Maximum KV cache size in tokens |
| `--prefill-step-size` | | Tokens per prefill step (try 512 or 256 for large models) |
| **Model** | | |
| `--adapter-path` | | LoRA adapter path |
| `--trust-remote-code` | false | Trust remote code from HF Hub |
| `--reload` | false | Auto-reload on file changes (dev only) |

**Testing the server:**

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]}
    ],
    "max_tokens": 256
  }'
```

---

## Video Generate

Generate text from video input. Wraps `mlx_vlm.video_generate`.

```bash
ollama-forge vlm video-generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --video recording.mp4 \
  --prompt "Summarize what happens in this video" \
  --max-tokens 200 \
  --fps 2.0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--video` | *(required)* | Path to video file (repeatable) |
| `--prompt` | `Describe this video.` | Text prompt |
| `--system` | | System prompt |
| `--max-tokens` | 100 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-pixels` | | Maximum pixel dimensions (width height, e.g. `224 224`) |
| `--max-frames` | | Maximum number of frames to extract |
| `--fps` | 1.0 | Frames per second for extraction |
| `--verbose` | true | Print verbose output |

---

## Convert

Convert a HuggingFace VLM to MLX format for faster local inference.

```bash
ollama-forge vlm convert \
  --hf-path Qwen/Qwen2-VL-2B-Instruct \
  --mlx-path ./my_mlx_model \
  --quantize --q-bits 4 --q-mode affine
```

**Quantization modes:**

| Mode | Description |
|------|-------------|
| `affine` | Default affine quantization |
| `mxfp4` | Microsoft MXFP 4-bit format |
| `nvfp4` | NVIDIA FP4 format |
| `mxfp8` | Microsoft MXFP 8-bit format |

**Dequantize back to full precision:**

```bash
ollama-forge vlm convert \
  --hf-path ./quantized_model \
  --mlx-path ./full_model \
  --dequantize
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-path` | *(required)* | HuggingFace repo ID or local path |
| `--mlx-path` | `mlx_model` | Output directory |
| `--quantize` | false | Quantize during conversion |
| `--q-bits` | 4 | Quantization bits |
| `--q-group-size` | 64 | Quantization group size |
| `--q-mode` | `affine` | Quantization mode: `affine`, `mxfp4`, `nvfp4`, `mxfp8` |
| `--dtype` | | Output dtype (e.g. `float16`) |
| `--upload-repo` | | Upload converted model to HuggingFace |
| `--revision` | | HuggingFace revision (branch/tag/commit) |
| `--dequantize` | false | Dequantize a quantized model |
| `--trust-remote-code` | false | Trust remote code from HF Hub |
| `--quant-predicate` | | Mixed-bit quantization recipe string |

### Metal GPU timeouts during conversion

Conversion of multimodal HF checkpoints (notably Gemma-3n / 4 E2B and
abliterated variants) can abort with:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Caused GPU Timeout Error
(00000002:kIOGPUCommandBufferCallbackErrorTimeout)
```

This is PyTorch's MPS allocator contending with MLX on the same Metal
device. PyTorch is only used as a weight loader during conversion, but its
default high-watermark ratio lets the allocator issue `waitUntilCompleted`
syncs that block on MLX's in-flight command buffers long enough to trip
Apple's GPU watchdog. Models whose processors touch torch (Gemma-3n's audio
tower mel filterbank is the usual trigger) hit this; text-only conversions
don't.

`ollama-forge` already sets the fix at CLI import time in `cli.py`:

```python
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
```

If you invoke `ollama_forge.vlm.vlm_convert()` directly from a long-running
Python process that has already touched MPS, set those env vars **before**
importing torch — they're read lazily on first MPS use and cannot be
changed afterward. The nuclear alternative is `PYTORCH_MPS_DISABLE=1` or
`torch.set_default_device("cpu")`, which forces torch fully onto CPU for
the conversion.

---

## Quantize

Convenience wrapper around `vlm convert` with `--quantize` enabled.

```bash
ollama-forge vlm quantize \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --bits 4 \
  --q-mode mxfp4 \
  --output ./quantized_model
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--output` | `mlx_model_quantized` | Output directory |
| `--bits` | 4 | Quantization bits |
| `--group-size` | 64 | Group size |
| `--q-mode` | `affine` | Quantization mode: `affine`, `mxfp4`, `nvfp4`, `mxfp8` |
| `--dtype` | | Output dtype |
| `--upload-repo` | | Upload to HuggingFace |
| `--trust-remote-code` | false | Trust remote code from HF Hub |

---

## Fine-tune

Fine-tune a VLM using LoRA, QLoRA, full fine-tuning, or ORPO via `mlx_vlm.lora`.

```bash
ollama-forge vlm finetune \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --dataset ./my_dataset \
  --output-path ./my_adapter \
  --epochs 3 \
  --lora-rank 8
```

**ORPO (preference-based training):**

```bash
ollama-forge vlm finetune \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --dataset ./preference_data \
  --train-mode orpo \
  --beta 0.1 \
  --epochs 2
```

**Full fine-tune with gradient checkpointing:**

```bash
ollama-forge vlm finetune \
  --model mlx-community/Qwen2-VL-2B-Instruct-bf16 \
  --dataset ./my_dataset \
  --full-finetune \
  --grad-checkpoint \
  --train-vision \
  --max-seq-length 4096
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--dataset` | *(required)* | Dataset path (JSONL) or HF dataset name |
| `--output-path` | `vlm_adapter` | Output adapter directory |
| **Training** | | |
| `--learning-rate` | 2e-5 | Learning rate |
| `--batch-size` | 4 | Training batch size |
| `--epochs` | | Number of epochs (overrides `--iters` if set) |
| `--iters` | 1000 | Training iterations (used when `--epochs` not set) |
| `--max-seq-length` | 2048 | Maximum sequence length |
| `--gradient-accumulation-steps` | 1 | Gradient accumulation steps |
| `--grad-checkpoint` | false | Enable gradient checkpointing (saves memory) |
| `--grad-clip` | | Gradient clipping max norm |
| **LoRA** | | |
| `--lora-rank` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--lora-dropout` | 0.0 | LoRA dropout |
| **Training mode** | | |
| `--train-vision` | false | Also train the vision encoder |
| `--full-finetune` | false | Full fine-tune instead of LoRA |
| `--train-on-completions` | false | Only train on completion tokens |
| `--train-mode` | `sft` | Training mode: `sft` or `orpo` |
| **ORPO** | | |
| `--beta` | 0.1 | ORPO odds-ratio weight |
| `--eps` | 1e-8 | ORPO numerical stability epsilon |
| **Dataset** | | |
| `--split` | `train` | Dataset split |
| `--dataset-config` | | HF dataset config name |
| `--image-resize-shape` | | Resize images to (width height) |
| `--custom-prompt-format` | | Custom JSON prompt template |
| `--assistant-id` | 77091 | Assistant token ID for completions training |
| **Reporting** | | |
| `--steps-per-report` | 10 | Steps between training reports |
| `--steps-per-eval` | 200 | Steps between evaluations |
| `--steps-per-save` | 100 | Steps between adapter saves |
| `--val-batches` | 25 | Validation batches |
| **Resume** | | |
| `--adapter-path` | | Resume from an existing adapter |

Use `--adapter-path` with `vlm generate`, `vlm chat`, or `vlm serve` to load a fine-tuned adapter for inference.

---

## TurboQuant KV cache

TurboQuant KV cache quantization reduces memory usage by ~76% with minimal quality loss, enabling much longer context processing. It uses random rotation and codebook quantization, optimized with Metal kernels on Apple Silicon.

**How to use:**

Add `--kv-bits` and `--kv-quant-scheme turboquant` to any inference command:

```bash
# Generate with TurboQuant KV
ollama-forge vlm generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --prompt "Analyze this document" --image doc.png \
  --kv-bits 3.5 --kv-quant-scheme turboquant

# Serve with TurboQuant KV
ollama-forge vlm serve \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --kv-bits 3.5 --kv-quant-scheme turboquant

# Chat with TurboQuant KV
ollama-forge vlm chat \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --kv-bits 3.5 --kv-quant-scheme turboquant
```

Fractional `--kv-bits` values (e.g. 3.5) automatically select the TurboQuant scheme. Integer values default to `uniform` unless `--kv-quant-scheme turboquant` is explicitly set.

Additional control:
- `--kv-group-size` — group size for uniform quantization
- `--quantized-kv-start` — skip quantizing the first N tokens (for important prompt context)
- `--max-kv-size` — cap KV cache at N tokens
- `--prefill-step-size` — reduce peak memory during prefill (try 512 or 256)

---

## Security eval integration

The [Security Eval](Security-Eval) UI supports local VLM evaluation. Enable the **"Use local VLM"** toggle in the Quick test or Run tabs, provide a model path, and the eval pipeline queries the VLM directly — no external server needed.

---

## Supported models

mlx-vlm supports 55+ architectures. Popular models available as pre-quantized MLX weights on HuggingFace:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`
- `mlx-community/Phi-3.5-vision-instruct-4bit`
- `mlx-community/Pixtral-12B-2409-4bit`
- `mlx-community/deepseek-vl2-small-4bit`

Browse the full list at [mlx-community on HuggingFace](https://huggingface.co/mlx-community?search=vlm).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mlx-vlm is required for VLM commands` | Install: `pip install 'mlx-vlm>=0.4.4'`. Only works on Apple Silicon. |
| Model download slow | Set `HF_HUB_ENABLE_HF_TRANSFER=1` (auto-enabled by ollama-forge) for faster downloads. |
| Out of memory | Use a smaller quantized model (4-bit), reduce `--max-tokens`, use `--kv-bits 3.5`, or try `--prefill-step-size 256`. |
| Adapter not loading | Ensure `--adapter-path` points to a directory with `adapter_config.json` and weight files. |
| TurboQuant not activating | Use fractional `--kv-bits` (e.g. `3.5`) or explicitly add `--kv-quant-scheme turboquant`. |
| Video generation fails | Ensure the model supports video input (e.g. Qwen2.5-VL). Check `--max-frames` and `--fps` settings. |
| `kIOGPUCommandBufferCallbackErrorTimeout` during `vlm convert` | PyTorch's MPS allocator is contending with MLX. The CLI sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and `PYTORCH_ENABLE_MPS_FALLBACK=0` at import time to prevent this — see [Metal GPU timeouts during conversion](#metal-gpu-timeouts-during-conversion). |
