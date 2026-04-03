# VLM (Vision Language Models)

Multimodal (image + audio + text) inference, conversion, and fine-tuning on Apple Silicon via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Supports 55+ model architectures including Qwen2-VL, LLaVA, Phi-3 Vision, Pixtral, and more.

> **Requires Apple Silicon** (M1/M2/M3/M4). mlx-vlm is automatically installed on macOS arm64. On other platforms the `vlm` commands print an install hint and exit gracefully.

---

## Quick start

```bash
# Generate text from an image
ollama-forge vlm generate \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --prompt "Describe this image in detail" \
  --image photo.jpg

# Interactive multimodal chat
ollama-forge vlm chat --model mlx-community/Qwen2-VL-2B-Instruct-4bit

# Start an OpenAI-compatible server
ollama-forge vlm serve --model mlx-community/Qwen2-VL-2B-Instruct-4bit --port 8080
```

---

## Commands

| Command | Description |
|---------|-------------|
| `vlm generate` | One-shot generation from text + images/audio |
| `vlm chat` | Interactive multimodal chat (attach images per turn) |
| `vlm serve` | OpenAI-compatible HTTP server (`/v1/chat/completions`) |
| `vlm convert` | Convert a HuggingFace VLM to MLX format |
| `vlm quantize` | Quantize a HuggingFace VLM to MLX (convenience wrapper) |
| `vlm finetune` | Fine-tune a VLM with LoRA/QLoRA/full via mlx-vlm |

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

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--prompt` | *(required)* | Text prompt |
| `--image` | | Image path or URL (repeatable) |
| `--audio` | | Audio file path |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.0 | Sampling temperature |
| `--top-p` | | Nucleus sampling threshold |
| `--kv-bits` | | KV cache quantization bits |
| `--enable-thinking` | false | Enable thinking/reasoning mode |
| `--thinking-budget` | | Max thinking tokens |
| `--adapter-path` | | LoRA adapter path |
| `--verbose` | false | Show throughput and memory stats |

---

## Chat

Interactive multimodal chat session. Attach images during conversation by typing an image path or URL.

```bash
ollama-forge vlm chat \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --system "You are a helpful visual assistant." \
  --temperature 0.7
```

**In-session commands:**
- Type an image path or URL to attach it to your next message
- `quit` or `exit` to end the session

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--max-tokens` | 512 | Maximum tokens per response |
| `--temperature` | 0.7 | Sampling temperature |
| `--system` | | System prompt |
| `--kv-bits` | | KV cache quantization bits |
| `--adapter-path` | | LoRA adapter path |

---

## Serve

Start an OpenAI-compatible HTTP server for VLM inference.

```bash
ollama-forge vlm serve \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8080
```

This wraps `mlx_vlm.server` and exposes `/v1/chat/completions`. Compatible with any OpenAI client library.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 8080 | Port |
| `--kv-bits` | | KV cache quantization bits |
| `--adapter-path` | | LoRA adapter path |

---

## Convert

Convert a HuggingFace VLM to MLX format for faster local inference.

```bash
ollama-forge vlm convert \
  --hf-path Qwen/Qwen2-VL-2B-Instruct \
  --mlx-path ./my_mlx_model \
  --quantize --q-bits 4
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-path` | *(required)* | HuggingFace repo ID or local path |
| `--mlx-path` | `mlx_model` | Output directory |
| `--quantize` | false | Quantize during conversion |
| `--q-bits` | 4 | Quantization bits |
| `--q-group-size` | 64 | Quantization group size |
| `--dtype` | | Output dtype (e.g. `float16`) |
| `--upload-repo` | | Upload converted model to HuggingFace |

---

## Quantize

Convenience wrapper around `vlm convert` with `--quantize` enabled.

```bash
ollama-forge vlm quantize \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --bits 4 \
  --output ./quantized_model
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--output` | `mlx_model_quantized` | Output directory |
| `--bits` | 4 | Quantization bits |
| `--group-size` | 64 | Group size |
| `--dtype` | | Output dtype |
| `--upload-repo` | | Upload to HuggingFace |

---

## Fine-tune

Fine-tune a VLM using LoRA, QLoRA, or full fine-tuning via `mlx_vlm.lora`.

```bash
ollama-forge vlm finetune \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --dataset ./my_dataset \
  --output-path ./my_adapter \
  --epochs 3 \
  --lora-rank 8
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace repo ID or local path |
| `--dataset` | *(required)* | Dataset directory (must exist) |
| `--output-path` | `vlm_adapter` | Output adapter directory |
| `--learning-rate` | 2e-5 | Learning rate |
| `--batch-size` | 4 | Training batch size |
| `--epochs` | 1 | Number of epochs |
| `--lora-rank` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--lora-dropout` | 0.0 | LoRA dropout |
| `--train-vision` | false | Also train the vision encoder |
| `--full-finetune` | false | Full fine-tune instead of LoRA |
| `--gradient-accumulation-steps` | 1 | Gradient accumulation steps |
| `--grad-checkpoint` | false | Enable gradient checkpointing |
| `--adapter-path` | | Resume from an existing adapter |

Use the `--adapter-path` flag with `vlm generate` or `vlm chat` to load a fine-tuned adapter for inference.

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
| `mlx-vlm is required for VLM commands` | Install: `pip install 'mlx-vlm>=0.4.3'`. Only works on Apple Silicon. |
| Model download slow | Set `HF_HUB_ENABLE_HF_TRANSFER=1` (auto-enabled by ollama-forge) for faster downloads. |
| Out of memory | Use a smaller quantized model (4-bit) or reduce `--max-tokens`. |
| Adapter not loading | Ensure `--adapter-path` points to a directory with `adapter_config.json` and weight files. |
