# Hugging Face without GGUF

Ollama runs **GGUF** models. If the Hugging Face repo only has PyTorch/Safetensors (no GGUF), you need to convert first.

---

## Steps

1. **Download the model** — e.g.:
   ```bash
   huggingface-cli download <org>/<model> --local-dir ./models/<model>
   ```
2. **Convert to GGUF** — Use [llama.cpp](https://github.com/ggerganov/llama.cpp). Example:
   ```bash
   python convert-hf-to-gguf.py ./models/<model> --outfile ./models/<model>.gguf
   ```
   Use the script that matches your architecture (Llama, Mistral, Qwen, etc.).
3. **(Optional) Quantize** — Reduce size/VRAM:
   ```bash
   ./quantize model.gguf model-Q4_K_M.gguf Q4_K_M
   ```
   (Requires llama.cpp built; see [Installation](Installation).)
4. **Create Ollama model** — Use ollama-tools:
   ```bash
   uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model
   ollama run my-model
   ```

---

## When GGUF is already on the Hub

If the repo **already has GGUF files**, use ollama-tools directly — no conversion step:

```bash
uv run ollama-tools fetch <repo_id> --name my-model
ollama run my-model
```

See [Fetch & Convert](Fetch-and-Convert).
