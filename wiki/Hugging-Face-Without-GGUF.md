# Hugging Face without GGUF

Ollama runs **GGUF** models. If the Hugging Face repo only has PyTorch/Safetensors (no GGUF), use the `import` command — it downloads, converts, quantizes, and creates the Ollama model in one step.

---

## One command: `import`

Requires llama.cpp (run `ollama-forge setup-llama-cpp` first).

```bash
uv run ollama-forge import meta-llama/Llama-3.2-1B-Instruct --name llama3.2-1b
ollama run llama3.2-1b
```

**From a local checkpoint:**

```bash
uv run ollama-forge import ./my-model-checkpoint --name my-model
```

**With options:**

```bash
uv run ollama-forge import Qwen/Qwen2.5-7B-Instruct --name qwen2.5 \
  --quant Q5_K_M --template-from qwen2.5:7b
```

Key flags: `--quant` (default Q4_K_M), `--no-requantize`, `--outtype` (default bf16), `--template-from`, `--system`, `--temperature`, `--num-ctx`. Run `ollama-forge import --help` for all options.

---

## Manual steps (alternative)

If you prefer to do it step by step:

1. **Download the model** — e.g.:
   ```bash
   huggingface-cli download <org>/<model> --local-dir ./models/<model>
   ```
2. **Convert to GGUF** — Use [llama.cpp](https://github.com/ggerganov/llama.cpp). Example:
   ```bash
   python convert-hf-to-gguf.py ./models/<model> --outfile ./models/<model>.gguf
   ```
3. **(Optional) Quantize** — Reduce size/VRAM:
   ```bash
   ./quantize model.gguf model-Q4_K_M.gguf Q4_K_M
   ```
   (Requires llama.cpp built; see [Installation](Installation).)
4. **Create Ollama model** — Use ollama-forge:
   ```bash
   uv run ollama-forge convert --gguf /path/to/model.gguf --name my-model
   ollama run my-model
   ```

---

## When GGUF is already on the Hub

If the repo **already has GGUF files**, use ollama-forge directly — no conversion step:

```bash
uv run ollama-forge fetch <repo_id> --name my-model
ollama run my-model
```

See [Fetch & Convert](Fetch-and-Convert).
