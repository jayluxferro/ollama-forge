# Fetch, Import & Convert

Get a model from Hugging Face and create an Ollama model in one command. Three paths depending on what you have:

---

## fetch: Hugging Face → Ollama (one command)

Use when the Hugging Face repo **already has GGUF files** (e.g. TheBloke-style repos).

```bash
uv run ollama-forge fetch TheBloke/Llama-2-7B-GGUF --name my-model
ollama run my-model
```

**Pick a specific quantization** when the repo has multiple GGUF files:

```bash
uv run ollama-forge fetch TheBloke/Llama-2-7B-GGUF --name my-model --quant Q4_K_M
# or Q8_0, Q5_K_M, etc.
```

**Other options:**

- `--gguf-file <filename>` — Use a specific file when the repo has many.
- `--revision main` — Branch/tag/revision of the repo.
- `--system "..."`, `--temperature 0.7`, `--num-ctx 4096`, `--top-p 0.9`, `--repeat-penalty 1.1` — Modelfile options.

**Gated or private repos:** Set `HF_TOKEN` or run `huggingface-cli login`.

---

## import: HF safetensors → GGUF → Ollama (one command)

Use when the Hugging Face repo has **safetensors/bin weights but no GGUF files** (e.g. `meta-llama/Llama-3.2-1B-Instruct`). Requires llama.cpp (`setup-llama-cpp`).

```bash
uv run ollama-forge import meta-llama/Llama-3.2-1B-Instruct --name llama3.2-1b
ollama run llama3.2-1b
```

**From a local checkpoint directory:**

```bash
uv run ollama-forge import ./my-model-checkpoint --name my-model
```

**With options:**

```bash
uv run ollama-forge import Qwen/Qwen2.5-7B-Instruct --name qwen2.5 \
  --quant Q5_K_M --template-from qwen2.5:7b --system "You are helpful."
```

**Key options:**

- `--quant Q4_K_M` (default) — Quantization type. Use `--no-requantize` to keep full-size GGUF.
- `--outtype bf16` (default) — GGUF output precision (`f32`/`f16`/`bf16`/`q8_0`/`auto`).
- `--template-from <ollama_model>` — Copy chat template from an existing Ollama model (for tool support).
- `--output-dir <path>` — Where to download/save files (default: auto temp dir).
- `--revision main` — HF repo branch/tag.
- `--system "..."`, `--temperature`, `--num-ctx`, `--top-p`, `--repeat-penalty` — Modelfile parameters.
- `--llama-cpp-dir <path>` — Path to llama.cpp clone (auto-detected from `./llama.cpp` or `~/llama.cpp`).

**Gated or private repos:** Set `HF_TOKEN` or run `huggingface-cli login`.

---

## convert: GGUF file → Ollama model

Use when you **already have a .gguf file** on disk (e.g. downloaded manually or produced by llama.cpp).

```bash
uv run ollama-forge convert --gguf /path/to/model.gguf --name my-model
ollama run my-model
```

**Shrink first (quantize)** — Reduces size/VRAM; requires llama.cpp `quantize` (or `llama-quantize`) on PATH:

```bash
uv run ollama-forge convert --gguf /path/to/model.gguf --name my-model --quantize Q4_K_M
ollama run my-model
```

**Other options:**

- `--system "..."`, `--temperature`, `--num_ctx`, `--top_p`, `--repeat_penalty` — Modelfile parameters.

See [Quantization](Quantization) for quant types and trade-offs.

---

## Clean up Hugging Face cache

Models downloaded by `fetch` and `fetch-adapter` are stored in the Hugging Face Hub cache (e.g. `~/.cache/huggingface/hub`). To list and remove them:

```bash
# List cached repos and sizes
uv run ollama-forge hf-cache ls

# Remove one or more repos (frees disk space)
uv run ollama-forge hf-cache rm TheBloke/Llama-2-7B-GGUF
uv run ollama-forge hf-cache rm repo1 repo2 --yes   # skip confirmation
uv run ollama-forge hf-cache rm repo1 --dry-run     # preview only
```
