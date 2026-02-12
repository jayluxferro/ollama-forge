# Fetch & Convert

Get a GGUF from Hugging Face and create an Ollama model in one command, or turn a local GGUF file into an Ollama model.

---

## fetch: Hugging Face → Ollama (one command)

Use when the Hugging Face repo **already has GGUF files** (e.g. TheBloke-style repos).

```bash
uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model
ollama run my-model
```

**Pick a specific quantization** when the repo has multiple GGUF files:

```bash
uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model --quant Q4_K_M
# or Q8_0, Q5_K_M, etc.
```

**Other options:**

- `--gguf-file <filename>` — Use a specific file when the repo has many.
- `--revision main` — Branch/tag/revision of the repo.
- `--system "..."`, `--temperature 0.7`, `--num-ctx 4096` — Modelfile options.

**Gated or private repos:** Set `HF_TOKEN` or run `huggingface-cli login`.

---

## convert: GGUF file → Ollama model

Use when you **already have a .gguf file** on disk (e.g. downloaded manually or produced by llama.cpp).

```bash
uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model
ollama run my-model
```

**Shrink first (quantize)** — Reduces size/VRAM; requires llama.cpp `quantize` (or `llama-quantize`) on PATH:

```bash
uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model --quantize Q4_K_M
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
uv run ollama-tools hf-cache ls

# Remove one or more repos (frees disk space)
uv run ollama-tools hf-cache rm TheBloke/Llama-2-7B-GGUF
uv run ollama-tools hf-cache rm repo1 repo2 --yes   # skip confirmation
uv run ollama-tools hf-cache rm repo1 --dry-run     # preview only
```
