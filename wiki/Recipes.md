# Recipes

Build an Ollama model from a **single YAML or JSON file** with one command: `ollama-forge build recipe.yaml`. To check a recipe without building, use `ollama-forge build recipe.yaml --validate-only`.

---

## Recipe format

- **Required:** `name` (Ollama model name), and **exactly one of:**
  - **base** — Existing Ollama model or path → create-from-base.
  - **gguf** — Path to a local `.gguf` file → convert.
  - **hf_repo** — Hugging Face repo id → fetch (download GGUF and create model).
- **Optional (all sources):** `system`, `temperature`, `num_ctx`, `top_p`, `repeat_penalty`.
- **With base:** `adapter` (path to adapter directory).
- **With gguf:** `quantize` (e.g. `Q4_K_M`).
- **With hf_repo:** `gguf_file`, `quant`, `revision`.

---

## Example: from base (custom prompt/settings)

```yaml
name: my-assistant
base: llama3.2
system: You are a concise coding assistant.
temperature: 0.7
num_ctx: 4096
top_p: 0.9
repeat_penalty: 1.1
```

```bash
uv run ollama-forge build recipe.yaml
ollama run my-assistant
```

---

## Example: from Hugging Face

```yaml
name: my-model
hf_repo: TheBloke/Llama-2-7B-GGUF
quant: Q4_K_M
temperature: 0.6
num_ctx: 8192
```

---

## Example: from GGUF path

```yaml
name: my-converted
gguf: /path/to/model.gguf
quantize: Q4_K_M
temperature: 0.6
```

---

## Example: base + adapter

```yaml
name: my-finetuned
base: llama3.2
adapter: /path/to/adapter
system: You are a domain expert.
```

---

## Example recipes in repo

In this repo, see `examples/recipes/`:

- **simple-base.yaml** — From existing Ollama base (system, temperature).
- **from-gguf.yaml** — From a GGUF file (set `gguf` to your path).
- **from-hf.yaml** — From a Hugging Face repo (set `hf_repo`; optional `gguf_file`).

Build any of them:

```bash
uv run ollama-forge build examples/recipes/from-hf.yaml
ollama run <name-from-recipe>
```
