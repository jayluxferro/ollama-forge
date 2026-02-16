# Auto & Plan

**auto** — Detect the type of source (recipe, GGUF file, HF repo, base model, adapter) and run the right flow.  
**plan** — Preview major operations without executing them (dry-run).

---

## auto: one command for any source

Pass a path or identifier; the tool detects what it is and runs the appropriate command.

| Source | Detection | Action |
|--------|-----------|--------|
| Recipe file | `.yaml` / `.json` path | `build` |
| GGUF file | Path to `.gguf` | `convert` (optional `--quantize`) |
| Hugging Face repo id | e.g. `TheBloke/Llama-2-7B-GGUF` | `fetch` |
| Base model name | e.g. `llama3.2` (existing Ollama model) | `create-from-base` |
| Adapter directory | Path to dir with adapter files | `retrain` |
| Adapter HF repo | Repo with adapter files | `fetch-adapter` |

Examples:

```bash
# Recipe → build
uv run ollama-forge auto ./examples/recipes/from-hf.yaml

# GGUF file → convert
uv run ollama-forge auto /path/to/model.gguf --name my-model --quantize Q4_K_M

# HF repo → fetch
uv run ollama-forge auto TheBloke/Llama-2-7B-GGUF --name my-model --quant Q4_K_M

# Base model → create-from-base
uv run ollama-forge auto llama3.2 --name my-assistant --system "You are helpful."

# Adapter dir → retrain
uv run ollama-forge auto /path/to/adapter_dir --base llama3.2 --name my-finetuned

# Adapter HF repo → fetch-adapter
uv run ollama-forge auto user/my-lora-adapter --base llama3.2 --name my-finetuned
```

**Preview only (no execution):**

```bash
uv run ollama-forge auto TheBloke/Llama-2-7B-GGUF --plan
```

When **--name** (and in HF mode, **--quant**) is missing, **auto** prompts in interactive terminals with safe defaults. For non-interactive scripts use **--no-prompt** to avoid prompts and use defaults.

---

## plan: dry-run from one place

Preview what would happen for quickstart, auto, doctor-fix, or adapters-apply.

```bash
uv run ollama-forge plan quickstart --profile balanced --name my-model
uv run ollama-forge plan auto TheBloke/Llama-2-7B-GGUF --name my-model
uv run ollama-forge plan doctor-fix --fix-llama-cpp
uv run ollama-forge plan adapters-apply --base llama3.2 --query "llama lora adapter"
```

**JSON output** (for scripting/CI):

```bash
uv run ollama-forge plan auto TheBloke/Llama-2-7B-GGUF --name my-model --json
```

Plan does not run the underlying commands; it only shows the steps and, with **--json**, machine-readable output.
