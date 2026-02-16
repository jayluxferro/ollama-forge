# ollama-tools

Get models from Hugging Face, convert them, add adapters, and run them in [Ollama](https://ollama.com) — without needing deep expertise. One place for fetch, convert, adapters, and recipes.

**Quick start:**
```bash
uv sync && uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model && ollama run my-model
```

Or the shortest path: `uv run ollama-tools start --name my-model` then `ollama run my-model`.

---

## Documentation (Wiki)

Detailed guides live in the [**wiki/**](wiki/Home.md):

| Topic | Description |
|-------|--------------|
| [Installation](wiki/Installation.md) | Setup, `check`, `doctor`, `setup-llama-cpp` |
| [Quick Start](wiki/Quick-Start.md) | `start` / `quickstart`, profiles, task presets |
| [Auto & Plan](wiki/Auto-and-Plan.md) | Auto-detect source, dry-run planner |
| [Fetch & Convert](wiki/Fetch-and-Convert.md) | GGUF from HF, GGUF file → Ollama |
| [Recipes](wiki/Recipes.md) | One-file YAML/JSON build |
| [Modelfile](wiki/Modelfile.md) | Ollama Modelfile basics |
| [Adapters](wiki/Adapters.md) | LoRA: search, recommend, fetch-adapter, retrain |
| [Training Data](wiki/Training-Data.md) | JSONL validate, prepare, train script |
| [Retrain Pipeline](wiki/Retrain-Pipeline.md) | Data → adapter → Ollama |
| [Abliterate](wiki/Abliterate.md) | Refusal removal |
| [Downsizing](wiki/Downsizing.md) | Teacher → student distillation |
| [Hugging Face without GGUF](wiki/Hugging-Face-Without-GGUF.md) | When the repo has no GGUF |
| [Quantization](wiki/Quantization.md) | Smaller/faster GGUF (Q4_K_M, Q8_0, etc.) |
| [CI / Automation](wiki/CI-Automation.md) | Example GitHub Actions |
| [Command Reference](wiki/Command-Reference.md) | All commands |

---

## Why ollama-tools

- **One place** — Fetch, convert, adapters, recipes; no scattered scripts.
- **Simple** — Clear commands and docs; try things without being an ML expert.
- **Local-first** — Get models running in Ollama on your machine.

---

## Setup (one-time)

- **Python 3.10+** and **[uv](https://docs.astral.sh/uv/)** (or pip).
- **From repo:** `uv sync` then `uv run ollama-tools` (or `uv tool install .` to use `ollama-tools` from anywhere).
- **Ollama** — [Install](https://ollama.com) and ensure `ollama` is on your PATH.
- **Verify:** `uv run ollama-tools check` — see what’s installed. `uv run ollama-tools doctor` for diagnosis; `doctor --fix` to apply safe fixes. See [Installation](wiki/Installation.md) for optional llama.cpp (finetune/quantize).

---

## Commands at a glance

| What you want | Command |
|---------------|---------|
| Easiest one-command start | `start` or `quickstart [--name my-model]` |
| Auto-detect source and run | `auto <source> [--name my-model]` |
| Preview operations (dry-run) | `plan <quickstart\|auto\|doctor-fix\|adapters-apply> ...` |
| GGUF from HF → Ollama | `fetch <repo_id> --name <name>` |
| GGUF file → Ollama | `convert --gguf <path> --name <name>` |
| Find / use adapters | `adapters search`, `adapters recommend`, `fetch-adapter`, `retrain` |
| One-file config build | `build recipe.yaml` |
| Check / fix environment | `check`, `doctor [--fix]` |
| Install llama.cpp | `setup-llama-cpp` |

Full list: [Command Reference](wiki/Command-Reference.md). Run `ollama-tools --help` for options.

---

## Simplest workflows

**Beginner (one command):**
```bash
uv run ollama-tools start --name my-model
ollama run my-model
```
Uses default model + balanced profile. Use `--profile fast|balanced|quality|low-vram` and `--task chat|coding|creative`. See [Quick Start](wiki/Quick-Start.md).

**Auto (any source):** Recipe, GGUF path, HF repo, base model, or adapter — the tool detects and runs the right flow:
```bash
uv run ollama-tools auto ./recipe.yaml
uv run ollama-tools auto TheBloke/Llama-2-7B-GGUF --name my-model
uv run ollama-tools auto llama3.2 --name my-assistant --system "You are helpful."
```
See [Auto & Plan](wiki/Auto-and-Plan.md).

**Fetch from Hugging Face:** When the repo has GGUF files:
```bash
uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model
ollama run my-model
```
Use `--quant Q4_K_M` to pick size. See [Fetch & Convert](wiki/Fetch-and-Convert.md).

**Local GGUF:** `uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model`. Optional `--quantize Q4_K_M` (needs llama.cpp on PATH). See [Quantization](wiki/Quantization.md).

**Recipe (one file):** `uv run ollama-tools build recipe.yaml`. See [Recipes](wiki/Recipes.md) for format and examples.

**Adapters:** `adapters search "llama lora"`, then `fetch-adapter <repo> --base <base> --name <name>`, or `retrain --base <base> --adapter <path> --name <name>`. See [Adapters](wiki/Adapters.md).

**Training data → model:** Validate JSONL, prepare for trainer, generate script: `train --data ./data/ --base llama3.2 --name my-model --write-script train.sh`. See [Training Data](wiki/Training-Data.md) and [Retrain Pipeline](wiki/Retrain-Pipeline.md).

---

## Other topics

- **Hugging Face repo without GGUF** — Convert with llama.cpp first, then `convert`. [Wiki](wiki/Hugging-Face-Without-GGUF.md).
- **Refusal removal (abliterate)** — `abliterate compute-dir`; optional deps: `uv sync --extra abliterate`. [Wiki](wiki/Abliterate.md).
- **Downsizing (distillation)** — `downsize --teacher <hf> --student <hf> --name <name>`. [Wiki](wiki/Downsizing.md).
- **LLM security evaluation** — Run prompt sets against Ollama/serve, score refusal/compliance, get ASR and KPIs: `security-eval run <prompt_set>`. Optional UI: `uv sync --extra security-eval-ui` then `security-eval ui`. [docs/LLM-SECURITY-EVAL-REVIEW.md](docs/LLM-SECURITY-EVAL-REVIEW.md), [src/ollama_tools/security_eval/README.md](src/ollama_tools/security_eval/README.md).
- **CI** — Example GitHub Actions in [CI / Automation](wiki/CI-Automation.md).
