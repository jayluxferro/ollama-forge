# ollama-forge

[![PyPI](https://img.shields.io/pypi/v/ollama-forge.svg)](https://pypi.org/project/ollama-forge/)

Get models from Hugging Face, convert them, add adapters, and run them in [Ollama](https://ollama.com) — without needing deep expertise. One place for fetch, convert, adapters, and recipes.

**Install:** `pip install ollama-forge` or `uv tool install ollama-forge` — [PyPI](https://pypi.org/project/ollama-forge/). From this repo: `uv sync` then `uv run ollama-forge`; or `uv tool install .` to use the CLI from anywhere.

**Quick start:**
```bash
ollama-forge fetch TheBloke/Llama-2-7B-GGUF --name my-model && ollama run my-model
```

Or the shortest path: `ollama-forge start --name my-model` then `ollama run my-model`.

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
| [Security Eval](wiki/Security-Eval.md) | LLM security evaluation: prompt sets, KPIs, UI |
| [Downsizing](wiki/Downsizing.md) | Teacher → student distillation |
| [Hugging Face without GGUF](wiki/Hugging-Face-Without-GGUF.md) | When the repo has no GGUF |
| [Quantization](wiki/Quantization.md) | Smaller/faster GGUF (Q4_K_M, Q8_0, etc.) |
| [TurboQuant](wiki/TurboQuant.md) | Extreme 2-4 bit quantization, KV cache compression, layer-adaptive, temporal decay |
| [VLM (Vision Language Models)](wiki/VLM.md) | Multimodal inference (image/audio/video), TurboQuant KV cache, VisionFeatureCache, fine-tuning (LoRA/ORPO) on Apple Silicon via mlx-vlm >=0.4.4 |
| [CI / Automation](wiki/CI-Automation.md) | Example GitHub Actions |
| [Command Reference](wiki/Command-Reference.md) | All commands |

---

## Why ollama-forge

- **One place** — Fetch, convert, adapters, recipes; no scattered scripts.
- **Simple** — Clear commands and docs; try things without being an ML expert.
- **Local-first** — Get models running in Ollama on your machine.

---

## Setup (one-time)

- **Python 3.10+**. **From PyPI:** `pip install ollama-forge` or `uv tool install ollama-forge` ([PyPI](https://pypi.org/project/ollama-forge/)). **From repo:** `uv sync` then `uv run ollama-forge`; use `uv tool install .` from the repo root to put `ollama-forge` on your PATH.
- **Ollama** — [Install](https://ollama.com) and ensure `ollama` is on your PATH.
- **Verify:** `ollama-forge check` — see what’s installed. `ollama-forge doctor` for diagnosis; `doctor --fix` to apply safe fixes. See [Installation](wiki/Installation.md) for optional llama.cpp (finetune/quantize).
- **Dependencies:** install with `uv sync` to get the full local toolchain.
- **Optional:** Run Ruff and tests before commit/push: `git config core.hooksPath .githooks`. See [.githooks/README.md](.githooks/README.md). To fix lint before pushing without hooks: `uv run ruff check src tests --fix && uv run ruff format src tests`.

---

## Commands at a glance

| What you want | Command |
|---------------|---------|
| Easiest one-command start | `start` or `quickstart [--name my-model]` |
| Auto-detect source and run | `auto <source> [--name my-model]` |
| Preview operations (dry-run) | `plan <quickstart\|auto\|doctor-fix\|adapters-apply> ...` |
| GGUF from HF → Ollama | `fetch <repo_id> --name <name>` |
| HF safetensors → GGUF → Ollama | `import <repo_or_dir> --name <name>` |
| GGUF file → Ollama | `convert --gguf <path> --name <name>` |
| Find / use adapters | `adapters search`, `adapters recommend`, `fetch-adapter`, `retrain` |
| One-file config build | `build recipe.yaml` |
| Serve GGUF directly (llama.cpp) | `serve <model.gguf> [--port 11434] [-ngl -1]` |
| Chat with a running server | `chat [--base-url <url>]` |
| Quantize HF model (TurboQuant) | `turboquant quantize <model> --bits 3` |
| Serve TurboQuant model | `turboquant serve <model.tqf> --port 8811` |
| Chat with TurboQuant model | `turboquant chat <model.tqf>` |
| VLM generate (image+text) | `vlm generate --model <model> --prompt "..." --image photo.jpg` |
| VLM generate + TurboQuant KV | `vlm generate --model <model> --prompt "..." --kv-bits 3.5` |
| VLM interactive chat (cached) | `vlm chat --model <model> --vision-cache-size 20` |
| VLM video understanding | `vlm video-generate --model <model> --video clip.mp4` |
| VLM Gradio web UI | `vlm chat-ui --model <model>` |
| VLM OpenAI-compatible server | `vlm serve --model <model> [--port 11434] [--kv-bits 3.5]` |
| VLM convert HF → MLX | `vlm convert --hf-path <repo_id> [--q-mode mxfp4]` |
| VLM quantize | `vlm quantize --model <repo_id> --bits 4 [--q-mode affine]` |
| VLM fine-tune (LoRA/ORPO) | `vlm finetune --model <model> --dataset <path> [--train-mode orpo]` |
| Download GGUF only (no Ollama) | `fetch <repo_id> --download-only` |
| Check / fix environment | `check`, `doctor [--fix]` |
| Install llama.cpp | `setup-llama-cpp` |

Full list: [Command Reference](wiki/Command-Reference.md). Run `ollama-forge --help` for options.

---

## Simplest workflows

**Beginner (one command):**
```bash
uv run ollama-forge start --name my-model
ollama run my-model
```
Uses default model + balanced profile. Use `--profile fast|balanced|quality|low-vram` and `--task chat|coding|creative`. See [Quick Start](wiki/Quick-Start.md).

**Auto (any source):** Recipe, GGUF path, HF repo, base model, or adapter — the tool detects and runs the right flow:
```bash
uv run ollama-forge auto ./recipe.yaml
uv run ollama-forge auto TheBloke/Llama-2-7B-GGUF --name my-model
uv run ollama-forge auto llama3.2 --name my-assistant --system "You are helpful."
```
See [Auto & Plan](wiki/Auto-and-Plan.md).

**Fetch from Hugging Face:** When the repo has GGUF files:
```bash
uv run ollama-forge fetch TheBloke/Llama-2-7B-GGUF --name my-model
ollama run my-model
```
Use `--quant Q4_K_M` to pick size. For gated or private repos, set `HF_TOKEN` or run `huggingface-cli login`. See [Fetch & Convert](wiki/Fetch-and-Convert.md).

**Local GGUF:** `uv run ollama-forge convert --gguf /path/to/model.gguf --name my-model`. Optional `--quantize Q4_K_M` (needs llama.cpp on PATH). See [Quantization](wiki/Quantization.md).

**Recipe (one file):** `uv run ollama-forge build recipe.yaml`. See [Recipes](wiki/Recipes.md) for format and examples. Sampling options (`temperature`, `top_p`, `repeat_penalty`) are available on fetch, convert, build, and create-from-base ([Modelfile](wiki/Modelfile.md), [Recipes](wiki/Recipes.md)).

**Serve a model directly (without Ollama):** Download a GGUF and spin up an OpenAI-compatible server:
```bash
uv run ollama-forge fetch janhq/Jan-code-4b-gguf --download-only
uv run ollama-forge serve /path/to/model.gguf
# In another terminal:
uv run ollama-forge chat
```
`serve` auto-detects GPU (Metal on Apple Silicon, CUDA on NVIDIA, CPU fallback). `fetch --download-only` also handles repos with safetensors — it converts to GGUF automatically.

**Adapters:** `adapters search "llama lora"`, then `fetch-adapter <repo> --base <base> --name <name>`, or `retrain --base <base> --adapter <path> --name <name>`. See [Adapters](wiki/Adapters.md).

**Training data → model:** Validate JSONL, prepare for trainer, generate script: `train --data ./data/ --base llama3.2 --name my-model --write-script train.sh`. See [Training Data](wiki/Training-Data.md) and [Retrain Pipeline](wiki/Retrain-Pipeline.md).

---

## Other topics

- **Hugging Face repo without GGUF** — Convert with llama.cpp first, then `convert`. [Wiki](wiki/Hugging-Face-Without-GGUF.md).
- **Refusal removal (abliterate)** — Quickstart: `abliterate easy --model <id> --name <name>` or `abliterate wizard` for prompts. Install deps with `uv sync`. For agents with tool support use the lightweight **proxy**: `abliterate proxy --name <name>`. [Wiki](wiki/Abliterate.md).
- **Downsizing (distillation)** — `downsize --teacher <hf> --student <hf> --name <name>`. [Wiki](wiki/Downsizing.md).
- **LLM security evaluation** — Run prompt sets against Ollama/serve, score refusal/compliance, get ASR and KPIs: `security-eval run <prompt_set>`. Install deps with `uv sync`, then run `security-eval ui` for the UI. [Wiki: Security Eval](wiki/Security-Eval.md).
- **TurboQuant** — Extreme 2-4 bit quantization with near-optimal quality. No llama.cpp needed. `turboquant quantize <model> --bits 3` then `turboquant serve` or `turboquant chat`. Includes asymmetric K/V cache compression, layer-adaptive precision (protects quality-critical layers), and temporal decay (progressively compresses old KV tokens). [Wiki](wiki/TurboQuant.md).
- **VLM (Vision Language Models)** — Multimodal (image + audio + video + text) inference on Apple Silicon via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) >=0.4.4. Generate, chat, serve (OpenAI-compatible), video understanding, convert, quantize (affine/mxfp4/nvfp4/mxfp8), and fine-tune (LoRA/QLoRA/full/ORPO). Features TurboQuant KV cache (`--kv-bits 3.5` for 76% memory savings) and VisionFeatureCache (`--vision-cache-size 20` for 11x+ multi-turn speedup). `vlm generate --model <model> --prompt "Describe this" --image photo.jpg`. [Wiki](wiki/VLM.md).
- **CI** — Example GitHub Actions in [CI / Automation](wiki/CI-Automation.md).
