# Command Reference

Full list of commands and what they do. Run `ollama-forge --help` for the latest.

---

## Commands

| Command | Description |
|---------|-------------|
| **check** | Verify ollama, Hugging Face, Python deps, and llama.cpp |
| **doctor** | Diagnose environment and optionally apply common fixes |
| **plan** | Preview major operations without executing (quickstart, auto, doctor-fix, adapters-apply) |
| **quickstart** | One-command setup: fetch a default model and create an Ollama model |
| **start** | Alias for quickstart with beginner defaults |
| **auto** | Auto-detect source (recipe, gguf, HF repo, base, adapter) and run the right flow |
| **setup-llama-cpp** | Clone and build llama.cpp (finetune, quantize); add build dir to PATH |
| **create-from-base** | Create a new model from a base model (Modelfile) |
| **convert** | Create an Ollama model from a GGUF file |
| **fetch** | Download a GGUF from Hugging Face and create an Ollama model |
| **import** | Download HF safetensors, convert to GGUF, optionally quantize, and create an Ollama model |
| **fetch-adapter** | Download an adapter from HF and create an Ollama model (base + adapter) |
| **build** | Build an Ollama model from a recipe file (YAML or JSON) |
| **validate-recipe** | Validate a recipe file (schema and paths) without building |
| **validate-training-data** | Validate JSONL training data (file(s) or directory); accepts Alpaca or messages format |
| **prepare-training-data** | Convert JSONL to plain text for trainers (e.g. llama.cpp); accepts both formats |
| **convert-training-data-format** | Convert JSONL to Alpaca-style (e.g. from TeichAI/datagen) — input file, `-o` output |
| **train-data** | Training data helpers: **init** — create directory with README + sample.jsonl |
| **train-resolve-base** | Suggest how to get a base GGUF for finetune/train-run (e.g. train-resolve-base llama3.2) |
| **train** | Generate a training script (to run pipeline in one go, use **finetune** or **train-run**) |
| **train-run** | E2E pipeline: validate → prepare → finetune (if --base-gguf) → retrain |
| **finetune** | Alias for train-run (same args): one command to run the full training pipeline |
| **retrain** | Create an Ollama model from base + adapter (adapter: dir or .bin/.gguf file) |
| **abliterate** | Refusal removal: compute-dir, run, chat, serve, **proxy** (lightweight tokenizer proxy for agents), evaluate, optimize, fix-ollama-template, download-lists |
| **adapters** | Find and use adapters (search, recommend) |
| **turboquant** | TurboQuant: extreme quantization + inference (quantize, serve, chat, info) — no llama.cpp needed |
| **downsize** | Downsize via distillation (teacher, student, name → steps) |
| **security-eval** | Run prompt sets against Ollama (run), or optional UI (ui) for refusal/compliance KPIs |
| **vlm** | Vision Language Model commands: generate, chat, serve, convert, quantize, finetune (Apple Silicon via mlx-vlm) |
| **serve** | Start llama-server to serve a GGUF model via an OpenAI-compatible API (auto GPU: Metal/CUDA/CPU) |
| **chat** | Interactive chat with a running llama-server or any OpenAI-compatible endpoint |
| **hf-cache** | List or remove Hugging Face Hub local cache (ls, rm) |

---

## Quick reference table (from README)

| What you want | Command |
|---------------|---------|
| Short alias for beginner start | `start [--name my-model]` |
| Easiest one-command start | `quickstart [--name my-model]` |
| Auto-detect source type | `auto <source> [--name my-model]` |
| Preview operations | `plan <quickstart\|auto\|doctor-fix\|adapters-apply> ...` |
| Get GGUF from HF and create model | `fetch <repo_id> --name <name>` |
| HF safetensors → GGUF → Ollama | `import <repo_or_dir> --name <name>` |
| GGUF file → Ollama model | `convert --gguf <path> --name <name>` |
| Find adapters on HF | `adapters search "llama lora"` |
| Adapter recommendations (optional auto-apply) | `adapters recommend [--base llama3.2] [--apply]` |
| Get adapter from HF | `fetch-adapter <repo_id> --base <base> --name <name>` |
| Customize model (prompt, params, adapter) | `create-from-base`, `retrain`, or `build recipe.yaml` |
| Scaffold training data dir | `train-data init -o ./data` |
| Validate training data | `validate-training-data <file(s) or dir>` |
| JSONL → trainer format | `prepare-training-data <file(s) or dir> -o out.txt` |
| Messages JSONL → Alpaca | `convert-training-data-format dataset.jsonl -o alpaca.jsonl` |
| Run full training pipeline | `finetune` or `train-run --data <path> --base <base> --name <name> [--base-gguf <path>]` |
| Get base GGUF suggestions | `train-resolve-base <base_name>` (e.g. llama3.2) |
| Generate training pipeline script | `train --data <path> --base <base> --name <name> --write-script train.sh` |
| Check environment | `check` |
| Diagnose / auto-fix | `doctor [--fix] [--plan] [--fix-llama-cpp]` |
| Install llama.cpp | `setup-llama-cpp [--dir ./llama.cpp]` |
| Refusal removal | `abliterate compute-dir`; for agents with tools: `abliterate proxy --name <name>` |
| Downsize (distillation) | `downsize --teacher <hf_id> --student <hf_id> --name <name>` |
| One-file config build | `build recipe.yaml` |
| Validate recipe only | `validate-recipe recipe.yaml` |
| List HF cache | `hf-cache ls` |
| Quantize to TurboQuant | `turboquant quantize <hf_model_or_path> --bits 3` |
| Serve TurboQuant model | `turboquant serve <model.tqf> --port 8811` |
| Chat with TurboQuant model | `turboquant chat <model.tqf>` |
| Serve GGUF directly (no Ollama) | `serve <model.gguf> [--host 127.0.0.1] [--port 11434] [-ngl -1]` |
| Chat with a running server | `chat [--base-url http://127.0.0.1:11434] [--system "..."] [--temperature 0.7]` |
| VLM generate (image+text) | `vlm generate --model <model> --prompt "..." --image photo.jpg` |
| VLM interactive chat | `vlm chat --model <model> [--system "..."]` |
| VLM Gradio web UI | `vlm chat-ui --model <model>` |
| VLM OpenAI-compatible server | `vlm serve --model <model> [--port 8080]` |
| VLM convert HF → MLX | `vlm convert --hf-path <repo_id> [--quantize --q-bits 4]` |
| VLM quantize | `vlm quantize --model <repo_id> --bits 4 [--group-size 64]` |
| VLM fine-tune (LoRA) | `vlm finetune --model <model> --dataset <path> [--lora-rank 8]` |
| Download GGUF only (no Ollama) | `fetch <repo_id> --download-only` (auto-converts safetensors if no GGUF) |
| Remove HF cache repo(s) | `hf-cache rm <repo_id> [repo_id ...] [--yes]` |

---

## Subcommands

- **adapters** has subcommands: `search`, `recommend`.
- **abliterate** has subcommands: `compute-dir`, `run`, `chat`, `serve`, `proxy`, `evaluate`, `optimize`, `fix-ollama-template`, `download-lists`. Run `abliterate --help` for full list.
- **turboquant** has subcommands: `quantize`, `serve`, `chat`, `info`. Supports asymmetric K/V cache compression, layer-adaptive precision (`TURBO_LAYER_ADAPTIVE` env var), and temporal decay for long contexts. See [TurboQuant](TurboQuant) for details.
- **vlm** has subcommands: `generate`, `chat`, `chat-ui`, `serve`, `convert`, `quantize`, `finetune`. Requires Apple Silicon (mlx-vlm). See [VLM](VLM) for details.
- **serve** starts llama-server from llama.cpp; auto-discovers the binary from PATH or `./llama.cpp/build/bin/`. GPU offloading is automatic (Metal on Apple Silicon, CUDA on NVIDIA, CPU fallback). Pass extra llama-server flags after `--`.
- **chat** connects to any OpenAI-compatible `/v1/chat/completions` endpoint. In-session commands: `/clear` (reset history), `quit` (exit).
- **Ports:** Ollama 11434, serve 11434, abliterate serve 11435, abliterate proxy 11436, turboquant serve 8811, vlm serve 8080 (defaults; override with `--port`).

Use `ollama-forge <command> --help` for options and examples.

---

## Exit codes (scripting)

For scripts and CI, rely on the process exit code:

| Code | Meaning |
|------|--------|
| **0** | Success |
| **1** | Error (invalid args, missing deps, failed step). Stderr has a short message and often "Next:" steps. |

Subcommands (e.g. `abliterate run`, `train-run`) do not use other codes; any failure is 1. Use `--json` where available (e.g. `check --json`, `doctor --json`, `abliterate evaluate --json`) for machine-readable output.
