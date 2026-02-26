# ollama-forge

**Get models from Hugging Face, convert them, add adapters, and run them in [Ollama](https://ollama.com) — in a simple way, without needing deep expertise.**

All the tools live here: fetch, convert, adapters, recipes. You run a few commands and get a model you can use locally.

---

## Quick start

```bash
uv sync && uv run ollama-forge fetch TheBloke/Llama-2-7B-GGUF --name my-model && ollama run my-model
```

Or the shortest path for beginners:

```bash
uv run ollama-forge start --name my-model
ollama run my-model
```

---

## Why this project

- **One place** — Fetch from Hugging Face, convert to GGUF, use adapters, customize with recipes; no need to hunt down scattered scripts.
- **Simple** — Clear commands and docs so you can try things without being an ML expert.
- **Local-first** — Everything is aimed at getting models running in Ollama on your machine.

---

## Wiki contents

| Topic | Description |
|-------|-------------|
| [Installation](Installation) | Setup, check, doctor, setup-llama-cpp |
| [Quick Start](Quick-Start) | start, quickstart, profiles, task presets |
| [Auto & Plan](Auto-and-Plan) | Auto-detect source, plan (dry-run) |
| [Fetch & Convert](Fetch-and-Convert) | Get GGUF from HF, convert file to Ollama model |
| [Recipes](Recipes) | One-file YAML/JSON build |
| [Modelfile](Modelfile) | Ollama Modelfile basics |
| [Adapters](Adapters) | LoRA: search, recommend, fetch-adapter, retrain |
| [Training Data](Training-Data) | JSONL validate, prepare, train script |
| [Retrain Pipeline](Retrain-Pipeline) | Data → adapter → Ollama |
| [Abliterate](Abliterate) | Refusal removal |
| [Heretic integration](Heretic-Integration) | Per-layer directions, strength kernel, evaluate, optimize |
| [Security Eval](Security-Eval) | LLM security evaluation: prompt sets, KPIs, UI |
| [Downsizing](Downsizing) | Teacher → student distillation |
| [Hugging Face without GGUF](Hugging-Face-Without-GGUF) | When the repo has no GGUF |
| [Quantization](Quantization) | Smaller/faster GGUF (Q4_K_M, Q8_0, etc.) |
| [CI / Automation](CI-Automation) | Example GitHub Actions |
| [Command Reference](Command-Reference) | All commands at a glance |

---

## Commands at a glance

| What you want | Command |
|---------------|---------|
| Easiest one-command start | `start` or `quickstart [--name my-model]` |
| Auto-detect source and run | `auto <source> [--name my-model]` |
| Preview operations (dry-run) | `plan <quickstart\|auto\|doctor-fix\|adapters-apply> ...` |
| Get GGUF from HF and create model | `fetch <repo_id> --name <name>` |
| HF safetensors → GGUF → Ollama | `import <repo_or_dir> --name <name>` |
| GGUF file → Ollama model | `convert --gguf <path> --name <name>` |
| Find / use adapters | `adapters search`, `adapters recommend`, `fetch-adapter`, `retrain` |
| One-file config build | `build recipe.yaml` |
| Check / fix environment | `check`, `doctor [--fix]` |
| Install llama.cpp (finetune, quantize) | `setup-llama-cpp` |

Run `ollama-forge --help` for the full list.
