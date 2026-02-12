# Command Reference

Full list of commands and what they do. Run `ollama-tools --help` for the latest.

---

## Commands

| Command | Description |
|---------|-------------|
| **check** | Verify ollama, Hugging Face, optional deps, and llama.cpp |
| **doctor** | Diagnose environment and optionally apply common fixes |
| **plan** | Preview major operations without executing (quickstart, auto, doctor-fix, adapters-apply) |
| **quickstart** | One-command setup: fetch a default model and create an Ollama model |
| **start** | Alias for quickstart with beginner defaults |
| **auto** | Auto-detect source (recipe, gguf, HF repo, base, adapter) and run the right flow |
| **setup-llama-cpp** | Clone and build llama.cpp (finetune, quantize); add build dir to PATH |
| **create-from-base** | Create a new model from a base model (Modelfile) |
| **convert** | Create an Ollama model from a GGUF file |
| **fetch** | Download a GGUF from Hugging Face and create an Ollama model |
| **fetch-adapter** | Download an adapter from HF and create an Ollama model (base + adapter) |
| **build** | Build an Ollama model from a recipe file (YAML or JSON) |
| **validate-training-data** | Validate JSONL training data (file(s) or directory) |
| **prepare-training-data** | Convert JSONL to plain text for trainers (e.g. llama.cpp) |
| **train** | Generate a training script (data path → runnable pipeline) |
| **retrain** | Create an Ollama model from base + adapter (after training) |
| **abliterate** | Refusal removal (compute-dir; then use Sumandora or export to GGUF) |
| **adapters** | Find and use adapters (search, recommend) |
| **downsize** | Downsize via distillation (teacher, student, name → steps) |
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
| GGUF file → Ollama model | `convert --gguf <path> --name <name>` |
| Find adapters on HF | `adapters search "llama lora"` |
| Adapter recommendations (optional auto-apply) | `adapters recommend [--base llama3.2] [--apply]` |
| Get adapter from HF | `fetch-adapter <repo_id> --base <base> --name <name>` |
| Customize model (prompt, params, adapter) | `create-from-base`, `retrain`, or `build recipe.yaml` |
| Validate training data | `validate-training-data <file(s) or dir>` |
| JSONL → trainer format | `prepare-training-data <file(s) or dir> -o out.txt` |
| Generate training pipeline script | `train --data <path> --base <base> --name <name> --write-script train.sh` |
| Check environment | `check` |
| Diagnose / auto-fix | `doctor [--fix] [--plan] [--fix-llama-cpp]` |
| Install llama.cpp | `setup-llama-cpp [--dir ./llama.cpp]` |
| Refusal removal | `abliterate compute-dir` |
| Downsize (distillation) | `downsize --teacher <hf_id> --student <hf_id> --name <name>` |
| One-file config build | `build recipe.yaml` |
| List HF cache | `hf-cache ls` |
| Remove HF cache repo(s) | `hf-cache rm <repo_id> [repo_id ...] [--yes]` |

---

## Subcommands

- **adapters** has subcommands: `search`, `recommend`.
- **abliterate** has subcommands: `compute-dir` (and others; run `abliterate --help`).

Use `ollama-tools <command> --help` for options and examples.
