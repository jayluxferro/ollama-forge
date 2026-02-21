# Ollama Tools — Roadmap

**Project name options:** `ollama-tools`, `ollama-model-factory`, `ollama-forge`. Current working name: **ollama-tools**.

---

## Main goal

**Make it simple.** The project exists so a person can try and do everything — fetch from Hugging Face, convert, use adapters, customize with recipes — in a **very simple way**, with **all tools provided here**, **without extraordinary expertise**. Prefer: one place for all tools, clear commands, sensible defaults, and docs that assume minimal prior knowledge.

---

## Goals (summary)

1. **Create new models** from existing local Ollama models (Modelfile-based: params, system prompt, template).
2. **Retrain / fine-tune** local models and consume the result in Ollama (e.g. LoRA → Modelfile `ADAPTER`).
3. **Abliterated models** — remove refusal behavior (harmful/harmless refusals) using the [refusal-direction technique](https://github.com/Sumandora/remove-refusals-with-transformers), then export for Ollama.
4. **Hugging Face → Ollama** — convert HF models (e.g. via GGUF) and package them as Ollama models.
5. **Downsize models** — produce smaller models from larger ones (e.g. 30B → 3B) via distillation or pruning, then export for Ollama.

---

## Scope clarification

| Goal | What it means technically |
|------|----------------------------|
| **New models from local** | Use Ollama Modelfile: `FROM <base>`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, optional `ADAPTER`. No training. |
| **Retraining** | Training happens outside Ollama (llama.cpp finetune, Axolotl, Unsloth, etc.); we focus on **ingesting** adapters/checkpoints into Ollama (Modelfile, CLI, or small helpers). |
| **Abliterated models** | **Refusal removal**: compute refusal direction from harmful vs harmless prompts (pure HF Transformers), apply to model weights, then export to GGUF and package for Ollama. Key reference: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers). |
| **HF → Ollama** | Pipeline: HF model → GGUF (e.g. llama.cpp convert script) → Modelfile `FROM /path/to/model.gguf` → `ollama create`. Can wrap in CLI/scripts and support HF Hub by name. |
| **Downsize models** | **Distillation**: teacher (e.g. 30B) → student (e.g. 3B) via logit or data distillation; train student to match teacher outputs, then export student to GGUF → Ollama. **Pruning**: optional path (layer/head pruning, slimming) for moderate size reduction; for large ratios (e.g. 10×), distillation is the main approach. |

---

## Phased roadmap

### Phase 1 — Foundation (done)

- [x] **Project layout**: Python + uv, `src/ollama_tools/`, CLI entrypoint `ollama-tools`, `docs/`.
- [x] **Tests**: pytest + ruff; `tests/`, `pyproject.toml` config.
- [x] **Modelfile workflows**: `docs/MODELFILE.md`; CLI `create-from-base`.
- [x] **HF → GGUF → Ollama**: `docs/HF_TO_OLLAMA.md`; CLI `convert --gguf <path> --name <name>` (HF→GGUF via llama.cpp).
- [x] **Adapter ingestion**: `docs/ADAPTER.md`; `create-from-base --adapter <path>`.

**Deliverables:** Docs (MODELFILE, HF_TO_OLLAMA, ADAPTER), CLI `create-from-base` and `convert`, README prerequisites.

---

### Phase 2 — Retraining pipeline (done)

- [x] **Training data format**: docs/TRAINING_DATA.md; JSONL instruction/input/output; validate-training-data CLI.
- [x] **Trainer integration**: docs/RETRAIN.md — llama.cpp finetune, Axolotl, Unsloth.
- [x] **End-to-end**: retrain --base --adapter --name (create Ollama model from base + adapter after training).
- [x] **Non-QLoRA**: Documented in ADAPTER.md and RETRAIN.md; prefer full LoRA.

**Deliverables:** Clear retrain path (data → adapter → Ollama model); validate-training-data and retrain subcommands.

---

### Phase 3 — Abliterated models (refusal removal) (done)

- [x] **Define “abliterated”**: Refusal removal — strip harmful/harmless refusal behavior using the refusal-direction method (no TransformerLens; pure HF Transformers). Reference: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers).
- [x] **Integrate or wrap**: docs/ABLITERATE.md; CLI `abliterate compute-dir` (optional deps: uv sync --extra abliterate); get_layers() supports model.model.layers and model.transformer.h.
- [x] **Pipeline**: HF base → compute refusal dir (our CLI or Sumandora) → apply (Sumandora inference) → export to GGUF → ollama-tools convert.
- [x] **Re-import**: Documented in ABLITERATE.md; `ollama-tools convert --gguf <path> --name <name>`.

**Deliverables:** docs/ABLITERATE.md; abliterate compute-dir CLI; doc on limitations and supported architectures.

---

### Phase 4 — Downsizing (e.g. 30B → 3B) (done)

- [x] **Distillation pipeline**: docs/DOWNSIZE.md — teacher/student, data/logit distillation, TRL GKD, Axolotl/Unsloth.
- [x] **Student export**: Documented (student → GGUF via llama.cpp → ollama-tools convert).
- [x] **CLI/recipe**: `downsize` and `downsize pipeline` print pipeline steps.
- [x] **Pruning (optional)**: Documented in DOWNSIZE.md for moderate ratios; prefer distillation for large ratios.

**Deliverables:** docs/DOWNSIZE.md; downsize CLI; doc on trade-offs and hardware.

---

### Phase 5 — Polish and extensions (done)

- [x] **Config-driven runs**: Recipe (YAML/JSON); `ollama-tools build <recipe>`.
- [x] **Quantization options**: docs/QUANTIZATION.md — GGUF quants, when to use, llama.cpp quantize.
- [x] **CI / automation**: docs/CI_EXAMPLE.md — example GitHub Action and local script.
- [x] **Naming**: Left to user; project name remains ollama-tools unless changed.

---

## Out of scope (for now)

- Training from scratch (only fine-tuning / adapters).
- Hosting or serving beyond “use with local Ollama”.
- GUI; focus is CLI/scripts and docs.

---

## Tech stack

- **Language:** Python (≥3.10).
- **Package manager:** [uv](https://docs.astral.sh/uv/) — use `uv sync` to install, `uv run ollama-tools` to run the CLI.
- **Layout:** `src/ollama_tools/` package, CLI entrypoint `ollama-tools`.

## Dependencies and references

- **Ollama**: [Modelfile](https://docs.ollama.com/modelfile), [Import](https://docs.ollama.com/import).
- **GGUF / conversion**: llama.cpp (convert-*.py, quantize), Hugging Face `huggingface-cli`.
- **Fine-tuning**: llama.cpp finetune (LoRA), Axolotl, Unsloth; Ollama `ADAPTER` in Modelfile.
- **Abliterated (refusal removal)**: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) — harmful/harmless refusal direction with pure HF Transformers; then export to GGUF for Ollama.
- **Downsizing**: Knowledge distillation (teacher → student) for large ratio (e.g. 30B → 3B); optional pruning for moderate reduction. HF `transformers`, custom training, or distillation frameworks; then student → GGUF → Ollama.
- **MCP**: Optional use of Ollama MCP tools (e.g. list_models, run) for validation or scripting from Cursor.

---

## Next steps

1. Confirm project name (optional).
2. ~~Set up repo layout~~ Done (Python + uv, `src/ollama_tools/`).
3. Implement Phase 1: Modelfile helper + HF → Ollama conversion path.
4. Update `.cursor/skills/ollama-tools/memory.md` as decisions and progress are made.
