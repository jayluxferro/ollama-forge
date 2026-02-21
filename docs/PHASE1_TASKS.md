# Phase 1 — Foundation: sub-tasks

| # | Task | Status |
|---|------|--------|
| 1.1 | **Tests & ruff** — Add minimal test, ruff config, run pytest and ruff | Done |
| 1.2 | **Modelfile docs** — Document "create model from base" (FROM, PARAMETER, SYSTEM, TEMPLATE) | Done |
| 1.3 | **Modelfile CLI** — `create-from-base` subcommand: generate Modelfile + run `ollama create` | Done |
| 1.4 | **HF → Ollama docs** — Document steps: HF model → GGUF (llama.cpp) → Modelfile → `ollama create` | Done |
| 1.5 | **HF → Ollama CLI** — `convert` subcommand: GGUF path → Modelfile → `ollama create` | Done |
| 1.6 | **Adapter ingestion** — Document ADAPTER in Modelfile; `create-from-base --adapter` helper | Done |

**Deliverables:** Docs (`docs/MODELFILE.md`, `docs/HF_TO_OLLAMA.md`, `docs/ADAPTER.md`), CLI commands `create-from-base` and `convert`, README prerequisites (Ollama, llama.cpp, Python + uv).
