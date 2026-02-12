# Phase 2 — Retraining pipeline: sub-tasks

| # | Task | Status |
|---|------|--------|
| 2.1 | **Training data format** — Document standard format (JSONL instruction); optional validation | Done |
| 2.2 | **Trainer integration** — Document llama.cpp finetune; link Axolotl/Unsloth | Done |
| 2.3 | **End-to-end retrain** — CLI `retrain` (base + adapter → Ollama model) | Done |
| 2.4 | **Non-QLoRA** — Document prefer full LoRA for Ollama (in ADAPTER / RETRAIN) | Done |

**Deliverables:** Clear retrain path (data → adapter → Ollama model); optional `retrain` subcommand or script.
