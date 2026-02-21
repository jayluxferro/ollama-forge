# Retraining pipeline: data → adapter → Ollama model

End-to-end path: prepare training data → run a trainer to produce an adapter → create an Ollama model with that adapter.

---

## 1. Training data format

Use the standard **JSONL instruction format** (see TRAINING_DATA.md): each line is a JSON object with `instruction`, optional `input`, and `output`.

Validate before training:

```bash
ollama-tools validate-training-data train.jsonl
```

Convert or export this format for your trainer (llama.cpp expects plain text; Axolotl/Unsloth often accept similar schemas).

---

## 2. Run a trainer to produce an adapter

Training is done **outside** ollama-tools. Choose one:

### llama.cpp finetune (LoRA)

- **Repo:** [llama.cpp examples/finetune](https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune)
- **Input:** Base model (GGUF), training data (plain text; convert from JSONL if needed).
- **Output:** LoRA adapter (e.g. `.bin` or directory). Prefer **full LoRA** (non-quantized) so Ollama can load it reliably.
- **Note:** Only llama-based architectures are supported.

### Axolotl

- **Repo:** [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) — config-driven LoRA/QLoRA training.
- **Output:** PEFT-style adapter directory (e.g. `adapter_config.json`, `adapter_model.safetensors`). For Ollama, train **full LoRA** (not QLoRA) when possible.

### Unsloth

- **Repo:** [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA training.
- **Output:** Adapter in a format you can export for Ollama; prefer non-quantized LoRA.

---

## 3. Prefer full LoRA (non-quantized) for Ollama

Ollama works best with **non-quantized** LoRA adapters. QLoRA adapters may not behave correctly. When using Axolotl or similar, train with full LoRA if your hardware allows, and export the adapter (not the quantized base) for use with Ollama.

---

## 4. Create the Ollama model from base + adapter

After you have an adapter directory (or compatible path):

```bash
ollama-tools retrain --base llama3.2 --adapter /path/to/adapter --name my-finetuned
```

Or use `create-from-base` with `--adapter`:

```bash
ollama-tools create-from-base --base llama3.2 --name my-finetuned --adapter /path/to/adapter
```

Then run:

```bash
ollama run my-finetuned
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Prepare JSONL (instruction/input/output); run `validate-training-data`. |
| 2 | Run trainer (llama.cpp, Axolotl, or Unsloth); prefer full LoRA. |
| 3 | Run `ollama-tools retrain --base <base> --adapter <path> --name <name>`. |
