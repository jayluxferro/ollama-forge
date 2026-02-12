# Abliterated models (refusal removal)

Strip harmful/harmless **refusal behavior** from an LLM by computing a "refusal direction" in activation space and subtracting it (direction ablation). Then export the modified model for Ollama.

**Reference:** [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) — pure Hugging Face Transformers (no TransformerLens).

---

## Pipeline overview

1. **Compute refusal direction** — Run forward passes on harmful vs harmless instructions; take the difference of mean hidden states at a chosen layer/position; normalize. Saves a vector (e.g. `.pt`).
2. **Apply ablation** — At inference (or by editing weights), subtract the projection of activations onto this direction. The reference repo does this at **inference time** by inserting ablation layers.
3. **Export for Ollama** — To get a single model file for Ollama you need an **abliterated checkpoint** (modified weights or equivalent). Then: convert to GGUF (llama.cpp) → `ollama-tools convert --gguf <path> --name <name>`.

---

## Reference implementation (Sumandora)

- **compute_refusal_dir.py** — Loads HF model, reads `harmful.txt` / `harmless.txt`, runs generation with `output_hidden_states=True`, computes `refusal_dir = (harmful_mean - harmless_mean).norm()`, saves to `.pt`.
- **inference.py** — Loads model + refusal dir, inserts an `AblationDecoderLayer` before each transformer layer that subtracts the projection onto `refusal_dir` from activations; runs interactive chat. Does **not** save the modified model; ablation is at inference time.

So: use their repo for **interactive refusal-removed inference**. For a **standalone Ollama model** you need either (a) a fork/script that bakes the ablation into weights and saves, or (b) export the base model to GGUF and note that full ablation would require applying the edit in HF space before conversion.

---

## Layer access (supported architectures)

Refusal direction is computed from a specific layer’s hidden state. Different Hugging Face models expose layers differently:

- **Llama, Falcon, many others:** `model.model.layers`
- **Qwen (some):** `model.transformer.h`

If you get an error like `model.model.layers` not found, try the other. The Sumandora README notes that some Qwen implementations use `model.transformer.h`.

---

## Limitations

- **Inference-time vs saved:** The reference implementation applies ablation at inference. Producing a single GGUF file with ablation "baked in" requires either merging the ablation into weights or saving a model that includes the ablation layers and converting (architecture may need to match what llama.cpp expects).
- **Quantization:** Sumandora uses 4-bit in places; for export to GGUF you may want full precision or a quantized GGUF from an ablated checkpoint.
- **Hardware:** Running the HF model for compute_refusal_dir and inference needs sufficient GPU/CPU and memory.

---

## Using ollama-tools

- After you have an **abliterated GGUF** (from whatever pipeline):
  ```bash
  ollama-tools convert --gguf /path/to/abliterated.gguf --name my-abliterated
  ollama run my-abliterated
  ```
- To **compute the refusal direction** (saves a `.pt` file for use with Sumandora or your own inference):
  ```bash
  uv run ollama-tools abliterate compute-dir --model <hf_id> --harmful harmful.txt --harmless harmless.txt --output refusal.pt
  ```
  Requires optional deps: `uv sync --extra abliterate`. Then use Sumandora’s `inference.py` (point it at your `.pt`) or your own script to apply the ablation.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Compute refusal direction: `ollama-tools abliterate compute-dir --model <hf_id> --harmful harmful.txt --harmless harmless.txt --output refusal.pt` (requires `uv sync --extra abliterate`), or Sumandora’s `compute_refusal_dir.py`. |
| 2 | Apply ablation (Sumandora `inference.py` for inference; or custom save for export). |
| 3 | If you have an abliterated checkpoint: convert to GGUF (llama.cpp) → `ollama-tools convert --gguf <path> --name <name>`. |
