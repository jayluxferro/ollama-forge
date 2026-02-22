# Abliterated models (refusal removal)

Strip harmful/harmless **refusal behavior** from an LLM by computing a "refusal direction" in activation space and subtracting it (direction ablation). Then export the modified model for Ollama.

**Reference:** [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) — pure Hugging Face Transformers (no TransformerLens).

**User-facing docs:** See the wiki [Abliterate](wiki/Abliterate) for commands, options, and usage (proxy, serve, chat, evaluate, config file, etc.).

---

## Built-in pipeline (ollama-tools)

ollama-tools provides a **full pipeline** in one command:

1. **Compute refusal direction** — Run forward passes on harmful vs harmless instructions; take the difference of mean hidden states at a chosen layer/position; normalize. Saves a `.pt` file.
2. **Apply ablation** — Bake the ablation into the model weights (edit linear layers) and save a full Hugging Face checkpoint (in-weight, not inference-time).
3. **Export to GGUF** — Run llama.cpp `convert_hf_to_gguf.py` on the abliterated checkpoint.
4. **Create Ollama model** — Build a Modelfile from the GGUF (optional requantization), derive or merge chat template, and run `ollama create`.

**One command:** `uv run ollama-forge abliterate run --model <hf_id_or_path> --name my-abliterated` (requires `uv sync --extra abliterate`). Use `--from-checkpoint` to resume, or `--config <file>` for repeatable YAML/JSON config. After creation: `abliterate chat`, `serve`, or `proxy` (see wiki for tool/function-calling).

**Disk space:** A full `abliterate run` writes a full-precision (bf16) checkpoint and then a GGUF (requantized by default). Ensure sufficient free disk space for both—roughly 2× the model size in GB for the checkpoint, plus the final GGUF size. The checkpoint is under `./abliterate-<name>/checkpoint` (or `--output-dir`). See wiki for memory and requantization options.

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

- **Full pipeline → Ollama model:** `ollama-forge abliterate run --model <id> --name <name>` (see wiki for options and `--config`).
- **Compute direction only (.pt):** `ollama-forge abliterate compute-dir --model <id> --harmful harmful.txt --harmless harmless.txt --output refusal.pt`. Then use Sumandora’s `inference.py` or your own script for inference-time ablation.
- **After run:** `ollama-forge abliterate chat --name <name>`, `serve`, or `proxy` (see wiki).

---

## Summary

| Goal | Command |
|------|--------|
| Full pipeline → Ollama model | `ollama-forge abliterate run --model <id> --name <name>` |
| Compute direction only (.pt) | `ollama-forge abliterate compute-dir ... --output refusal.pt` |
| Chat / serve / proxy after run | `ollama-forge abliterate chat/serve/proxy --name <name>` (see wiki) |
