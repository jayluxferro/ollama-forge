# Quantization options (GGUF)

When converting Hugging Face or other models to GGUF for Ollama, you can **quantize** to reduce size and VRAM. Quantization trades some quality for smaller files and lower memory use.

---

## Common GGUF quant types (llama.cpp)

| Quant | Typical use | Size / quality |
|-------|-------------|----------------|
| **Q4_0** | Smallest, lowest quality | 4-bit, fast |
| **Q4_K_M** | Good default for many models | 4-bit, better than Q4_0 |
| **Q5_0**, **Q5_K_S**, **Q5_K_M** | Higher quality, larger | 5-bit |
| **Q8_0** | Near full precision, larger | 8-bit |
| **F16**, **F32** | Full precision (no quant) | Original size |

K variants (e.g. K_M, K_S) use mixed quantization for better quality at similar size.

---

## When to quantize

- **Before Ollama:** Convert HF → GGUF with llama.cpp, then run `quantize` (or use a pre-quantized GGUF from the Hub). Then `ollama-tools convert --gguf <path> --name <name>`.
- **After conversion:** Ollama runs whatever GGUF you give it; quantization is done at the GGUF level (llama.cpp `quantize` tool), not by ollama-tools.

---

## How to quantize (llama.cpp)

1. Produce a full or high-precision GGUF (e.g. F16 or Q8_0) with llama.cpp's convert script.
2. Run the **quantize** tool:
   ```bash
   ./quantize path/to/model.gguf path/to/model-Q4_K_M.gguf Q4_K_M
   ```
3. Use the quantized file with Ollama: `ollama-tools convert --gguf path/to/model-Q4_K_M.gguf --name my-model`.

---

## Trade-offs

- **Lower bit (e.g. Q4):** Smaller file, less VRAM, may lose nuance or instruction-following.
- **Higher bit (e.g. Q8, F16):** Larger file, more VRAM, closer to original quality.
- **QLoRA / adapter:** If you use LoRA adapters with Ollama, prefer **non-quantized** base/adapters when possible (see RETRAIN.md / ADAPTER.md).

---

## Summary

Quantization is applied when **creating the GGUF** (llama.cpp convert + quantize), not inside ollama-tools. Choose a quant that fits your hardware and quality needs; then pass the resulting GGUF to `ollama-tools convert`.
