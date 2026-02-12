# Quantization

Quantization reduces GGUF size and VRAM use; it trades some quality for smaller/faster models.

---

## When to use

- **When fetching from HF:** Use `fetch --quant Q4_K_M` (or `Q8_0`, etc.) to pick that variant when the repo has multiple GGUF files.
- **When you have a GGUF file:** Use `convert --quantize Q4_K_M` to shrink it before creating the Ollama model. Requires llama.cpp `quantize` or `llama-quantize` on PATH (see [Installation](Installation)).
- **Adapters:** Prefer non-quantized LoRA when possible.

---

## Common quant types (llama.cpp)

| Quant | Use case | Size / quality |
|-------|----------|----------------|
| **Q4_0** | Smallest, lowest quality | 4-bit, fast |
| **Q4_K_M** | Good default | 4-bit, better than Q4_0 |
| **Q5_K_M**, **Q5_K_S** | Higher quality, larger | 5-bit |
| **Q8_0** | Near full precision | 8-bit, larger |
| **F16**, **F32** | Full precision | Original size |

K variants (e.g. K_M, K_S) use mixed quantization for better quality at similar size.

---

## Trade-offs

- **Lower bit (e.g. Q4):** Smaller file, less VRAM; may lose nuance or instruction-following.
- **Higher bit (e.g. Q8, F16):** Larger file, more VRAM; closer to original quality.

Quantization is applied when **creating the GGUF** (llama.cpp convert + quantize), not inside ollama-tools. Choose a quant that fits your hardware and quality needs.
