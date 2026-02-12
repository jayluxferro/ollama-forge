# Abliterate (refusal removal)

Strip refusal behavior by computing a "refusal direction" and applying ablation. Reference: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers).

---

## Optional dependency

```bash
uv sync --extra abliterate
```

---

## Built-in lists (no files needed)

The tool ships with default harmful/harmless instruction lists. Run without `--harmful`/`--harmless`:

```bash
uv run ollama-tools abliterate compute-dir --model <hf_id> --output refusal.pt
```

---

## Your own lists

- **Single files:** `--harmful harmful.txt --harmless harmless.txt` (one instruction per line; lines starting with `#` are skipped).
- **Directories:** `--harmful-dir ./harmful/ --harmless-dir ./harmless/` to use all `.txt` files in those directories.

```bash
uv run ollama-tools abliterate compute-dir --model <hf_id> --harmful harmful.txt --harmless harmless.txt --output refusal.pt
```

---

## After computing

Use Sumandora's `inference.py` (or your script) with the `.pt` file. If you get an abliterated checkpoint, convert to GGUF (llama.cpp) then:

```bash
ollama-tools convert --gguf /path/to/abliterated.gguf --name my-abliterated
ollama run my-abliterated
```
