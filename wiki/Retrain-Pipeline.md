# Retrain Pipeline

End-to-end: **data → adapter → Ollama model**.

**Default trainer:** `train-run` and `finetune` use **llama.cpp** as the default trainer (run `finetune` or `llama-finetune` when `--base-gguf` is set). The generated script from `train --write-script` also targets llama.cpp by default. To use another trainer (e.g. Axolotl, Unsloth), run it manually after `prepare-training-data`, then run `retrain --base <base> --adapter <path> --name <name>`.

---

## Steps

1. **Validate data** — `validate-training-data <file(s) or directory>`.
2. **Optional: generate pipeline** — `train --data <path> --base <base> --name <name> --write-script train.sh` then run the script.
3. **Or manually:** `prepare-training-data` → run trainer (llama.cpp, Axolotl, Unsloth) → `retrain --base <base> --adapter <path> --name <name>`.
4. **Run** — `ollama run <name>`.

---

## One-command script

```bash
uv run ollama-forge train --data ./data/ --base llama3.2 --name my-model --write-script train.sh
./train.sh
```

With llama.cpp on PATH and a base GGUF, you can add `--base-gguf /path/to/base.gguf --run-trainer` so the script runs the trainer step. See [Training Data](Training-Data).

---

## After training

You have an adapter (directory or file). Create the Ollama model:

```bash
uv run ollama-forge retrain --base llama3.2 --adapter /path/to/adapter --name my-finetuned
ollama run my-finetuned
```

**Adapter path:** Can be a **directory** (PEFT with `adapter_config.json` + weights, or a directory containing a single `.bin`/`.gguf` from llama.cpp finetune) or a **file** (`.bin` or `.gguf`). See [Adapters](Adapters) for format and compatibility.
