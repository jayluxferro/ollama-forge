# Training Data

Use **JSONL** (one JSON object per line) for training data. ollama-forge accepts two formats:

- **Alpaca-style:** `instruction` (required), `output` (required), `input` (optional).
- **Messages-style** (e.g. [TeichAI/datagen](https://github.com/TeichAI/datagen)): `messages` array with `user` and `assistant` roles; system is optional. Validate and prepare accept this format directly.

---

## Validate

Check one file, several files, or a whole directory:

```bash
uv run ollama-forge validate-training-data train.jsonl
uv run ollama-forge validate-training-data data/part1.jsonl data/part2.jsonl
uv run ollama-forge validate-training-data ./data/
```

---

## Prepare (JSONL → plain text)

Convert JSONL to the format trainers expect (e.g. llama.cpp plain text with blocks):

```bash
uv run ollama-forge prepare-training-data ./data/ -o train_prepared.txt --format llama.cpp
```

Use `train_prepared.txt` with llama.cpp finetune (e.g. `--sample-start '### Instruction'`).

---

## Train script (full pipeline)

Generate a script that validates data, prepares it, and optionally runs the trainer:

```bash
uv run ollama-forge train --data ./data/ --base llama3.2 --name my-model --write-script train.sh
./train.sh
```

To have the script **run the trainer** (no manual llama.cpp step), pass a base GGUF and `--run-trainer`. You need llama.cpp's `finetune` on PATH (see [Installation](Installation)):

```bash
uv run ollama-forge train --data ./data/ --base llama3.2 --name my-model --base-gguf /path/to/base.gguf --run-trainer --write-script train.sh
./train.sh
```

---

## JSONL format

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | Yes | The task or question. |
| `input` | No | Additional context (can be empty string). |
| `output` | Yes | The desired response. |

Example (Alpaca-style):

```jsonl
{"instruction": "Summarize the following.", "input": "Long document...", "output": "Short summary."}
{"instruction": "What is 2+2?", "input": "", "output": "4."}
```

Minimal: `{"instruction": "Say hello.", "output": "Hello!"}`

---

## Generating training data

Tools like [TeichAI/datagen](https://github.com/TeichAI/datagen) generate JSONL from a list of prompts by calling an LLM (e.g. via OpenRouter): one prompt per line in a TXT file → one JSONL line per response. datagen outputs a **`messages`** array (user/assistant roles). ollama-forge **accepts this format directly** in `validate-training-data` and `prepare-training-data`. Optionally convert to Alpaca-style JSONL on disk:

```bash
ollama-forge convert-training-data-format dataset.jsonl -o dataset_alpaca.jsonl
```

See [docs/DATAGEN-ANALYSIS.md](../docs/DATAGEN-ANALYSIS.md) for details.

---

## Next step

After training you get an adapter. Use [Retrain Pipeline](Retrain-Pipeline) to create an Ollama model: `retrain --base <base> --adapter <path> --name <name>`.
