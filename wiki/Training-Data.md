# Training Data

Use **JSONL** (one JSON object per line) for training data. ollama-forge accepts two formats:

- **Alpaca-style:** `instruction` (required), `output` (required), `input` (optional).
- **Messages-style** (e.g. [TeichAI/datagen](https://github.com/TeichAI/datagen)): `messages` array with `user` and `assistant` roles; system is optional. Validate and prepare accept this format directly.

---

## Quick start (training)

1. **Put data in JSONL** — `instruction` + `output` (and optional `input`), or `messages`. Use `train-data init -o ./data` to create a sample directory.
2. **Validate** — `ollama-forge validate-training-data <path>`.
3. **Run pipeline** — `ollama-forge finetune --data <path> --base <base> --name <name>` (or `train-run`). If you have a base GGUF and llama.cpp `finetune` on PATH, add `--base-gguf <path>` to run training and retrain in one go. Need a GGUF? Run `ollama-forge train-resolve-base <base>`.
4. **Run model** — `ollama run <name>`.

See [Retrain Pipeline](Retrain-Pipeline) for adapter → Ollama. Adapter can be a directory (PEFT or a single .bin/.gguf from llama.cpp) or a .bin/.gguf file path.

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

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **finetune not found** | You need llama.cpp's `finetune` on PATH to run training in one command. Run `ollama-forge setup-llama-cpp`, then add the build directory to your PATH. Or run the training steps manually (prepare → run finetune yourself → retrain). |
| **Adapter directory invalid** | `retrain --adapter` accepts: (1) a **directory** with PEFT files (`adapter_config.json` + `adapter_model.safetensors` or `.bin`), or (2) a directory with **exactly one** `.bin` or `.gguf` file (llama.cpp output), or (3) a **file** path to a `.bin` or `.gguf` adapter. Pass the directory or file path directly. |
| **No .jsonl files found** | Pass a **directory** that contains `.jsonl` files, or list the files explicitly: `--data file1.jsonl file2.jsonl`. Use `train-data init -o ./data` to create a sample directory. |
| **Need a base GGUF** | Run `ollama-forge train-resolve-base <base_model_name>` (e.g. `llama3.2`) for suggestions on where to get a GGUF for `--base-gguf`. |
