# TeichAI/datagen — relevance for ollama-forge

**Source:** [TeichAI/datagen](https://github.com/TeichAI/datagen) — CLI to generate JSONL datasets from a TXT file using LLMs (OpenRouter by default).

## Summary

**Relevant as an upstream data source** for the training pipeline. datagen produces JSONL from prompts + LLM; ollama-forge consumes JSONL for validate → prepare → train-run/retrain. ollama-forge **accepts both** Alpaca-style and datagen’s `messages` format (validate/prepare); a converter command writes Alpaca-style JSONL to a file when needed.

## What datagen does

- **Input:** TXT file, one prompt per line (optionally `--system`).
- **Process:** Calls OpenRouter (or custom API) for each line; gets assistant response.
- **Output:** JSONL where each line is:
  ```json
  { "messages": [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }
  ```
  or with system (if `--store-system true`):
  ```json
  { "messages": [ {"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }
  ```
- **Features:** Concurrency, YAML config, OpenRouter provider/sort, optional reasoning, progress bar.

## What ollama-forge expects (training data)

- **Format:** Alpaca-style JSONL — `instruction` (required), `output` (required), `input` (optional).
- **Example:** `{"instruction": "Summarize the following.", "input": "Long doc...", "output": "Short summary."}`
- **Pipeline:** `validate-training-data` → `prepare-training-data` → (finetune) → `retrain` / `train-run`.

## Fit

| Aspect | Notes |
|--------|--------|
| **Use case** | datagen = “prompts → LLM → JSONL”; ollama-forge = “JSONL → validate → prepare → train → Ollama”. So datagen can **generate** the JSONL that ollama-forge then uses. |
| **Format** | datagen outputs `messages`; ollama-forge accepts both in validate/prepare and provides `convert-training-data-format` to write Alpaca JSONL. |
| **Ecosystem** | datagen is Node/OpenRouter; ollama-forge is Python/Ollama. No code dependency; they integrate via files (TXT → datagen → JSONL → ollama-forge). |

## Implementation (done)

- **Validate & prepare** accept **messages** format: each line may be `{ "messages": [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }`. Internally normalized to Alpaca for prepare/train.
- **Converter command:** `ollama-forge convert-training-data-format <input.jsonl> -o <output.jsonl>` reads JSONL (Alpaca or messages) and writes Alpaca-style JSONL.
- **Pipeline:** datagen → `dataset.jsonl` → `validate-training-data` → `prepare-training-data` (or `train-run`) works without a separate conversion step.

## Options (superseded by implementation above)

1. **Document only**  
   In [Training Data](wiki/Training-Data.md), add a short “Generating training data” subsection: mention datagen (and similar tools), note the format difference, and give a conversion example (e.g. jq or a one-off script).

2. **Support `messages` in ollama-forge**  
   In `training_data.py`: when validating or converting, if an object has `messages` but not `instruction`/`output`, map the last user message to `instruction`, last assistant message to `output`, and optionally system to `input` (or leave `input` empty). Then existing validate/prepare/train-run flow works on datagen output with no extra step.

3. **Small converter script**  
   Provide a script (or doc snippet) that reads datagen JSONL and writes Alpaca-style JSONL (e.g. `instruction` = user content, `output` = assistant content).

**Recommendation:** (1) plus (2) if you want zero-friction use of datagen output; otherwise (1) plus (3) is enough.
