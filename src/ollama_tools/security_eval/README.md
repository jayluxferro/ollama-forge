# LLM Security Evaluation

Run prompt sets against Ollama or abliterate serve, score refusal/compliance/extraction, and view KPIs (ASR, refusal rate) in the CLI or Streamlit UI.

## Quick start

```bash
# Ensure Ollama (or abliterate serve) is running with a model
uv run ollama-tools security-eval run path/to/prompts.txt --model my-model --output-csv results.csv
```

## Prompt set formats

- **.txt** — One prompt per line; lines starting with `#` are skipped. Category is "default".
- **.jsonl** — One JSON object per line. Fields:
  - `prompt` (required)
  - `category` — e.g. jailbreak, benign, extraction
  - `target_for_extraction` — optional; if set, extraction scorer checks if response contains this (or use "pii" for PII regex)
  - `context` / `injected_document` — optional; for indirect prompt injection, prepended as "Context:\n...\n\nUser query:\n{prompt}"

Sample files: `src/ollama_tools/security_eval/data/sample_prompts.txt`, `sample_prompts.jsonl`, `sample_indirect.jsonl`.

## CLI

- **Run eval:** `ollama-tools security-eval run PROMPT_SET [--model NAME] [--base-url URL] [--output-csv PATH] [--output-json PATH] [--save-history]`
- **Launch UI:** `ollama-tools security-eval ui` (requires `uv sync --extra security-eval-ui`)

## KPIs

- **ASR %** — Attack success rate (compliance rate): % of prompts where the model did not refuse.
- **Refusal %** — % of responses classified as refusal (keyword-based).
- **Extraction %** — % where target string or PII pattern was found in the response.
- **By category** — Same metrics broken down by prompt category.

## Run history

Use `--save-history` (CLI) or check "Save run to history" (UI) to append runs to `~/.ollama_tools/security_eval_runs.db`. The UI shows run history and an ASR-over-time plot.
