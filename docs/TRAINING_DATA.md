# Training data format for fine-tuning

A standard instruction format for preparing data that common trainers (llama.cpp, Axolotl, Unsloth) can consume or convert from.

---

## JSONL instruction format (recommended)

Use one JSON object per line (JSONL). Each line is an example with optional fields:

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | Yes | The task or question. |
| `input` | No | Additional context (can be empty string). |
| `output` | Yes | The desired response. |

**Example (Alpaca-style):**

```jsonl
{"instruction": "Summarize the following.", "input": "Long document...", "output": "Short summary."}
{"instruction": "What is 2+2?", "input": "", "output": "4."}
```

**Minimal (instruction + output only):**

```jsonl
{"instruction": "Say hello.", "output": "Hello!"}
```

Tools can derive a single prompt string, e.g. `instruction` + `input` (if present), with `output` as the target for the model.

---

## Conversion for trainers

- **llama.cpp finetune** — Expects plain text. Convert JSONL to a text file (e.g. one example per block, with a separator like `\n\n` or a custom prefix). Your pipeline can script this.
- **Axolotl / Unsloth** — Often accept Alpaca/instruction JSONL or similar; check their docs for the exact schema (e.g. `instruction`, `response`, or `messages`).
- **ollama-tools** — Use `ollama-tools validate-training-data <file.jsonl>` (if implemented) to check format before training.

---

## Creating JSONL from other sources

- Export from a spreadsheet: one row → one JSON object; columns → `instruction`, `input`, `output`.
- From chat logs: map user turns to `instruction` or `input`, assistant turns to `output`.
- Keep encoding UTF-8 and escape newlines in strings if required by your trainer.
