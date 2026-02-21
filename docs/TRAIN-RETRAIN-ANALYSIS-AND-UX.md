# Train and Retrain: Implementation Analysis and UX Suggestions

## 1. Implementation overview

### 1.1 Flow

| Step | Command / component | What it does |
|------|---------------------|--------------|
| Validate | `validate-training-data <path>` | Checks JSONL lines for required fields (Alpaca or messages). |
| Prepare | `prepare-training-data <path> -o <file> --format llama.cpp` | Converts JSONL → plain text with `### Instruction` / `### Input` / `### Response` blocks. |
| Train (external) | llama.cpp `finetune` (or Axolotl/Unsloth) | Consumes prepared text + base GGUF; outputs adapter (e.g. LoRA). |
| Retrain | `retrain --base <name> --adapter <path> --name <name>` | Builds Modelfile `FROM base` + `ADAPTER path`, runs `ollama create`. |

**Two entry points:**

- **`train`** — Generates a **bash script** that runs: validate → prepare → (optional) finetune → prints the retrain command. User runs the script manually. Optional `--base-gguf` + `--run-trainer` makes the script invoke `finetune` if on PATH.
- **`train-run`** — Runs the **pipeline in-process**: validate → prepare → (if `--base-gguf` and `finetune` on PATH) run finetune → then `create-from-base` (retrain). Single command, no script.

### 1.2 Data format (how data is supposed to be)

**Input (JSONL):**

- **Alpaca-style (preferred for docs):**  
  `instruction` (required), `output` (required), `input` (optional).  
  Example: `{"instruction": "Summarize.", "input": "Long text...", "output": "Short."}`
- **Messages-style (e.g. TeichAI/datagen):**  
  `messages`: array of `{role: "user"|"assistant"|"system", content: "..."}`.  
  Normalized internally to instruction (= last user), input (= system parts), output (= last assistant).

**Validation** (`training_data.validate_line`): Accepts either schema; rejects unknown keys only for Alpaca; for messages, requires at least one user and one assistant with non-empty content.

**Prepare (llama.cpp format):** Each record becomes one block:

```
### Instruction:
<instruction>

### Input:
<input or "(none)">

### Response:
<output>
```

Blocks are concatenated with `\n` (no separator between blocks). llama.cpp `finetune` is called with `--sample-start '### Instruction'` so it splits samples by that prefix.

**No other formats are implemented.** Axolotl/Unsloth are mentioned in docs but no `--format axolotl` or similar; only `llama.cpp` is wired.

### 1.3 Adapter format and mismatch

- **Retrain / create-from-base** expects an **adapter directory** and uses `_verify_adapter_and_base()`:
  - Directory must exist.
  - Must contain **either** `adapter_config.json` **or** at least one of `adapter_model.safetensors` / `adapter_model.bin` (PEFT-style).
- **llama.cpp finetune** typically writes a **single file** (e.g. `--lora-out adapter_out` may create `adapter_out` as a file or a directory with one `.bin`). That is **not** PEFT layout.
- **Ollama** Modelfile `ADAPTER` accepts:
  - A **path to a GGUF adapter file**, or
  - A **directory** containing Safetensors adapter files.

So: if the user runs `train-run` with llama.cpp, the resulting adapter may be a single `.bin` (or one file in a directory). The current check (config + safetensors/bin in a PEFT layout) can **fail** for raw llama.cpp output, and the UX doesn’t explain that llama.cpp output might need to be used as a file path (e.g. `ADAPTER /path/to/lora.gguf`) or converted.

---

## 2. Abstracting “llama.cpp as adapter” for simpler UX

### 2.1 Adapter source abstraction

Introduce a small **adapter source** notion so that “llama.cpp” is a first-class path:

- **PEFT directory** — current behavior: directory with `adapter_config.json` and/or `adapter_model.safetensors` / `adapter_model.bin`.
- **llama.cpp output** — directory that may contain a single LoRA file (e.g. `ggml-adapter-f32.bin`, `lora.bin`, or a `.gguf` adapter), or a path to a single file.

**Implementation sketch:**

- In `_verify_adapter_and_base()` (or a shared helper), if the path is a **file** (e.g. `.bin`, `.gguf`), accept it as a valid adapter (Ollama can take a file path).
- If the path is a **directory**:
  - First check PEFT (current logic).
  - Else check for “llama.cpp style”: a single known file (e.g. `*.bin`, `*.gguf`) in that directory. If exactly one, treat the **file** as the adapter path for `build_modelfile(..., adapter=file_path)`.
- In `train` / `train-run`, when `--adapter-output` is used and the trainer is llama.cpp, document (or auto-set) that the adapter may be a single file; after finetune, **resolve** the adapter to either the directory or the single file and pass that to retrain.

This way, when the user chooses “llama.cpp” they don’t have to know about PEFT; the tool accepts the directory (or file) that llama.cpp produces.

### 2.2 Trainer backend abstraction

- Add a **`--trainer`** (or `--backend`) option: `llama.cpp` (default), and later `axolotl` / `unsloth` if needed.
- When `--trainer llama.cpp`:
  - **Prepare:** keep current `--format llama.cpp` (or make it default and implicit).
  - **Invoke:** `finetune --train-data <prepared> --sample-start '### Instruction' --model-base <base_gguf> --lora-out <adapter_out>` (current behavior).
  - **Post-step:** after training, **resolve adapter** as above (directory with one .bin/.gguf → use that file for ADAPTER; or use directory if it becomes PEFT-like).
- Script generation (`train --write-script`) and `train-run` both use this backend so the UX is “I use llama.cpp” vs “I use something else” instead of raw commands.

### 2.3 Single “finetune” command (optional)

- One command that means “data + base + name + optional base GGUF” and does:
  - validate → prepare (for the chosen trainer) → run trainer if possible → retrain.
- Example:  
  `ollama-forge finetune --data ./data/ --base llama3.2 --name my-model [--base-gguf /path/to/base.gguf]`  
  with default trainer `llama.cpp`. This is essentially `train-run` with a clearer name and sensible defaults (e.g. `--format llama.cpp`, `--prepared-output`, `--adapter-output` under a single `--workdir`).

---

## 3. Data format: explicit contract and UX

### 3.1 Document the single “canonical” input

- **Canonical:** JSONL with Alpaca-style `instruction` / `output` / `input` **or** messages-style `messages`. Everything else (e.g. other trainers) is converted from this.
- In CLI help and docs, state: “Training data: one JSON per line with `instruction` + `output`, or `messages` (user/assistant). Use `validate-training-data` to check.”

### 3.2 Accept directory or list of files everywhere

- Already: `--data` accepts file(s) or directory; `get_jsonl_paths_or_exit` expands directories to `*.jsonl`. Keep this and document it (e.g. “Pass a directory to use all .jsonl inside”).

### 3.3 Optional: “data init” for greenfield users

- `ollama-forge train-data init --out ./data/` could create:
  - `./data/README.md` (field names, minimal example).
  - `./data/sample.jsonl` with 2–3 example lines (Alpaca + optional messages).
- Reduces “what do I put in the file?” friction.

---

## 4. UX suggestions (efficient and easy)

### 4.1 One obvious path for “I want to finetune and run in Ollama”

- **Primary path:** `train-run --data <path> --base <name> --name <name> [--base-gguf <gguf>]` with defaults:
  - `--format llama.cpp` (default).
  - `--prepared-output train_prepared.txt`, `--adapter-output adapter_out` (or under a single `--workdir`).
- If `--base-gguf` is omitted: after prepare, print exactly what to do next (run finetune with which args, then run retrain with which args). No script required unless user asks for it.
- If `--base-gguf` is set and `finetune` is on PATH: run finetune then retrain automatically and print `ollama run <name>`.

### 4.2 “train” vs “train-run” naming

- **Current:** `train` = generate script; `train-run` = run pipeline. Many users will expect “train” to actually train.
- **Suggestion:** Rename or alias so the **default** is “run”:
  - Option A: `train` runs the pipeline; `train --write-script only` or `train-script` generates the script.
  - Option B: Keep both but in docs and `--help` lead with “Use `train-run` to do everything in one go; use `train` to generate a script.”

### 4.3 Auto-detect and guide (doctor / setup)

- **`doctor`** already checks for `finetune` on PATH. Extend:
  - If user has `--data` and `--base-gguf` but no `finetune`: “To run training in one command, install llama.cpp and add finetune to PATH. See: …” with link or `ollama-forge setup-llama-cpp`.
- After **prepare**, print one line: “Prepared N samples. Next: run your trainer (e.g. llama.cpp finetune) or use train-run with --base-gguf to run it automatically.”

### 4.4 Base GGUF discovery

- **Friction:** User has a base model name (e.g. `llama3.2`) but finetune needs a **GGUF file**. Today they must find it themselves.
- **Suggestions:**
  - Document: “Use a GGUF of the same model (e.g. from Hugging Face or convert from HF).”
  - Optional: `ollama-forge train-resolve-base llama3.2` (or similar) that:
    - Checks if Ollama has the model and suggests using its blob/cache path if it’s a single GGUF, **or**
    - Prints a one-liner to download a known GGUF (e.g. via `hf_repo` + optional quant) so they can pass it as `--base-gguf`.  
  This stays optional so the CLI doesn’t depend on HF in the hot path.

### 4.5 Progress and feedback

- **Validate:** Already reports “OK: N valid line(s)”. If invalid, list first few errors and “Run validate-training-data for full list.”
- **Prepare:** Already reports “Wrote … (X bytes)”. Add: “N samples” so user sees sample count.
- **Finetune (subprocess):** Don’t buffer stdout/stderr so llama.cpp progress is visible. If possible, add a one-line prefix like “Running finetune (llama.cpp)…” so it’s clear which step is running.
- **Retrain:** After `ollama create`, print: “Created model '<name>'. Run: ollama run <name>.”

### 4.6 Script generation

- When generating the script (`train --write-script`):
  - Use **single quoted** `'### Instruction'` in the script so it works in bash.
  - If `--base-gguf` is set, use a variable (e.g. `BASE_GGUF`) and set it in the script so the user can edit one place.
  - Add a short comment at the top: “Data: … Base: … Name: … Adapter output: …” so the script is self-describing.

### 4.7 Retrain UX

- **Retrain** is already simple: `retrain --base X --adapter Y --name Z`. Improvements:
  - If adapter path is a **file** (e.g. `lora.gguf`), accept it (see adapter abstraction above).
  - In `_verify_adapter_and_base`, if verification fails, suggest: “If this is a llama.cpp LoRA, pass the .bin or .gguf file path, or the directory containing it.”
  - Optional: `retrain --base X --adapter Y --name Z --system "..."` and other Modelfile options are already there; document them in `retrain --help`.

### 4.8 Docs and discoverability

- **Quick start:** One “Training” section: (1) Put data in JSONL (instruction/input/output or messages). (2) `validate-training-data <path>`. (3) `train-run --data <path> --base <base> --name <name>`; if you have a base GGUF and finetune on PATH, add `--base-gguf <path>` to run training and retrain in one go. (4) `ollama run <name>`.
- **Command reference:** List `validate-training-data`, `prepare-training-data`, `train`, `train-run`, `retrain` in one block with one-line descriptions and “See Training data / Retrain pipeline” links.
- **Troubleshooting:** “finetune not found” → setup-llama-cpp / PATH; “adapter dir invalid” → PEFT vs llama.cpp output and how to pass file path; “no .jsonl” → use a directory of .jsonl or pass files explicitly.

---

## 5. Summary

| Topic | Current state | Recommendation |
|-------|----------------|-----------------|
| **Data format** | JSONL Alpaca or messages; prepare → llama.cpp blocks. | Document as single canonical input; optional `train-data init` for samples. |
| **Adapter format** | PEFT dir (config + safetensors/bin). llama.cpp output not accepted as-is. | Accept file path or dir with single .bin/.gguf; resolve llama.cpp output for retrain. |
| **Trainer** | Only llama.cpp wired (prepare + finetune args). | Add `--trainer llama.cpp` (default) and adapter resolution so “llama.cpp” is a clear, simple path. |
| **Entry points** | `train` (script) vs `train-run` (run). | Lead with `train-run` as the one-command path; alias or rename so “train” doesn’t only mean “print script”. |
| **Base GGUF** | User must provide path. | Document; optional helper to suggest or download a GGUF for a base name. |
| **Progress** | Some messages; finetune can be silent. | Don’t buffer finetune output; add short step labels and “N samples” where useful. |
| **Script** | Generated with variables. | Keep; add self-describing comment and correct quoting. |
| **Retrain** | Simple; adapter check strict. | Accept file adapter; clearer error when adapter is llama.cpp output. |

Implementing the **adapter abstraction** (file or dir with single LoRA file) and **trainer=llama.cpp** resolution will make the “use llama.cpp and get an Ollama model” path straightforward without changing the underlying data format or prepare step.
