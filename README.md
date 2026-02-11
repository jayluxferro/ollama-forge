# ollama-tools

**Get models from Hugging Face, convert them, add adapters, and run them in [Ollama](https://ollama.com) — in a simple way, without needing deep expertise.**  
All the tools live here: fetch, convert, adapters, recipes. You run a few commands and get a model you can use locally.

**Quick start:** `uv sync && uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model && ollama run my-model`

---

## Why this project

- **One place** — Fetch from Hugging Face, convert to GGUF, use adapters, customize with recipes; no need to hunt down scattered scripts.
- **Simple** — Clear commands and docs so you can try things without being an ML expert.
- **Local-first** — Everything is aimed at getting models running in Ollama on your machine.

---

## Setup (one-time)

- Install **Python 3.10+** and **[uv](https://docs.astral.sh/uv/)** (or pip).
- **From this repo:** `uv sync` then `uv run ollama-tools` (or `uv run ollama-tools --help`).
- **Run from anywhere:** From the repo run `uv tool install .` so `ollama-tools` is on your PATH; or `pip install -e .` for a development install.
- Install **[Ollama](https://ollama.com)** and ensure `ollama` is on your PATH.
- Run **`uv run ollama-tools check`** to see what’s installed (ollama, Hugging Face, optional deps, llama.cpp). Use **`uv run ollama-tools setup-llama-cpp`** to clone and build llama.cpp so you can use `finetune` and `quantize`.

---

## Commands at a glance

| What you want | Command |
|---------------|---------|
| Get a GGUF from Hugging Face and create a model | `fetch <repo_id> --name <name>` (use `--quant Q4_K_M` to pick size) |
| Turn a GGUF file into an Ollama model | `convert --gguf <path> --name <name>` (use `--quantize Q4_K_M` to shrink first) |
| Find adapters on Hugging Face | `adapters search "llama lora"` |
| Get an adapter from HF and create a model | `fetch-adapter <repo_id> --base <base> --name <name>` |
| Customize a model (prompt, params, adapter) | `create-from-base`, `retrain`, or `build recipe.yaml` |
| Validate training data (JSONL) | `validate-training-data <file(s) or dir>` |
| Convert JSONL → trainer format | `prepare-training-data <file(s) or dir> -o out.txt` |
| Generate training pipeline script | `train --data <file(s) or dir> --base <base> --name <name> --write-script train.sh` |
| Check environment (ollama, HF, llama.cpp) | `check` |
| Install llama.cpp (clone + build) | `setup-llama-cpp [--dir ./llama.cpp]` |
| Refusal removal (abliterate) | `abliterate compute-dir` (optional: use built-in harmful/harmless lists) |
| Downsize (e.g. 30B→3B) | `downsize --teacher <hf_id> --student <hf_id> --name <name> [--quantize Q4_K_M]` |
| One-file config build | `build recipe.yaml` |

---

## Simplest workflows

**Get a model from Hugging Face (one command):**
```bash
uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model
ollama run my-model
```
Use `--quant Q4_K_M` (or `Q8_0`, etc.) to pick that size when the repo has many GGUF files. For gated or private models, set `HF_TOKEN` or run `huggingface-cli login`.

**Use a GGUF file you already have (optionally quantize to save VRAM):**
```bash
uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model
# Or shrink first (requires llama.cpp 'quantize' on PATH):
uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model --quantize Q4_K_M
ollama run my-model
```

**Create a custom model from an existing Ollama model (system prompt, temperature):**
```bash
uv run ollama-tools create-from-base --base llama3.2 --name my-assistant --system "You are helpful and concise."
ollama run my-assistant
```

**Use a recipe file (one file, one command):**
```bash
uv run ollama-tools build recipe.yaml
ollama run <name-from-recipe>
```

**Use an adapter from Hugging Face:**
```bash
uv run ollama-tools fetch-adapter username/adapter-repo --base llama3.2 --name my-finetuned
ollama run my-finetuned
```

**Use an adapter you have locally:**
```bash
uv run ollama-tools retrain --base llama3.2 --adapter /path/to/adapter --name my-finetuned
ollama run my-finetuned
```

---

## Recipe format

Build from a YAML or JSON file: `ollama-tools build recipe.yaml`.

- **Required:** `name` (Ollama model name), and **exactly one of:**
  - `base` — Existing Ollama model or path (create-from-base).
  - `gguf` — Path to a local .gguf file (convert).
  - `hf_repo` — Hugging Face repo id; downloads a GGUF and creates the model (same as `fetch`).
- **Optional:** `system`, `temperature`, `num_ctx`; with `base`: `adapter`; with `hf_repo`: `gguf_file`, `revision`.

**Example (from base):**
```yaml
name: my-assistant
base: llama3.2
system: You are a concise coding assistant.
temperature: 0.7
num_ctx: 4096
```

**Example (from Hugging Face):**
```yaml
name: my-model
hf_repo: TheBloke/Llama-2-7B-GGUF
temperature: 0.6
num_ctx: 8192
```

**Example (from GGUF path):**
```yaml
name: my-converted
gguf: /path/to/model.gguf
temperature: 0.6
```

**Example (base + adapter):**
```yaml
name: my-finetuned
base: llama3.2
adapter: /path/to/adapter
system: You are a domain expert.
```

Example recipe files are in `examples/recipes/` in this repo.

---

## Modelfile (Ollama)

Ollama uses a **Modelfile** to define a model. You can write one by hand or let ollama-tools generate it.

- **FROM** — Base model name (e.g. `llama3.2`), or path to a GGUF file, or directory with Safetensors.
- **PARAMETER** — e.g. `temperature 0.7`, `num_ctx 4096`, `top_p 0.9`, `repeat_penalty 1.1`.
- **SYSTEM** — System message (role, tone). Use triple quotes for multi-line.
- **ADAPTER** — Path to a LoRA/adapter directory (use with a base model).

Example:
```modelfile
FROM llama3.2
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM You are a concise coding assistant.
```

Then: `ollama create my-assistant -f Modelfile` and `ollama run my-assistant`.  
Reference: [Ollama Modelfile](https://docs.ollama.com/modelfile).

---

## Hugging Face → Ollama (when there’s no GGUF)

Ollama runs **GGUF** models. If the Hugging Face repo only has PyTorch/Safetensors:

1. **Download the model** — e.g. `huggingface-cli download <org>/<model> --local-dir ./models/<model>`.
2. **Convert to GGUF** — Use [llama.cpp](https://github.com/ggerganov/llama.cpp): e.g. `python convert-hf-to-gguf.py ./models/<model> --outfile ./models/<model>.gguf`. Use the script that matches your architecture (Llama, Mistral, Qwen, etc.).
3. **(Optional) Quantize** — e.g. `./quantize model.gguf model-Q4_K_M.gguf Q4_K_M` to reduce size/VRAM.
4. **Create Ollama model** — `ollama-tools convert --gguf /path/to/model.gguf --name my-model`.

For repos that already have GGUF files, use `ollama-tools fetch <repo_id> --name my-model` (see above).

---

## Adapters (LoRA) — what they are and where to get them

**What is an adapter?** A small add-on trained on top of a base model (e.g. Llama, Mistral) that changes how it behaves — for a style, a task, or a dataset. You use “base model + adapter” together in Ollama.

**Where to get one:**
- **From Hugging Face** — Many people share adapters. Search on [huggingface.co/models](https://huggingface.co/models) for e.g. “llama lora” or “mistral adapter”, or use the tool:
  ```bash
  uv run ollama-tools adapters search "llama 3 lora"
  ```
  This lists adapters and the exact `fetch-adapter` command for each. Pick a repo, then:
  ```bash
  uv run ollama-tools fetch-adapter <repo_id> --base llama3.2 --name my-finetuned
  ollama run my-finetuned
  ```
- **Train your own** — Use [llama.cpp finetune](https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), or [Unsloth](https://github.com/unslothai/unsloth). Then use the adapter locally:
  ```bash
  uv run ollama-tools retrain --base llama3.2 --adapter /path/to/adapter --name my-finetuned
  ```

**Details:** The HF repo (or your folder) must be in a format Ollama accepts (e.g. PEFT with `adapter_model.safetensors`). Prefer full LoRA (non-quantized). The base model must match the adapter (same architecture). See [Ollama Modelfile — ADAPTER](https://docs.ollama.com/modelfile#adapter) if something fails.

---

## Training data: passing a lot of data

Use **JSONL**: one JSON object per line. Fields: `instruction` (required), `output` (required), `input` (optional).

**You can pass one file, several files, or a whole directory:**

```bash
# One file
uv run ollama-tools validate-training-data train.jsonl

# Many files
uv run ollama-tools validate-training-data data/part1.jsonl data/part2.jsonl

# Whole directory (all .jsonl inside)
uv run ollama-tools validate-training-data ./data/
```

**Convert your JSONL to the format trainers expect** (e.g. llama.cpp wants plain text with blocks):

```bash
uv run ollama-tools prepare-training-data ./data/ -o train_prepared.txt --format llama.cpp
```

Then use `train_prepared.txt` with llama.cpp finetune (use `--sample-start '### Instruction'`).

**One command to get a full pipeline script** — pass your data path, base model, and output name; the tool writes a script that validates data, prepares it, and optionally runs the trainer if llama.cpp is installed:

```bash
uv run ollama-tools train --data ./data/ --base llama3.2 --name my-model --write-script train.sh
./train.sh
```

To have the script **run the trainer for you** (no manual llama.cpp step), pass a base GGUF and `--run-trainer`. You need llama.cpp’s `finetune` on PATH (see below):

```bash
uv run ollama-tools train --data ./data/ --base llama3.2 --name my-model --base-gguf /path/to/base.gguf --run-trainer --write-script train.sh
./train.sh
```

**Get llama.cpp (finetune, quantize) automatically:** Run `ollama-tools setup-llama-cpp` to clone and build llama.cpp, then add the build directory to your PATH. After that, `check` will report finetune/quantize as OK and the train script can run the trainer step.

**Verify your environment:** Run `ollama-tools check` to see what’s installed (ollama, Hugging Face, optional deps, llama.cpp).

---

## Retrain pipeline (data → adapter → Ollama)

1. **Validate data** — `validate-training-data <file(s) or directory>`.
2. **Optional: generate pipeline** — `train --data <path> --base <base> --name <name> --write-script train.sh` then run the script.
3. **Or manually:** `prepare-training-data` → run trainer (llama.cpp, Axolotl, Unsloth) → `retrain --base <base> --adapter <path> --name <name>`.
4. **Run** — `ollama run <name>`.

---

## Refusal removal (abliterate)

Strip refusal behavior by computing a “refusal direction” and applying ablation. Reference: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers).

**You don’t need to supply harmful/harmless lists yourself** — the tool ships with built-in default lists (instructions that typically elicit refusals vs neutral ones). You can still use your own:

- **Automatic (built-in lists):** Run without `--harmful`/`--harmless`; the tool uses bundled defaults.
- **Your own files:** `--harmful harmful.txt --harmless harmless.txt` (one instruction per line; lines starting with `#` are skipped).
- **Directories:** `--harmful-dir ./harmful/ --harmless-dir ./harmless/` to use all `.txt` files in those directories.

Example (optional deps: `uv sync --extra abliterate`):
```bash
# Use built-in lists (no files needed)
uv run ollama-tools abliterate compute-dir --model <hf_id> --output refusal.pt

# Or use your own
uv run ollama-tools abliterate compute-dir --model <hf_id> --harmful harmful.txt --harmless harmless.txt --output refusal.pt
```

Then use Sumandora’s `inference.py` (or your script) with the `.pt` file. If you get an abliterated checkpoint, convert to GGUF (llama.cpp) then `ollama-tools convert --gguf <path> --name my-abliterated`.

---

## Downsizing (e.g. 30B → 3B)

Produce a smaller model via **knowledge distillation** (teacher → student). You tell the tool which models and it gives you the exact steps.

**Simple usage:** Pass teacher and student (Hugging Face repo ids) and the name for your final model. The tool prints the commands to run (download → distill → convert). Optionally pass a quantization so the last step uses a smaller GGUF.

```bash
uv run ollama-tools downsize --teacher org/30b-model --student org/3b-model --name my-downsized --quantize Q4_K_M
```

To save the steps to a file and run them yourself:

```bash
uv run ollama-tools downsize --teacher org/30b --student org/3b --name my-downsized --quantize Q4_K_M --write-script downsize.sh
```

Distillation itself (training the student to mimic the teacher) is done with external tools (e.g. [TRL GKD](https://huggingface.co/docs/trl/main/en/gkd_trainer), Axolotl); the script tells you exactly what to run. The last step is always `ollama-tools convert` (with `--quantize` if you passed it).

---

## Quantization (smaller / faster GGUF)

- **When fetching from HF:** Use `fetch --quant Q4_K_M` (or `Q8_0`, etc.) to pick that variant when the repo has multiple GGUF files.
- **When you have a GGUF file:** Use `convert --quantize Q4_K_M` to shrink it before creating the Ollama model (requires llama.cpp `quantize` on PATH). Common types: **Q4_0** (smallest), **Q4_K_M** (good default), **Q5_K_M**, **Q8_0** (near full precision).
- For adapters, prefer non-quantized LoRA when possible.

---

## CI / automation (example)

Build from a recipe in CI (e.g. GitHub Actions). Install Ollama and ollama-tools, then:

```yaml
- name: Install Ollama
  run: |
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
    sleep 5
- name: Build model from recipe
  run: uv run ollama-tools build recipes/my-model.yaml
```

If the recipe uses `base: llama3.2`, pull it first (e.g. `ollama pull llama3.2`). If it uses `gguf: path/to/file.gguf`, ensure that file exists from a previous step or artifact.

---

## All commands

Run `uv run ollama-tools --help` for the full list. Quick start hint is shown there as well.
