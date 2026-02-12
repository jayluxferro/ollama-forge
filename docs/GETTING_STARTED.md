# Getting started — simple way, no deep expertise

This guide assumes you just want to **try things** without learning every detail. All tools are in this project; you run a few commands.

---

## 1. I want to use a model from Hugging Face in Ollama

**Easiest:** Use a model that already has a **GGUF** file on the Hub. One command:

```bash
uv sync
uv run ollama-tools fetch TheBloke/Llama-2-7B-GGUF --name my-model
ollama run my-model
```

If the repo has more than one `.gguf` file, one is chosen automatically (use `--gguf-file <filename>` to pick another). For gated or private models, set `HF_TOKEN` or run `huggingface-cli login`. Optional: `--revision main`, `--system "..."`, `--temperature 0.7`, `--num-ctx 4096`.

**If you already downloaded a GGUF** (e.g. from the model's Files tab):

```bash
uv run ollama-tools convert --gguf /path/to/model.gguf --name my-model
ollama run my-model
```

**If the model only has non-GGUF files:** Convert it to GGUF first (e.g. with [llama.cpp](https://github.com/ggerganov/llama.cpp)); then use `convert` as above.

---

## 2. I want to customize an existing Ollama model (different prompt, temperature)

You already have a model in Ollama (e.g. `llama3.2`). You want a **new** model that’s the same base but with a custom system prompt or settings.

```bash
uv run ollama-tools create-from-base --base llama3.2 --name my-assistant --system "You are a friendly coding assistant. Keep answers short."
ollama run my-assistant
```

Optional: add `--temperature 0.7` or `--num-ctx 4096` if you want. No training, no adapters — just a new “flavor” of the model.

---

## 3. I have (or will get) an adapter and want to use it in Ollama

**Adapter on Hugging Face:** Download and create in one go:

```bash
uv run ollama-tools fetch-adapter username/adapter-repo --base llama3.2 --name my-finetuned
ollama run my-finetuned
```

Optional: `--output /path/to/dir` to keep the adapter on disk; `--revision main`. The adapter repo must be in a format Ollama accepts (e.g. PEFT directory or llama.cpp adapter); see ADAPTER.md for details.

**Adapter already on your machine** (trained locally or from a friend):

```bash
uv run ollama-tools retrain --base llama3.2 --adapter /path/to/adapter --name my-finetuned
ollama run my-finetuned
```

No need to write a Modelfile yourself; the tool does it.

---

## 4. I want one config file to build a model (recipe)

You’d rather edit one file than remember flags. Use a **recipe** (YAML or JSON).

Create `my-recipe.yaml`:

```yaml
name: my-assistant
base: llama3.2
system: You are helpful and concise.
temperature: 0.7
```

Then:

```bash
uv run ollama-tools build my-recipe.yaml
ollama run my-assistant
```

Same idea works for a model from a GGUF file: put `gguf: /path/to/model.gguf` instead of `base` in the recipe. Example recipe files are in `examples/recipes/` in this repo.

---

## 5. I want to prepare training data for fine-tuning

You have a list of instruction/response pairs. You want to check they’re in the right format before training (with another tool).

```bash
uv run ollama-tools validate-training-data my-data.jsonl
```

Each line of `my-data.jsonl` should look like: `{"instruction": "...", "output": "..."}` (and optionally `"input": "..."`). The tool tells you if something’s wrong. Training itself is done elsewhere (e.g. llama.cpp, Axolotl); after you get an adapter, use `retrain` (step 3).

---

## 6. I want to try refusal removal (abliterate) or downsizing

- **Refusal removal:** Optional extra deps + one command to compute a “refusal direction”; the rest is in the docs. Start with `uv run ollama-tools abliterate --help`.
- **Downsizing (e.g. 30B → 3B):** Run `uv run ollama-tools downsize` to see the pipeline. The heavy work (distillation) is done by other tools; we document the steps and you use `convert` for the final student model.

---

## Where to look next

- **All commands:** `uv run ollama-tools --help`
- **Recipe format:** RECIPE.md
- **Hugging Face → Ollama (full path):** HF_TO_OLLAMA.md
- **Adapters / retraining:** ADAPTER.md, RETRAIN.md
- **Quantization (smaller/faster models):** QUANTIZATION.md

Everything is in this repo so you can do it in a simple way without extraordinary expertise.
