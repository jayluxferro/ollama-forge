# Modelfile (Ollama)

Ollama uses a **Modelfile** to define a model. You can write one by hand or let ollama-forge generate it (e.g. via `convert`, `fetch`, `create-from-base`, `build`).

---

## Basics

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

Then:

```bash
ollama create my-assistant -f Modelfile
ollama run my-assistant
```

---

## Refreshing the chat template

Older or custom models can use a chat template that breaks with the Chat API or tools (e.g. empty message slices). You can fix this **without changing the model's weights** by taking the template from a newer model.

**Keep your old model, fix only the template (recommended):**

```bash
# model-a = your old model, model-b = any model with a working template (e.g. same family, newer)
uv run ollama-forge refresh-template --name model-a --base model-b --template-only
```

This keeps **model-a's weights** (FROM is unchanged) and only replaces the **TEMPLATE** with model-b's. No tweaks; tools and Chat API should work. Use `--output-name model-a-fixed` if you want to keep the original and create a new tag. Safe for abliterated or otherwise modified models — only the prompt template is updated, not the weights.

**Recreate from a base (replace weights and template):**

If you instead want to rebuild from a named base (same weights as the base, latest template):

```bash
uv run ollama-forge refresh-template --name my-model --base llama3.2
```

Pull the base first if needed: `ollama pull model-b` or `ollama pull llama3.2`. Your model's SYSTEM, PARAMETER, and ADAPTER are kept; with `--template-only` only **TEMPLATE** is updated; without it **FROM** and **TEMPLATE** are updated.

---

## Reference

Full syntax and options: [Ollama Modelfile](https://docs.ollama.com/modelfile).

Adapter usage: [Ollama Modelfile — ADAPTER](https://docs.ollama.com/modelfile#adapter).
