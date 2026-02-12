# Modelfile (Ollama)

Ollama uses a **Modelfile** to define a model. You can write one by hand or let ollama-tools generate it (e.g. via `convert`, `fetch`, `create-from-base`, `build`).

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

## Reference

Full syntax and options: [Ollama Modelfile](https://docs.ollama.com/modelfile).

Adapter usage: [Ollama Modelfile — ADAPTER](https://docs.ollama.com/modelfile#adapter).
