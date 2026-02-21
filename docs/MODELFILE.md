# Modelfile: create model from base

Create a new Ollama model from an existing one by customizing parameters, system prompt, and (optionally) the prompt template. No training required.

**Reference:** [Ollama Modelfile](https://docs.ollama.com/modelfile)

---

## Quick example

```modelfile
FROM llama3.2
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM You are a concise coding assistant. Answer in short, clear steps.
```

Save as `Modelfile` (or `Modelfile.my-assistant`), then:

```bash
ollama create my-assistant -f Modelfile.my-assistant
ollama run my-assistant
```

---

## Instructions

| Instruction | Required | Description |
|-------------|----------|-------------|
| **FROM** | Yes | Base model. Can be a model name (`llama3.2`), a path to a GGUF file (`/path/to/model.gguf`), or a directory containing a Safetensors model. |
| **PARAMETER** | No | Runtime parameters. Can be repeated. |
| **SYSTEM** | No | System message (role, tone, boundaries). Use triple quotes for multi-line. |
| **TEMPLATE** | No | Full prompt template. Usually left default from base; override only if you know the model’s template format. |
| **ADAPTER** | No | Path to a LoRA/adapter to apply on top of the base. See [Adapter ingestion](ADAPTER.md). |

---

## Common parameters

| Parameter | Example | Meaning |
|-----------|---------|---------|
| `temperature` | `0.7` | Lower = more deterministic; higher = more varied. |
| `num_ctx` | `4096` | Context window size (tokens). |
| `top_p` | `0.9` | Nucleus sampling. |
| `repeat_penalty` | `1.1` | Reduces repetition. |
| `stop` | `"</s>"` | Stop generating when this string is produced. |

---

## Multi-line SYSTEM

```modelfile
FROM llama3.2
SYSTEM """
You are a helpful assistant for a scientific lab.
- Be precise and cite sources when possible.
- If unsure, say so clearly.
"""
PARAMETER temperature 0.3
```

---

## Creating the model

1. Save the Modelfile (e.g. `Modelfile` or `Modelfile.my-name`).
2. Run: `ollama create <model-name> -f <path-to-Modelfile>`.
3. Run: `ollama run <model-name>`.

To inspect an existing model’s Modelfile: `ollama show <model-name> --modelfile`.

---

## Next

- **Adapter (LoRA):** See [Adapter ingestion](ADAPTER.md) for base + adapter.
- **From GGUF/Hugging Face:** See [HF → Ollama](HF_TO_OLLAMA.md) for converting external models to Ollama.
