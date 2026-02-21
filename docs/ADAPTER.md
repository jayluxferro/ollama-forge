# Adapter (LoRA) ingestion

Use a LoRA or other adapter on top of a base model and create an Ollama model. No retraining in this step — you provide an existing adapter (e.g. from llama.cpp finetune, Axolotl, or Unsloth).

**Reference:** [Ollama Modelfile — ADAPTER](https://docs.ollama.com/modelfile#adapter)

---

## Modelfile

In a Modelfile, use `FROM` for the base model and `ADAPTER` for the path to the adapter directory:

```modelfile
FROM llama3.2
ADAPTER /path/to/adapter
PARAMETER temperature 0.7
```

Then create and run:

```bash
ollama create my-finetuned -f Modelfile
ollama run my-finetuned
```

---

## Adapter format

- **Path:** `ADAPTER` points to a **directory** containing the adapter files (e.g. `adapter_config.json` and `adapter_model.safetensors` for Hugging Face PEFT-style, or the format produced by your training tool).
- **Compatibility:** Prefer **full LoRA** (non-quantized). Ollama works best with non-quantized adapters; QLoRA may not behave correctly. See RETRAIN.md for the full retrain pipeline.
- **Base model:** The base model (e.g. `llama3.2`) must be compatible with the adapter (same architecture and usually same base checkpoint).

---

## Using ollama-tools

**fetch-adapter** downloads an adapter from Hugging Face and creates the model in one step. The HF repo must contain the adapter in a format Ollama accepts (e.g. PEFT-style directory with `adapter_model.safetensors`, or llama.cpp adapter output). If your repo is raw training output or another format, export to the expected layout first; see Ollama’s Modelfile docs.

The **create-from-base** command supports `--adapter` for a local path:

```bash
ollama-tools create-from-base --base llama3.2 --name my-finetuned --adapter /path/to/adapter
```

Optional: add `--system`, `--temperature`, `--num-ctx`, or `--out-modelfile` as needed.

---

## Getting an adapter

Adapters are produced by training, not by ollama-tools. Common options:

- **llama.cpp** — [finetune](https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune) (LoRA); output is typically a single file or directory you can point `ADAPTER` at if the format is supported.
- **Axolotl / Unsloth** — train LoRA; export in a format Ollama accepts (e.g. PEFT directory with `adapter_model.safetensors`).

Check Ollama’s docs for the exact adapter formats and layout they support.
