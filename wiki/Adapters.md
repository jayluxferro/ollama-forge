# Adapters (LoRA)

Use a LoRA or other adapter on top of a base model and create an Ollama model. No retraining in this step — you provide an existing adapter (from Hugging Face or local training).

---

## What is an adapter?

A small add-on trained on top of a base model (e.g. Llama, Mistral) that changes how it behaves — for a style, a task, or a dataset. You use **base model + adapter** together in Ollama.

---

## Where to get one

- **Hugging Face** — Many shared adapters. Search on [huggingface.co/models](https://huggingface.co/models) (e.g. "llama lora") or use the tool:
  ```bash
  uv run ollama-forge adapters search "llama 3 lora"
  ```
  Then use the suggested `fetch-adapter` command.
- **Train your own** — e.g. [llama.cpp finetune](https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [Unsloth](https://github.com/unslothai/unsloth). Then use the adapter locally with `retrain`.

---

## fetch-adapter: Hugging Face → Ollama

Download an adapter from Hugging Face and create the model in one step:

```bash
uv run ollama-forge fetch-adapter username/adapter-repo --base llama3.2 --name my-finetuned
ollama run my-finetuned
```

Options: `--output /path/to/dir` to keep the adapter on disk; `--revision main`. The HF repo must be in a format Ollama accepts (e.g. PEFT with `adapter_model.safetensors`).

---

## retrain: local adapter → Ollama

Use an adapter you already have on disk:

```bash
uv run ollama-forge retrain --base llama3.2 --adapter /path/to/adapter --name my-finetuned
ollama run my-finetuned
```

**Adapter format:** A directory (PEFT with `adapter_model.safetensors`, or a directory containing a single `.bin`/`.gguf` from llama.cpp finetune) or a single `.bin`/`.gguf` file. Use `--template-from <ollama_model>` to copy the chat template from an existing Ollama model (e.g. the base) for tool/function-calling support.

---

## adapters recommend

Get recommendations for a base model and optionally apply the top one:

```bash
# Show recommendations
uv run ollama-forge adapters recommend --base llama3.2

# Apply top recommendation
uv run ollama-forge adapters recommend --base llama3.2 --apply --name my-finetuned
ollama run my-finetuned

# Preview only (dry-run)
uv run ollama-forge adapters recommend --base llama3.2 --apply --plan
```

---

## Format and compatibility

- **Path:** Adapter must be a **directory** with the adapter files (e.g. PEFT: `adapter_config.json`, `adapter_model.safetensors`).
- **Compatibility:** Prefer **non-quantized** LoRA; QLoRA may not behave correctly with Ollama.
- **Base:** The base model must match the adapter (same architecture). See [Ollama Modelfile — ADAPTER](https://docs.ollama.com/modelfile#adapter) if something fails.
