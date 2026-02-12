# Hugging Face → Ollama

Convert a Hugging Face model to run in Ollama. Pipeline: **HF model → GGUF → Modelfile → `ollama create`**.

Ollama runs models in **GGUF** format. Hugging Face hosts models in PyTorch/Safetensors, so conversion is required. The standard path uses [llama.cpp](https://github.com/ggerganov/llama.cpp) conversion scripts.

---

## Prerequisites

- **Ollama** — [install](https://ollama.com)
- **Python 3.10+** and **uv** (for `ollama-tools` CLI)
- **llama.cpp** — for converting HF → GGUF (clone and build, or use their Python convert scripts in a venv)
- **Hugging Face** — `huggingface-hub` and `transformers` if using Python-based conversion

---

## Option A: Pre-converted GGUF from Hugging Face

Many models on the Hub already provide GGUF files (e.g. in a `gguf` or repo file list).

1. Download the `.gguf` file (e.g. via [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli) or the repo page).
2. Create a Modelfile that points at it:

   ```modelfile
   FROM /path/to/downloaded/model.gguf
   ```

3. Create and run:

   ```bash
   ollama create my-model -f Modelfile
   ollama run my-model
   ```

---

## Option B: Convert HF model to GGUF with llama.cpp

For a model that exists only in PyTorch/Safetensors on the Hub:

1. **Clone and set up llama.cpp** (for conversion scripts and build):

   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   pip install -r requirements.txt   # for convert-*.py scripts
   ```

2. **Download the Hugging Face model** (e.g. into `./models/hf`):

   ```bash
   huggingface-cli download <org>/<model-name> --local-dir ./models/hf/<model-name>
   ```

3. **Convert to GGUF** — llama.cpp provides model-specific scripts. For Llama/Mistral-style:

   ```bash
   python convert-hf-to-gguf.py ./models/hf/<model-name> --outfile ./models/gguf/<model-name>.gguf
   ```

   Check the [llama.cpp repo](https://github.com/ggerganov/llama.cpp#convert-pytorch-models-to-gguf) for the exact script name for your architecture (e.g. `convert-hf-to-gguf.py`, `convert.py`).

4. **(Optional) Quantize** to reduce size and VRAM:

   ```bash
   ./quantize ./models/gguf/<model-name>.gguf ./models/gguf/<model-name>-Q4_K_M.gguf Q4_K_M
   ```

5. **Create Ollama model** from the GGUF:

   ```modelfile
   FROM /path/to/models/gguf/<model-name>-Q4_K_M.gguf
   ```

   Then:

   ```bash
   ollama create my-model -f Modelfile
   ```

---

## Option C: Using ollama-tools convert (GGUF → Ollama)

After you have a GGUF file (e.g. from Option B), create the Ollama model in one step:

```bash
ollama-tools convert --gguf /path/to/model.gguf --name my-model
```

Optional: `--system`, `--temperature`, `--num-ctx`, `--out-modelfile`. Full HF → GGUF conversion still requires llama.cpp; this command handles the **GGUF → Ollama** step.

---

## Architecture support

Conversion scripts in llama.cpp are often model-family specific (Llama, Mistral, Qwen, etc.). Check the [llama.cpp README](https://github.com/ggerganov/llama.cpp) and `convert-*.py` scripts for your model. If no script exists, you may need to add support or use a third-party converter.

---

## Summary

| Step              | Tool / action                          |
|-------------------|----------------------------------------|
| Get HF model      | `huggingface-cli download` or Hub UI   |
| HF → GGUF         | llama.cpp `convert-hf-to-gguf.py` etc. |
| Optional quantize | llama.cpp `quantize`                   |
| GGUF → Ollama     | Modelfile `FROM path/to/model.gguf` + `ollama create` |

For the GGUF → Ollama step, use: `ollama-tools convert --gguf <path> --name <name>`.
