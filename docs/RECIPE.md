# Model recipe (config-driven build)

Build an Ollama model from a **recipe file** (YAML or JSON) with a single command: `ollama-tools build <recipe.yaml>`.

---

## Recipe format

Required:

- **name** — Name for the new Ollama model.
- Exactly one of:
  - **base** — Create-from-base (existing Ollama model or path).
  - **gguf** — Convert from a local .gguf file path.
  - **hf_repo** — Download a GGUF from this Hugging Face repo and create the model (same as `fetch`).

Optional (same as create-from-base / convert / fetch):

- **system** — System message.
- **temperature** — Float (e.g. 0.7).
- **num_ctx** — Context window size in tokens.
- **top_p** — Float (e.g. 0.9).
- **repeat_penalty** — Float (e.g. 1.1).
- **adapter** — Path to adapter (only with **base**).
- **quantize** — Quantize GGUF before create (only with **gguf**; e.g. `Q4_K_M`).
- **gguf_file** — Specific .gguf filename (only with **hf_repo**; use when repo has multiple).
- **quant** — Preferred quantization file when HF repo has multiple GGUFs (only with **hf_repo**; e.g. `Q4_K_M`).
- **revision** — HF repo revision (only with **hf_repo**; default: main).

---

## Example: create-from-base

**recipe.yaml:**

```yaml
name: my-assistant
base: llama3.2
system: You are a concise coding assistant.
temperature: 0.7
num_ctx: 4096
top_p: 0.9
repeat_penalty: 1.1
```

Build:

```bash
ollama-tools build recipe.yaml
```

---

## Example: base + adapter

```yaml
name: my-finetuned
base: llama3.2
adapter: /path/to/adapter
system: You are a domain expert.
```

---

## Example: from GGUF (convert)

```yaml
name: my-converted
gguf: /path/to/model.gguf
quantize: Q4_K_M
temperature: 0.6
num_ctx: 8192
```

---

## Example: from Hugging Face (fetch)

```yaml
name: my-model
hf_repo: TheBloke/Llama-2-7B-GGUF
# optional: gguf_file: model-Q4_K_M.gguf
# optional: quant: Q4_K_M
# optional: revision: main
temperature: 0.6
num_ctx: 8192
```

---

## JSON format

Same keys; use a `.json` file:

```json
{
  "name": "my-assistant",
  "base": "llama3.2",
  "system": "You are helpful.",
  "temperature": 0.7
}
```

---

## YAML and JSON

Both YAML and JSON recipes are supported with the default install (PyYAML is included).

---

## CLI

```bash
ollama-tools build recipe.yaml
ollama-tools build recipe.yaml --out-modelfile Modelfile.generated
```

The recipe must contain exactly one of **base**, **gguf**, or **hf_repo**.
