# Example recipes

Copy any file here and edit it for your model. Then build:

```bash
uv sync
uv run ollama-forge build your-recipe.yaml
ollama run <name-from-recipe>
```

- **simple-base.yaml** — Custom model from an existing Ollama base (system prompt, temperature).
- **from-gguf.yaml** — Model from a GGUF file (set `gguf` to your file path).
- **from-hf.yaml** — Model from a Hugging Face repo (set `hf_repo`; optional `gguf_file` if multiple).

Recipe format: `name`, and one of `base`, `gguf`, or `hf_repo`. Optional: `system`, `temperature`, `num_ctx`, `adapter` (with base), `gguf_file` and `revision` (with hf_repo).
