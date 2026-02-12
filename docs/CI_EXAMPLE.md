# CI / automation (optional)

Example of building an Ollama model from a recipe in CI (e.g. GitHub Actions). Assumes Ollama is installed in the runner and you have a recipe file (and any base model or GGUF available).

---

## GitHub Actions (example)

```yaml
name: Build model from recipe

on:
  workflow_dispatch:
  push:
    paths:
      - 'recipes/*.yaml'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Install Ollama
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          ollama serve &
          sleep 5

      - name: Install ollama-tools
        run: uv sync --extra recipe

      - name: Build model from recipe
        run: uv run ollama-tools build recipes/my-model.yaml
        env:
          # If recipe uses a base model, pull it first:
          # OLLAMA_MODELS: ...
```

---

## Local script

```bash
#!/usr/bin/env bash
# scripts/build-from-recipe.sh
set -e
cd "$(dirname "$0")/.."
uv sync --extra recipe
uv run ollama-tools build "$1"
```

Usage: `./scripts/build-from-recipe.sh recipes/my-model.yaml`.

---

## Notes

- **Base models:** If the recipe uses `base: llama3.2`, ensure that model is available (e.g. `ollama pull llama3.2` before build, or use a runner image that pre-pulls it).
- **GGUF in recipe:** If the recipe uses `gguf: path/to/file.gguf`, the file must exist (e.g. from a previous step or artifact).
- **Pushing models:** Ollama’s registry and push flow are separate; this example only builds locally. To publish, add steps that use Ollama’s push or your own artifact storage.
