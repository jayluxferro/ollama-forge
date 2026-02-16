# CI / Automation

Example: build a model from a recipe in CI (e.g. GitHub Actions).

---

## Example (GitHub Actions)

Install Ollama and ollama-forge, then build from a recipe:

```yaml
- name: Install Ollama
  run: |
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
    sleep 5

- name: Build model from recipe
  run: uv run ollama-forge build recipes/my-model.yaml
```

- If the recipe uses **base: llama3.2**, pull the base first (e.g. `ollama pull llama3.2`).
- If the recipe uses **gguf: path/to/file.gguf**, ensure that file exists from a previous step or artifact.

---

## plan --json for scripting

Use the **plan** command with **--json** to get machine-readable output for automation:

```bash
uv run ollama-forge plan auto TheBloke/Llama-2-7B-GGUF --name my-model --json
uv run ollama-forge plan quickstart --profile balanced --name my-model --json
```

See [Auto & Plan](Auto-and-Plan).
