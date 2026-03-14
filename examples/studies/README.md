# Study Examples

These are starter configs for the generic study workflow:

```bash
uv run ollama-forge study validate examples/studies/quick-wikitext.yaml
uv run ollama-forge study plan examples/studies/quick-wikitext.yaml
uv run --no-sync ollama-forge study run examples/studies/quick-wikitext.yaml
```

Use `ollama-forge study interactive` to generate a new config guided by hardware and preset choices.
