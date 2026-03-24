# Quick Start

The fastest way to get a working model with one command: **start** or **quickstart**.

---

## One command

```bash
uv run ollama-forge start --name my-model
ollama run my-model
```

`start` is an alias for `quickstart` with beginner defaults. By default it uses `TheBloke/Llama-2-7B-GGUF` with the **balanced** profile. Override with `--repo-id` and `--quant` if you want a different model or size.

---

## Profiles

Profiles set default quantization and model parameters so you don't have to remember flags.

| Profile | Use case | Typical quant | Context |
|---------|----------|---------------|---------|
| **fast** | Quick, smaller | Q4_0 | Lower |
| **balanced** | General default | Q4_K_M | Moderate |
| **quality** | Higher quality | Q8_0 | Larger |
| **low-vram** | Constrained memory | Q4_0 | Smaller |

Examples:

```bash
uv run ollama-forge quickstart --profile low-vram --name my-lite-model
uv run ollama-forge quickstart --profile quality --name my-best-model
# Override profile with explicit flags:
uv run ollama-forge quickstart --profile balanced --quant Q8_0 --num-ctx 8192 --name my-tuned-model
```

---

## Task presets

Task presets set a default **system prompt** (e.g. chat, coding, creative).

```bash
uv run ollama-forge start --task chat --name my-chat-model
uv run ollama-forge quickstart --task coding --name my-coder
uv run ollama-forge quickstart --task creative --name my-writer
# Override task with custom system prompt:
uv run ollama-forge quickstart --task coding --system "You are terse." --name my-coder
```

Use `--system "..."` to override the task's prompt.

---

## Serve directly (without Ollama)

You can skip Ollama entirely and run a GGUF model with llama-server using `serve`:

```bash
# Download a GGUF (no Ollama model created)
uv run ollama-forge fetch janhq/Jan-code-4b-gguf --download-only

# Serve it — GPU is used automatically (Metal / CUDA / CPU fallback)
uv run ollama-forge serve /path/to/model.gguf
```

This starts an OpenAI-compatible API at `http://127.0.0.1:11434`. Then chat with it:

```bash
# In another terminal
uv run ollama-forge chat
```

`chat` supports `/clear` to reset conversation and `quit` to exit. Use `--system "You are helpful."` to set a system prompt.

If the HF repo has no GGUF files (only safetensors), `fetch --download-only` will automatically download and convert to GGUF:

```bash
uv run ollama-forge fetch huihui-ai/some-model --download-only
```

You can also pass a HF repo ID directly to `serve` — it resolves the GGUF from the local HF cache:

```bash
uv run ollama-forge serve janhq/Jan-code-4b-gguf
```

---

## Summary

- **start** = quickstart with defaults; good for first-time users.
- **quickstart** = same behavior, with **--profile** and **--task** for presets.
- **serve** = run a GGUF model directly via llama-server (no Ollama needed).
- After quickstart/start, run **ollama run &lt;name&gt;** to use your model.
