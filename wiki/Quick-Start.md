# Quick Start

The fastest way to get a working model with one command: **start** or **quickstart**.

---

## One command

```bash
uv run ollama-tools start --name my-model
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
uv run ollama-tools quickstart --profile low-vram --name my-lite-model
uv run ollama-tools quickstart --profile quality --name my-best-model
# Override profile with explicit flags:
uv run ollama-tools quickstart --profile balanced --quant Q8_0 --num-ctx 8192 --name my-tuned-model
```

---

## Task presets

Task presets set a default **system prompt** (e.g. chat, coding, creative).

```bash
uv run ollama-tools start --task chat --name my-chat-model
uv run ollama-tools quickstart --task coding --name my-coder
uv run ollama-tools quickstart --task creative --name my-writer
# Override task with custom system prompt:
uv run ollama-tools quickstart --task coding --system "You are terse." --name my-coder
```

Use `--system "..."` to override the task's prompt.

---

## Summary

- **start** = quickstart with defaults; good for first-time users.
- **quickstart** = same behavior, with **--profile** and **--task** for presets.
- After the command finishes, run **ollama run &lt;name&gt;** to use your model.
