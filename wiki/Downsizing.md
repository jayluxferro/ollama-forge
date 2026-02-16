# Downsizing (knowledge distillation)

Produce a smaller model via **knowledge distillation** (teacher → student). You pass teacher and student (Hugging Face repo ids) and the tool gives you the exact steps.

---

## Simple usage

```bash
uv run ollama-forge downsize --teacher org/30b-model --student org/3b-model --name my-downsized --quantize Q4_K_M
```

The tool prints the commands to run: download → distill → convert. Optionally pass **--quantize** so the last step uses a smaller GGUF.

---

## Write script and run yourself

```bash
uv run ollama-forge downsize --teacher org/30b --student org/3b --name my-downsized --quantize Q4_K_M --write-script downsize.sh
```

Then run `downsize.sh` (or run the steps manually). Distillation itself (training the student to mimic the teacher) is done with external tools (e.g. [TRL GKD](https://huggingface.co/docs/trl/main/en/gkd_trainer), Axolotl); the script tells you what to run. The last step is always `ollama-forge convert` (with `--quantize` if you passed it).
