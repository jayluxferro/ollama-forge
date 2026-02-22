# Minimal downsize (distillation) example

After you have a **student GGUF** (from TRL GKD, Axolotl, or other distillation), create an Ollama model in two ways:

## Option 1: Recipe (with variable)

```bash
# Set the path to your student GGUF (e.g. after exporting from Hugging Face)
export student_gguf=/path/to/student.gguf
uv run ollama-forge build examples/recipes/downsize-student.yaml
ollama run my-downsized
```

Edit `examples/recipes/downsize-student.yaml` and set `gguf` to your file, or use the `variables` section and `{{ student_gguf }}` with the env var above.

## Option 2: One-shot convert

```bash
uv run ollama-forge convert --gguf /path/to/student.gguf --name my-downsized --quantize Q4_K_M
ollama run my-downsized
```

## Full pipeline (teacher → student → Ollama)

1. Get teacher and student model IDs (e.g. from Hugging Face).
2. Run distillation externally (see [wiki/Downsizing.md](../../wiki/Downsizing.md)).
3. Export student to GGUF, then use Option 1 or 2 above.

To print the full pipeline steps:

```bash
uv run ollama-forge downsize --teacher org/teacher --student org/student --name my-downsized --write-script downsize.sh
```
