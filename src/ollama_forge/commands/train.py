"""Train, retrain, and training-data command handlers for ollama-forge."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ollama_forge.config_loader import apply_config_to_args, load_config
from ollama_forge.log import get_logger
from ollama_forge.run_helpers import get_jsonl_paths_or_exit, print_actionable_error
from ollama_forge.training_data import (
    convert_jsonl_to_plain_text,
    convert_messages_to_alpaca_jsonl,
    validate_training_data_paths,
)

log = get_logger()


def _training_data_schema_json() -> dict:
    """Return JSON schema for accepted training data (Alpaca + messages)."""
    return {
        "description": "Training data: one JSON object per line (JSONL). Alpaca or messages format.",
        "oneOf": [
            {
                "type": "object",
                "required": ["instruction", "output"],
                "properties": {
                    "instruction": {"type": "string", "description": "Required. The user/task prompt."},
                    "output": {"type": "string", "description": "Required. The desired assistant response."},
                    "input": {"type": "string", "description": "Optional. Additional context."},
                },
                "additionalProperties": True,
            },
            {
                "type": "object",
                "required": ["messages"],
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "At least one user and one assistant message with string content.",
                        "items": {
                            "type": "object",
                            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                        },
                    },
                },
                "additionalProperties": True,
            },
        ],
    }


def _cmd_validate_training_data(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Validate JSONL training data (instruction/input/output format)."""
    if getattr(args, "schema", False):
        print(json.dumps(_training_data_schema_json(), indent=2))
        return 0
    paths = get_jsonl_paths_or_exit(
        args.data,
        next_steps=[
            "Pass one or more .jsonl files",
            "Or pass a directory that contains .jsonl files",
        ],
    )
    if paths is None:
        return 1
    ok, errors, count = validate_training_data_paths(paths)
    if ok:
        print(f"OK: {count} valid line(s) in {len(paths)} file(s)")
        return 0
    for msg in errors:
        print(msg, file=sys.stderr)
    print_actionable_error(
        "validation failed",
        next_steps=[
            "Fix the errors above (invalid JSONL lines or missing fields)",
            "Run: ollama-forge validate-training-data <path>",
        ],
    )
    return 1


def _cmd_prepare_training_data(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Convert JSONL training data to plain text for trainers (e.g. llama.cpp)."""
    if getattr(args, "list_formats", False):
        print(
            "Supported formats (use with --format <name> or comma-separated for multiple):\n"
            "  llama.cpp   - ### Instruction / ### Input / ### Response blocks (llama.cpp finetune --train-data)\n"
            "  alpaca_plain - instruction\\noutput per sample (minimal; some scripts)\n"
            "When multiple formats are given, each is written to <output_stem>_<format>.txt",
            file=sys.stderr,
        )
        return 0
    paths = get_jsonl_paths_or_exit(
        args.data,
        next_steps=[
            "Pass one or more .jsonl files",
            "Or pass a directory that contains .jsonl files",
        ],
    )
    if paths is None:
        return 1
    ok, errors, _ = validate_training_data_paths(paths)
    if not ok:
        for msg in errors:
            print(msg, file=sys.stderr)
        print_actionable_error(
            "validation failed; fix errors before preparing",
            next_steps=[
                "Run: ollama-forge validate-training-data <path> to see errors",
                "Fix invalid JSONL lines or missing fields, then re-run prepare-training-data",
            ],
        )
        return 1
    formats = [f.strip() for f in getattr(args, "format", "llama.cpp").split(",") if f.strip()]
    if not formats:
        formats = ["llama.cpp"]
    out_path = Path(args.output)
    written: list[tuple[str, Path, int]] = []
    for fmt in formats:
        dest = out_path
        if len(formats) > 1:
            stem = out_path.stem
            suffix = out_path.suffix
            parent = out_path.parent
            dest = parent / f"{stem}_{fmt.replace('.', '_')}{suffix}"
        try:
            n_samples = convert_jsonl_to_plain_text(paths, dest, format_name=fmt)
            written.append((fmt, dest, n_samples))
        except OSError as e:
            print_actionable_error(
                "failed to write output file",
                cause=str(e),
                next_steps=[
                    "Check parent directory exists and is writable",
                    "Try a different output path with -o/--output",
                ],
            )
            return 1
    for fmt, dest, n_samples in written:
        size = dest.stat().st_size if dest.is_file() else 0
        print(f"Wrote {n_samples} sample(s) → {dest} ({size} bytes) [{fmt}]")
    if written and written[0][0] == "llama.cpp":
        print(
            "Use with llama.cpp finetune: --train-data ... --sample-start '### Instruction'",
            file=sys.stderr,
        )
    return 0


def _cmd_convert_training_data_format(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Convert JSONL from messages format (e.g. TeichAI/datagen) to Alpaca-style instruction/output."""
    input_spec = args.input
    if isinstance(input_spec, list):
        input_spec = input_spec[0] if input_spec else ""
    path_in = Path(input_spec)
    if not path_in.is_file():
        print_actionable_error(
            f"Input file not found: {path_in}",
            next_steps=[
                "Pass a .jsonl file (e.g. from datagen --out dataset.jsonl)",
                "Run: ollama-forge convert-training-data-format <input.jsonl> -o <output.jsonl>",
            ],
        )
        return 1
    path_out = Path(args.output)
    try:
        count = convert_messages_to_alpaca_jsonl(path_in, path_out)
    except OSError as e:
        print_actionable_error(
            "failed to write output file",
            cause=str(e),
            next_steps=["Check output path and permissions"],
        )
        return 1
    print(f"Wrote {count} Alpaca-style record(s) to {path_out}")
    return 0


def _cmd_train_data_init(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Create a directory with README and sample.jsonl for training data."""
    out_dir = Path(getattr(args, "out", "./data")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    template = getattr(args, "template", "alpaca")
    readme = """# Training data

Use JSONL (one JSON object per line). Each line must have:

- **Alpaca-style:** `instruction` (required), `output` (required), `input` (optional).
- **Messages-style:** `messages`: array of `{role: "user"|"assistant"|"system", content: "..."}`.

Validate: `ollama-forge validate-training-data ./data/`
Prepare: `ollama-forge prepare-training-data ./data/ -o train_prepared.txt --format llama.cpp`
"""
    if template == "chat":
        sample = (
            '{"messages": [{"role": "user", "content": "What is 2+2?"},'
            ' {"role": "assistant", "content": "4."}]}\n'
            '{"messages": [{"role": "user", "content": "Say hello."},'
            ' {"role": "assistant", "content": "Hello! How can I help you?"}]}\n'
            '{"messages": [{"role": "system", "content": "You are helpful."},'
            ' {"role": "user", "content": "Summarize briefly."},'
            ' {"role": "assistant", "content": "Short summary."}]}\n'
        )
    else:
        sample = """{"instruction": "What is 2+2?", "input": "", "output": "4."}
{"instruction": "Say hello.", "output": "Hello! How can I help you?"}
{"instruction": "Summarize in one sentence.", "input": "Long document text here...", "output": "Short summary."}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "sample.jsonl").write_text(sample, encoding="utf-8")
    print(f"Created {out_dir}/README.md and {out_dir}/sample.jsonl (template={template})")
    print("Add your own .jsonl files, then run: ollama-forge validate-training-data", str(out_dir))
    return 0


def _cmd_train_resolve_base(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Suggest how to get a base GGUF for finetune/train-run (model name → GGUF path)."""
    base_name = (getattr(args, "base_name", None) or "").strip()
    if not base_name:
        log.info("Usage: ollama-forge train-resolve-base <base_model_name>")
        log.info("Example: ollama-forge train-resolve-base llama3.2")
        return 1
    print(f"For --base-gguf you need a GGUF file matching the base model '{base_name}'.")
    print("")
    print("Options:")
    print("  1. Download from Hugging Face (creates an Ollama model; GGUF is in HF cache):")
    print("     ollama-forge fetch <repo_id> --name <name>")
    print("     Example: ollama-forge fetch bartowski/Llama-3.2-3B-Instruct-GGUF --name llama3.2-base")
    print("     Then use the downloaded GGUF path from the HF cache, or re-export from Ollama.")
    print("  2. Search Hugging Face for your model + 'GGUF' and download a .gguf file.")
    print("     Pass that path to finetune/train-run: --base-gguf /path/to/model.gguf")
    print("")
    print("After you have a GGUF path:")
    print(f"  ollama-forge finetune --data <path> --base {base_name} --name <out_name> --base-gguf /path/to/model.gguf")
    return 0


def _cmd_train(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Generate a training script: validate → prepare data → trainer → retrain."""
    paths = get_jsonl_paths_or_exit(
        args.data,
        error_msg="Error: no .jsonl files found at --data. Use a file or directory.",
        next_steps=[
            "Pass one or more .jsonl files",
            "Or pass a directory that contains .jsonl files",
        ],
    )
    if paths is None:
        return 1
    data_input = args.data if isinstance(args.data, list) else [args.data]
    first = Path(data_input[0])
    data_spec = str(first.resolve())
    base = args.base
    name = args.name
    base_gguf = getattr(args, "base_gguf", None)
    run_trainer = getattr(args, "run_trainer", False)
    base_gguf_var = f'BASE_GGUF="{base_gguf}"' if base_gguf else 'BASE_GGUF=""  # set to your base .gguf path'
    run_finetune_block = ""
    trainer_ctx = int(getattr(args, "num_ctx", 0) or 4096)
    trainer_batch = 128
    trainer_ubatch = 64
    if base_gguf and run_trainer:
        run_finetune_block = """
FINETUNE_BIN="$(command -v finetune || command -v llama-finetune || true)"
if [ -n "$FINETUNE_BIN" ]; then
  echo "Step 3: Running llama.cpp finetune..."
  if "$FINETUNE_BIN" --help 2>&1 | grep -q -- "--train-data"; then
    "$FINETUNE_BIN" --train-data "$PREPARED" --sample-start '### Instruction' \\
      --model-base "$BASE_GGUF" --lora-out "$ADAPTER_DIR" || true
  else
    "$FINETUNE_BIN" --model "$BASE_GGUF" --file "$PREPARED" \\
      --output "$ADAPTER_DIR/adapter.gguf" --epochs 1 \\
      --ctx-size {trainer_ctx} --batch-size {trainer_batch} --ubatch-size {trainer_ubatch} || true
  fi
else
  echo "Step 3: finetune/llama-finetune not on PATH. Run: ollama-forge setup-llama-cpp and add to PATH."
  echo "  Then run either:"
  echo "    finetune --train-data \$PREPARED --sample-start '### Instruction' \\"
  echo "      --model-base \$BASE_GGUF --lora-out \$ADAPTER_DIR"
  echo "  Or:"
  echo "    llama-finetune --model \$BASE_GGUF --file \$PREPARED --output \$ADAPTER_DIR/adapter.gguf --epochs 1"
fi
"""
    else:
        run_finetune_block = """
echo "Step 3: Run llama.cpp finetune (need base GGUF and finetune on PATH)."
echo "  finetune --train-data \$PREPARED --sample-start '### Instruction' \\"
echo "    --model-base <path-to-base.gguf> --lora-out \$ADAPTER_DIR"
echo "  Or re-run with --base-gguf <path> --run-trainer to run it automatically."
"""
    script = f"""#!/usr/bin/env bash
# Training pipeline: data → adapter → Ollama model
# Data: {data_spec}
# Base: {base}  Name: {name}  Prepared: train_prepared.txt  Adapter out: ./adapter_out
set -e
DATA="{data_spec}"
BASE="{base}"
NAME="{name}"
{base_gguf_var}
PREPARED="train_prepared.txt"
ADAPTER_DIR="./adapter_out"

echo "Step 1: Validating data..."
ollama-forge validate-training-data "$DATA"
echo "Step 2: Preparing data for llama.cpp (plain text)..."
ollama-forge prepare-training-data "$DATA" -o "$PREPARED" --format llama.cpp
{run_finetune_block}
echo "Step 4: After training, create Ollama model:"
echo "  ollama-forge retrain --base $BASE --adapter $ADAPTER_DIR --name $NAME"
echo ""
echo "Then: ollama run $NAME"
"""
    if getattr(args, "execute", False):
        data_list = data_input if isinstance(data_input, list) else [data_input]
        code = subprocess.run(
            ["ollama-forge", "validate-training-data"] + data_list,
            shell=False,
        )
        if code.returncode != 0:
            return code.returncode
        code = subprocess.run(
            ["ollama-forge", "prepare-training-data"]
            + data_list
            + ["-o", "train_prepared.txt", "--format", "llama.cpp"],
            shell=False,
        )
        if code.returncode != 0:
            return code.returncode
        if base_gguf and run_trainer:
            finetune_bin = shutil.which("finetune") or shutil.which("llama-finetune")
            if finetune_bin:
                try:
                    help_result = subprocess.run(
                        [finetune_bin, "--help"],
                        capture_output=True,
                        text=True,
                        shell=False,
                    )
                    help_text = f"{help_result.stdout}\n{help_result.stderr}"
                except OSError:
                    help_text = ""
                if "--train-data" in help_text:
                    cmd = [
                        finetune_bin,
                        "--train-data", "train_prepared.txt",
                        "--sample-start", "### Instruction",
                        "--model-base", base_gguf,
                        "--lora-out", "./adapter_out",
                    ]
                else:
                    cmd = [
                        finetune_bin,
                        "--model", base_gguf,
                        "--file", "train_prepared.txt",
                        "--output", "./adapter_out/adapter.gguf",
                        "--epochs", "1",
                        "--ctx-size", str(trainer_ctx),
                        "--batch-size", str(trainer_batch),
                        "--ubatch-size", str(trainer_ubatch),
                    ]
            else:
                cmd = ["finetune", "--help"]
            code = subprocess.run(cmd, shell=False)
            if code.returncode != 0:
                log.warning("finetune exited with %s; adapter may be incomplete.", code.returncode)
        print("Next: ollama-forge retrain --base", base, "--adapter ./adapter_out --name", name)
        return 0
    if getattr(args, "write_script", None):
        out_path = Path(args.write_script)
        out_path.write_text(script, encoding="utf-8")
        out_path.chmod(0o755)
        print(f"Wrote script to {out_path}. Run it: ./{out_path}")
        return 0
    print(script)
    return 0


_TRAIN_RUN_DEFAULTS: dict[str, object] = {
    "data": None,
    "base": None,
    "name": None,
    "base_gguf": None,
    "prepared_output": None,
    "adapter_output": None,
    "format": "llama.cpp",
    "trainer": "llama.cpp",
    "system": None,
    "temperature": None,
    "num_ctx": None,
    "top_p": None,
    "repeat_penalty": None,
    "out_modelfile": None,
    "skip_retrain": False,
}


def _cmd_train_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run the full pipeline: validate → prepare → (finetune if available) → retrain."""
    config_path = getattr(args, "config", None)
    if config_path:
        try:
            cfg = load_config(config_path)
            apply_config_to_args(args, cfg, only_if_default=_TRAIN_RUN_DEFAULTS)
        except (FileNotFoundError, ValueError, ImportError) as e:
            print_actionable_error(
                "Failed to load config file",
                cause=str(e),
                next_steps=["Check --config path and file format (YAML/JSON)"],
            )
            return 1
    paths = get_jsonl_paths_or_exit(
        args.data,
        error_msg="Error: no .jsonl files found at --data.",
        next_steps=["Pass one or more .jsonl files or a directory containing .jsonl"],
    )
    if paths is None:
        return 1
    ok, errors, _ = validate_training_data_paths(paths)
    if not ok:
        for msg in errors:
            print(msg, file=sys.stderr)
        print_actionable_error(
            "validation failed; fix errors before running pipeline",
            next_steps=[
                "Run: ollama-forge validate-training-data <path>",
                "Fix invalid JSONL lines or missing fields, then re-run train-run",
            ],
        )
        return 1
    data_spec = args.data[0] if isinstance(args.data, list) and args.data else args.data
    if isinstance(data_spec, list):
        data_spec = data_spec[0]
    prepared_path = Path(getattr(args, "prepared_output", None) or "train_prepared.txt")
    adapter_dir = Path(getattr(args, "adapter_output", None) or "adapter_out")
    base = args.base
    name = args.name
    base_gguf = getattr(args, "base_gguf", None)
    trainer_ctx = int(getattr(args, "num_ctx", 0) or 4096)
    trainer_batch = 128
    trainer_ubatch = 64
    # Step 2: prepare
    try:
        n_samples = convert_jsonl_to_plain_text(paths, prepared_path, format_name=getattr(args, "format", "llama.cpp"))
    except OSError as e:
        print_actionable_error(
            "failed to write prepared data",
            cause=str(e),
            next_steps=["Use --prepared-output <path> or fix permissions"],
        )
        return 1
    log.info("Prepared %s sample(s) → %s", n_samples, prepared_path)
    print(
        "Next: run your trainer (e.g. llama.cpp finetune) or use --base-gguf to run it automatically.",
        file=sys.stderr,
    )
    # Step 3: finetune (if base_gguf and finetune on PATH)
    ran_finetune = False
    if base_gguf and Path(base_gguf).is_file():
        finetune_bin = shutil.which("finetune") or shutil.which("llama-finetune")
        if finetune_bin:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            try:
                help_result = subprocess.run(
                    [finetune_bin, "--help"],
                    capture_output=True,
                    text=True,
                )
                help_text = f"{help_result.stdout}\n{help_result.stderr}"
            except OSError:
                help_text = ""
            if "--train-data" in help_text:
                cmd = [
                    finetune_bin,
                    "--train-data",
                    str(prepared_path.resolve()),
                    "--sample-start",
                    "### Instruction",
                    "--model-base",
                    str(Path(base_gguf).resolve()),
                    "--lora-out",
                    str(adapter_dir.resolve()),
                ]
            else:
                cmd = [
                    finetune_bin,
                    "--model",
                    str(Path(base_gguf).resolve()),
                    "--file",
                    str(prepared_path.resolve()),
                    "--output",
                    str((adapter_dir / "adapter.gguf").resolve()),
                    "--epochs",
                    "1",
                    "--ctx-size",
                    str(trainer_ctx),
                    "--batch-size",
                    str(trainer_batch),
                    "--ubatch-size",
                    str(trainer_ubatch),
                ]
            log.info("Running finetune (llama.cpp)...")
            result = subprocess.run(cmd)  # stdout/stderr inherited; progress visible
            if result.returncode != 0:
                print_actionable_error(
                    "finetune failed",
                    next_steps=[
                        "Check --base-gguf and training data",
                        "Run finetune manually and then: ollama-forge retrain --base <base> --adapter <dir> --name <name>",  # noqa: E501
                    ],
                )
                return 1
            ran_finetune = True
        else:
            log.info("finetune not on PATH; skipping. Run: ollama-forge setup-llama-cpp")
    if not ran_finetune:
        intro = "Skipped finetune (need --base-gguf and finetune on PATH). After training:"
        cmd = f"ollama-forge retrain --base {base} --adapter {adapter_dir} --name {name}"
        log.info("%s %s", intro, cmd)
        return 0
    if getattr(args, "skip_retrain", False):
        log.info(
            "Skipping retrain (--skip-retrain). Adapter at %s; "
            "run: ollama-forge retrain --base %s --adapter %s --name %s",
            adapter_dir, base, adapter_dir, name,
        )
        return 0
    # Step 4: retrain (create-from-base with adapter)
    fake = argparse.Namespace(
        base=base,
        name=name,
        adapter=str(adapter_dir.resolve()),
        system=getattr(args, "system", None),
        temperature=getattr(args, "temperature", None),
        num_ctx=getattr(args, "num_ctx", None),
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        template_from=getattr(args, "template_from", None),
        out_modelfile=getattr(args, "out_modelfile", None),
    )
    from ollama_forge.cli import _cmd_create_from_base  # noqa: PLC0415
    return _cmd_create_from_base(parser, fake)


def _cmd_retrain(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Create an Ollama model from a base + adapter (after training)."""
    from ollama_forge.commands.model import _cmd_create_from_base  # noqa: PLC0415
    return _cmd_create_from_base(parser, args)


_MLX_TRAIN_TEMPLATE = """\
# forge mlx-train config  — mlx-lm LoRA fine-tuning on Apple Silicon
# Usage: ollama-forge mlx-train config.yaml

# ── Required ─────────────────────────────────────────────────────────────────
model: mlx-community/Llama-3.2-3B-Instruct-4bit  # HF repo or local MLX dir
data: ./data   # directory with train.jsonl and valid.jsonl (mlx_lm format)

# ── Training (mlx_lm lora options) ───────────────────────────────────────────
fine_tune_type: lora          # lora | dora | full
num_layers: 16                # adapter layers (-1 = all)
batch_size: 4
iters: 500
learning_rate: 1.0e-4
max_seq_length: 2048
grad_checkpoint: false
adapter_path: ./adapters      # output directory for adapter weights
optimizer: adam               # adam | adamw | muon | sgd | adafactor
mask_prompt: false

# ── Post-training steps (optional) ───────────────────────────────────────────
fuse: false                   # merge adapter into base model (mlx_lm fuse)
fused_path: ./fused_model     # where to save the fused model
export_gguf: false            # also export a GGUF (requires fuse: true)
gguf_path: ./fused_model/ggml-model-f16.gguf

# ── Ollama push (optional) ────────────────────────────────────────────────────
# ollama_name: my-finetuned-model       # creates Ollama model when set
# ollama_base: llama3.2:3b              # Modelfile FROM; defaults to 'model' above
"""


_MLX_TRAIN_DEFAULTS: dict[str, object] = {
    "fine_tune_type": "lora",
    "num_layers": 16,
    "batch_size": 4,
    "iters": 100,
    "learning_rate": 1e-4,
    "max_seq_length": 2048,
    "grad_checkpoint": False,
    "adapter_path": "./adapters",
    "optimizer": "adam",
    "mask_prompt": False,
    "fuse": False,
    "fused_path": "./fused_model",
    "export_gguf": False,
    "gguf_path": "./fused_model/ggml-model-f16.gguf",
    "ollama_name": None,
    "ollama_base": None,
}


def _cmd_mlx_train(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run mlx-lm LoRA fine-tuning from a YAML config file."""
    import shlex

    if getattr(args, "init", False):
        dest = getattr(args, "config", None) or "forge-train.yaml"
        p = Path(dest)
        if p.exists():
            print(f"Already exists: {p}", file=sys.stderr)
            return 1
        p.write_text(_MLX_TRAIN_TEMPLATE, encoding="utf-8")
        print(f"Wrote template: {p}")
        print(f"Edit it, then run: ollama-forge mlx-train {p}")
        return 0

    config_path = getattr(args, "config", None)
    if not config_path:
        parser.print_help()
        return 1

    try:
        import yaml  # type: ignore[import]
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError("config must be a YAML mapping")
    except FileNotFoundError:
        print_actionable_error(
            f"Config file not found: {config_path}",
            next_steps=["Create a template: ollama-forge mlx-train --init config.yaml"],
        )
        return 1
    except Exception as exc:
        print_actionable_error(f"Could not load config: {exc}", next_steps=[])
        return 1

    cfg: dict = {**_MLX_TRAIN_DEFAULTS, **loaded}

    model = cfg.get("model")
    data = cfg.get("data")
    if not model or not data:
        print_actionable_error(
            "config.yaml must include 'model' and 'data'",
            next_steps=[f"Edit {config_path} and set both fields"],
        )
        return 1

    adapter_path = cfg["adapter_path"]
    dry_run = getattr(args, "dry_run", False)

    # ── Step 1: mlx_lm lora --train ──────────────────────────────────────────
    lora_argv = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--model", str(model),
        "--data", str(data),
        "--adapter-path", str(adapter_path),
        "--fine-tune-type", str(cfg["fine_tune_type"]),
        "--num-layers", str(cfg["num_layers"]),
        "--batch-size", str(cfg["batch_size"]),
        "--iters", str(cfg["iters"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--max-seq-length", str(cfg["max_seq_length"]),
        "--optimizer", str(cfg["optimizer"]),
    ]
    if cfg.get("grad_checkpoint"):
        lora_argv.append("--grad-checkpoint")
    if cfg.get("mask_prompt"):
        lora_argv.append("--mask-prompt")

    if dry_run:
        print("Step 1 — train:")
        print(f"  {shlex.join(lora_argv)}")
    else:
        print(f"\nforge mlx-train  model={model}  iters={cfg['iters']}", file=sys.stderr)
        rc = subprocess.run(lora_argv).returncode
        if rc != 0:
            print_actionable_error(
                f"mlx_lm lora exited with {rc}",
                next_steps=["Check the output above for mlx_lm errors"],
            )
            return rc
        print(f"\nAdapter saved to: {adapter_path}", file=sys.stderr)

    # ── Step 2: mlx_lm fuse (optional) ───────────────────────────────────────
    fused_path = cfg["fused_path"]
    if cfg.get("fuse") or cfg.get("export_gguf"):
        fuse_argv = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", str(model),
            "--adapter-path", str(adapter_path),
            "--save-path", str(fused_path),
        ]
        if cfg.get("export_gguf"):
            fuse_argv += ["--export-gguf", "--gguf-path", str(cfg["gguf_path"])]
        if dry_run:
            print("Step 2 — fuse:")
            print(f"  {shlex.join(fuse_argv)}")
        else:
            print(f"\nForge mlx-train: fusing adapter → {fused_path}", file=sys.stderr)
            rc = subprocess.run(fuse_argv).returncode
            if rc != 0:
                print_actionable_error(
                    f"mlx_lm fuse exited with {rc}",
                    next_steps=["The adapter is still usable; fuse manually with mlx_lm fuse"],
                )
                return rc

    # ── Step 3: Ollama push (optional) ───────────────────────────────────────
    ollama_name = cfg.get("ollama_name")
    if ollama_name:
        # Use GGUF if we exported one, otherwise fall back to adapter (safetensors)
        from ollama_forge.commands.model import _cmd_create_from_base  # noqa: PLC0415

        if cfg.get("export_gguf"):
            # Import GGUF as a fresh Ollama model
            import argparse as _ap
            ns = _ap.Namespace(
                gguf=str(cfg["gguf_path"]),
                name=str(ollama_name),
                system=None,
                template_from=None,
                adapter=None,
                base=None,
                out_modelfile=None,
                temperature=None,
                num_ctx=None,
                top_p=None,
                repeat_penalty=None,
            )
        else:
            ollama_base = cfg.get("ollama_base") or str(model)
            import argparse as _ap
            ns = _ap.Namespace(
                base=str(ollama_base),
                adapter=str(adapter_path),
                name=str(ollama_name),
                system=None,
                template_from=None,
                gguf=None,
                out_modelfile=None,
                temperature=None,
                num_ctx=None,
                top_p=None,
                repeat_penalty=None,
            )

        if dry_run:
            print(f"Step 3 — ollama push: {ollama_name}")
        else:
            print(f"\nforge mlx-train: pushing to Ollama as {ollama_name!r}", file=sys.stderr)
            rc = _cmd_create_from_base(parser, ns)
            if rc != 0:
                return rc

    if not dry_run:
        print(f"\nDone. Adapter: {adapter_path}", file=sys.stderr)
        if cfg.get("fuse"):
            print(f"  Fused model: {fused_path}", file=sys.stderr)
        if ollama_name:
            print(f"  Ollama model: ollama run {ollama_name}", file=sys.stderr)

    return 0


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Register train/retrain/training-data commands. Returns the train-data parser."""
    # validate-training-data
    p_validate = subparsers.add_parser(
        "validate-training-data",
        help="Validate JSONL training data (file(s) or directory)",
    )
    p_validate.add_argument(
        "data",
        nargs="*",
        help="Path(s) to .jsonl file(s) or a directory of .jsonl files",
    )
    p_validate.add_argument(
        "--schema",
        action="store_true",
        help="Print expected JSON schema (Alpaca + messages) and exit",
    )
    p_validate.set_defaults(handler=_cmd_validate_training_data)

    # prepare-training-data (convert JSONL → trainer format)
    p_prepare = subparsers.add_parser(
        "prepare-training-data",
        help="Convert JSONL to plain text for trainers (e.g. llama.cpp)",
    )
    p_prepare.add_argument(
        "data",
        nargs="+",
        help="Path(s) to .jsonl file(s) or a directory",
    )
    p_prepare.add_argument(
        "-o", "--output", required=True, help="Output file path (or directory when using multiple --format)",
    )
    p_prepare.add_argument(
        "--format",
        dest="format",
        default="llama.cpp",
        help="Output format(s), comma-separated (e.g. llama.cpp,alpaca_plain); each writes <output_stem>_<format>.txt",
    )
    p_prepare.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported formats and which trainer expects them, then exit",
    )
    p_prepare.set_defaults(handler=_cmd_prepare_training_data)

    # convert-training-data-format (messages → Alpaca JSONL, e.g. for datagen output)
    p_convert_fmt = subparsers.add_parser(
        "convert-training-data-format",
        help="Convert JSONL to Alpaca-style (e.g. from TeichAI/datagen messages format)",
    )
    p_convert_fmt.add_argument(
        "input",
        help="Input .jsonl file (Alpaca or messages format)",
    )
    p_convert_fmt.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .jsonl file (Alpaca-style instruction/input/output)",
    )
    p_convert_fmt.set_defaults(handler=_cmd_convert_training_data_format)

    # train-data (init: scaffold directory with README + sample JSONL)
    p_train_data = subparsers.add_parser(
        "train-data",
        help="Training data helpers (init: create sample directory)",
    )
    train_data_sub = p_train_data.add_subparsers(dest="train_data_cmd", required=True)
    p_train_data_init = train_data_sub.add_parser(
        "init",
        help="Create a directory with README and sample.jsonl for training data",
    )
    p_train_data_init.add_argument(
        "--out",
        "-o",
        default="./data",
        metavar="DIR",
        help="Output directory (default: ./data)",
    )
    p_train_data_init.add_argument(
        "--template",
        choices=["alpaca", "chat"],
        default="alpaca",
        help="Sample template: alpaca (instruction/input/output) or chat (messages); default: alpaca",
    )
    p_train_data_init.set_defaults(handler=_cmd_train_data_init)

    # train-resolve-base (suggest how to get base GGUF for a model name)
    p_train_resolve_base = subparsers.add_parser(
        "train-resolve-base",
        help="Suggest how to get a base GGUF for finetune/train-run (e.g. llama3.2)",
    )
    p_train_resolve_base.add_argument(
        "base_name",
        nargs="?",
        default="",
        help="Base model name (e.g. llama3.2); omit to show usage",
    )
    p_train_resolve_base.set_defaults(handler=_cmd_train_resolve_base)

    # train (generate script: data → prepare → trainer → retrain)
    p_train = subparsers.add_parser(
        "train",
        help="Generate a training script. To run the pipeline in one go, use 'finetune' or 'train-run' instead.",
    )
    p_train.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="Training data: .jsonl file(s) or directory of .jsonl",
    )
    p_train.add_argument("--base", required=True, help="Base model name (e.g. llama3.2)")
    p_train.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_train.add_argument(
        "--base-gguf",
        help="Path to base GGUF for finetune; use with --run-trainer to run training in script",
    )
    p_train.add_argument(
        "--run-trainer",
        action="store_true",
        help="Generated script will run finetune if on PATH (requires --base-gguf)",
    )
    p_train.add_argument(
        "--trainer",
        default="llama.cpp",
        help="Trainer backend (default: llama.cpp). Only llama.cpp is wired today.",
    )
    p_train.add_argument(
        "--write-script",
        metavar="PATH",
        help="Write the pipeline script to this file",
    )
    p_train.add_argument(
        "--execute",
        action="store_true",
        help="Run validate → prepare → (finetune if --base-gguf and --run-trainer); then print retrain command",
    )
    p_train.set_defaults(handler=_cmd_train)

    # retrain (base + adapter → Ollama model)
    p_retrain = subparsers.add_parser(
        "retrain",
        help="Create an Ollama model from base + adapter (run after training)",
    )
    p_retrain.add_argument("--base", required=True, help="Base model name or path")
    p_retrain.add_argument(
        "--adapter",
        required=True,
        help="Path to adapter: directory (PEFT or single .bin/.gguf) or .bin/.gguf file (e.g. llama.cpp finetune)",
    )
    p_retrain.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_retrain.add_argument("--system", help="System message")
    p_retrain.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_retrain.add_argument("--num-ctx", type=int, help="Context window size in tokens")
    p_retrain.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_retrain.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_retrain.add_argument(
        "--template-from",
        metavar="OLLAMA_MODEL",
        help="Ollama model to copy chat template from (for tool/Chat API support)",
    )
    p_retrain.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_retrain.set_defaults(handler=_cmd_retrain)

    # train-run (e2e: validate → prepare → finetune → retrain)
    p_train_run = subparsers.add_parser(
        "train-run",
        help="Run full pipeline: validate → prepare → finetune (if --base-gguf and finetune on PATH) → retrain",
    )
    p_train_run.add_argument("--data", required=True, nargs="+", help="Training data: .jsonl file(s) or directory")
    p_train_run.add_argument("--base", required=True, help="Base model name for retrain (e.g. llama3.2)")
    p_train_run.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_train_run.add_argument(
        "--base-gguf", help="Path to base GGUF; if set and finetune on PATH, run finetune then retrain"
    )  # noqa: E501
    p_train_run.add_argument(
        "--prepared-output", default=None, help="Output path for prepared text (default: train_prepared.txt)"
    )  # noqa: E501
    p_train_run.add_argument(
        "--adapter-output", default=None, help="Output dir for LoRA adapter (default: adapter_out)"
    )  # noqa: E501
    p_train_run.add_argument("--format", default="llama.cpp", help="Prepare format (default: llama.cpp)")
    p_train_run.add_argument("--trainer", default="llama.cpp", help="Trainer backend (default: llama.cpp)")
    p_train_run.add_argument("--system", help="System message for final model")
    p_train_run.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_train_run.add_argument("--num-ctx", type=int, help="Context window size in tokens")
    p_train_run.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_train_run.add_argument("--repeat-penalty", type=float, help="Repeat penalty (e.g. 1.1)")
    p_train_run.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_train_run.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Stop after prepare (and finetune if run); do not create Ollama model (run retrain manually later)",
    )
    p_train_run.add_argument(
        "--config",
        metavar="FILE",
        help="Load options from YAML/JSON file (CLI overrides config); repeatable runs",
    )
    p_train_run.set_defaults(handler=_cmd_train_run)

    # finetune (alias for train-run: one command to run the full pipeline)
    p_finetune = subparsers.add_parser(
        "finetune",
        help="Same as train-run. Use train --write-script to generate a script instead.",
    )
    p_finetune.add_argument("--data", required=True, nargs="+", help="Training data: .jsonl file(s) or directory")
    p_finetune.add_argument("--base", required=True, help="Base model name for retrain (e.g. llama3.2)")
    p_finetune.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_finetune.add_argument("--base-gguf", help="Base GGUF path; with finetune on PATH, runs finetune then retrain")
    p_finetune.add_argument("--prepared-output", default=None, help="Prepared text path (default: train_prepared.txt)")
    p_finetune.add_argument("--adapter-output", default=None, help="Output dir for LoRA adapter (default: adapter_out)")
    p_finetune.add_argument("--format", default="llama.cpp", help="Prepare format (default: llama.cpp)")
    p_finetune.add_argument("--trainer", default="llama.cpp", help="Trainer backend (default: llama.cpp)")
    p_finetune.add_argument("--system", help="System message for final model")
    p_finetune.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_finetune.add_argument("--num-ctx", type=int, help="Context window size in tokens")
    p_finetune.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_finetune.add_argument("--repeat-penalty", type=float, help="Repeat penalty (e.g. 1.1)")
    p_finetune.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_finetune.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Stop after prepare/finetune; do not create Ollama model",
    )
    p_finetune.add_argument(
        "--config",
        metavar="FILE",
        help="Load options from YAML/JSON file (CLI overrides config); repeatable runs",
    )
    p_finetune.set_defaults(handler=_cmd_train_run)

    # mlx-train (YAML-driven mlx-lm LoRA fine-tuning on Apple Silicon)
    p_mlx = subparsers.add_parser(
        "mlx-train",
        help=(
            "Fine-tune a model on Apple Silicon via mlx-lm LoRA. "
            "Pass a config.yaml (see --init to generate a template)."
        ),
    )
    p_mlx.add_argument(
        "config",
        nargs="?",
        metavar="CONFIG_YAML",
        help="Path to config.yaml (see --init for a template)",
    )
    p_mlx.add_argument(
        "--init",
        action="store_true",
        help="Write a template config.yaml and exit (default name: forge-train.yaml)",
    )
    p_mlx.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the mlx_lm commands that would run, without running them",
    )
    p_mlx.set_defaults(handler=_cmd_mlx_train)

    return p_train_data
