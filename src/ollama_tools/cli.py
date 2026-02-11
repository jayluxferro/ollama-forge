"""CLI entrypoint for ollama-tools."""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ollama_tools.hf_fetch import download_adapter, download_gguf, list_gguf_files, pick_one_gguf
from ollama_tools.modelfile import build_modelfile
from ollama_tools.recipe import load_recipe
from ollama_tools.run_helpers import (
    check_item,
    get_jsonl_paths_or_exit,
    require_ollama,
    run_cmd,
    run_ollama_create,
    write_temp_text_file,
)
from ollama_tools.training_data import (
    convert_jsonl_to_plain_text,
    validate_training_data_paths,
)


def _cmd_create_from_base(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    content = build_modelfile(
        args.base,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        adapter=args.adapter,
    )
    return run_ollama_create(args.name, content, out_path=args.out_modelfile)


def _cmd_fetch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Download a GGUF from Hugging Face and create an Ollama model (one command)."""
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    try:
        if args.gguf_file:
            gguf_path = download_gguf(
                args.repo_id,
                args.gguf_file,
                revision=args.revision,
            )
        else:
            gguf_files = list_gguf_files(args.repo_id, revision=args.revision)
            if not gguf_files:
                print(
                    f"No .gguf files found in {args.repo_id}. "
                    "Use a repo that has GGUF files, or convert the model first.",
                    file=sys.stderr,
                )
                return 1
            chosen = pick_one_gguf(gguf_files, prefer_quant=getattr(args, "quant", None))
            if len(gguf_files) > 1:
                print(
                    f"Multiple .gguf files found; using {chosen!r}. "
                    "Use --gguf-file <filename> to pick another.",
                    file=sys.stderr,
                )
            gguf_path = download_gguf(args.repo_id, chosen, revision=args.revision)
        print(f"Downloaded to {gguf_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error downloading: {e}", file=sys.stderr)
        return 1
    # Run convert with the downloaded path
    fake = argparse.Namespace(
        gguf=gguf_path,
        name=args.name,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        out_modelfile=args.out_modelfile,
    )
    return _cmd_convert(parser, fake)


def _cmd_fetch_adapter(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Download an adapter from Hugging Face and create an Ollama model (base + adapter)."""
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    adapter_dir = Path(args.output) if args.output else Path(
        tempfile.mkdtemp(prefix="ollama-adapter-")
    )
    try:
        download_adapter(
            args.repo_id,
            revision=args.revision,
            local_dir=adapter_dir,
        )
        print(f"Downloaded adapter to {adapter_dir}", file=sys.stderr)
    except Exception as e:
        print(f"Error downloading adapter: {e}", file=sys.stderr)
        return 1
    fake = argparse.Namespace(
        base=args.base,
        name=args.name,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        adapter=str(adapter_dir),
        out_modelfile=args.out_modelfile,
    )
    return _cmd_create_from_base(parser, fake)


def _cmd_convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Create an Ollama model from a GGUF file (e.g. after HF→GGUF via llama.cpp)."""
    gguf = Path(args.gguf).resolve()
    if not gguf.is_file():
        print(f"Error: GGUF file not found: {gguf}", file=sys.stderr)
        return 1
    gguf_to_use = str(gguf)
    if getattr(args, "quantize", None):
        q = args.quantize
        quantize_bin = shutil.which("quantize")
        if not quantize_bin:
            print(
                "Error: --quantize requires llama.cpp 'quantize' on PATH. "
                "Build llama.cpp and add it to PATH, or use a pre-quantized GGUF.",
                file=sys.stderr,
            )
            return 1
        out_gguf = gguf.parent / f"{gguf.stem}-{q}.gguf"
        code = run_cmd(
            [quantize_bin, str(gguf), str(out_gguf), q],
            not_found_message="Error: quantize not found.",
            process_error_message="Quantize failed: {e}",
        )
        if code != 0:
            return code
        print(f"Quantized to {out_gguf}", file=sys.stderr)
        gguf_to_use = str(out_gguf)
    content = build_modelfile(
        gguf_to_use,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
    )
    return run_ollama_create(args.name, content, out_path=args.out_modelfile)


def _cmd_validate_training_data(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Validate JSONL training data (instruction/input/output format)."""
    paths = get_jsonl_paths_or_exit(args.data)
    if paths is None:
        return 1
    ok, errors, count = validate_training_data_paths(paths)
    if ok:
        print(f"OK: {count} valid line(s) in {len(paths)} file(s)")
        return 0
    for msg in errors:
        print(msg, file=sys.stderr)
    return 1


def _cmd_prepare_training_data(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Convert JSONL training data to plain text for trainers (e.g. llama.cpp)."""
    paths = get_jsonl_paths_or_exit(args.data)
    if paths is None:
        return 1
    ok, errors, _ = validate_training_data_paths(paths)
    if not ok:
        for msg in errors:
            print(msg, file=sys.stderr)
        print("Fix errors above before preparing.", file=sys.stderr)
        return 1
    out = Path(args.output)
    try:
        convert_jsonl_to_plain_text(
            paths, out, format_name=args.format
        )
    except OSError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {len(paths)} file(s) → {out} ({out.stat().st_size} bytes)")
    print(
        "Use with llama.cpp finetune: --train-data ... --sample-start '### Instruction'",
        file=sys.stderr,
    )
    return 0


def _cmd_train(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Generate a training script: validate → prepare data → trainer → retrain."""
    paths = get_jsonl_paths_or_exit(
        args.data,
        error_msg="Error: no .jsonl files found at --data. Use a file or directory.",
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
    if base_gguf:
        base_gguf_var = f'BASE_GGUF="{base_gguf}"'
    else:
        base_gguf_var = 'BASE_GGUF=""  # set to your base .gguf path'
    run_finetune_block = ""
    if base_gguf and run_trainer:
        run_finetune_block = """
if command -v finetune >/dev/null 2>&1; then
  echo "Step 3: Running llama.cpp finetune..."
  finetune --train-data "$PREPARED" --sample-start '### Instruction' \\
    --model-base "$BASE_GGUF" --lora-out "$ADAPTER_DIR" || true
else
  echo "Step 3: finetune not on PATH. Run: ollama-tools setup-llama-cpp and add to PATH."
  echo "  Then: finetune --train-data \$PREPARED --sample-start '### Instruction' \\"
  echo "    --model-base \$BASE_GGUF --lora-out \$ADAPTER_DIR"
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
# Generated by ollama-tools train --data ... --base {base} --name {name}
set -e
DATA="{data_spec}"
BASE="{base}"
NAME="{name}"
{base_gguf_var}
PREPARED="train_prepared.txt"
ADAPTER_DIR="./adapter_out"

echo "Step 1: Validating data..."
ollama-tools validate-training-data "$DATA"
echo "Step 2: Preparing data for llama.cpp (plain text)..."
ollama-tools prepare-training-data "$DATA" -o "$PREPARED" --format llama.cpp
{run_finetune_block}
echo "Step 4: After training, create Ollama model:"
echo "  ollama-tools retrain --base $BASE --adapter $ADAPTER_DIR --name $NAME"
echo ""
echo "Then: ollama run $NAME"
"""
    if getattr(args, "write_script", None):
        out_path = Path(args.write_script)
        out_path.write_text(script, encoding="utf-8")
        out_path.chmod(0o755)
        print(f"Wrote script to {out_path}. Run it: ./{out_path}")
        return 0
    print(script)
    return 0


def _cmd_retrain(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Create an Ollama model from a base + adapter (after training)."""
    return _cmd_create_from_base(parser, args)


def _namespace_for_fetch(
    recipe: dict, out_modelfile: str | None
) -> argparse.Namespace:
    """Build a Namespace for _cmd_fetch from a recipe dict."""
    return argparse.Namespace(
        repo_id=recipe["hf_repo"],
        name=recipe["name"],
        gguf_file=recipe.get("gguf_file"),
        revision=recipe.get("revision", "main"),
        system=recipe.get("system"),
        temperature=recipe.get("temperature"),
        num_ctx=recipe.get("num_ctx"),
        out_modelfile=out_modelfile,
    )


def _namespace_for_convert(
    recipe: dict, gguf_path: Path, out_modelfile: str | None
) -> argparse.Namespace:
    """Build a Namespace for _cmd_convert from a recipe dict."""
    return argparse.Namespace(
        gguf=str(gguf_path),
        name=recipe["name"],
        system=recipe.get("system"),
        temperature=recipe.get("temperature"),
        num_ctx=recipe.get("num_ctx"),
        out_modelfile=out_modelfile,
    )


def _namespace_for_create_from_base(
    recipe: dict, out_modelfile: str | None
) -> argparse.Namespace:
    """Build a Namespace for _cmd_create_from_base from a recipe dict."""
    return argparse.Namespace(
        base=recipe["base"],
        name=recipe["name"],
        system=recipe.get("system"),
        temperature=recipe.get("temperature"),
        num_ctx=recipe.get("num_ctx"),
        adapter=recipe.get("adapter"),
        out_modelfile=out_modelfile,
    )


def _cmd_build(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Build an Ollama model from a recipe file (YAML/JSON)."""
    try:
        recipe = load_recipe(args.recipe)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    out_modelfile = getattr(args, "out_modelfile", None)
    if "hf_repo" in recipe:
        return _cmd_fetch(parser, _namespace_for_fetch(recipe, out_modelfile))
    if "gguf" in recipe:
        gguf = Path(recipe["gguf"]).resolve()
        if not gguf.is_file():
            print(f"Error: GGUF file not found: {gguf}", file=sys.stderr)
            return 1
        return _cmd_convert(
            parser, _namespace_for_convert(recipe, gguf, out_modelfile)
        )
    return _cmd_create_from_base(
        parser, _namespace_for_create_from_base(recipe, out_modelfile)
    )


def _cmd_check(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Verify ollama, HF, optional deps, and llama.cpp; print what's missing."""
    ok = True
    ok = check_item(
        "ollama",
        bool(shutil.which("ollama")),
        "install from https://ollama.com and add to PATH",
    ) and ok
    try:
        from huggingface_hub import HfApi
        HfApi()
        hf_ok = True
    except ImportError:
        hf_ok = False
    ok = check_item("huggingface_hub", hf_ok, "run: uv sync") and ok
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("HF_TOKEN: set (for gated/private repos)")
    else:
        print("HF_TOKEN: not set (optional; needed for gated/private Hugging Face)")
    try:
        import yaml  # noqa: F401
        yaml_ok = True
    except ImportError:
        yaml_ok = False
    ok = check_item(
        "pyyaml", yaml_ok, "run: uv sync (included by default)"
    ) and ok
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        abliterate_ok = True
    except ImportError:
        abliterate_ok = False
    check_item(
        "abliterate deps",
        abliterate_ok,
        "run: uv sync --extra abliterate",
    )
    finetune = shutil.which("finetune") or shutil.which("llama-finetune")
    quantize = shutil.which("quantize")
    check_item(
        "llama.cpp finetune",
        bool(finetune),
        "for train/run-trainer, build llama.cpp and add finetune to PATH, "
        "or run: ollama-tools setup-llama-cpp",
    )
    check_item(
        "llama.cpp quantize",
        bool(quantize),
        "optional for convert --quantize",
    )
    return 0 if ok else 1


def _cmd_setup_llama_cpp(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Clone and build llama.cpp; print instructions to add to PATH."""
    target_dir = Path(args.dir or "llama.cpp").resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        print(
            f"Directory already exists and is non-empty: {target_dir}. "
            "Use --dir <other> or remove it.",
            file=sys.stderr,
        )
        return 1
    url = "https://github.com/ggerganov/llama.cpp"
    print(f"Cloning {url} into {target_dir}...", file=sys.stderr)
    code = run_cmd(
        ["git", "clone", "--depth", "1", url, str(target_dir)],
        not_found_message="Error: git not found. Install git and try again.",
        process_error_message="Error: git clone failed: {e}",
    )
    if code != 0:
        return code
    build_dir = target_dir / "build"
    build_dir.mkdir(exist_ok=True)
    print("Building (cmake)...", file=sys.stderr)
    code = run_cmd(
        ["cmake", ".."],
        not_found_message="Error: cmake not found. Install cmake and try again.",
        process_error_message="Error: cmake failed: {e}",
        cwd=build_dir,
    )
    if code != 0:
        return code
    code = run_cmd(
        ["cmake", "--build", ".", "--config", "Release"],
        not_found_message="Error: cmake not found.",
        process_error_message="Error: build failed: {e}",
        cwd=build_dir,
    )
    if code != 0:
        return code
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        bin_dir = build_dir  # some layouts put binaries in build/
    print(f"\nDone. Add to PATH: export PATH=\"{bin_dir}:$PATH\"")
    print("Then you can use: finetune, quantize, and other llama.cpp tools.")
    return 0


def _cmd_adapters_search(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Search Hugging Face for adapters and show how to use them."""
    from huggingface_hub import HfApi

    query = getattr(args, "query", None) or "lora adapter"
    limit = getattr(args, "limit", 10)
    api = HfApi()
    print(f"Searching Hugging Face for '{query}' (max {limit})...\n", file=sys.stderr)
    try:
        models = list(api.list_models(search=query, limit=limit))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not models:
        print("No adapters found. Try another search (e.g. 'llama lora', 'mistral adapter').")
        return 0
    print("Adapters you can use with fetch-adapter:\n")
    for m in models:
        repo = m.id
        print(f"  {repo}")
        print(f"    → ollama-tools fetch-adapter {repo} --base <BASE_MODEL> --name <NAME>")
    print("\nReplace <BASE_MODEL> with the model the adapter was trained for (e.g. llama3.2).")
    return 0


def _cmd_downsize_pipeline(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Print the downsize (distillation) pipeline steps, or generate a script."""
    teacher = getattr(args, "teacher", None)
    student = getattr(args, "student", None)
    name = getattr(args, "name", None)
    quantize = getattr(args, "quantize", None)
    write_script = getattr(args, "write_script", None)

    if teacher and student and name:
        # Generate concrete steps or script
        q_flag = f" --quantize {quantize}" if quantize else ""
        steps = f"""# Downsize: {teacher} (teacher) → {student} (student) → Ollama model '{name}'

1. Download teacher and student (for distillation):
   huggingface-cli download {teacher} --local-dir ./teacher
   huggingface-cli download {student} --local-dir ./student

2. Run distillation (train student to mimic teacher). Example with TRL GKD:
   pip install trl
   # See https://huggingface.co/docs/trl (GKD trainer); then export student to GGUF with llama.cpp.

3. Create Ollama model from the student GGUF:
   ollama-tools convert --gguf <path/to/student.gguf> --name {name}{q_flag}
   ollama run {name}
"""
        if write_script:
            path = Path(write_script)
            path.write_text(steps, encoding="utf-8")
            print(f"Wrote steps to {path}. Run the commands in order.")
            return 0
        print(steps)
        return 0

    # Default: generic pipeline
    steps = """
Downsize pipeline (teacher → student → Ollama):

1. Choose teacher (large) and student (small) model — e.g. 30B and 3B from same family.
2. Run distillation externally (TRL GKD, Axolotl, Unsloth, or custom).
3. Export student to GGUF (llama.cpp), then:
   ollama-tools convert --gguf <path/to/student.gguf> --name my-downsized [--quantize Q4_K_M]
   ollama run my-downsized

Simpler: use --teacher, --student, --name (and optional --quantize, --write-script).
"""
    print(steps.strip())
    return 0


def _collect_instructions_from_path(path: str | Path) -> list[str]:
    """Collect non-empty lines from a file or from all .txt files in a directory."""
    p = Path(path)
    lines: list[str] = []
    if p.is_file():
        for line in p.open(encoding="utf-8"):
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    elif p.is_dir():
        for f in sorted(p.glob("*.txt")):
            for line in f.open(encoding="utf-8"):
                s = line.strip()
                if s and not s.startswith("#"):
                    lines.append(s)
    return lines


def _resolve_abliterate_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    """Resolve harmful/harmless to two file paths. Returns (harmful_path, harmless_path, temp_files)."""  # noqa: E501
    from ollama_tools.abliterate_defaults import HARMFUL_DEFAULT, HARMLESS_DEFAULT

    data_dir = Path(__file__).parent / "data"
    default_harmful_file = data_dir / "abliterate_harmful_default.txt"
    default_harmless_file = data_dir / "abliterate_harmless_default.txt"

    harmful_path: Path
    harmless_path: Path
    temp_files: list[Path] = []

    if getattr(args, "harmful_dir", None) and getattr(args, "harmless_dir", None):
        h_lines = _collect_instructions_from_path(args.harmful_dir)
        l_lines = _collect_instructions_from_path(args.harmless_dir)
        if not h_lines or not l_lines:
            raise FileNotFoundError(
                "No instructions in --harmful-dir and/or --harmless-dir"
            )
        harmful_path = write_temp_text_file(
            "\n".join(h_lines), suffix=".txt", prefix="ollama-harmful-"
        )
        harmless_path = write_temp_text_file(
            "\n".join(l_lines), suffix=".txt", prefix="ollama-harmless-"
        )
        temp_files = [harmful_path, harmless_path]
    elif getattr(args, "harmful", None) and getattr(args, "harmless", None):
        harmful_path = Path(args.harmful)
        harmless_path = Path(args.harmless)
    elif default_harmful_file.is_file() and default_harmless_file.is_file():
        harmful_path = default_harmful_file
        harmless_path = default_harmless_file
        print(
            "Using bundled default harmful/harmless lists. "
            "Pass --harmful/--harmless or --harmful-dir/--harmless-dir for custom.",
            file=sys.stderr,
        )
    else:
        harmful_path = write_temp_text_file(
            HARMFUL_DEFAULT.strip(), suffix=".txt", prefix="ollama-harmful-"
        )
        harmless_path = write_temp_text_file(
            HARMLESS_DEFAULT.strip(), suffix=".txt", prefix="ollama-harmless-"
        )
        temp_files = [harmful_path, harmless_path]
        print(
            "Using built-in default harmful/harmless lists. "
            "Pass --harmful/--harmless or --harmful-dir/--harmless-dir for custom.",
            file=sys.stderr,
        )

    return harmful_path, harmless_path, temp_files


def _cmd_abliterate_compute_dir(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    """Compute refusal direction for abliteration (requires: uv sync --extra abliterate)."""
    try:
        from ollama_tools.abliterate import compute_refusal_dir
    except ImportError as e:
        print(
            "Error: abliterate requires optional deps. Run: uv sync --extra abliterate",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 1
    try:
        harmful_path, harmless_path, temp_files = _resolve_abliterate_inputs(args)
        try:
            compute_refusal_dir(
                args.model,
                str(harmful_path),
                str(harmless_path),
                args.output,
                num_instructions=args.num_instructions,
                layer_frac=args.layer_frac,
            )
            print(f"Saved refusal direction to {args.output}")
            return 0
        finally:
            for t in temp_files:
                Path(t).unlink(missing_ok=True)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ollama-tools",
        description="Create, retrain, ablate, and convert models for local Ollama.",
        epilog="Quick start: ollama-tools fetch <HF_REPO> --name my-model",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # check (environment)
    p_check = subparsers.add_parser(
        "check",
        help="Verify ollama, Hugging Face, optional deps, and llama.cpp",
    )
    p_check.set_defaults(handler=_cmd_check)

    # setup-llama-cpp (clone and build)
    p_setup = subparsers.add_parser(
        "setup-llama-cpp",
        help="Clone and build llama.cpp (finetune, quantize); add build dir to PATH",
    )
    p_setup.add_argument(
        "--dir",
        default=None,
        help="Directory to clone into (default: ./llama.cpp)",
    )
    p_setup.set_defaults(handler=_cmd_setup_llama_cpp)

    # create-from-base
    p_create = subparsers.add_parser(
        "create-from-base",
        help="Create a new model from a base model (Modelfile)",
    )
    p_create.add_argument(
        "--base",
        required=True,
        help="Base model name or path (e.g. llama3.2 or /path/to/model.gguf)",
    )
    p_create.add_argument("--name", required=True, help="Name for the new model")
    p_create.add_argument("--system", help="System message (role/instructions)")
    p_create.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_create.add_argument("--num-ctx", type=int, help="Context window size in tokens (e.g. 4096)")
    p_create.add_argument("--adapter", help="Path to LoRA/adapter directory")
    p_create.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_create.set_defaults(handler=_cmd_create_from_base)

    # convert (GGUF → Ollama; use after HF→GGUF via llama.cpp)
    p_convert = subparsers.add_parser(
        "convert",
        help="Create an Ollama model from a GGUF file (e.g. after converting HF with llama.cpp)",
    )
    p_convert.add_argument(
        "--gguf",
        required=True,
        help="Path to the .gguf model file",
    )
    p_convert.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_convert.add_argument("--system", help="System message (role/instructions)")
    p_convert.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_convert.add_argument("--num-ctx", type=int, help="Context window size in tokens (e.g. 4096)")
    p_convert.add_argument(
        "--quantize",
        help="Quantize the GGUF first (e.g. Q4_K_M); requires llama.cpp 'quantize' on PATH",
    )
    p_convert.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_convert.set_defaults(handler=_cmd_convert)

    # fetch (HF repo → download GGUF → create Ollama model)
    p_fetch = subparsers.add_parser(
        "fetch",
        help="Download a GGUF from Hugging Face and create an Ollama model (one command)",
    )
    p_fetch.add_argument(
        "repo_id",
        help="Hugging Face repo id (e.g. TheBloke/Llama-2-7B-GGUF)",
    )
    p_fetch.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_fetch.add_argument(
        "--gguf-file",
        help="Specific .gguf filename if repo has multiple (optional)",
    )
    p_fetch.add_argument(
        "--quant",
        help="Prefer this quantization when repo has multiple GGUF files (e.g. Q4_K_M, Q8_0)",
    )
    p_fetch.add_argument("--revision", default="main", help="Repo revision (default: main)")
    p_fetch.add_argument("--system", help="System message (role/instructions)")
    p_fetch.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_fetch.add_argument("--num-ctx", type=int, help="Context window size in tokens (e.g. 4096)")
    p_fetch.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_fetch.set_defaults(handler=_cmd_fetch)

    # fetch-adapter (HF adapter repo → download → create-from-base)
    p_fetch_adapter = subparsers.add_parser(
        "fetch-adapter",
        help="Download an adapter from Hugging Face and create an Ollama model (base + adapter)",
    )
    p_fetch_adapter.add_argument(
        "repo_id",
        help="Hugging Face repo id of the adapter (e.g. user/my-lora)",
    )
    p_fetch_adapter.add_argument("--base", required=True, help="Base model name or path")
    p_fetch_adapter.add_argument("--name", required=True, help="Name for the new model")
    p_fetch_adapter.add_argument("--revision", default="main", help="Repo revision (default: main)")
    p_fetch_adapter.add_argument(
        "--output",
        help="Directory to download adapter into (default: temp dir)",
    )
    p_fetch_adapter.add_argument("--system", help="System message")
    p_fetch_adapter.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_fetch_adapter.add_argument("--num-ctx", type=int, help="Context window size in tokens")
    p_fetch_adapter.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_fetch_adapter.set_defaults(handler=_cmd_fetch_adapter)

    # build (from recipe YAML/JSON)
    p_build = subparsers.add_parser(
        "build",
        help="Build an Ollama model from a recipe file (YAML or JSON)",
    )
    p_build.add_argument(
        "recipe",
        help="Path to recipe file (.yaml, .yml, or .json) with name and base or gguf",
    )
    p_build.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_build.set_defaults(handler=_cmd_build)

    # validate-training-data
    p_validate = subparsers.add_parser(
        "validate-training-data",
        help="Validate JSONL training data (file(s) or directory)",
    )
    p_validate.add_argument(
        "data",
        nargs="+",
        help="Path(s) to .jsonl file(s) or a directory of .jsonl files",
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
    p_prepare.add_argument("-o", "--output", required=True, help="Output file path")
    p_prepare.add_argument(
        "--format",
        default="llama.cpp",
        help="Output format (default: llama.cpp)",
    )
    p_prepare.set_defaults(handler=_cmd_prepare_training_data)

    # train (generate script: data → prepare → trainer → retrain)
    p_train = subparsers.add_parser(
        "train",
        help="Generate a training script: pass your data path, get a runnable pipeline",
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
        "--write-script",
        metavar="PATH",
        help="Write the pipeline script to this file",
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
        help="Path to adapter directory (from llama.cpp finetune, Axolotl, etc.)",
    )
    p_retrain.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_retrain.add_argument("--system", help="System message")
    p_retrain.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_retrain.add_argument("--num-ctx", type=int, help="Context window size in tokens")
    p_retrain.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_retrain.set_defaults(handler=_cmd_retrain)

    # abliterate (refusal removal)
    p_abliterate = subparsers.add_parser(
        "abliterate",
        help="Refusal removal (abliteration); use compute-dir then Sumandora or export to GGUF",
    )
    abliterate_sub = p_abliterate.add_subparsers(dest="abliterate_command")
    p_compute = abliterate_sub.add_parser(
        "compute-dir",
        help="Compute refusal direction from harmful/harmless instructions "
        "(needs: uv sync --extra abliterate)",
    )
    p_compute.add_argument("--model", required=True, help="Hugging Face model id")
    p_compute.add_argument(
        "--harmful",
        help="Path to file with harmful instructions (one per line)",
    )
    p_compute.add_argument(
        "--harmless",
        help="Path to file with harmless instructions (one per line)",
    )
    p_compute.add_argument(
        "--harmful-dir",
        help="Directory of .txt files with harmful instructions (alternative to --harmful)",
    )
    p_compute.add_argument(
        "--harmless-dir",
        help="Directory of .txt files with harmless instructions (alternative to --harmless)",
    )
    p_compute.add_argument(
        "--output",
        required=True,
        help="Output path for .pt file",
    )
    p_compute.add_argument(
        "--num-instructions",
        type=int,
        default=32,
        help="Number of instructions to sample (default: 32)",
    )
    p_compute.add_argument(
        "--layer-frac",
        type=float,
        default=0.6,
        help="Layer index as fraction of depth (default: 0.6)",
    )
    p_compute.set_defaults(handler=_cmd_abliterate_compute_dir)

    # adapters (search Hugging Face for adapters)
    p_adapters = subparsers.add_parser(
        "adapters",
        help="Find and use adapters (e.g. search Hugging Face)",
    )
    adapters_sub = p_adapters.add_subparsers(dest="adapters_command")
    p_adapters_search = adapters_sub.add_parser(
        "search",
        help="Search Hugging Face for adapters; shows fetch-adapter commands",
    )
    p_adapters_search.add_argument(
        "query",
        nargs="?",
        default="lora adapter",
        help="Search query (default: lora adapter)",
    )
    p_adapters_search.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of results (default: 10)",
    )
    p_adapters_search.set_defaults(handler=_cmd_adapters_search)

    # downsize (distillation: large → small model)
    p_downsize = subparsers.add_parser(
        "downsize",
        help="Downsize via distillation; use --teacher, --student, --name for exact steps",
    )
    p_downsize.add_argument(
        "--teacher",
        help="Hugging Face repo id of the teacher (large) model",
    )
    p_downsize.add_argument(
        "--student",
        help="Hugging Face repo id of the student (small) model",
    )
    p_downsize.add_argument(
        "--name",
        help="Name for the final Ollama model (use with --teacher and --student)",
    )
    p_downsize.add_argument(
        "--quantize",
        help="Quantization for the student GGUF (e.g. Q4_K_M) before convert",
    )
    p_downsize.add_argument(
        "--write-script",
        metavar="PATH",
        help="Write the step-by-step commands to this file",
    )
    downsize_sub = p_downsize.add_subparsers(dest="downsize_command")
    p_ds_pipeline = downsize_sub.add_parser(
        "pipeline",
        help="Print the generic downsize pipeline steps",
    )
    p_ds_pipeline.set_defaults(handler=_cmd_downsize_pipeline)

    parsed = parser.parse_args()
    if not parsed.command:
        parser.print_help()
        return 0
    if parsed.command == "abliterate" and not getattr(parsed, "abliterate_command", None):
        p_abliterate.print_help()
        return 0
    if parsed.command == "adapters" and not getattr(parsed, "adapters_command", None):
        p_adapters.print_help()
        return 0
    if parsed.command == "downsize" and not getattr(parsed, "downsize_command", None):
        _cmd_downsize_pipeline(parser, parsed)
        return 0
    handler = getattr(parsed, "handler", None)
    if handler is None:
        return 0
    return handler(parser, parsed)


if __name__ == "__main__":
    sys.exit(main())
