"""CLI entrypoint for ollama-forge."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

from ollama_forge.hf_fetch import download_adapter, download_gguf, list_gguf_files, pick_one_gguf
from ollama_forge.modelfile import build_modelfile, merge_modelfile_with_reference_template
from ollama_forge.recipe import load_recipe
from ollama_forge.run_helpers import (
    check_item,
    get_jsonl_paths_or_exit,
    print_actionable_error,
    require_ollama,
    run_cmd,
    run_ollama_create,
    run_ollama_show_modelfile,
    write_temp_text_file,
)
from ollama_forge.training_data import (
    convert_jsonl_to_plain_text,
    validate_training_data_paths,
)


def _which_quantize() -> str | None:
    """Resolve llama.cpp quantize binary (quantize or llama-quantize)."""
    return shutil.which("quantize") or shutil.which("llama-quantize")


def _hf_token_available() -> bool:
    """True if HF token is set via env, .env, or huggingface-cli login."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


_QUICKSTART_PROFILES: dict[str, dict[str, float | int | str]] = {
    "fast": {
        "quant": "Q4_0",
        "temperature": 0.8,
        "num_ctx": 2048,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
    },
    "balanced": {
        "quant": "Q4_K_M",
        "temperature": 0.7,
        "num_ctx": 4096,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    },
    "quality": {
        "quant": "Q8_0",
        "temperature": 0.6,
        "num_ctx": 8192,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
    },
    "low-vram": {
        "quant": "Q4_0",
        "temperature": 0.7,
        "num_ctx": 2048,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    },
}

_QUICKSTART_TASK_SYSTEMS: dict[str, str] = {
    "chat": "You are a helpful and concise assistant.",
    "coding": "You are a senior coding assistant. Provide practical, safe code solutions.",
    "creative": "You are a creative assistant. Write vivid, engaging, and original responses.",
}


def _cmd_create_from_base(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    content = build_modelfile(
        args.base,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        adapter=args.adapter,
    )
    return run_ollama_create(args.name, content, out_path=args.out_modelfile)


def _cmd_refresh_template(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Recreate a model using the base model's latest chat template (fixes Chat API issues)."""
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    name = getattr(args, "name", None)
    base = getattr(args, "base", None)
    output_name = getattr(args, "output_name", None) or name
    current = run_ollama_show_modelfile(name)
    if not current:
        print_actionable_error(
            f"Could not get Modelfile for model {name!r}",
            next_steps=[
                f"Ensure the model exists: ollama run {name}",
                "Use a model name that is already created in Ollama",
            ],
        )
        return 1
    reference = run_ollama_show_modelfile(base)
    if not reference:
        print_actionable_error(
            f"Could not get Modelfile for base {base!r}",
            next_steps=[
                f"Pull the base model first: ollama pull {base}",
                "Use a base model name that exists in Ollama (e.g. llama3.2)",
            ],
        )
        return 1
    template_only = getattr(args, "template_only", False)
    merged = merge_modelfile_with_reference_template(current, reference, base, template_only=template_only)
    return run_ollama_create(output_name, merged, out_path=getattr(args, "out_modelfile", None))


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
                print_actionable_error(
                    f"no .gguf files found in {args.repo_id}",
                    next_steps=[
                        "Use a repo that already includes GGUF files",
                        "Or convert from HF to GGUF first, then run: ollama-forge convert --gguf <path> --name <name>",
                    ],
                )
                return 1
            chosen = pick_one_gguf(gguf_files, prefer_quant=getattr(args, "quant", None))
            if len(gguf_files) > 1:
                print(
                    f"Multiple .gguf files found; using {chosen!r}. Use --gguf-file <filename> to pick another.",
                    file=sys.stderr,
                )
            gguf_path = download_gguf(args.repo_id, chosen, revision=args.revision)
        print(f"Downloaded to {gguf_path}", file=sys.stderr)
    except Exception as e:
        print_actionable_error(
            "download failed",
            cause=str(e),
            next_steps=[
                "If the repo is gated/private, run: huggingface-cli login",
                "Or set: HF_TOKEN=<your_token>",
                "Try: ollama-forge check",
            ],
        )
        return 1
    # Run convert with the downloaded path
    fake = argparse.Namespace(
        gguf=gguf_path,
        name=args.name,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        out_modelfile=args.out_modelfile,
    )
    return _cmd_convert(parser, fake)


def _cmd_quickstart(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Beginner one-command path: fetch a default GGUF and create an Ollama model."""
    repo_id = getattr(args, "repo_id", "TheBloke/Llama-2-7B-GGUF")
    name = getattr(args, "name", "my-model")
    profile = getattr(args, "profile", "balanced")
    cfg = _QUICKSTART_PROFILES[profile]
    quant = getattr(args, "quant", None) or str(cfg["quant"])
    temperature = (
        getattr(args, "temperature", None)
        if getattr(args, "temperature", None) is not None
        else float(cfg["temperature"])
    )
    num_ctx = getattr(args, "num_ctx", None) if getattr(args, "num_ctx", None) is not None else int(cfg["num_ctx"])
    top_p = getattr(args, "top_p", None) if getattr(args, "top_p", None) is not None else float(cfg["top_p"])
    repeat_penalty = (
        getattr(args, "repeat_penalty", None)
        if getattr(args, "repeat_penalty", None) is not None
        else float(cfg["repeat_penalty"])
    )
    task = getattr(args, "task", None)
    system = getattr(args, "system", None)
    system_source = "custom"
    if system is None and task in _QUICKSTART_TASK_SYSTEMS:
        system = _QUICKSTART_TASK_SYSTEMS[task]
        system_source = f"task:{task}"
    elif system is None:
        system_source = "none"
    if not getattr(args, "json", False):
        print("Quickstart plan:", file=sys.stderr)
        print(f"  model name: {name}", file=sys.stderr)
        print(f"  repo: {repo_id}@{getattr(args, 'revision', 'main')}", file=sys.stderr)
        print(f"  profile/task: {profile} / {task or 'none'}", file=sys.stderr)
        print(
            f"  quant/temp/ctx/top_p/repeat: {quant} / {temperature} / {num_ctx} / {top_p} / {repeat_penalty}",
            file=sys.stderr,
        )
        print(f"  system prompt source: {system_source}", file=sys.stderr)
    if getattr(args, "plan", False):
        action = f"ollama-forge fetch {repo_id} --name {name} --quant {quant}"
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "route": "quickstart",
                        "source": repo_id,
                        "name": name,
                        "profile": profile,
                        "task": task,
                        "revision": getattr(args, "revision", "main"),
                        "quant": quant,
                        "temperature": temperature,
                        "num_ctx": num_ctx,
                        "top_p": top_p,
                        "repeat_penalty": repeat_penalty,
                        "system_source": system_source,
                        "action": action,
                    }
                )
            )
        else:
            print(f"  action: {action}", file=sys.stderr)
        return 0
    fake = argparse.Namespace(
        repo_id=repo_id,
        name=name,
        gguf_file=None,
        quant=quant,
        revision=getattr(args, "revision", "main"),
        system=system,
        temperature=temperature,
        num_ctx=num_ctx,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        out_modelfile=getattr(args, "out_modelfile", None),
    )
    code = _cmd_fetch(parser, fake)
    if code == 0:
        print(f"Done. Run your model with: ollama run {name}")
    return code


def _cmd_start(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Alias for beginner defaults (same as quickstart)."""
    fake = argparse.Namespace(
        name=getattr(args, "name", "my-model"),
        profile=getattr(args, "profile", "balanced"),
        repo_id=getattr(args, "repo_id", "TheBloke/Llama-2-7B-GGUF"),
        quant=getattr(args, "quant", None),
        revision=getattr(args, "revision", "main"),
        task=getattr(args, "task", None),
        system=getattr(args, "system", None),
        temperature=getattr(args, "temperature", None),
        num_ctx=getattr(args, "num_ctx", None),
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        out_modelfile=getattr(args, "out_modelfile", None),
    )
    return _cmd_quickstart(parser, fake)


def _cmd_plan_quickstart(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Plan wrapper for quickstart."""
    fake = argparse.Namespace(
        name=args.name,
        profile=args.profile,
        repo_id=args.repo_id,
        quant=args.quant,
        revision=args.revision,
        task=args.task,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        out_modelfile=args.out_modelfile,
        plan=True,
        json=getattr(args, "json", False),
    )
    return _cmd_quickstart(parser, fake)


def _cmd_plan_auto(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Plan wrapper for auto routing."""
    fake = argparse.Namespace(
        source=args.source,
        name=args.name,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        base=args.base,
        adapter=args.adapter,
        output=args.output,
        gguf_file=args.gguf_file,
        quant=args.quant,
        quantize=args.quantize,
        revision=args.revision,
        no_prompt=args.no_prompt,
        out_modelfile=args.out_modelfile,
        plan=True,
        json=getattr(args, "json", False),
    )
    return _cmd_auto(parser, fake)


def _cmd_plan_doctor_fix(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Plan wrapper for doctor --fix."""
    fake = argparse.Namespace(
        fix=True,
        plan=True,
        fix_llama_cpp=args.fix_llama_cpp,
        llama_cpp_dir=args.llama_cpp_dir,
        json=getattr(args, "json", False),
    )
    return _cmd_doctor(parser, fake)


def _cmd_plan_adapters_apply(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Plan wrapper for adapters recommend --apply."""
    fake = argparse.Namespace(
        base=args.base,
        query=args.query,
        limit=args.limit,
        apply=True,
        plan=True,
        name=args.name,
        revision=args.revision,
        output=args.output,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        out_modelfile=args.out_modelfile,
        json=getattr(args, "json", False),
    )
    return _cmd_adapters_recommend(parser, fake)


def _detect_auto_source(source: str) -> str:
    """
    Detect source type for auto workflow.
    Returns one of: recipe, gguf, local_dir, hf_repo, base.
    """
    p = Path(source)
    if p.is_dir():
        return "local_dir"
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml", ".json"):
        return "recipe"
    if suffix == ".gguf":
        return "gguf"
    if "/" in source:
        return "hf_repo"
    return "base"


def _prompt_with_default(prompt: str, default: str) -> str:
    """Prompt in interactive terminals; return default when blank or non-interactive."""
    if not sys.stdin.isatty():
        return default
    try:
        value = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return value or default


def _is_local_adapter_dir(path: Path) -> bool:
    """Heuristic: detect common adapter artifact files in a local directory."""
    if not path.is_dir():
        return False
    return any(
        (path / filename).exists()
        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    )


def _repo_looks_like_adapter(repo_id: str, revision: str) -> bool:
    """Heuristic: detect adapter-like HF repos by file names."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return False
    try:
        files = list_repo_files(repo_id, revision=revision or "main")
    except Exception:
        return False
    names = {Path(f).name for f in files}
    has_adapter_marker = bool({"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"} & names)
    has_gguf = any(str(f).lower().endswith(".gguf") for f in files)
    return has_adapter_marker and not has_gguf


def _cmd_auto(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Auto route source to build/fetch/convert/create-from-base."""
    source = args.source
    source_type = _detect_auto_source(source)
    prompt_enabled = sys.stdin.isatty() and not getattr(args, "no_prompt", False)
    plan_only = getattr(args, "plan", False)

    def maybe_plan(route: str, detail: str) -> bool:
        if not plan_only:
            return False
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "route": route,
                        "source": source,
                        "action": detail,
                    }
                )
            )
        else:
            print("Auto plan:")
            print(f"  route: {route}")
            print(f"  source: {source}")
            print(f"  action: {detail}")
        return True

    name = args.name
    if source_type != "recipe" and not name:
        name = _prompt_with_default("Model name", "my-model") if prompt_enabled else "my-model"
    if source_type == "recipe":
        fake = argparse.Namespace(recipe=source, out_modelfile=args.out_modelfile)
        if maybe_plan("build", f"ollama-forge build {source}"):
            return 0
        return _cmd_build(parser, fake)
    if source_type == "gguf":
        fake = argparse.Namespace(
            gguf=source,
            name=name,
            quantize=args.quantize,
            system=args.system,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            out_modelfile=args.out_modelfile,
        )
        if maybe_plan(
            "convert",
            f"ollama-forge convert --gguf {source} --name {name}",
        ):
            return 0
        return _cmd_convert(parser, fake)
    if source_type == "local_dir":
        source_path = Path(source).resolve()
        if _is_local_adapter_dir(source_path):
            base = args.base
            if not base:
                base = _prompt_with_default("Base model for adapter", "llama3.2") if prompt_enabled else "llama3.2"
            fake = argparse.Namespace(
                base=base,
                adapter=str(source_path),
                name=name,
                system=args.system,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                out_modelfile=args.out_modelfile,
            )
            if maybe_plan(
                "retrain",
                f"ollama-forge retrain --base {base} --adapter {source_path} --name {name}",
            ):
                return 0
            return _cmd_retrain(parser, fake)
        print_actionable_error(
            f"unsupported local directory source: {source_path}",
            next_steps=[
                "Use auto with a recipe/.gguf/HF repo/base model",
                "Or provide an adapter directory (with adapter_config.json)",
            ],
        )
        return 1
    if source_type == "hf_repo":
        if _repo_looks_like_adapter(source, args.revision):
            base = args.base
            if not base:
                base = _prompt_with_default("Base model for adapter", "llama3.2") if prompt_enabled else "llama3.2"
            fake = argparse.Namespace(
                repo_id=source,
                base=base,
                name=name,
                revision=args.revision,
                output=args.output,
                system=args.system,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                out_modelfile=args.out_modelfile,
            )
            if maybe_plan(
                "fetch-adapter",
                f"ollama-forge fetch-adapter {source} --base {base} --name {name}",
            ):
                return 0
            return _cmd_fetch_adapter(parser, fake)
        quant = args.quant
        if quant is None and prompt_enabled:
            quant = _prompt_with_default("Preferred quantization", "Q4_K_M")
        fake = argparse.Namespace(
            repo_id=source,
            name=name,
            gguf_file=args.gguf_file,
            quant=quant,
            revision=args.revision,
            system=args.system,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            out_modelfile=args.out_modelfile,
        )
        if maybe_plan(
            "fetch",
            f"ollama-forge fetch {source} --name {name}",
        ):
            return 0
        return _cmd_fetch(parser, fake)
    fake = argparse.Namespace(
        base=source,
        name=name,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        adapter=args.adapter,
        out_modelfile=args.out_modelfile,
    )
    if maybe_plan(
        "create-from-base",
        f"ollama-forge create-from-base --base {source} --name {name}",
    ):
        return 0
    return _cmd_create_from_base(parser, fake)


def _cmd_fetch_adapter(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Download an adapter from Hugging Face and create an Ollama model (base + adapter)."""
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    adapter_dir = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="ollama-adapter-"))
    try:
        download_adapter(
            args.repo_id,
            revision=args.revision,
            local_dir=adapter_dir,
        )
        print(f"Downloaded adapter to {adapter_dir}", file=sys.stderr)
    except Exception as e:
        print_actionable_error(
            "adapter download failed",
            cause=str(e),
            next_steps=[
                "Confirm adapter repo id is correct on Hugging Face",
                "If gated/private, run: huggingface-cli login",
                "Then retry fetch-adapter",
            ],
        )
        return 1
    fake = argparse.Namespace(
        base=args.base,
        name=args.name,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        adapter=str(adapter_dir),
        out_modelfile=args.out_modelfile,
    )
    return _cmd_create_from_base(parser, fake)


def _cmd_convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Create an Ollama model from a GGUF file (e.g. after HF→GGUF via llama.cpp)."""
    gguf = Path(args.gguf).resolve()
    if not gguf.is_file():
        print_actionable_error(
            f"GGUF file not found: {gguf}",
            next_steps=[
                "Check the path and file extension (.gguf)",
                "Or fetch a GGUF from HF: ollama-forge fetch <repo_id> --name <name>",
            ],
        )
        return 1
    gguf_to_use = str(gguf)
    if getattr(args, "quantize", None):
        q = args.quantize
        quantize_bin = _which_quantize()
        if not quantize_bin:
            print_actionable_error(
                "--quantize requires llama.cpp 'quantize' or 'llama-quantize' on PATH",
                next_steps=[
                    "Run: ollama-forge setup-llama-cpp",
                    "Add its build/bin path to PATH",
                    "Or use a pre-quantized GGUF and skip --quantize",
                ],
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
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
    )
    return run_ollama_create(args.name, content, out_path=args.out_modelfile)


def _cmd_validate_training_data(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Validate JSONL training data (instruction/input/output format)."""
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
    return 1


def _cmd_prepare_training_data(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Convert JSONL training data to plain text for trainers (e.g. llama.cpp)."""
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
        print("Fix errors above before preparing.", file=sys.stderr)
        return 1
    out = Path(args.output)
    try:
        convert_jsonl_to_plain_text(paths, out, format_name=args.format)
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
    print(f"Wrote {len(paths)} file(s) → {out} ({out.stat().st_size} bytes)")
    print(
        "Use with llama.cpp finetune: --train-data ... --sample-start '### Instruction'",
        file=sys.stderr,
    )
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
  echo "Step 3: finetune not on PATH. Run: ollama-forge setup-llama-cpp and add to PATH."
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
# Generated by ollama-forge train --data ... --base {base} --name {name}
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


def _namespace_for_fetch(recipe: dict, out_modelfile: str | None) -> argparse.Namespace:
    """Build a Namespace for _cmd_fetch from a recipe dict."""
    return argparse.Namespace(
        repo_id=recipe["hf_repo"],
        name=recipe["name"],
        gguf_file=recipe.get("gguf_file"),
        quant=recipe.get("quant"),
        revision=recipe.get("revision", "main"),
        system=recipe.get("system"),
        temperature=recipe.get("temperature"),
        num_ctx=recipe.get("num_ctx"),
        top_p=recipe.get("top_p"),
        repeat_penalty=recipe.get("repeat_penalty"),
        out_modelfile=out_modelfile,
    )


def _namespace_for_convert(recipe: dict, gguf_path: Path, out_modelfile: str | None) -> argparse.Namespace:
    """Build a Namespace for _cmd_convert from a recipe dict."""
    return argparse.Namespace(
        gguf=str(gguf_path),
        name=recipe["name"],
        quantize=recipe.get("quantize"),
        system=recipe.get("system"),
        temperature=recipe.get("temperature"),
        num_ctx=recipe.get("num_ctx"),
        top_p=recipe.get("top_p"),
        repeat_penalty=recipe.get("repeat_penalty"),
        out_modelfile=out_modelfile,
    )


def _namespace_for_create_from_base(recipe: dict, out_modelfile: str | None) -> argparse.Namespace:
    """Build a Namespace for _cmd_create_from_base from a recipe dict."""
    return argparse.Namespace(
        base=recipe["base"],
        name=recipe["name"],
        system=recipe.get("system"),
        temperature=recipe.get("temperature"),
        num_ctx=recipe.get("num_ctx"),
        top_p=recipe.get("top_p"),
        repeat_penalty=recipe.get("repeat_penalty"),
        adapter=recipe.get("adapter"),
        out_modelfile=out_modelfile,
    )


def _cmd_build(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Build an Ollama model from a recipe file (YAML/JSON)."""
    try:
        recipe = load_recipe(args.recipe)
    except FileNotFoundError as e:
        print_actionable_error(
            str(e),
            next_steps=["Check recipe path and retry: ollama-forge build <recipe.yaml>"],
        )
        return 1
    except (ValueError, ImportError) as e:
        print_actionable_error(
            "invalid recipe",
            cause=str(e),
            next_steps=[
                "Run: ollama-forge build <recipe.yaml> --help",
                "Ensure recipe has name and exactly one of base/gguf/hf_repo",
            ],
        )
        return 1
    out_modelfile = getattr(args, "out_modelfile", None)
    if "hf_repo" in recipe:
        return _cmd_fetch(parser, _namespace_for_fetch(recipe, out_modelfile))
    if "gguf" in recipe:
        gguf = Path(recipe["gguf"]).resolve()
        if not gguf.is_file():
            print_actionable_error(
                f"GGUF file not found: {gguf}",
                next_steps=[
                    "Fix the gguf path in recipe",
                    "Or use hf_repo + optional quant in recipe instead",
                ],
            )
            return 1
        return _cmd_convert(parser, _namespace_for_convert(recipe, gguf, out_modelfile))
    return _cmd_create_from_base(parser, _namespace_for_create_from_base(recipe, out_modelfile))


def _cmd_check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Verify ollama, HF, optional deps, and llama.cpp; print what's missing."""
    ok = True
    ok = (
        check_item(
            "ollama",
            bool(shutil.which("ollama")),
            "install from https://ollama.com and add to PATH",
        )
        and ok
    )
    try:
        from huggingface_hub import HfApi

        HfApi()
        hf_ok = True
    except ImportError:
        hf_ok = False
    ok = check_item("huggingface_hub", hf_ok, "run: uv sync") and ok
    if _hf_token_available():
        print("HF_TOKEN: set (for gated/private repos)")
    else:
        print("HF_TOKEN: not set (optional; needed for gated/private Hugging Face)")
    try:
        import yaml  # noqa: F401

        yaml_ok = True
    except ImportError:
        yaml_ok = False
    ok = check_item("pyyaml", yaml_ok, "run: uv sync (included by default)") and ok
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
    quantize = _which_quantize()
    check_item(
        "llama.cpp finetune",
        bool(finetune),
        "for train/run-trainer, build llama.cpp and add finetune to PATH, or run: ollama-forge setup-llama-cpp",
    )
    check_item(
        "llama.cpp quantize",
        bool(quantize),
        "optional for convert --quantize",
    )
    return 0 if ok else 1


def _env_status() -> dict[str, bool]:
    """Collect environment readiness booleans used by check/doctor."""
    try:
        from huggingface_hub import HfApi

        HfApi()
        hf_ok = True
    except ImportError:
        hf_ok = False
    try:
        import yaml  # noqa: F401

        yaml_ok = True
    except ImportError:
        yaml_ok = False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        abliterate_ok = True
    except ImportError:
        abliterate_ok = False
    finetune = bool(shutil.which("finetune") or shutil.which("llama-finetune"))
    quantize = bool(_which_quantize())
    hf_token_set = _hf_token_available()
    return {
        "ollama": bool(shutil.which("ollama")),
        "huggingface_hub": hf_ok,
        "pyyaml": yaml_ok,
        "hf_token": hf_token_set,
        "abliterate_deps": abliterate_ok,
        "finetune": finetune,
        "quantize": quantize,
    }


def _cmd_doctor(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Diagnose environment and optionally apply common fixes."""
    status = _env_status()
    json_mode = bool(getattr(args, "json", False))
    if not json_mode:
        print("Doctor report:")
        check_item(
            "ollama",
            status["ollama"],
            "install from https://ollama.com and add to PATH",
        )
        check_item("huggingface_hub", status["huggingface_hub"], "run: uv sync")
        check_item("pyyaml", status["pyyaml"], "run: uv sync")
        if status["hf_token"]:
            print("HF_TOKEN: set (for gated/private repos)")
        else:
            print("HF_TOKEN: not set (optional; needed for gated/private Hugging Face)")
        check_item(
            "abliterate deps",
            status["abliterate_deps"],
            "run: uv sync --extra abliterate",
        )
        check_item(
            "llama.cpp finetune",
            status["finetune"],
            "run: ollama-forge setup-llama-cpp",
        )
        check_item(
            "llama.cpp quantize",
            status["quantize"],
            "run: ollama-forge setup-llama-cpp",
        )

    if not getattr(args, "fix", False):
        ok = status["ollama"] and status["huggingface_hub"] and status["pyyaml"]
        return 0 if ok else 1

    plan_only = getattr(args, "plan", False)
    if plan_only:
        planned: list[str] = []
        if not status["huggingface_hub"] or not status["pyyaml"]:
            planned.append("Run: uv sync")
        if getattr(args, "fix_llama_cpp", False) and (not status["finetune"] or not status["quantize"]):
            target_dir = getattr(args, "llama_cpp_dir", None) or "./llama.cpp"
            planned.append(f"Run: ollama-forge setup-llama-cpp --dir {target_dir}")
        if not planned:
            planned.append("No fix actions needed.")
        if getattr(args, "json", False):
            print(json.dumps({"route": "doctor-fix", "actions": planned}))
        else:
            print("\nFix plan:")
            for step in planned:
                print(f"  - {step}")
        return 0

    print("\nApplying fixes...", file=sys.stderr)
    if not status["huggingface_hub"] or not status["pyyaml"]:
        code = run_cmd(
            ["uv", "sync"],
            not_found_message="Error: uv not found. Install uv first: https://docs.astral.sh/uv/",
            process_error_message="Error: uv sync failed: {e}",
            not_found_next_steps=["Install uv, then run: uv sync"],
            process_error_next_steps=["Resolve errors above, then rerun: ollama-forge doctor --fix"],
        )
        if code != 0:
            return code
        print("Applied: uv sync", file=sys.stderr)

    if getattr(args, "fix_llama_cpp", False) and (not status["finetune"] or not status["quantize"]):
        code = _cmd_setup_llama_cpp(
            parser,
            argparse.Namespace(dir=getattr(args, "llama_cpp_dir", None)),
        )
        if code != 0:
            return code
    elif (not status["finetune"] or not status["quantize"]) and not getattr(args, "fix_llama_cpp", False):
        print(
            "Tip: add --fix-llama-cpp to auto-install llama.cpp tools.",
            file=sys.stderr,
        )

    if not status["ollama"]:
        print(
            "Cannot auto-install Ollama here. Install from https://ollama.com, then rerun doctor.",
            file=sys.stderr,
        )
        return 1

    final_status = _env_status()
    ok = final_status["ollama"] and final_status["huggingface_hub"] and final_status["pyyaml"]
    return 0 if ok else 1


def _cmd_setup_llama_cpp(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Clone and build llama.cpp; print instructions to add to PATH."""
    target_dir = Path(args.dir or "llama.cpp").resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        print(
            f"Directory already exists and is non-empty: {target_dir}. Use --dir <other> or remove it.",
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
    print(f'\nDone. Add to PATH: export PATH="{bin_dir}:$PATH"')
    print("Then you can use: finetune, quantize, and other llama.cpp tools.")
    return 0


def _cmd_adapters_search(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
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
        print(f"    → ollama-forge fetch-adapter {repo} --base <BASE_MODEL> --name <NAME>")
    print("\nReplace <BASE_MODEL> with the model the adapter was trained for (e.g. llama3.2).")
    return 0


def _score_adapter_repo(repo_id: str, base: str | None) -> int:
    """Simple ranking heuristic for adapter recommendations."""
    rid = repo_id.lower()
    score = 0
    if "adapter" in rid:
        score += 5
    if "lora" in rid:
        score += 4
    if "qlora" in rid:
        score += 3
    if "gguf" in rid:
        score -= 2
    if base:
        base_tokens = [t for t in base.lower().replace("-", " ").replace("_", " ").split() if t]
        if any(tok in rid for tok in base_tokens):
            score += 6
    return score


def _cmd_adapters_recommend(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Recommend adapter repos and optionally apply the top one."""
    from huggingface_hub import HfApi

    base = getattr(args, "base", None)
    query = getattr(args, "query", None) or (f"{base} lora adapter" if base else "lora adapter")
    limit = max(1, int(getattr(args, "limit", 5)))
    json_mode = bool(getattr(args, "json", False))
    api = HfApi()
    if not json_mode:
        print(f"Finding adapter recommendations for query: {query!r}", file=sys.stderr)
    try:
        candidates = list(api.list_models(search=query, limit=max(limit * 4, 20)))
    except Exception as e:
        print_actionable_error(
            "failed to search adapter recommendations",
            cause=str(e),
            next_steps=[
                "Check internet/Hugging Face connectivity",
                'Try a broader query: ollama-forge adapters recommend --query "lora adapter"',
            ],
        )
        return 1
    if not candidates:
        print("No adapter recommendations found.")
        return 0
    ranked = sorted(
        ((m.id, _score_adapter_repo(m.id, base)) for m in candidates),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    if not json_mode:
        print("Recommended adapters:\n")
        for repo, score in ranked:
            print(f"  {repo}  (score={score})")
            if base:
                print(f"    -> ollama-forge fetch-adapter {repo} --base {base} --name <NAME>")
            else:
                print(f"    -> ollama-forge fetch-adapter {repo} --base <BASE_MODEL> --name <NAME>")
    elif not getattr(args, "apply", False):
        print(
            json.dumps(
                {
                    "route": "adapters-recommend",
                    "base": base,
                    "query": query,
                    "recommendations": [{"repo": repo, "score": score} for repo, score in ranked],
                }
            )
        )
        return 0
    if not getattr(args, "apply", False):
        return 0
    top_repo = ranked[0][0]
    if not base:
        print_actionable_error(
            "--apply requires --base",
            next_steps=[
                f'Re-run with: ollama-forge adapters recommend --base <BASE_MODEL> --apply --query "{query}"',
            ],
        )
        return 1
    target_name = getattr(args, "name", None) or f"{base}-adapter"
    if getattr(args, "plan", False):
        action = f"ollama-forge fetch-adapter {top_repo} --base {base} --name {target_name}"
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "route": "adapters-apply",
                        "top_repo": top_repo,
                        "base": base,
                        "name": target_name,
                        "action": action,
                    }
                )
            )
        else:
            print("\nApply plan:")
            print(f"  top repo: {top_repo}")
            print(f"  base: {base}")
            print(f"  output model: {target_name}")
            print(f"  action: {action}")
        return 0
    print(f"\nApplying top recommendation: {top_repo} -> model {target_name!r}", file=sys.stderr)
    fake = argparse.Namespace(
        repo_id=top_repo,
        base=base,
        name=target_name,
        revision=getattr(args, "revision", "main"),
        output=getattr(args, "output", None),
        system=getattr(args, "system", None),
        temperature=getattr(args, "temperature", None),
        num_ctx=getattr(args, "num_ctx", None),
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        out_modelfile=getattr(args, "out_modelfile", None),
    )
    return _cmd_fetch_adapter(parser, fake)


def _cmd_hf_cache_ls(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """List Hugging Face Hub cache (repos and sizes)."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print_actionable_error(
            "huggingface_hub is required for hf-cache",
            next_steps=["Run: uv sync"],
        )
        return 1
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"Error scanning cache: {e}", file=sys.stderr)
        return 1
    verbosity = 1 if getattr(args, "revisions", False) else 0
    print(cache_info.export_as_table(verbosity=verbosity))
    return 0


def _cmd_hf_cache_rm(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Remove one or more repos from the Hugging Face Hub cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print_actionable_error(
            "huggingface_hub is required for hf-cache",
            next_steps=["Run: uv sync"],
        )
        return 1
    repo_ids = getattr(args, "repo_id", None)
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]
    repo_ids = repo_ids or []
    if not repo_ids:
        print("Error: provide at least one repo_id (e.g. TheBloke/Llama-2-7B-GGUF)", file=sys.stderr)
        return 1
    dry_run = getattr(args, "dry_run", False)
    yes = getattr(args, "yes", False)
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"Error scanning cache: {e}", file=sys.stderr)
        return 1
    revisions_to_delete: list[str] = []
    for repo in cache_info.repos:
        # repo.repo_id is e.g. "TheBloke/Llama-2-7B-GGUF" or "bert-base-cased"
        if repo.repo_id in repo_ids:
            for rev in repo.revisions:
                revisions_to_delete.append(rev.commit_hash)
    if not revisions_to_delete:
        print("No matching repos found in cache.", file=sys.stderr)
        return 1
    strategy = cache_info.delete_revisions(*revisions_to_delete)
    print(f"About to free {strategy.expected_freed_size_str}.", file=sys.stderr)
    if dry_run:
        print("Dry run: no files deleted.", file=sys.stderr)
        return 0
    if not yes:
        try:
            answer = input("Proceed? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer != "y":
            print("Cancelled.", file=sys.stderr)
            return 0
    strategy.execute()
    print(f"Freed {strategy.expected_freed_size_str}.", file=sys.stderr)
    return 0


def _cmd_security_eval_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run security eval: load prompt set, query model, score, print KPIs and optionally write CSV/JSON."""
    try:
        from ollama_forge.security_eval.run import run_eval
    except ImportError as e:
        print(f"Error: security-eval failed to import: {e}", file=sys.stderr)
        return 1
    prompt_set = getattr(args, "prompt_set", None)
    if not prompt_set:
        print("Error: prompt_set path required", file=sys.stderr)
        return 1
    base_url = getattr(args, "base_url", None) or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = getattr(args, "model", "llama3.2")
    output_csv = getattr(args, "output_csv", None)
    output_json = getattr(args, "output_json", None)
    system = getattr(args, "system", None)
    use_chat = not getattr(args, "no_chat", False)
    timeout = getattr(args, "timeout", 120.0)
    verbose = not getattr(args, "quiet", False)
    try:
        run_meta = run_eval(
            prompt_set,
            base_url=base_url,
            model=model,
            output_csv=output_csv,
            output_json=output_json,
            save_to_history=getattr(args, "save_history", False),
            use_chat=use_chat,
            system=system,
            timeout=timeout,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    kpis = run_meta.get("kpis") or {}
    print("\n--- KPIs ---", file=sys.stderr)
    print(f"  Total:        {kpis.get('total', 0)}", file=sys.stderr)
    print(f"  ASR %:        {kpis.get('asr_pct', 0):.1f}", file=sys.stderr)
    print(f"  Refusal %:    {kpis.get('refusal_rate_pct', 0):.1f}", file=sys.stderr)
    print(f"  Extraction %: {kpis.get('extraction_rate_pct', 0):.1f}", file=sys.stderr)
    print(f"  Errors:       {kpis.get('errors', 0)}", file=sys.stderr)
    by_cat = kpis.get("by_category") or {}
    if by_cat:
        print("  By category:", file=sys.stderr)
        for cat, v in by_cat.items():
            print(
                f"    {cat}: ASR={v.get('asr_pct', 0):.1f}% refusal={v.get('refusal_rate_pct', 0):.1f}%",
                file=sys.stderr,
            )
    return 0


def _cmd_security_eval_ui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Launch Streamlit UI for security evaluation."""
    app_dir = Path(__file__).resolve().parent
    app_path = app_dir / "security_eval" / "app.py"
    if not app_path.exists():
        print(f"Error: App not found at {app_path}", file=sys.stderr)
        return 1
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
            check=False,
        )
    except FileNotFoundError:
        print(
            "Error: Streamlit not found. Run: uv sync --extra security-eval-ui",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_downsize_pipeline(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
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
   ollama-forge convert --gguf <path/to/student.gguf> --name {name}{q_flag}
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
   ollama-forge convert --gguf <path/to/student.gguf> --name my-downsized [--quantize Q4_K_M]
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
    from ollama_forge.abliterate_defaults import HARMFUL_DEFAULT, HARMLESS_DEFAULT

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
            raise FileNotFoundError("No instructions in --harmful-dir and/or --harmless-dir")
        harmful_path = write_temp_text_file("\n".join(h_lines), suffix=".txt", prefix="ollama-harmful-")
        harmless_path = write_temp_text_file("\n".join(l_lines), suffix=".txt", prefix="ollama-harmless-")
        temp_files = [harmful_path, harmless_path]
    elif getattr(args, "harmful", None) and getattr(args, "harmless", None):
        harmful_path = Path(args.harmful)
        harmless_path = Path(args.harmless)
    elif default_harmful_file.is_file() and default_harmless_file.is_file():
        harmful_path = default_harmful_file
        harmless_path = default_harmless_file
        print(
            "Using bundled default harmful/harmless lists (Sumandora, HarmBench, etc.; up to 32 pairs). "
            "Pass --harmful/--harmless for custom lists.",
            file=sys.stderr,
        )
    else:
        harmful_path = write_temp_text_file(HARMFUL_DEFAULT.strip(), suffix=".txt", prefix="ollama-harmful-")
        harmless_path = write_temp_text_file(HARMLESS_DEFAULT.strip(), suffix=".txt", prefix="ollama-harmless-")
        temp_files = [harmful_path, harmless_path]
        print(
            "Using built-in default harmful/harmless lists. "
            "Pass --harmful/--harmless or --harmful-dir/--harmless-dir for custom.",
            file=sys.stderr,
        )

    return harmful_path, harmless_path, temp_files


# Online lists used by download-lists and bundled defaults
# (Sumandora, HarmBench, JailbreakBench, AdvBench, refusal_direction/Arditi et al.)
ABLITERATE_HARMFUL_URL = (
    "https://raw.githubusercontent.com/Sumandora/remove-refusals-with-transformers/master/harmful.txt"
)
ABLITERATE_HARMBENCH_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
ABLITERATE_JBB_HARMFUL_URL = (
    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/raw/main/data/harmful-behaviors.csv"
)
ABLITERATE_JBB_BENIGN_URL = (
    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/raw/main/data/benign-behaviors.csv"
)
ABLITERATE_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)
ABLITERATE_HARMLESS_URL = (
    "https://raw.githubusercontent.com/Sumandora/remove-refusals-with-transformers/master/harmless.txt"
)
# refusal_direction (Arditi et al. – arXiv:2406.11717): JSON with "instruction" key
ABLITERATE_REFUSAL_DIR_HARMFUL = (
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/harmful_train.json",
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/harmful_val.json",
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/harmful_test.json",
)
ABLITERATE_REFUSAL_DIR_HARMLESS = (
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/harmless_train.json",
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/harmless_val.json",
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/harmless_test.json",
)


def _abliterate_fetch_json_instructions(urls: tuple[str, ...]) -> list[str]:
    """Fetch JSON arrays from URLs; each item must have 'instruction' key. Return deduped list."""
    instructions: list[str] = []
    seen: set[str] = set()
    for url in urls:
        with urlopen(url, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        for item in data:
            if isinstance(item, dict) and "instruction" in item:
                s = (item["instruction"] or "").strip()
                if s and s not in seen:
                    seen.add(s)
                    instructions.append(s)
    return instructions


def _abliterate_merge_harmful_sources() -> list[str]:
    """Fetch and merge all harmful sources (Sumandora, HarmBench, JBB, AdvBench, refusal_direction)."""
    import csv

    with urlopen(ABLITERATE_HARMFUL_URL, timeout=60) as r:
        sumandora = r.read().decode("utf-8")
    lines = [s.strip() for s in sumandora.splitlines() if s.strip() and not s.strip().startswith("#")]
    seen = set(lines)
    with urlopen(ABLITERATE_HARMBENCH_URL, timeout=60) as r:
        reader = csv.reader(r.read().decode("utf-8").splitlines())
        next(reader)
        for row in reader:
            if row:
                b = row[0].strip()
                if b and b not in seen:
                    seen.add(b)
                    lines.append(b)
    with urlopen(ABLITERATE_JBB_HARMFUL_URL, timeout=60) as r:
        reader = csv.reader(r.read().decode("utf-8").splitlines())
        next(reader)
        for row in reader:
            if len(row) > 1:
                b = row[1].strip()  # Goal column
                if b and b not in seen:
                    seen.add(b)
                    lines.append(b)
    with urlopen(ABLITERATE_ADVBENCH_URL, timeout=60) as r:
        reader = csv.reader(r.read().decode("utf-8").splitlines())
        next(reader)
        for row in reader:
            if row:
                b = row[0].strip()  # goal column
                if b and b not in seen:
                    seen.add(b)
                    lines.append(b)
    for instr in _abliterate_fetch_json_instructions(ABLITERATE_REFUSAL_DIR_HARMFUL):
        if instr not in seen:
            seen.add(instr)
            lines.append(instr)
    return lines


def _abliterate_merge_harmless_sources() -> list[str]:
    """Fetch and merge harmless sources (Sumandora + JBB benign + refusal_direction)."""
    import csv

    with urlopen(ABLITERATE_HARMLESS_URL, timeout=60) as r:
        lines = [s.strip() for s in r.read().decode("utf-8").splitlines() if s.strip()]
    seen = set(lines)
    with urlopen(ABLITERATE_JBB_BENIGN_URL, timeout=60) as r:
        reader = csv.reader(r.read().decode("utf-8").splitlines())
        next(reader)
        for row in reader:
            if len(row) > 1:
                b = row[1].strip()  # Goal column
                if b and b not in seen:
                    seen.add(b)
                    lines.append(b)
    for instr in _abliterate_fetch_json_instructions(ABLITERATE_REFUSAL_DIR_HARMLESS):
        if instr not in seen:
            seen.add(instr)
            lines.append(instr)
    return lines


def _cmd_abliterate_download_lists(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Download harmful/harmless instruction lists (Sumandora, HarmBench, JailbreakBench, etc.)."""
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    harmful_path = out_dir / "harmful.txt"
    harmless_path = out_dir / "harmless.txt"
    try:
        harmful_lines = _abliterate_merge_harmful_sources()
        harmless_lines = _abliterate_merge_harmless_sources()
        harmful_path.write_text("\n".join(harmful_lines) + "\n", encoding="utf-8")
        harmless_path.write_text("\n".join(harmless_lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Error downloading lists: {e}", file=sys.stderr)
        return 1
    print(f"Saved harmful list:  {harmful_path} ({len(harmful_lines)} instructions)")
    print(f"Saved harmless list: {harmless_path} ({len(harmless_lines)} instructions)")
    print("Use with: --harmful", harmful_path, "--harmless", harmless_path)
    return 0


def _abliterate_output_dir_from_name(name: str) -> str:
    """Return default output dir from abliterate run --name (e.g. name -> abliterate-<sanitized>)."""
    sanitized = name.replace("/", "-").strip()
    while "  " in sanitized:
        sanitized = sanitized.replace("  ", " ")
    sanitized = sanitized.replace(" ", "-")
    return f"abliterate-{sanitized}"


def _cmd_abliterate_chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Interactive chat with abliterated checkpoint (HF tokenizer). If serve is running with same
    model, chat connects to it instead of loading checkpoint. Use --no-serve to always load locally."""
    try:
        from ollama_forge.abliterate import run_chat
        from ollama_forge.abliterate_serve import chat_via_serve
    except ImportError:
        print(
            "Error: abliterate chat requires optional deps. Run: uv sync --extra abliterate",
            file=sys.stderr,
        )
        return 1
    name = getattr(args, "name", None)
    checkpoint_arg = getattr(args, "checkpoint", None)
    if name and checkpoint_arg:
        print("Error: use either --name or --checkpoint, not both", file=sys.stderr)
        return 1
    if name:
        checkpoint = (Path(_abliterate_output_dir_from_name(name)) / "checkpoint").resolve()
        model_name = name
    elif checkpoint_arg:
        checkpoint = Path(checkpoint_arg).resolve()
        model_name = None
    else:
        print("Error: pass --name <model_name> (from abliterate run) or --checkpoint DIR", file=sys.stderr)
        return 1
    if not checkpoint.is_dir():
        print(
            f"Error: checkpoint dir not found: {checkpoint}",
            file=sys.stderr,
        )
        if name:
            print("  Run abliterate run first with that --name (checkpoint is saved by default).", file=sys.stderr)
        return 1

    # If we have a model name and user didn't pass --no-serve, try existing abliterate serve first
    use_serve = not getattr(args, "no_serve", False)
    if use_serve and model_name is not None:
        serve_url = getattr(args, "serve_url", None) or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")
        if chat_via_serve(
            serve_url,
            model_name,
            max_new_tokens=getattr(args, "max_new_tokens", None),
        ):
            return 0
        # Serve unreachable or model mismatch; fall back to local load
        print("No serve at that URL (or model mismatch). Using local checkpoint.", file=sys.stderr)

    try:
        run_chat(
            checkpoint,
            max_new_tokens=getattr(args, "max_new_tokens", None),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_abliterate_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Start Ollama-API-compatible server for abliterated model (HF tokenizer)."""
    try:
        from ollama_forge.abliterate_serve import serve_abliterated
    except ImportError:
        print(
            "Error: abliterate serve requires optional deps. Run: uv sync --extra abliterate",
            file=sys.stderr,
        )
        return 1
    name = getattr(args, "name", None)
    checkpoint_arg = getattr(args, "checkpoint", None)
    if name and checkpoint_arg:
        print("Error: use either --name or --checkpoint, not both", file=sys.stderr)
        return 1
    if name:
        checkpoint = Path(_abliterate_output_dir_from_name(name)) / "checkpoint"
        model_name = name
    elif checkpoint_arg:
        checkpoint = Path(checkpoint_arg)
        model_name = (
            checkpoint.name
            if checkpoint.name != "checkpoint"
            else (checkpoint.parent.name if checkpoint.parent else "abliterated")
        )
    else:
        print("Error: pass --name <model_name> (from abliterate run) or --checkpoint DIR", file=sys.stderr)
        return 1
    if not checkpoint.is_dir():
        print(f"Error: checkpoint dir not found: {checkpoint}", file=sys.stderr)
        if name:
            print("  Run abliterate run first with that --name.", file=sys.stderr)
        return 1
    try:
        serve_abliterated(
            str(checkpoint.resolve()),
            model_name=model_name,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 11435),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _abliterate_resolve_model(model_id: str) -> str:
    """
    Resolve --model to the path or Hugging Face repo id to load.
    Returns a path for a local .gguf file or local HF-format directory, otherwise the given model_id (HF repo).
    """
    p = Path(model_id)
    if p.is_file() and str(model_id).lower().endswith(".gguf"):
        return str(p.resolve())
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    return model_id


def _cmd_abliterate_compute_dir(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Compute refusal direction for abliteration (requires: uv sync --extra abliterate)."""
    try:
        from ollama_forge.abliterate import compute_refusal_dir
    except ImportError as e:
        print(
            "Error: abliterate requires optional deps. Run: uv sync --extra abliterate",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 1
    model_id = _abliterate_resolve_model(args.model)
    gguf_file_for_load = str(model_id) if str(model_id).lower().endswith(".gguf") else None
    if gguf_file_for_load:
        print(f"Using local GGUF at {model_id}", file=sys.stderr)
    try:
        harmful_path, harmless_path, temp_files = _resolve_abliterate_inputs(args)
        try:
            compute_refusal_dir(
                model_id,
                str(harmful_path),
                str(harmless_path),
                args.output,
                num_instructions=args.num_instructions,
                layer_fracs=tuple(getattr(args, "layer_fracs", [0.4, 0.5, 0.6])),
                n_directions=getattr(args, "num_directions", 1),
                load_in_8bit=getattr(args, "load_in_8bit", False),
                gguf_file=gguf_file_for_load,
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


def _cmd_abliterate_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """One command: compute direction, bake into weights, convert to GGUF, create Ollama model."""
    try:
        from ollama_forge.abliterate import apply_refusal_dir_and_save, compute_refusal_dir
    except ImportError:
        print(
            "Error: abliterate run requires optional deps. Run: uv sync --extra abliterate",
            file=sys.stderr,
        )
        return 1
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    model_id = getattr(args, "model", None)
    name = getattr(args, "name", None)
    if not model_id or not name:
        print("Error: --model and --name are required", file=sys.stderr)
        return 1
    model_id = _abliterate_resolve_model(model_id)
    gguf_file_for_load_run = str(model_id) if str(model_id).lower().endswith(".gguf") else None
    if gguf_file_for_load_run:
        print(f"Using local GGUF at {model_id}", file=sys.stderr)
    default_out = Path(_abliterate_output_dir_from_name(name)) if name else None
    output_dir = Path(
        getattr(args, "output_dir", None)
        or (default_out if default_out else tempfile.mkdtemp(prefix="ollama-forge-abliterate-"))
    )
    llama_cpp_dir = getattr(args, "llama_cpp_dir", None) and Path(args.llama_cpp_dir)
    if not llama_cpp_dir:
        for candidate in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
            if (candidate / "convert_hf_to_gguf.py").is_file():
                llama_cpp_dir = candidate
                break
    if not llama_cpp_dir or not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
        print(
            "Error: convert_hf_to_gguf.py not found. Set --llama-cpp-dir to your llama.cpp clone.",
            file=sys.stderr,
        )
        return 1
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refusal_pt = output_dir / "refusal_dir.pt"
    checkpoint_dir = output_dir / "checkpoint"
    gguf_path = output_dir / "model.gguf"
    try:
        harmful_path, harmless_path, temp_files = _resolve_abliterate_inputs(args)
        try:
            print("Computing refusal direction...", file=sys.stderr)
            compute_refusal_dir(
                model_id,
                str(harmful_path),
                str(harmless_path),
                str(refusal_pt),
                num_instructions=getattr(args, "num_instructions", 32),
                layer_fracs=tuple(getattr(args, "layer_fracs", [0.4, 0.5, 0.6])),
                n_directions=getattr(args, "num_directions", 1),
                load_in_8bit=getattr(args, "load_in_8bit", False),
                gguf_file=gguf_file_for_load_run,
            )
            # Free memory from first load before second load (apply_refusal_dir_and_save loads again).
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("Baking ablation into weights and saving checkpoint...", file=sys.stderr)
            apply_refusal_dir_and_save(
                model_id,
                refusal_pt,
                checkpoint_dir,
                verify=not getattr(args, "no_verify", False),
                gguf_file=gguf_file_for_load_run,
                strength=getattr(args, "strength", 1.0),
                skip_begin_layers=getattr(args, "skip_begin_layers", 0),
                skip_end_layers=getattr(args, "skip_end_layers", 0),
            )
        finally:
            for t in temp_files:
                Path(t).unlink(missing_ok=True)
    except Exception as e:
        import traceback

        msg = str(e).strip() or f"{type(e).__name__} (no message)"
        print(f"Error: {msg}", file=sys.stderr)
        if not str(e).strip():
            traceback.print_exc(file=sys.stderr)
        return 1
    print("Converting to GGUF...", file=sys.stderr)
    convert_script = (llama_cpp_dir / "convert_hf_to_gguf.py").resolve()
    checkpoint_abs = checkpoint_dir.resolve()
    gguf_path_abs = gguf_path.resolve()
    try:
        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(checkpoint_abs),
                "--outfile",
                str(gguf_path_abs),
            ],
            cwd=str(llama_cpp_dir.resolve()),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: GGUF conversion failed: {e}", file=sys.stderr)
        return 1
    if not gguf_path.is_file():
        print("Error: GGUF file was not produced", file=sys.stderr)
        return 1
    gguf_to_use = gguf_path
    requantize = not getattr(args, "no_requantize", False)
    if requantize:
        quant_type = getattr(args, "quant", "Q4_K_M")
        quantize_bin = _which_quantize()
        if not quantize_bin:
            print(
                "Error: requantize (default) requires llama.cpp 'quantize' or 'llama-quantize' on PATH",
                file=sys.stderr,
            )
            print("  Add llama.cpp build dir to PATH, or pass --no-requantize to keep full-size GGUF.", file=sys.stderr)
            return 1
        quant_gguf = gguf_path.parent / f"{gguf_path.stem}-{quant_type}.gguf"
        print(f"Quantizing to {quant_type}...", file=sys.stderr)
        try:
            subprocess.run(
                [quantize_bin, str(gguf_path), str(quant_gguf), quant_type],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: quantization failed: {e}", file=sys.stderr)
            return 1
        if quant_gguf.is_file():
            gguf_to_use = quant_gguf
    # Use absolute path so Ollama finds the GGUF when the Modelfile is in a temp dir
    gguf_for_modelfile = gguf_to_use.resolve()
    content = build_modelfile(str(gguf_for_modelfile))
    _model_path = Path(model_id)
    _is_local_hf = _model_path.is_dir() and (_model_path / "config.json").is_file()
    template_from = getattr(args, "template_from", None) or (None if _is_local_hf else model_id)
    if template_from:
        ref_content = run_ollama_show_modelfile(template_from)
        if ref_content:
            content = merge_modelfile_with_reference_template(
                content, ref_content, base=str(gguf_for_modelfile), template_only=True
            )
            print(
                f"Using chat template from Ollama model {template_from!r} (for tool/Chat API support)", file=sys.stderr
            )
        else:
            print(
                f"Note: no Ollama model {template_from!r} found; pull it first for tool support.",
                file=sys.stderr,
            )
    elif _is_local_hf:
        print(
            "Note: using local HF path; pass --template-from <ollama_model> for tool support.",
            file=sys.stderr,
        )
    if not getattr(args, "output_dir", None):
        print(
            f"To chat with correct tokenization (HF tokenizer): ollama-forge abliterate chat --name {name}",
            file=sys.stderr,
        )
    else:
        print(
            "To chat with correct tokenization (HF tokenizer): "
            f"ollama-forge abliterate chat --checkpoint {output_dir / 'checkpoint'}",
            file=sys.stderr,
        )
    return run_ollama_create(name, content)


def _load_env() -> None:
    """Load .env from ~/.env then cwd. Never override existing env (e.g. export in shell)."""
    load_dotenv(Path.home() / ".env")
    load_dotenv(override=False)  # do not overwrite shell exports


def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(
        prog="ollama-forge",
        description="Create, retrain, ablate, and convert models for local Ollama.",
        epilog="Quick start: ollama-forge fetch <HF_REPO> --name my-model",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # check (environment)
    p_check = subparsers.add_parser(
        "check",
        help="Verify ollama, Hugging Face, optional deps, and llama.cpp",
    )
    p_check.set_defaults(handler=_cmd_check)

    # doctor (diagnose + optional fixes)
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Diagnose environment and optionally apply common fixes",
    )
    p_doctor.add_argument(
        "--fix",
        action="store_true",
        help="Apply lightweight fixes (e.g. uv sync)",
    )
    p_doctor.add_argument(
        "--plan",
        action="store_true",
        help="With --fix, show planned fix actions without executing",
    )
    p_doctor.add_argument(
        "--fix-llama-cpp",
        action="store_true",
        help="Also run setup-llama-cpp when finetune/quantize are missing",
    )
    p_doctor.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Directory for setup-llama-cpp when --fix-llama-cpp is used",
    )
    p_doctor.set_defaults(handler=_cmd_doctor)

    # plan (global dry-run wrappers for key flows)
    p_plan = subparsers.add_parser(
        "plan",
        help="Preview major operations without executing them",
    )
    plan_sub = p_plan.add_subparsers(dest="plan_command")

    p_plan_quickstart = plan_sub.add_parser(
        "quickstart",
        help="Preview quickstart/start resolved settings and action",
    )
    p_plan_quickstart.add_argument(
        "--name",
        default="my-model",
        help="Name for the new Ollama model (default: my-model)",
    )
    p_plan_quickstart.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "low-vram"],
        default="balanced",
        help="Preset for quant/parameters (default: balanced)",
    )
    p_plan_quickstart.add_argument(
        "--task",
        choices=sorted(_QUICKSTART_TASK_SYSTEMS.keys()),
        default=None,
        help="Task preset for default system prompt",
    )
    p_plan_quickstart.add_argument(
        "--repo-id",
        default="TheBloke/Llama-2-7B-GGUF",
        help="Hugging Face GGUF repo to use (default: TheBloke/Llama-2-7B-GGUF)",
    )
    p_plan_quickstart.add_argument(
        "--quant",
        default=None,
        help="Override profile quantization (e.g. Q4_K_M)",
    )
    p_plan_quickstart.add_argument(
        "--revision",
        default="main",
        help="Repo revision (default: main)",
    )
    p_plan_quickstart.add_argument("--system", help="System message (role/instructions)")
    p_plan_quickstart.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_plan_quickstart.add_argument(
        "--num-ctx",
        type=int,
        help="Context window size in tokens (e.g. 4096)",
    )
    p_plan_quickstart.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_plan_quickstart.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_plan_quickstart.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_plan_quickstart.add_argument(
        "--json",
        action="store_true",
        help="Output plan as JSON",
    )
    p_plan_quickstart.set_defaults(handler=_cmd_plan_quickstart)

    p_plan_auto = plan_sub.add_parser(
        "auto",
        help="Preview auto route/action for a source",
    )
    p_plan_auto.add_argument(
        "source",
        help="Source input: recipe path, .gguf path, HF repo id, or local base model name",
    )
    p_plan_auto.add_argument("--name", default=None, help="Model name for non-recipe flows")
    p_plan_auto.add_argument("--system", help="System message (role/instructions)")
    p_plan_auto.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_plan_auto.add_argument(
        "--num-ctx",
        type=int,
        help="Context window size in tokens (e.g. 4096)",
    )
    p_plan_auto.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_plan_auto.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_plan_auto.add_argument("--base", help="Base model for adapter sources")
    p_plan_auto.add_argument("--adapter", help="Path to LoRA/adapter directory (base mode)")
    p_plan_auto.add_argument("--output", help="Adapter download directory in adapter repo mode")
    p_plan_auto.add_argument("--gguf-file", help="Specific .gguf filename for HF repos")
    p_plan_auto.add_argument("--quant", help="Preferred quantization for HF repo mode")
    p_plan_auto.add_argument("--quantize", help="Quantize GGUF first in gguf mode")
    p_plan_auto.add_argument("--revision", default="main", help="Repo revision (default: main)")
    p_plan_auto.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts and use defaults for missing values",
    )
    p_plan_auto.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_plan_auto.add_argument(
        "--json",
        action="store_true",
        help="Output plan as JSON",
    )
    p_plan_auto.set_defaults(handler=_cmd_plan_auto)

    p_plan_doctor = plan_sub.add_parser(
        "doctor-fix",
        help="Preview doctor --fix actions",
    )
    p_plan_doctor.add_argument(
        "--fix-llama-cpp",
        action="store_true",
        help="Include setup-llama-cpp in plan when tools are missing",
    )
    p_plan_doctor.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Directory for setup-llama-cpp when --fix-llama-cpp is used",
    )
    p_plan_doctor.add_argument(
        "--json",
        action="store_true",
        help="Output plan as JSON",
    )
    p_plan_doctor.set_defaults(handler=_cmd_plan_doctor_fix)

    p_plan_adapters = plan_sub.add_parser(
        "adapters-apply",
        help="Preview adapters recommend --apply action",
    )
    p_plan_adapters.add_argument("--base", required=True, help="Base model for apply")
    p_plan_adapters.add_argument("--query", default=None, help="Search query override")
    p_plan_adapters.add_argument("--limit", type=int, default=5, help="Max recommendations")
    p_plan_adapters.add_argument("--name", default=None, help="Output model name")
    p_plan_adapters.add_argument("--revision", default="main", help="Repo revision")
    p_plan_adapters.add_argument("--output", default=None, help="Adapter download directory")
    p_plan_adapters.add_argument("--system", help="System message")
    p_plan_adapters.add_argument("--temperature", type=float, help="Temperature")
    p_plan_adapters.add_argument("--num-ctx", type=int, help="Context window")
    p_plan_adapters.add_argument("--top-p", type=float, help="Top-p")
    p_plan_adapters.add_argument("--repeat-penalty", type=float, help="Repeat penalty")
    p_plan_adapters.add_argument("--out-modelfile", default=None, help="Write Modelfile path")
    p_plan_adapters.add_argument(
        "--json",
        action="store_true",
        help="Output plan as JSON",
    )
    p_plan_adapters.set_defaults(handler=_cmd_plan_adapters_apply)

    # quickstart (beginner one-command flow)
    p_quickstart = subparsers.add_parser(
        "quickstart",
        help="Beginner one-command setup: fetch a default model and create an Ollama model",
    )
    p_quickstart.add_argument(
        "--name",
        default="my-model",
        help="Name for the new Ollama model (default: my-model)",
    )
    p_quickstart.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "low-vram"],
        default="balanced",
        help="Quickstart preset for quant + generation params (default: balanced)",
    )
    p_quickstart.add_argument(
        "--task",
        choices=sorted(_QUICKSTART_TASK_SYSTEMS.keys()),
        default=None,
        help="Task preset that sets a default system prompt (overridden by --system)",
    )
    p_quickstart.add_argument(
        "--repo-id",
        default="TheBloke/Llama-2-7B-GGUF",
        help="Hugging Face GGUF repo to use (default: TheBloke/Llama-2-7B-GGUF)",
    )
    p_quickstart.add_argument(
        "--quant",
        default=None,
        help="Override profile quantization (e.g. Q4_K_M)",
    )
    p_quickstart.add_argument("--revision", default="main", help="Repo revision (default: main)")
    p_quickstart.add_argument("--system", help="System message (role/instructions)")
    p_quickstart.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_quickstart.add_argument(
        "--num-ctx",
        type=int,
        help="Context window size in tokens (e.g. 4096)",
    )
    p_quickstart.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_quickstart.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_quickstart.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_quickstart.set_defaults(handler=_cmd_quickstart)

    # start (alias for quickstart defaults)
    p_start = subparsers.add_parser(
        "start",
        help="Alias for quickstart with beginner defaults",
    )
    p_start.add_argument(
        "--name",
        default="my-model",
        help="Name for the new Ollama model (default: my-model)",
    )
    p_start.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "low-vram"],
        default="balanced",
        help="Preset for quant/parameters (default: balanced)",
    )
    p_start.add_argument(
        "--task",
        choices=sorted(_QUICKSTART_TASK_SYSTEMS.keys()),
        default=None,
        help="Task preset that sets a default system prompt (overridden by --system)",
    )
    p_start.add_argument(
        "--repo-id",
        default="TheBloke/Llama-2-7B-GGUF",
        help="Hugging Face GGUF repo to use (default: TheBloke/Llama-2-7B-GGUF)",
    )
    p_start.add_argument(
        "--quant",
        default=None,
        help="Override profile quantization (e.g. Q4_K_M)",
    )
    p_start.add_argument("--revision", default="main", help="Repo revision (default: main)")
    p_start.add_argument("--system", help="System message (role/instructions)")
    p_start.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_start.add_argument(
        "--num-ctx",
        type=int,
        help="Context window size in tokens (e.g. 4096)",
    )
    p_start.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_start.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_start.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_start.set_defaults(handler=_cmd_start)

    # auto (detect source type and route)
    p_auto = subparsers.add_parser(
        "auto",
        help="Auto-detect source (recipe, gguf, hf repo, base) and run the right flow",
    )
    p_auto.add_argument(
        "source",
        help="Source input: recipe path, .gguf path, HF repo id, or local base model name",
    )
    p_auto.add_argument(
        "--name",
        default=None,
        help="Name for created model when source is not a recipe (interactive/default: my-model)",
    )
    p_auto.add_argument("--system", help="System message (role/instructions)")
    p_auto.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_auto.add_argument("--num-ctx", type=int, help="Context window size in tokens (e.g. 4096)")
    p_auto.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_auto.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_auto.add_argument(
        "--base",
        help="Base model for adapter sources (auto-detected adapter repo/dir)",
    )
    p_auto.add_argument("--adapter", help="Path to LoRA/adapter directory (base mode)")
    p_auto.add_argument(
        "--output",
        help="Directory to download adapter into for HF adapter repos",
    )
    p_auto.add_argument("--gguf-file", help="Specific .gguf filename for HF repos")
    p_auto.add_argument(
        "--quant",
        help="Preferred quantization for HF repo mode (e.g. Q4_K_M)",
    )
    p_auto.add_argument(
        "--quantize",
        help="Quantize GGUF first in gguf mode (e.g. Q4_K_M)",
    )
    p_auto.add_argument("--revision", default="main", help="Repo revision (default: main)")
    p_auto.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts and use defaults for missing values",
    )
    p_auto.add_argument(
        "--plan",
        action="store_true",
        help="Show detected route and planned action without executing",
    )
    p_auto.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_auto.set_defaults(handler=_cmd_auto)

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
    p_create.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_create.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
    p_create.add_argument("--adapter", help="Path to LoRA/adapter directory")
    p_create.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_create.set_defaults(handler=_cmd_create_from_base)

    # refresh-template (recreate model with base's latest chat template)
    p_refresh = subparsers.add_parser(
        "refresh-template",
        help="Recreate a model using the base model's latest chat template (fixes Chat API issues)",
    )
    p_refresh.add_argument(
        "--name",
        required=True,
        help="Name of the existing model to refresh (must exist in Ollama)",
    )
    p_refresh.add_argument(
        "--base",
        required=True,
        help="Base model to take the template from (e.g. llama3.2); pull first with ollama pull",
    )
    p_refresh.add_argument(
        "--output-name",
        help="Name for the recreated model (default: same as --name, overwrites)",
    )
    p_refresh.add_argument(
        "--template-only",
        action="store_true",
        help="Only replace TEMPLATE; keep weights. Use when updating old model's template for tools/Chat API.",
    )
    p_refresh.add_argument(
        "--out-modelfile",
        help="Also write the merged Modelfile to this path",
    )
    p_refresh.set_defaults(handler=_cmd_refresh_template)

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
    p_convert.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_convert.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
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
    p_fetch.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_fetch.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
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
    p_fetch_adapter.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_fetch_adapter.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
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
    p_retrain.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_retrain.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty (e.g. 1.1)",
    )
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
        help="Compute refusal direction from harmful/harmless instructions (needs: uv sync --extra abliterate)",
    )
    p_compute.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id, or path to local HF-format dir or .gguf file",
    )
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
        "--layer-fracs",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6],
        metavar="F",
        help="Layer fractions to try; best layer by gap norm is used (default: 0.4 0.5 0.6)",
    )
    p_compute.add_argument(
        "--num-directions",
        type=int,
        default=1,
        help="Number of refusal directions from SVD (default: 1)",
    )
    p_compute.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (bitsandbytes) to avoid OOM on large/MXFP4 models",
    )
    p_compute.set_defaults(handler=_cmd_abliterate_compute_dir)

    p_run = abliterate_sub.add_parser(
        "run",
        help="Compute, apply, convert to GGUF, requantize (default), and create Ollama model",
    )
    p_run.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id, or path to local HF-format dir or .gguf file",
    )
    p_run.add_argument("--name", required=True, help="Name for the Ollama model")
    p_run.add_argument(
        "--output-dir",
        help="Directory for checkpoint and GGUF (default: ./abliterate-<name>, or temp if no --name)",
    )
    p_run.add_argument(
        "--llama-cpp-dir",
        help="Path to llama.cpp clone (for convert_hf_to_gguf.py); default: ./llama.cpp or ~/llama.cpp",
    )
    p_run.add_argument("--harmful", help="Path to file with harmful instructions (one per line)")
    p_run.add_argument("--harmless", help="Path to file with harmless instructions (one per line)")
    p_run.add_argument("--harmful-dir", help="Directory of .txt files with harmful instructions")
    p_run.add_argument("--harmless-dir", help="Directory of .txt files with harmless instructions")
    p_run.add_argument(
        "--num-instructions",
        type=int,
        default=32,
        help="Number of instructions for direction (default: 32)",
    )
    p_run.add_argument(
        "--layer-fracs",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6],
        metavar="F",
        help="Layer fractions to try; best layer by gap norm is used (default: 0.4 0.5 0.6)",
    )
    p_run.add_argument(
        "--num-directions",
        type=int,
        default=1,
        help="Number of refusal directions from SVD (default: 1)",
    )
    p_run.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (bitsandbytes) to avoid OOM on large/MXFP4 models",
    )
    p_run.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip forward-pass verification after ablation (default: verify)",
    )
    p_run.add_argument(
        "--strength",
        type=float,
        default=1.0,
        metavar="ALPHA",
        help="Ablation strength 0 < ALPHA <= 1 (default: 1.0). Use 0.5–0.7 on small models to reduce coherence loss.",
    )
    p_run.add_argument(
        "--skip-begin-layers",
        type=int,
        default=0,
        metavar="N",
        help="Number of layers to skip at the start (default: 0 for full abliteration).",
    )
    p_run.add_argument(
        "--skip-end-layers",
        type=int,
        default=0,
        metavar="N",
        help="Number of layers to skip at the end (default: 0 for full abliteration).",
    )
    p_run.add_argument(
        "--no-requantize",
        action="store_true",
        help="Skip quantizing GGUF (default: quantize to --quant to keep size similar to input)",
    )
    p_run.add_argument(
        "--quant",
        default="Q4_K_M",
        help="GGUF quantization when requantizing (default: Q4_K_M); requires quantize on PATH",
    )
    p_run.add_argument(
        "--template-from",
        metavar="OLLAMA_MODEL",
        help="Ollama model to copy chat template from (default: same as --model; pull first for tools)",
    )
    p_run.set_defaults(handler=_cmd_abliterate_run)

    p_download = abliterate_sub.add_parser(
        "download-lists",
        help="Download harmful/harmless lists (Sumandora, HarmBench, JailbreakBench, AdvBench, refusal_direction)",
    )
    p_download.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write harmful.txt and harmless.txt (default: current dir)",
    )
    p_download.set_defaults(handler=_cmd_abliterate_download_lists)

    p_chat = abliterate_sub.add_parser(
        "chat",
        help="Interactive chat using abliterated checkpoint (HF tokenizer; use when GGUF/Ollama output is garbled)",
    )
    p_chat.add_argument(
        "--name",
        metavar="NAME",
        help="Ollama/model name from abliterate run (finds checkpoint in ./abliterate-<name>/checkpoint)",
    )
    p_chat.add_argument(
        "--checkpoint",
        metavar="DIR",
        help="Path to checkpoint dir (alternative to --name)",
    )
    p_chat.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max new tokens per reply (default: from model config max_position_embeddings, capped at 8192)",
    )
    p_chat.add_argument(
        "--serve-url",
        metavar="URL",
        default=None,
        help="Abliterate serve URL first (default: OLLAMA_HOST or http://127.0.0.1:11435); if reachable, chat uses it",
    )
    p_chat.add_argument(
        "--no-serve",
        action="store_true",
        help="Do not try an existing serve; always load the checkpoint locally",
    )
    p_chat.set_defaults(handler=_cmd_abliterate_chat)

    p_serve = abliterate_sub.add_parser(
        "serve",
        help="Ollama-API-compatible server for abliterated model (HF tokenizer); agents use OLLAMA_HOST to point here",
    )
    p_serve.add_argument(
        "--name",
        metavar="NAME",
        help="Ollama/model name from abliterate run (checkpoint in ./abliterate-<name>/checkpoint)",
    )
    p_serve.add_argument(
        "--checkpoint",
        metavar="DIR",
        help="Path to checkpoint directory (alternative to --name)",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (use 0.0.0.0 if clients run in Docker/another host)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Bind port (default: 11435; use a different port if Ollama runs on 11434)",
    )
    p_serve.set_defaults(handler=_cmd_abliterate_serve)

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

    p_adapters_recommend = adapters_sub.add_parser(
        "recommend",
        help="Recommend likely adapter repos (optionally apply top result)",
    )
    p_adapters_recommend.add_argument(
        "--base",
        default=None,
        help="Base model name/path to bias recommendations and use with --apply",
    )
    p_adapters_recommend.add_argument(
        "--query",
        default=None,
        help="Search query override (default uses --base if given)",
    )
    p_adapters_recommend.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max recommendations to show (default: 5)",
    )
    p_adapters_recommend.add_argument(
        "--apply",
        action="store_true",
        help="Apply top recommendation via fetch-adapter (requires --base)",
    )
    p_adapters_recommend.add_argument(
        "--plan",
        action="store_true",
        help="When used with --apply, show the planned fetch-adapter command only",
    )
    p_adapters_recommend.add_argument(
        "--name",
        default=None,
        help="Output model name when using --apply (default: <base>-adapter)",
    )
    p_adapters_recommend.add_argument(
        "--revision",
        default="main",
        help="Repo revision when using --apply (default: main)",
    )
    p_adapters_recommend.add_argument(
        "--output",
        default=None,
        help="Adapter download directory when using --apply",
    )
    p_adapters_recommend.add_argument("--system", help="System message for --apply")
    p_adapters_recommend.add_argument(
        "--temperature",
        type=float,
        help="Temperature for --apply (e.g. 0.7)",
    )
    p_adapters_recommend.add_argument(
        "--num-ctx",
        type=int,
        help="Context window for --apply (e.g. 4096)",
    )
    p_adapters_recommend.add_argument("--top-p", type=float, help="Top-p for --apply")
    p_adapters_recommend.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty for --apply",
    )
    p_adapters_recommend.add_argument(
        "--out-modelfile",
        default=None,
        help="Also write Modelfile when using --apply",
    )
    p_adapters_recommend.set_defaults(handler=_cmd_adapters_recommend)

    # hf-cache (list / remove Hugging Face Hub cache)
    p_hf_cache = subparsers.add_parser(
        "hf-cache",
        help="List or remove Hugging Face Hub local cache (models downloaded by fetch/fetch-adapter)",
    )
    hf_cache_sub = p_hf_cache.add_subparsers(dest="hf_cache_command")
    p_hf_cache_ls = hf_cache_sub.add_parser("ls", help="List cached repos and sizes")
    p_hf_cache_ls.add_argument(
        "--revisions",
        action="store_true",
        help="Show one row per revision (default: one row per repo)",
    )
    p_hf_cache_ls.set_defaults(handler=_cmd_hf_cache_ls)
    p_hf_cache_rm = hf_cache_sub.add_parser(
        "rm",
        help="Remove repo(s) from cache (frees disk space)",
    )
    p_hf_cache_rm.add_argument(
        "repo_id",
        nargs="+",
        help="Repo id(s) to remove (e.g. TheBloke/Llama-2-7B-GGUF)",
    )
    p_hf_cache_rm.add_argument("--dry-run", action="store_true", help="Show what would be freed, do not delete")
    p_hf_cache_rm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_hf_cache_rm.set_defaults(handler=_cmd_hf_cache_rm)

    # security-eval (LLM security evaluation: run prompt sets, score, KPIs)
    p_security_eval = subparsers.add_parser(
        "security-eval",
        help="LLM security evaluation: run prompt sets, score refusal/compliance, output KPIs and CSV",
    )
    se_sub = p_security_eval.add_subparsers(dest="security_eval_command")
    p_se_run = se_sub.add_parser("run", help="Run eval: load prompt set, query model, score, write CSV/JSON")
    p_se_run.add_argument(
        "prompt_set",
        metavar="PROMPT_SET",
        help="Path to .txt (one prompt/line) or .jsonl (prompt, category, target_for_extraction)",
    )
    p_se_run.add_argument(
        "--model",
        default="llama3.2",
        help="Model name (default: llama3.2)",
    )
    p_se_run.add_argument(
        "--base-url",
        default=None,
        help="Ollama or abliterate serve URL (default: OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    p_se_run.add_argument(
        "--output-csv",
        metavar="PATH",
        help="Write per-prompt results to CSV",
    )
    p_se_run.add_argument(
        "--output-json",
        metavar="PATH",
        help="Write full run (results + KPIs + metadata) to JSON",
    )
    p_se_run.add_argument(
        "--save-history",
        action="store_true",
        help="Append run to SQLite history (~/.ollama_forge/security_eval_runs.db) for plots over time",
    )
    p_se_run.add_argument(
        "--system",
        help="Optional system prompt to send with each request",
    )
    p_se_run.add_argument(
        "--no-chat",
        action="store_true",
        help="Use /api/generate instead of /api/chat",
    )
    p_se_run.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    p_se_run.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less progress output",
    )
    p_se_run.set_defaults(handler=_cmd_security_eval_run)
    p_se_ui = se_sub.add_parser(
        "ui", help="Launch Streamlit UI for security evaluation (requires: uv sync --extra security-eval-ui)"
    )
    p_se_ui.set_defaults(handler=_cmd_security_eval_ui)

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
    if parsed.command == "plan" and not getattr(parsed, "plan_command", None):
        p_plan.print_help()
        return 0
    if parsed.command == "downsize" and not getattr(parsed, "downsize_command", None):
        _cmd_downsize_pipeline(parser, parsed)
        return 0
    if parsed.command == "hf-cache" and not getattr(parsed, "hf_cache_command", None):
        p_hf_cache.print_help()
        return 0
    if parsed.command == "security-eval" and not getattr(parsed, "security_eval_command", None):
        p_security_eval.print_help()
        return 0
    handler = getattr(parsed, "handler", None)
    if handler is None:
        return 0
    return handler(parser, parsed)


if __name__ == "__main__":
    sys.exit(main())
