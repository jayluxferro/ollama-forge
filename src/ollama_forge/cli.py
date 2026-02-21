"""CLI entrypoint for ollama-forge."""

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

from ollama_forge.hf_fetch import download_adapter, download_gguf, list_gguf_files, pick_one_gguf
from ollama_forge.modelfile import (
    build_modelfile,
    get_stop_tokens_from_checkpoint,
    merge_modelfile_with_reference_template,
    modelfile_append_num_predict,
    modelfile_append_stop_parameters,
    modelfile_append_template,
    template_body_from_modelfile,
    template_from_hf_checkpoint,
    template_from_hf_checkpoint_with_reason,
)
from ollama_forge.recipe import load_recipe
from ollama_forge.run_helpers import (
    check_item,
    get_jsonl_paths_or_exit,
    ping_ollama,
    print_actionable_error,
    require_ollama,
    run_cmd,
    run_ollama_create,
    run_ollama_show_modelfile,
    write_temp_text_file,
)
from ollama_forge.training_data import (
    convert_jsonl_to_plain_text,
    convert_messages_to_alpaca_jsonl,
    validate_training_data_paths,
)


def _which_quantize() -> str | None:
    """Resolve llama.cpp quantize binary (quantize or llama-quantize)."""
    return shutil.which("quantize") or shutil.which("llama-quantize")


def _prompt_for_value(prompt: str, default: str) -> str:
    """When stdin is a TTY, prompt the user; return default if empty. When not TTY, return default."""
    if not sys.stdin.isatty():
        return default
    try:
        line = input(prompt).strip()
        return line if line else default
    except (EOFError, KeyboardInterrupt):
        return default


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


def _verify_adapter_and_base(adapter: str | None, base: str) -> tuple[str | None, list[str] | None]:
    """Verify adapter dir and base exist; return (error_summary, next_steps) on failure, (None, None) on success."""
    if adapter:
        ad = Path(adapter).resolve()
        if not ad.is_dir():
            return (
                f"Adapter path is not a directory: {ad}",
                [
                    "Check the adapter path (e.g. from fetch-adapter or training output)",
                    "Run: ollama-forge create-from-base --base <base> --name <name> --adapter <path>",
                ],
            )
        has_config = (ad / "adapter_config.json").is_file()
        has_weights = (ad / "adapter_model.safetensors").is_file() or (ad / "adapter_model.bin").is_file()
        if not has_config and not has_weights:
            return (
                f"Adapter directory has no adapter_config.json or adapter_model.*: {ad}",
                [
                    "Use a LoRA/PEFT adapter directory (e.g. from fetch-adapter or ollama-forge retrain)",
                    "Run: ollama-forge fetch-adapter <repo_id> --base <base> --name <name>",
                ],
            )
    base_path = Path(base)
    if "/" in base or "\\" in base:
        if not base_path.exists():
            return (
                f"Base path does not exist: {base_path.resolve()}",
                [
                    "Use an existing base model path or Ollama model name",
                    "Run: ollama-forge create-from-base --base <path_or_name> --name <name>",
                ],
            )
    return (None, None)


def _cmd_create_from_base(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    adapter = getattr(args, "adapter", None)
    if adapter:
        err, steps = _verify_adapter_and_base(adapter, args.base)
        if err:
            print_actionable_error(err, next_steps=steps)
            return 1
    content = build_modelfile(
        args.base,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        adapter=adapter,
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
    repo_id = getattr(args, "repo_id", None)
    name = getattr(args, "name", None)
    if repo_id is None or name is None:
        if sys.stdin.isatty():
            if repo_id is None:
                repo_id = _prompt_for_value(
                    "Repo ID [TheBloke/Llama-2-7B-GGUF]: ",
                    "TheBloke/Llama-2-7B-GGUF",
                )
            if name is None:
                name = _prompt_for_value("Model name [my-model]: ", "my-model")
        if repo_id is None or name is None:
            print_actionable_error(
                "repo_id and --name are required",
                next_steps=[
                    "Run: ollama-forge fetch <repo_id> --name <name>",
                    "Or run interactively (from a TTY) to be prompted for missing values",
                ],
            )
            return 1
        args.repo_id = repo_id
        args.name = name
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
                    f"We auto-picked {chosen!r}; use --gguf-file <filename> to override.",
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
    repo_id = getattr(args, "repo_id", None)
    base = getattr(args, "base", None)
    name = getattr(args, "name", None)
    if repo_id is None or base is None or name is None:
        if sys.stdin.isatty():
            if repo_id is None:
                repo_id = _prompt_for_value("Adapter repo ID (e.g. user/my-lora): ", "")
            if base is None:
                base = _prompt_for_value("Base model name or path: ", "llama3.2")
            if name is None:
                name = _prompt_for_value("Output model name [my-adapter]: ", "my-adapter")
        if not repo_id or not base or not name:
            print_actionable_error(
                "repo_id, --base, and --name are required",
                next_steps=[
                    "Run: ollama-forge fetch-adapter <repo_id> --base <base> --name <name>",
                    "Or run interactively (from a TTY) to be prompted for missing values",
                ],
            )
            return 1
        args.repo_id = repo_id
        args.base = base
        args.name = name
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


def _cmd_convert_training_data_format(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
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


def _cmd_train_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run the full pipeline: validate → prepare → (finetune if available) → retrain."""
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
    first_path = Path(data_spec).resolve()
    prepared_path = Path(getattr(args, "prepared_output", None) or "train_prepared.txt")
    adapter_dir = Path(getattr(args, "adapter_output", None) or "adapter_out")
    base = args.base
    name = args.name
    base_gguf = getattr(args, "base_gguf", None)
    # Step 2: prepare
    try:
        convert_jsonl_to_plain_text(paths, prepared_path, format_name=getattr(args, "format", "llama.cpp"))
    except OSError as e:
        print_actionable_error(
            "failed to write prepared data",
            cause=str(e),
            next_steps=["Use --prepared-output <path> or fix permissions"],
        )
        return 1
    print(f"Prepared data → {prepared_path}", file=sys.stderr)
    # Step 3: finetune (if base_gguf and finetune on PATH)
    ran_finetune = False
    if base_gguf and Path(base_gguf).is_file():
        finetune_bin = shutil.which("finetune") or shutil.which("llama-finetune")
        if finetune_bin:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                finetune_bin,
                "--train-data", str(prepared_path.resolve()),
                "--sample-start", "### Instruction",
                "--model-base", str(Path(base_gguf).resolve()),
                "--lora-out", str(adapter_dir.resolve()),
            ]
            print("Running finetune...", file=sys.stderr)
            result = subprocess.run(cmd)
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
            print("finetune not on PATH; skipping. Run: ollama-forge setup-llama-cpp", file=sys.stderr)
    if not ran_finetune:
        intro = "Skipped finetune (need --base-gguf and finetune on PATH). After training:"
        cmd = f"ollama-forge retrain --base {base} --adapter {adapter_dir} --name {name}"
        print(intro, cmd, file=sys.stderr)
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
        out_modelfile=getattr(args, "out_modelfile", None),
    )
    return _cmd_create_from_base(parser, fake)


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
    if getattr(args, "validate_only", False):
        source = "base" if "base" in recipe else ("gguf" if "gguf" in recipe else "hf_repo")
        print(f"Recipe valid: name={recipe['name']!r}, source={source}")
        return 0
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


def _cmd_validate_recipe(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Validate a recipe file (schema and paths) without building."""
    try:
        recipe = load_recipe(args.recipe)
    except FileNotFoundError as e:
        print_actionable_error(
            str(e),
            next_steps=["Check recipe path and retry: ollama-forge validate-recipe <recipe.yaml>"],
        )
        return 1
    except (ValueError, ImportError) as e:
        print_actionable_error(
            "invalid recipe",
            cause=str(e),
            next_steps=[
                "Ensure recipe has name and exactly one of base/gguf/hf_repo",
                "See docs/RECIPE.md and wiki/Recipes.md",
            ],
        )
        return 1
    source = "base" if "base" in recipe else ("gguf" if "gguf" in recipe else "hf_repo")
    print(f"Recipe valid: name={recipe['name']!r}, source={source}")
    return 0


def _cmd_check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Verify ollama, HF, optional deps, and llama.cpp; print what's missing."""
    if getattr(args, "json", False):
        status = _env_status()
        print(json.dumps(status))
        ok = status["ollama"] and status["huggingface_hub"] and status["pyyaml"]
        return 0 if ok else 1
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
    if json_mode and not getattr(args, "fix", False):
        print(json.dumps(status))
        ok = status["ollama"] and status["huggingface_hub"] and status["pyyaml"]
        return 0 if ok else 1
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
    if json_mode:
        print(json.dumps(final_status))
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
        print_actionable_error(
            "adapter search failed",
            cause=str(e),
            next_steps=["Check network and HF access", "Try: ollama-forge adapters search '<query>' --limit 10"],
        )
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
        print_actionable_error(
            "failed to scan Hugging Face cache",
            cause=str(e),
            next_steps=[
                "Check that ~/.cache/huggingface/hub exists and is readable",
                "Run: ollama-forge hf-cache ls",
            ],
        )
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
        print_actionable_error(
            "no repo_id provided",
            next_steps=[
                "Provide at least one repo id: ollama-forge hf-cache rm <repo_id> [repo_id ...]",
                "Example: ollama-forge hf-cache rm TheBloke/Llama-2-7B-GGUF",
            ],
        )
        return 1
    dry_run = getattr(args, "dry_run", False)
    yes = getattr(args, "yes", False)
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print_actionable_error(
            "failed to scan Hugging Face cache",
            cause=str(e),
            next_steps=[
                "Check that ~/.cache/huggingface/hub exists and is readable",
                "Run: ollama-forge hf-cache ls",
            ],
        )
        return 1
    revisions_to_delete: list[str] = []
    for repo in cache_info.repos:
        # repo.repo_id is e.g. "TheBloke/Llama-2-7B-GGUF" or "bert-base-cased"
        if repo.repo_id in repo_ids:
            for rev in repo.revisions:
                revisions_to_delete.append(rev.commit_hash)
    if not revisions_to_delete:
        print_actionable_error(
            "no matching repos found in cache",
            next_steps=[
                "Run: ollama-forge hf-cache ls (with --revisions) to see cached repo ids",
                "Use exact repo_id(s) from that list, e.g. TheBloke/Llama-2-7B-GGUF",
            ],
        )
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
        print_actionable_error(
            "security-eval failed to import",
            cause=str(e),
            next_steps=["Run: uv sync --extra security-eval", "Then: ollama-forge security-eval run <prompt_set>"],
        )
        return 1
    prompt_set = getattr(args, "prompt_set", None)
    if not prompt_set:
        print_actionable_error(
            "prompt_set path required",
            next_steps=[
                "Run: ollama-forge security-eval run <path_to_.txt_or_.jsonl> [--model <name>]",
                "Example: ollama-forge security-eval run ./prompts.txt --model llama3.2",
            ],
        )
        return 1
    base_url = getattr(args, "base_url", None) or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = getattr(args, "model", "llama3.2")
    output_csv = getattr(args, "output_csv", None)
    output_json = getattr(args, "output_json", None)
    system = getattr(args, "system", None)
    use_chat = not getattr(args, "no_chat", False)
    timeout = getattr(args, "timeout", 120.0)
    verbose = not getattr(args, "quiet", False)
    if not getattr(args, "no_check_ollama", False) and not ping_ollama(base_url):
        print_actionable_error(
            "Ollama (or serve) is not reachable at " + base_url,
            next_steps=[
                "Start Ollama: ollama serve (or start abliterate serve and set OLLAMA_HOST)",
                "Or set --base-url to your Ollama/serve URL",
                "Or skip this check: ollama-forge security-eval run <path> --no-check-ollama",
            ],
        )
        return 1
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
            retries=getattr(args, "retries", 2),
            max_prompts=getattr(args, "max_prompts", None),
            refusal_keywords_path=getattr(args, "refusal_keywords", None),
        )
    except FileNotFoundError as e:
        print_actionable_error(
            "prompt set file not found",
            cause=str(e),
            next_steps=["Check the path to your .txt or .jsonl prompt set", "Run: ollama-forge security-eval run <path> --help"],  # noqa: E501
        )
        return 1
    except ValueError as e:
        print_actionable_error(
            "invalid prompt set or options",
            cause=str(e),
            next_steps=["Check prompt set format (one prompt per line or JSONL)", "Run: ollama-forge security-eval run --help"],  # noqa: E501
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "security-eval run failed",
            cause=str(e),
            next_steps=["Ensure Ollama is running (ollama serve) or set --base-url", "Run: ollama-forge security-eval run --help"],  # noqa: E501
        )
        return 1
    kpis = run_meta.get("kpis") or {}
    print("\n--- KPIs ---", file=sys.stderr)
    print(f"  Total:        {kpis.get('total', 0)}", file=sys.stderr)
    print(f"  ASR %:        {kpis.get('asr_pct', 0):.1f}", file=sys.stderr)
    print(f"  Refusal %:    {kpis.get('refusal_rate_pct', 0):.1f}", file=sys.stderr)
    print(f"  Extraction %: {kpis.get('extraction_rate_pct', 0):.1f}", file=sys.stderr)
    print(f"  Errors:       {kpis.get('errors', 0)}", file=sys.stderr)
    if kpis.get("avg_latency_sec") is not None:
        print(f"  Avg latency:  {kpis['avg_latency_sec']:.2f}s", file=sys.stderr)
    if kpis.get("expected_refusal_accuracy_pct") is not None:
        print(f"  Expected-refusal accuracy: {kpis['expected_refusal_accuracy_pct']:.1f}%", file=sys.stderr)
    if kpis.get("benign_refusal_rate_pct") is not None:
        print(f"  Benign refusal rate: {kpis['benign_refusal_rate_pct']:.1f}%", file=sys.stderr)
    if kpis.get("error_counts"):
        print("  Error breakdown:", file=sys.stderr)
        for msg, count in sorted(kpis["error_counts"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {count}x {msg}", file=sys.stderr)
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
        print_actionable_error(
            f"security-eval UI app not found at {app_path}",
            next_steps=[
                "Ensure the security_eval package is installed with app.py",
                "Run: uv sync --extra security-eval-ui",
            ],
        )
        return 1
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
            check=False,
        )
    except FileNotFoundError:
        print_actionable_error(
            "Streamlit not found",
            next_steps=[
                "Run: uv sync --extra security-eval-ui",
                "Then: ollama-forge security-eval ui",
            ],
        )
        return 1
    return 0


def _cmd_security_eval_compare(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Compare two security-eval run JSON files side-by-side."""
    path_a = Path(getattr(args, "run_a", ""))
    path_b = Path(getattr(args, "run_b", ""))
    if not path_a.is_file():
        print_actionable_error("Run A file not found", cause=str(path_a), next_steps=["Use path from security-eval run --output-json"])  # noqa: E501
        return 1
    if not path_b.is_file():
        print_actionable_error("Run B file not found", cause=str(path_b), next_steps=["Use path from security-eval run --output-json"])  # noqa: E501
        return 1
    try:
        run_a = json.loads(path_a.read_text(encoding="utf-8"))
        run_b = json.loads(path_b.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print_actionable_error("Failed to load run JSON", cause=str(e), next_steps=["Ensure files are valid JSON from security-eval run --output-json"])  # noqa: E501
        return 1
    kpis_a = run_a.get("kpis") or {}
    kpis_b = run_b.get("kpis") or {}
    label_a = run_a.get("model", "A") + " @ " + (run_a.get("timestamp_iso", "")[:10] or "?")
    label_b = run_b.get("model", "B") + " @ " + (run_b.get("timestamp_iso", "")[:10] or "?")
    print("\n--- Compare ---", file=sys.stderr)
    print(f"  {'KPI':<28} {label_a[:24]:<24} {label_b[:24]:<24}", file=sys.stderr)
    print("  " + "-" * 76, file=sys.stderr)
    for key, name in [
        ("total", "Total"),
        ("asr_pct", "ASR %"),
        ("refusal_rate_pct", "Refusal %"),
        ("extraction_rate_pct", "Extraction %"),
        ("errors", "Errors"),
        ("avg_latency_sec", "Avg latency (s)"),
        ("expected_refusal_accuracy_pct", "Expected-refusal acc %"),
        ("benign_refusal_rate_pct", "Benign refusal %"),
    ]:
        va = kpis_a.get(key)
        vb = kpis_b.get(key)
        sa = f"{va:.1f}" if isinstance(va, (int, float)) and key.endswith("_pct") else (f"{va:.2f}" if isinstance(va, float) else str(va) if va is not None else "—")  # noqa: E501
        sb = f"{vb:.1f}" if isinstance(vb, (int, float)) and key.endswith("_pct") else (f"{vb:.2f}" if isinstance(vb, float) else str(vb) if vb is not None else "—")  # noqa: E501
        print(f"  {name:<28} {sa:<24} {sb:<24}", file=sys.stderr)
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
    curated_harmful_file = data_dir / "abliterate_harmful_curated.txt"
    curated_harmless_file = data_dir / "abliterate_harmless_curated.txt"
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
    elif curated_harmful_file.is_file() and curated_harmless_file.is_file():
        # Merge curated (first) with bundled merged list; dedupe so curated takes precedence.
        curated_h = _collect_instructions_from_path(curated_harmful_file)
        curated_l = _collect_instructions_from_path(curated_harmless_file)
        default_h = _collect_instructions_from_path(default_harmful_file) if default_harmful_file.is_file() else []
        default_l = _collect_instructions_from_path(default_harmless_file) if default_harmless_file.is_file() else []
        seen_h = frozenset(curated_h)
        seen_l = frozenset(curated_l)
        harmful_lines = curated_h + [x for x in default_h if x not in seen_h]
        harmless_lines = curated_l + [x for x in default_l if x not in seen_l]
        harmful_path = write_temp_text_file("\n".join(harmful_lines) + "\n", suffix=".txt", prefix="ollama-harmful-")
        harmless_path = write_temp_text_file("\n".join(harmless_lines) + "\n", suffix=".txt", prefix="ollama-harmless-")
        temp_files = [harmful_path, harmless_path]
        n_h, n_l = len(harmful_lines), len(harmless_lines)
        print(
            f"Using curated + merged harmful/harmless lists ({n_h} harmful, {n_l} harmless). "
            "Pass --harmful/--harmless for custom lists.",
            file=sys.stderr,
        )
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


def _abliterate_fetch_url(url: str, timeout: int = 60) -> bytes:
    """Fetch URL and return raw bytes. Used for parallel download."""
    with urlopen(url, timeout=timeout) as r:
        return r.read()


def _abliterate_fetch_json_instructions_from_bytes(raw_list: list[bytes]) -> list[str]:
    """Parse JSON arrays from raw bytes; each item must have 'instruction' key. Return deduped list."""
    instructions: list[str] = []
    seen: set[str] = set()
    for raw in raw_list:
        data = json.loads(raw.decode("utf-8"))
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    urls_with_timeout: list[tuple[str, int]] = [
        (ABLITERATE_HARMFUL_URL, 60),
        (ABLITERATE_HARMBENCH_URL, 60),
        (ABLITERATE_JBB_HARMFUL_URL, 60),
        (ABLITERATE_ADVBENCH_URL, 60),
    ]
    for _ in ABLITERATE_REFUSAL_DIR_HARMFUL:
        urls_with_timeout.append((_, 90))
    results: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(urls_with_timeout))) as executor:
        futures = {executor.submit(_abliterate_fetch_url, u, t): u for u, t in urls_with_timeout}
        for fut in as_completed(futures):
            url = futures[fut]
            with contextlib.suppress(Exception):
                results[url] = fut.result()
    sumandora = results.get(ABLITERATE_HARMFUL_URL, b"").decode("utf-8")
    lines = [s.strip() for s in sumandora.splitlines() if s.strip() and not s.strip().startswith("#")]
    seen = set(lines)
    for url, key in [
        (ABLITERATE_HARMBENCH_URL, 0),
        (ABLITERATE_JBB_HARMFUL_URL, 1),
        (ABLITERATE_ADVBENCH_URL, 0),
    ]:
        raw = results.get(url, b"")
        reader = csv.reader(raw.decode("utf-8").splitlines())
        next(reader, None)
        for row in reader:
            if len(row) > key:
                b = row[key].strip()
                if b and b not in seen:
                    seen.add(b)
                    lines.append(b)
    json_bytes = [results.get(u, b"") for u in ABLITERATE_REFUSAL_DIR_HARMFUL]
    for instr in _abliterate_fetch_json_instructions_from_bytes(json_bytes):
        if instr not in seen:
            seen.add(instr)
            lines.append(instr)
    return lines


def _abliterate_merge_harmless_sources() -> list[str]:
    """Fetch and merge harmless sources (Sumandora + JBB benign + refusal_direction)."""
    import csv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    urls_with_timeout: list[tuple[str, int]] = [
        (ABLITERATE_HARMLESS_URL, 60),
        (ABLITERATE_JBB_BENIGN_URL, 60),
    ]
    for u in ABLITERATE_REFUSAL_DIR_HARMLESS:
        urls_with_timeout.append((u, 90))
    results: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(urls_with_timeout))) as executor:
        futures = {executor.submit(_abliterate_fetch_url, u, t): u for u, t in urls_with_timeout}
        for fut in as_completed(futures):
            url = futures[fut]
            with contextlib.suppress(Exception):
                results[url] = fut.result()
    raw_harmless = results.get(ABLITERATE_HARMLESS_URL, b"")
    lines = [s.strip() for s in raw_harmless.decode("utf-8").splitlines() if s.strip()]
    seen = set(lines)
    raw_jbb = results.get(ABLITERATE_JBB_BENIGN_URL, b"")
    reader = csv.reader(raw_jbb.decode("utf-8").splitlines())
    next(reader, None)
    for row in reader:
        if len(row) > 1:
            b = row[1].strip()
            if b and b not in seen:
                seen.add(b)
                lines.append(b)
    json_bytes = [results.get(u, b"") for u in ABLITERATE_REFUSAL_DIR_HARMLESS]
    for instr in _abliterate_fetch_json_instructions_from_bytes(json_bytes):
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
        print_actionable_error(
            "failed to download harmful/harmless lists",
            cause=str(e),
            next_steps=[
                "Check network access and list URLs in ollama_forge/data/",
                "Or pass --harmful <path> --harmless <path> to use local files",
            ],
        )
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
        print_actionable_error(
            "abliterate chat requires optional deps",
            next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate chat --name <name>"],
        )
        return 1
    name = getattr(args, "name", None)
    checkpoint_arg = getattr(args, "checkpoint", None)
    if name and checkpoint_arg:
        print_actionable_error(
            "use either --name or --checkpoint, not both",
            next_steps=["Run: ollama-forge abliterate chat --name <name> OR --checkpoint <dir>"],
        )
        return 1
    if name:
        checkpoint = (Path(_abliterate_output_dir_from_name(name)) / "checkpoint").resolve()
        model_name = name
    elif checkpoint_arg:
        checkpoint = Path(checkpoint_arg).resolve()
        model_name = None
    else:
        print_actionable_error(
            "pass --name <model_name> (from abliterate run) or --checkpoint DIR",
            next_steps=[
                "Run: ollama-forge abliterate chat --name <name> (after abliterate run)",
                "Or: ollama-forge abliterate chat --checkpoint <path_to_checkpoint>",
            ],
        )
        return 1
    if not checkpoint.is_dir():
        print_actionable_error(
            f"checkpoint dir not found: {checkpoint}",
            next_steps=(
                ["Run abliterate run first with that --name (checkpoint is saved by default)."]
                if name
                else ["Ensure --checkpoint points to a directory containing the abliterated checkpoint."]
            ),
        )
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
            device="cpu" if getattr(args, "device", None) == "cpu" else None,
        )
    except FileNotFoundError as e:
        print_actionable_error(
            "checkpoint or resource not found",
            cause=str(e),
            next_steps=["Check that the checkpoint directory is complete", "Run: ollama-forge abliterate chat --checkpoint <dir>"],  # noqa: E501
        )
        return 1
    except Exception as e:
        msg = str(e).strip()
        next_steps = ["Run: ollama-forge abliterate chat --name <name> --device cpu"] if ("histogram_mps" in msg or "not implemented" in msg.lower()) else ["Check the checkpoint path and try --device cpu"]  # noqa: E501
        print_actionable_error("abliterate chat failed", cause=msg, next_steps=next_steps)
        return 1
    return 0


def _cmd_abliterate_proxy(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Start lightweight prompt proxy (HF tokenizer -> Ollama /api/generate)."""
    try:
        from ollama_forge.abliterate_proxy import run_proxy
    except ImportError:
        print_actionable_error(
            "abliterate proxy requires transformers",
            next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate proxy --name <name>"],
        )
        return 1
    config_file = getattr(args, "config", None)
    add_models = getattr(args, "add_model", None) or []
    name = getattr(args, "name", None)
    checkpoint_arg = getattr(args, "checkpoint", None)
    if (config_file or add_models) and (name or checkpoint_arg):
        print_actionable_error(
            "use either --config/--add-model (multi-model) or --name/--checkpoint (single), not both",
            next_steps=[
                "Single: ollama-forge abliterate proxy --name <name>",
                "Multi: ollama-forge abliterate proxy --config <file> or --add-model name:path [--add-model ...]",
            ],
        )
        return 1
    models_list: list[tuple[str, str]] = []
    if config_file:
        config_path = Path(config_file)
        if not config_path.is_file():
            print_actionable_error(
                f"Config file not found: {config_path}",
                next_steps=["Use a YAML file with 'models: [{name: <name>, checkpoint: <path>}, ...]'", "Run: ollama-forge abliterate proxy --help"],  # noqa: E501
            )
            return 1
        try:
            import yaml
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print_actionable_error(
                "Failed to load proxy config YAML",
                cause=str(e),
                next_steps=["Use valid YAML with 'models: [{name: <name>, checkpoint: <path>}, ...]'"],
            )
            return 1
        for entry in data.get("models") or []:
            if isinstance(entry, dict):
                n, p = entry.get("name"), entry.get("checkpoint")
                if n and p:
                    models_list.append((str(n).strip(), str(p).strip()))
        if not models_list:
            print_actionable_error(
                "Config file has no valid 'models' entries (expect name and checkpoint per entry)",
                next_steps=["Example YAML: models: [{name: my-model, checkpoint: ./abliterate-my/checkpoint}]"],
            )
            return 1
    if models_list:
        if not getattr(args, "no_check_ollama", False):
            ollama_target = getattr(args, "ollama_target", None) or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
            if not ping_ollama(ollama_target):
                print_actionable_error(
                    "Ollama is not reachable at " + ollama_target,
                    next_steps=[
                        "Start Ollama: ollama serve",
                        "Or set OLLAMA_HOST / --ollama-target to your Ollama URL",
                        "Or skip this check: ollama-forge abliterate proxy ... --no-check-ollama",
                    ],
                )
                return 1
        try:
            run_proxy(
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 11436),
                ollama_target=getattr(args, "ollama_target", None),
                models=models_list,
            )
        except FileNotFoundError as e:
            print_actionable_error(
                "checkpoint or resource not found",
                cause=str(e),
                next_steps=["Check paths in --config or --add-model", "Run: ollama-forge abliterate proxy --help"],
            )
            return 1
        except Exception as e:
            print_actionable_error(
                "abliterate proxy failed",
                cause=str(e),
                next_steps=["Check the checkpoint paths", "Run: ollama-forge abliterate proxy --help"],
            )
            return 1
        return 0
    if add_models:
        # Multi-model: parse "name:path" pairs
        for spec in add_models:
            if ":" not in spec:
                print_actionable_error(
                    "each --add-model must be 'name:path'",
                    next_steps=["Example: ollama-forge abliterate proxy --add-model my-model:/path/to/checkpoint"],
                )
                return 1
            n, p = spec.split(":", 1)
            n, p = n.strip(), p.strip()
            if not n or not p:
                print_actionable_error(
                    "each --add-model must be 'name:path' (non-empty name and path)",
                    next_steps=["Example: ollama-forge abliterate proxy --add-model my-model:/path/to/checkpoint"],
                )
                return 1
            models_list.append((n, p))
        if not getattr(args, "no_check_ollama", False):
            ollama_target = getattr(args, "ollama_target", None) or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
            if not ping_ollama(ollama_target):
                print_actionable_error(
                    "Ollama is not reachable at " + ollama_target,
                    next_steps=[
                        "Start Ollama: ollama serve",
                        "Or set OLLAMA_HOST / --ollama-target to your Ollama URL",
                        "Or skip this check: ollama-forge abliterate proxy ... --no-check-ollama",
                    ],
                )
                return 1
        try:
            run_proxy(
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 11436),
                ollama_target=getattr(args, "ollama_target", None),
                models=models_list,
            )
        except FileNotFoundError as e:
            print_actionable_error(
                "checkpoint or resource not found",
                cause=str(e),
                next_steps=["Check each path in --add-model name:path", "Run: ollama-forge abliterate proxy --help"],
            )
            return 1
        except Exception as e:
            print_actionable_error(
                "abliterate proxy failed",
                cause=str(e),
                next_steps=["Check the checkpoint paths", "Run: ollama-forge abliterate proxy --help"],
            )
            return 1
        return 0
    # Single-model mode
    if name and checkpoint_arg:
        print_actionable_error(
            "use either --name or --checkpoint, not both",
            next_steps=["Run: ollama-forge abliterate proxy --name <name> OR --checkpoint <dir>"],
        )
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
        print_actionable_error(
            "pass --name <model_name> (from abliterate run) or --checkpoint DIR or --add-model name:path",
            next_steps=[
                "Run: ollama-forge abliterate proxy --name <name> (after abliterate run)",
                "Or: ollama-forge abliterate proxy --checkpoint <path_to_checkpoint>",
                "Or: ollama-forge abliterate proxy --add-model name:path [--add-model name2:path2 ...]",
            ],
        )
        return 1
    if not checkpoint.is_dir():
        print_actionable_error(
            f"checkpoint dir not found: {checkpoint}",
            next_steps=(
                ["Run abliterate run first with that --name."] if name else ["Ensure --checkpoint points to the abliterated checkpoint directory."]  # noqa: E501
            ),
        )
        return 1
    # Optional: fail fast if Ollama (proxy target) is not reachable
    if not getattr(args, "no_check_ollama", False):
        ollama_target = getattr(args, "ollama_target", None) or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if not ping_ollama(ollama_target):
            print_actionable_error(
                "Ollama is not reachable at " + ollama_target,
                next_steps=[
                    "Start Ollama: ollama serve",
                    "Or set OLLAMA_HOST / --ollama-target to your Ollama URL",
                    "Or skip this check: ollama-forge abliterate proxy ... --no-check-ollama",
                ],
            )
            return 1
    try:
        run_proxy(
            checkpoint_dir=str(checkpoint.resolve()),
            model_name=model_name,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 11436),
            ollama_target=getattr(args, "ollama_target", None),
        )
    except FileNotFoundError as e:
        print_actionable_error(
            "checkpoint or resource not found",
            cause=str(e),
            next_steps=["Check that the checkpoint directory is complete", "Run: ollama-forge abliterate proxy --checkpoint <dir>"],  # noqa: E501
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "abliterate proxy failed",
            cause=str(e),
            next_steps=["Check the checkpoint path", "Run: ollama-forge abliterate proxy --help"],
        )
        return 1
    return 0


def _cmd_abliterate_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Start Ollama-API-compatible server for abliterated model (HF tokenizer)."""
    try:
        from ollama_forge.abliterate_serve import serve_abliterated
    except ImportError:
        print_actionable_error(
            "abliterate serve requires optional deps",
            next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate serve --name <name>"],
        )
        return 1
    name = getattr(args, "name", None)
    checkpoint_arg = getattr(args, "checkpoint", None)
    if name and checkpoint_arg:
        print_actionable_error(
            "use either --name or --checkpoint, not both",
            next_steps=["Run: ollama-forge abliterate serve --name <name> OR --checkpoint <dir>"],
        )
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
        print_actionable_error(
            "pass --name <model_name> (from abliterate run) or --checkpoint DIR",
            next_steps=[
                "Run: ollama-forge abliterate serve --name <name> (after abliterate run)",
                "Or: ollama-forge abliterate serve --checkpoint <path_to_checkpoint>",
            ],
        )
        return 1
    if not checkpoint.is_dir():
        print_actionable_error(
            f"checkpoint dir not found: {checkpoint}",
            next_steps=(
                ["Run abliterate run first with that --name."] if name else ["Ensure --checkpoint points to the abliterated checkpoint directory."]  # noqa: E501
            ),
        )
        return 1
    try:
        serve_abliterated(
            str(checkpoint.resolve()),
            model_name=model_name,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 11435),
            device="cpu" if getattr(args, "device", None) == "cpu" else None,
        )
    except FileNotFoundError as e:
        print_actionable_error(
            "checkpoint or resource not found",
            cause=str(e),
            next_steps=["Check that the checkpoint directory is complete", "Run: ollama-forge abliterate serve --checkpoint <dir>"],  # noqa: E501
        )
        return 1
    except Exception as e:
        msg = str(e).strip()
        next_steps = ["Run: ollama-forge abliterate serve --name <name> --device cpu"] if ("histogram_mps" in msg or "not implemented" in msg.lower()) else ["Check the checkpoint path and try --device cpu"]  # noqa: E501
        print_actionable_error("abliterate serve failed", cause=msg, next_steps=next_steps)
        return 1
    return 0


def _cmd_abliterate_evaluate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run harmful prompts through abliterated checkpoint and count refusals."""
    try:
        from ollama_forge.abliterate import evaluate_abliteration
    except ImportError:
        print_actionable_error(
            "abliterate evaluate requires optional deps",
            next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate evaluate --checkpoint <dir> --harmful <path> --harmless <path>"],  # noqa: E501
        )
        return 1
    try:
        metrics = evaluate_abliteration(
            args.checkpoint,
            args.harmful,
            refusal_markers_path=getattr(args, "refusal_markers", None),
            num_prompts=getattr(args, "num_prompts", 50),
            max_new_tokens=getattr(args, "max_new_tokens", 256),
        )
        print(f"Refusals: {metrics['refusal_count']} / {metrics['total']} ({metrics['refusal_rate']:.1%})")
        return 0
    except FileNotFoundError as e:
        print_actionable_error(
            "checkpoint or prompt file not found",
            cause=str(e),
            next_steps=["Check --checkpoint, --harmful, and --harmless paths", "Run: ollama-forge abliterate evaluate --help"],  # noqa: E501
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "abliterate evaluate failed",
            cause=str(e),
            next_steps=["Check paths and refusal markers", "Run: ollama-forge abliterate evaluate --help"],
        )
        return 1


def _cmd_abliterate_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Optuna search over ablation params to minimize refusal rate."""
    try:
        from ollama_forge.abliterate import optimize_abliteration
    except ImportError as e:
        print_actionable_error(
            "abliterate optimize requires optional deps",
            cause=str(e),
            next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate optimize --model <id> --harmful <path> --harmless <path>"],  # noqa: E501
        )
        return 1
    model_id = _abliterate_resolve_model(args.model)
    gguf_file = str(model_id) if str(model_id).lower().endswith(".gguf") else None
    try:
        best = optimize_abliteration(
            model_id,
            args.refusal_pt,
            args.harmful,
            Path(args.output_dir),
            harmless_path=getattr(args, "harmless", None),
            n_trials=getattr(args, "n_trials", 20),
            num_eval_prompts=getattr(args, "num_eval_prompts", 30),
            refusal_markers_path=getattr(args, "refusal_markers", None),
            gguf_file=gguf_file,
        )
        print(f"Best refusal_rate: {best['refusal_rate']:.2%}", file=sys.stderr)
        print("Best params:", best)
        eval_set = getattr(args, "eval_prompt_set", None)
        if eval_set and Path(eval_set).exists():
            try:
                from ollama_forge.security_eval.run import run_eval
                eval_base = getattr(args, "eval_base_url", None) or "http://127.0.0.1:11434"
                print("Running security eval (ensure serve has best model loaded)...", file=sys.stderr)
                run_meta = run_eval(
                    eval_set,
                    base_url=eval_base,
                    model=getattr(args, "eval_model", None) or "abliterated",
                    max_prompts=getattr(args, "eval_max_prompts", 50),
                    verbose=True,
                )
                k = run_meta.get("kpis") or {}
                print(
                    f"Eval ASR: {k.get('asr_pct', 0):.1f}% Refusal: {k.get('refusal_rate_pct', 0):.1f}%",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"Security eval failed: {e}", file=sys.stderr)
        return 0
    except FileNotFoundError as e:
        print_actionable_error(
            "refusal_pt, harmful, harmless, or output path not found",
            cause=str(e),
            next_steps=["Check --refusal_pt, --harmful, --harmless, and --output-dir", "Run: ollama-forge abliterate optimize --help"],  # noqa: E501
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "abliterate optimize failed",
            cause=str(e),
            next_steps=["Check paths and refusal markers", "Run: ollama-forge abliterate optimize --help"],
        )
        return 1


def _cmd_abliterate_fix_ollama_template(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Update an existing Ollama abliterated model's chat template from the checkpoint (fix garbled ollama run)."""
    name = getattr(args, "name", None)
    if not name:
        print_actionable_error(
            "--name <ollama_model> is required",
            next_steps=[
                "Run: ollama-forge abliterate fix-ollama-template --name <ollama_model_name>",
                "Example: ollama-forge abliterate fix-ollama-template --name openai/gpt-oss-20b-abliterated",
            ],
        )
        return 1
    checkpoint_arg = getattr(args, "checkpoint", None)
    if checkpoint_arg:
        checkpoint_dir = Path(checkpoint_arg).resolve()
    else:
        checkpoint_dir = Path(_abliterate_output_dir_from_name(name)) / "checkpoint"
    if not checkpoint_dir.is_dir():
        print_actionable_error(
            f"checkpoint not found at {checkpoint_dir}",
            next_steps=[
                "Run abliterate run first with that --name, or use --checkpoint DIR",
                "Run: ollama-forge abliterate fix-ollama-template --name <name> --checkpoint <dir>",
            ],
        )
        return 1
    template_from = getattr(args, "template_from", None)
    if template_from:
        ref_content = run_ollama_show_modelfile(template_from)
        template_body = template_body_from_modelfile(ref_content) if ref_content else None
        if not template_body:
            print_actionable_error(
                f"could not get template from Ollama model {template_from!r}",
                next_steps=["Pull or create that model first: ollama pull " + template_from.split("/")[-1], "Then re-run fix-ollama-template"],  # noqa: E501
            )
            return 1
        print(f"Using chat template from Ollama model {template_from!r}.", file=sys.stderr)
    else:
        template_body, reason = template_from_hf_checkpoint_with_reason(str(checkpoint_dir))
        if not template_body:
            print_actionable_error(
                "could not derive chat template from checkpoint tokenizer",
                cause=reason or "Unknown",
                next_steps=["Use --template-from <ollama_model> to copy template from another model", "Run: ollama-forge abliterate fix-ollama-template --help"],  # noqa: E501
            )
            return 1
    content = run_ollama_show_modelfile(name)
    if not content:
        print_actionable_error(
            f"Ollama model {name!r} not found",
            next_steps=["Pull or create the model first: ollama pull <model> or ollama create <name>", "Then run: ollama-forge abliterate fix-ollama-template --name " + name],  # noqa: E501
        )
        return 1
    content = modelfile_append_template(content, template_body)
    stop_tokens = get_stop_tokens_from_checkpoint(checkpoint_dir)
    if stop_tokens:
        content = modelfile_append_stop_parameters(content, stop_tokens)
    content = modelfile_append_num_predict(content, 2048)
    print("Updating Ollama model with chat template derived from checkpoint...", file=sys.stderr)
    return run_ollama_create(name, content)


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
        print_actionable_error(
            "abliterate compute-dir requires optional deps",
            cause=str(e),
            next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate compute-dir --model <id> --output <dir>"],  # noqa: E501
        )
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
                per_layer_directions=getattr(args, "per_layer_directions", False),
            )
            print(f"Saved refusal direction to {args.output}")
            return 0
        finally:
            for t in temp_files:
                Path(t).unlink(missing_ok=True)
    except FileNotFoundError as e:
        print_actionable_error(
            "model, harmful, harmless, or output path not found",
            cause=str(e),
            next_steps=["Check --model, --harmful, --harmless, and --output paths", "Run: ollama-forge abliterate compute-dir --help"],  # noqa: E501
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "abliterate compute-dir failed",
            cause=str(e),
            next_steps=["Ensure optional deps are installed: uv sync --extra abliterate", "Run: ollama-forge abliterate compute-dir --help"],  # noqa: E501
        )
        return 1


def _cmd_abliterate_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """One command: compute direction, bake into weights, convert to GGUF, create Ollama model."""
    from_checkpoint_dir = getattr(args, "from_checkpoint", None)
    name = getattr(args, "name", None)
    if not name:
        print_actionable_error(
            "--name is required",
            next_steps=["Run: ollama-forge abliterate run --model <id> --name <name> or --from-checkpoint <dir> --name <name>"],  # noqa: E501
        )
        return 1
    if from_checkpoint_dir:
        checkpoint_dir = Path(from_checkpoint_dir).resolve()
        if not checkpoint_dir.is_dir():
            print_actionable_error(
                f"--from-checkpoint path is not a directory: {checkpoint_dir}",
                next_steps=["Point --from-checkpoint to an existing abliterate checkpoint directory (e.g. ./abliterate-<name>/checkpoint)"],  # noqa: E501
            )
            return 1
        if not (checkpoint_dir / "config.json").is_file():
            print_actionable_error(
                f"Checkpoint directory has no config.json: {checkpoint_dir}",
                next_steps=["Use a directory produced by abliterate run (compute + apply) or abliterate compute-dir + apply"],  # noqa: E501
            )
            return 1
        output_dir = checkpoint_dir.parent
        gguf_path = output_dir / "model.gguf"
        model_id = getattr(args, "model", None)  # optional when resuming; used later for template_from
    else:
        try:
            from ollama_forge.abliterate import apply_refusal_dir_and_save, compute_refusal_dir
        except ImportError:
            print_actionable_error(
                "abliterate run requires optional deps",
                next_steps=["Run: uv sync --extra abliterate", "Then: ollama-forge abliterate run --model <id> --name <name>"],  # noqa: E501
            )
            return 1
        model_id = getattr(args, "model", None)
        if not model_id:
            print_actionable_error(
                "--model is required (or use --from-checkpoint to resume from a checkpoint)",
                next_steps=[
                    "Run: ollama-forge abliterate run --model <hf_repo_or_path> --name <ollama_model_name>",
                    "Example: ollama-forge abliterate run --model meta-llama/Llama-2-7b-hf --name my-abliterated",
                ],
            )
            return 1
        checkpoint_dir = output_dir = gguf_path = None  # set below
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    if not from_checkpoint_dir:
        model_id = _abliterate_resolve_model(model_id)
        gguf_file_for_load_run = str(model_id) if str(model_id).lower().endswith(".gguf") else None
        if gguf_file_for_load_run:
            print(f"Using local GGUF at {model_id}", file=sys.stderr)
        default_out = Path(_abliterate_output_dir_from_name(name)) if name else None
        output_dir = Path(
            getattr(args, "output_dir", None)
            or (default_out if default_out else tempfile.mkdtemp(prefix="ollama-forge-abliterate-"))
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        refusal_pt = output_dir / "refusal_dir.pt"
        checkpoint_dir = output_dir / "checkpoint"
        gguf_path = output_dir / "model.gguf"
    llama_cpp_dir = getattr(args, "llama_cpp_dir", None) and Path(args.llama_cpp_dir)
    if not llama_cpp_dir:
        for candidate in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
            if (candidate / "convert_hf_to_gguf.py").is_file():
                llama_cpp_dir = candidate
                break
    if not llama_cpp_dir or not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
        print_actionable_error(
            "convert_hf_to_gguf.py not found",
            next_steps=[
                "Clone llama.cpp and set --llama-cpp-dir to the clone path",
                "Or run: ollama-forge setup-llama-cpp",
                "Then: ollama-forge abliterate run --model <id> --name <name> --llama-cpp-dir <path>",
            ],
        )
        return 1
    if from_checkpoint_dir:
        print("Resuming from checkpoint: converting to GGUF...", file=sys.stderr)
    else:
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
                    per_layer_directions=getattr(args, "per_layer_directions", False),
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
                    atten_strength=getattr(args, "atten_strength", None),
                    mlp_strength=getattr(args, "mlp_strength", None),
                    direction_index=getattr(args, "direction_index", None),
                    strength_kernel=getattr(args, "strength_kernel", "constant"),
                    kernel_center_frac=getattr(args, "kernel_center_frac", 0.5),
                    kernel_width_frac=getattr(args, "kernel_width_frac", 0.4),
                    skip_begin_layers=getattr(args, "skip_begin_layers", 0),
                    skip_end_layers=getattr(args, "skip_end_layers", 0),
                )
            finally:
                for t in temp_files:
                    Path(t).unlink(missing_ok=True)
        except Exception as e:
            import traceback

            msg = str(e).strip() or f"{type(e).__name__} (no message)"
            print_actionable_error(
                "abliterate run failed (compute or bake step)",
                cause=msg,
                next_steps=[
                    "Check --model (HF repo or path), --harmful, --harmless paths",
                    "Run: ollama-forge abliterate run --help",
                ],
            )
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
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print_actionable_error(
            "GGUF conversion timed out after 3600s",
            next_steps=[
                "Try a smaller model or increase system resources",
                "Re-run: ollama-forge abliterate run --model <id> --name <name> --llama-cpp-dir <path>",
            ],
        )
        return 1
    except subprocess.CalledProcessError as e:
        print_actionable_error(
            "GGUF conversion failed",
            cause=str(e),
            next_steps=[
                "Ensure llama.cpp convert_hf_to_gguf.py runs in that directory",
                "Run: ollama-forge setup-llama-cpp; add build dir to PATH",
            ],
        )
        return 1
    if not gguf_path.is_file():
        print_actionable_error(
            "GGUF file was not produced",
            next_steps=[
                "Check disk space and llama.cpp convert script output",
                "Re-run: ollama-forge abliterate run --model <id> --name <name>",
            ],
        )
        return 1
    gguf_to_use = gguf_path
    requantize = not getattr(args, "no_requantize", False)
    if requantize:
        quant_type = getattr(args, "quant", "Q4_K_M")
        quantize_bin = _which_quantize()
        if not quantize_bin:
            print_actionable_error(
                "requantize (default) requires llama.cpp quantize on PATH",
                next_steps=[
                    "Run: ollama-forge setup-llama-cpp; add the build dir to PATH",
                    "Or pass --no-requantize to keep full-size GGUF (no quantize step)",
                ],
            )
            return 1
        quant_gguf = gguf_path.parent / f"{gguf_path.stem}-{quant_type}.gguf"
        print(f"Quantizing to {quant_type}...", file=sys.stderr)
        try:
            subprocess.run(
                [quantize_bin, str(gguf_path), str(quant_gguf), quant_type],
                check=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            print_actionable_error(
                "quantization timed out after 3600s",
                next_steps=[
                    "Try --no-requantize to skip quantize and use full-size GGUF",
                    "Or re-run with more time / smaller quant type",
                ],
            )
            return 1
        except subprocess.CalledProcessError as e:
            print_actionable_error(
                "quantization failed",
                cause=str(e),
                next_steps=[
                    "Ensure llama.cpp quantize (or llama-quantize) is on PATH",
                    "Or pass --no-requantize to keep full-size GGUF",
                ],
            )
            return 1
        if quant_gguf.is_file():
            gguf_to_use = quant_gguf
    # Use absolute path so Ollama finds the GGUF when the Modelfile is in a temp dir
    gguf_for_modelfile = gguf_to_use.resolve()
    content = build_modelfile(str(gguf_for_modelfile))
    _model_path = Path(model_id) if model_id else checkpoint_dir
    _is_local_hf = _model_path.is_dir() and (_model_path / "config.json").is_file()
    template_from = getattr(args, "template_from", None) or (None if _is_local_hf else (model_id if model_id else None))
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
    # Detect model family for better diagnostics and template selection
    try:
        from ollama_forge.model_family import get_family_name
        family_name = get_family_name(checkpoint_dir)
        if family_name:
            print(f"Detected model family: {family_name}", file=sys.stderr)
    except ImportError:
        family_name = None

    # If we still have no TEMPLATE, derive from the checkpoint's HF tokenizer so Ollama uses the same format.
    if not re.search(r"TEMPLATE\s+\"\"\"", content, re.IGNORECASE):
        hf_template = template_from_hf_checkpoint(checkpoint_dir)
        if hf_template:
            content = modelfile_append_template(content, hf_template)
            stop_tokens = get_stop_tokens_from_checkpoint(checkpoint_dir)
            if stop_tokens:
                content = modelfile_append_stop_parameters(content, stop_tokens)
            content = modelfile_append_num_predict(content, 2048)
            print("Using chat template derived from checkpoint (HF format) for Ollama.", file=sys.stderr)
    if not getattr(args, "output_dir", None):
        print(
            f"To chat with correct tokenization (HF tokenizer): ollama-forge abliterate chat --name {name}",
            file=sys.stderr,
        )
        print(
            f"For agents with tool support: ollama-forge abliterate proxy --name {name}",
            file=sys.stderr,
        )
    else:
        print(
            "To chat with correct tokenization (HF tokenizer): "
            f"ollama-forge abliterate chat --checkpoint {output_dir / 'checkpoint'}",
            file=sys.stderr,
        )
        print(
            "For agents with tool support: "
            f"ollama-forge abliterate proxy --checkpoint {output_dir / 'checkpoint'}",
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
    p_check.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable status (same shape as doctor --json)",
    )
    p_check.set_defaults(handler=_cmd_check)

    # doctor (diagnose + optional fixes)
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Diagnose environment and optionally apply common fixes",
    )
    p_doctor.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable status for CI/scripting",
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
        nargs="?",
        default=None,
        help="Hugging Face repo id (e.g. TheBloke/Llama-2-7B-GGUF); prompted if missing and running in a TTY",
    )
    p_fetch.add_argument(
        "--name",
        default=None,
        help="Name for the new Ollama model; prompted if missing when running in a TTY (default: my-model)",
    )
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
        nargs="?",
        default=None,
        help="Hugging Face repo id of the adapter (e.g. user/my-lora); prompted if missing in a TTY",
    )
    p_fetch_adapter.add_argument(
        "--base",
        default=None,
        help="Base model name or path; prompted if missing in a TTY",
    )
    p_fetch_adapter.add_argument(
        "--name",
        default=None,
        help="Name for the new model; prompted if missing in a TTY",
    )
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
    p_build.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate recipe (load and check schema), do not build",
    )
    p_build.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_build.set_defaults(handler=_cmd_build)

    # validate-recipe (standalone preflight)
    p_validate_recipe = subparsers.add_parser(
        "validate-recipe",
        help="Validate a recipe file (schema and paths) without building",
    )
    p_validate_recipe.add_argument(
        "recipe",
        help="Path to recipe file (.yaml, .yml, or .json)",
    )
    p_validate_recipe.set_defaults(handler=_cmd_validate_recipe)

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

    # train-run (e2e: validate → prepare → finetune → retrain)
    p_train_run = subparsers.add_parser(
        "train-run",
        help="Run full pipeline: validate → prepare → finetune (if --base-gguf and finetune on PATH) → retrain",
    )
    p_train_run.add_argument("--data", required=True, nargs="+", help="Training data: .jsonl file(s) or directory")
    p_train_run.add_argument("--base", required=True, help="Base model name for retrain (e.g. llama3.2)")
    p_train_run.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_train_run.add_argument("--base-gguf", help="Path to base GGUF; if set and finetune on PATH, run finetune then retrain")  # noqa: E501
    p_train_run.add_argument("--prepared-output", default=None, help="Output path for prepared text (default: train_prepared.txt)")  # noqa: E501
    p_train_run.add_argument("--adapter-output", default=None, help="Output dir for LoRA adapter (default: adapter_out)")  # noqa: E501
    p_train_run.add_argument("--format", default="llama.cpp", help="Prepare format (default: llama.cpp)")
    p_train_run.add_argument("--system", help="System message for final model")
    p_train_run.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_train_run.add_argument("--num-ctx", type=int, help="Context window size in tokens")
    p_train_run.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_train_run.add_argument("--repeat-penalty", type=float, help="Repeat penalty (e.g. 1.1)")
    p_train_run.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_train_run.set_defaults(handler=_cmd_train_run)

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
    p_compute.add_argument(
        "--per-layer-directions",
        action="store_true",
        help="Compute one refusal direction per layer (Heretic-style); save format usable with --direction-index",
    )
    p_compute.set_defaults(handler=_cmd_abliterate_compute_dir)

    p_run = abliterate_sub.add_parser(
        "run",
        help="Compute, apply, convert to GGUF, requantize (default), and create Ollama model",
    )
    p_run.add_argument(
        "--model",
        help="Hugging Face model id, or path to local HF-format dir or .gguf file (omit when using --from-checkpoint)",
    )
    p_run.add_argument("--name", required=True, help="Name for the Ollama model")
    p_run.add_argument(
        "--from-checkpoint",
        metavar="DIR",
        help="Resume from an existing checkpoint dir (skip compute/apply; run GGUF conversion and create)",
    )
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
        "--per-layer-directions",
        action="store_true",
        help="Compute one refusal direction per layer (Heretic-style); use with --direction-index on apply",
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
        "--atten-strength",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Strength for attention layers only (default: same as --strength). Heretic-style: can set lower --mlp-strength to preserve quality.",  # noqa: E501
    )
    p_run.add_argument(
        "--mlp-strength",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Strength for MLP layers only (default: same as --strength). Use e.g. 0.5 to soften MLP ablation and reduce coherence loss.",  # noqa: E501
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
        "--direction-index",
        type=float,
        default=None,
        metavar="IDX",
        help="With per-layer directions: layer index (int) or blend (float) to use one effective direction for all layers.",  # noqa: E501
    )
    p_run.add_argument(
        "--strength-kernel",
        choices=("constant", "linear_peak", "gaussian"),
        default="constant",
        help="Layer-dependent strength: constant (default), linear_peak (peak at center), gaussian.",
    )
    p_run.add_argument(
        "--kernel-center-frac",
        type=float,
        default=0.5,
        metavar="F",
        help="Center of strength kernel as layer fraction (default: 0.5).",
    )
    p_run.add_argument(
        "--kernel-width-frac",
        type=float,
        default=0.4,
        metavar="F",
        help="Width of strength kernel (default: 0.4).",
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
    p_chat.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Device for model (default: auto). Use cpu to avoid MPS errors on Apple Silicon (e.g. histogram_mps).",
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
        help="Bind port (default: 11435; Ollama: 11434, abliterate proxy: 11436)",
    )
    p_serve.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Device for model (default: auto). Use cpu to avoid MPS errors on Apple Silicon.",
    )
    p_serve.set_defaults(handler=_cmd_abliterate_serve)

    p_evaluate = abliterate_sub.add_parser(
        "evaluate",
        help="Run harmful prompts through abliterated checkpoint and count refusals (refusal_markers)",
    )
    p_evaluate.add_argument(
        "--checkpoint",
        metavar="DIR",
        required=True,
        help="Path to abliterated checkpoint directory",
    )
    p_evaluate.add_argument(
        "--harmful",
        metavar="FILE",
        required=True,
        help="Path to file with harmful prompts (one per line)",
    )
    p_evaluate.add_argument(
        "--refusal-markers",
        metavar="FILE",
        default=None,
        help="Path to file with refusal marker substrings (default: bundled list)",
    )
    p_evaluate.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        metavar="N",
        help="Max number of harmful prompts to run (default: 50)",
    )
    p_evaluate.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        metavar="N",
        help="Max new tokens per response (default: 256)",
    )
    p_evaluate.set_defaults(handler=_cmd_abliterate_evaluate)

    p_optimize = abliterate_sub.add_parser(
        "optimize",
        help="Optuna search over ablation params to minimize refusal rate (requires optuna)",
    )
    p_optimize.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or path (same as used for compute-dir)",
    )
    p_optimize.add_argument(
        "--refusal-pt",
        required=True,
        metavar="FILE",
        help="Path to refusal direction .pt (from compute-dir)",
    )
    p_optimize.add_argument(
        "--harmful",
        required=True,
        metavar="FILE",
        help="Path to harmful prompts for evaluation",
    )
    p_optimize.add_argument(
        "--harmless",
        default=None,
        metavar="FILE",
        help="Path to harmless prompts (optional; only needed if re-computing direction)",
    )
    p_optimize.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory to write best params JSON (default: current dir)",
    )
    p_optimize.add_argument(
        "--n-trials",
        type=int,
        default=20,
        metavar="N",
        help="Number of Optuna trials (default: 20)",
    )
    p_optimize.add_argument(
        "--num-eval-prompts",
        type=int,
        default=30,
        metavar="N",
        help="Number of prompts per evaluation (default: 30)",
    )
    p_optimize.add_argument(
        "--refusal-markers",
        default=None,
        metavar="FILE",
        help="Path to refusal markers file (default: bundled)",
    )
    p_optimize.add_argument(
        "--eval-prompt-set",
        default=None,
        metavar="PATH",
        help="After optimize, run security-eval with this prompt set (serve must have best model)",
    )
    p_optimize.add_argument(
        "--eval-base-url",
        default="http://127.0.0.1:11434",
        metavar="URL",
        help="Base URL for post-optimize security eval (default: 127.0.0.1:11434)",
    )
    p_optimize.add_argument(
        "--eval-model",
        default=None,
        metavar="NAME",
        help="Model name for post-optimize eval (default: abliterated)",
    )
    p_optimize.add_argument(
        "--eval-max-prompts",
        type=int,
        default=50,
        metavar="N",
        help="Max prompts for post-optimize security eval (default: 50)",
    )
    p_optimize.set_defaults(handler=_cmd_abliterate_optimize)

    p_fix_template = abliterate_sub.add_parser(
        "fix-ollama-template",
        help="Update an existing Ollama abliterated model's chat template from checkpoint (fix garbled ollama run)",
    )
    p_fix_template.add_argument(
        "--name",
        metavar="NAME",
        required=True,
        help="Ollama model name (e.g. openai/gpt-oss-20b-abliterated)",
    )
    p_fix_template.add_argument(
        "--checkpoint",
        metavar="DIR",
        help="Checkpoint dir (default: abliterate-<name>/checkpoint)",
    )
    p_fix_template.add_argument(
        "--template-from",
        metavar="OLLAMA_MODEL",
        help="Use template from this Ollama model (e.g. gemma3:270m) instead of deriving from checkpoint",
    )
    p_fix_template.set_defaults(handler=_cmd_abliterate_fix_ollama_template)

    p_proxy = abliterate_sub.add_parser(
        "proxy",
        help="Lightweight prompt proxy: formats with HF tokenizer, forwards to Ollama (supports tools)",
    )
    p_proxy.add_argument(
        "--name",
        metavar="NAME",
        help="Model name from abliterate run (uses abliterate-<name>/checkpoint)",
    )
    p_proxy.add_argument(
        "--checkpoint",
        metavar="DIR",
        help="Direct path to abliterated checkpoint (HF format)",
    )
    p_proxy.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    p_proxy.add_argument(
        "--port",
        type=int,
        default=11436,
        help="Bind port (default: 11436; Ollama: 11434, abliterate serve: 11435)",
    )
    p_proxy.add_argument(
        "--ollama-target",
        metavar="URL",
        help="Ollama URL to forward to (default: OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    p_proxy.add_argument(
        "--no-check-ollama",
        action="store_true",
        help="Skip checking that Ollama is reachable before starting proxy (default: check)",
    )
    p_proxy.add_argument(
        "--config",
        metavar="FILE",
        help="YAML config file listing models (e.g. models: [{name: my-model, checkpoint: /path}]); cannot use with --name/--checkpoint",  # noqa: E501
    )
    p_proxy.add_argument(
        "--add-model",
        action="append",
        metavar="NAME:PATH",
        help="Register a model (name:checkpoint_path). Repeat for multiple models; cannot use with --name/--checkpoint",
    )
    p_proxy.set_defaults(handler=_cmd_abliterate_proxy)

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
        "--no-check-ollama",
        action="store_true",
        help="Skip checking that Ollama/serve is reachable before running (default: check)",
    )
    p_se_run.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per prompt on transient API errors (default: 2)",
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
    p_se_run.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N prompts (for quick smoke runs)",
    )
    p_se_run.add_argument(
        "--refusal-keywords",
        metavar="PATH",
        help="Path to file with custom refusal keywords (one per line, # comments). Default: built-in list.",
    )
    p_se_run.set_defaults(handler=_cmd_security_eval_run)
    p_se_ui = se_sub.add_parser(
        "ui", help="Launch Streamlit UI for security evaluation (requires: uv sync --extra security-eval-ui)"
    )
    p_se_ui.set_defaults(handler=_cmd_security_eval_ui)
    p_se_compare = se_sub.add_parser(
        "compare",
        help="Compare two security-eval run JSON files side-by-side (KPIs)",
    )
    p_se_compare.add_argument(
        "run_a",
        metavar="RUN_A.json",
        help="Path to first run JSON (from security-eval run --output-json)",
    )
    p_se_compare.add_argument(
        "run_b",
        metavar="RUN_B.json",
        help="Path to second run JSON",
    )
    p_se_compare.set_defaults(handler=_cmd_security_eval_compare)

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
