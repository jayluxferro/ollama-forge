"""Fetch, fetch-adapter, quickstart, and start command group."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ollama_forge.hf_fetch import (
    download_adapter,
    download_gguf,
    list_gguf_files,
    pick_one_gguf,
    verify_gguf_checksum,
)
from ollama_forge.run_helpers import (
    _gguf_cache_dir_for_repo,
    _llama_cpp_lib_env,
    _resolve_gguf_from_forge_cache,
    _resolve_llama_cpp_dir_from_arg,
    _which_quantize_full,
    print_actionable_error,
    require_ollama,
)


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


# Adapter file extensions Ollama accepts (GGUF or llama.cpp LoRA .bin)
_ADAPTER_FILE_SUFFIXES = (".bin", ".gguf")


def _resolve_adapter_path(adapter: str) -> str | None:
    """
    Resolve adapter path to the value to pass to Modelfile ADAPTER.
    - If path is a file with .bin or .gguf: return that path.
    - If path is a directory: PEFT (adapter_config.json or adapter_model.*) → return dir;
      else if exactly one .bin or .gguf in dir → return that file path (llama.cpp style).
    Returns None if path does not exist or is invalid.
    """
    ad = Path(adapter).resolve()
    if not ad.exists():
        return None
    if ad.is_file():
        return str(ad) if ad.suffix.lower() in _ADAPTER_FILE_SUFFIXES else None
    # Directory
    has_config = (ad / "adapter_config.json").is_file()
    has_peft_weights = (ad / "adapter_model.safetensors").is_file() or (ad / "adapter_model.bin").is_file()
    if has_config or has_peft_weights:
        return str(ad)
    # llama.cpp style: single .bin or .gguf in directory
    lora_files = [f for f in ad.iterdir() if f.is_file() and f.suffix.lower() in _ADAPTER_FILE_SUFFIXES]
    if len(lora_files) == 1:
        return str(lora_files[0])
    return None


def _verify_adapter_and_base(
    adapter: str | None,
    base: str,
) -> tuple[str | None, str | None, list[str] | None]:
    """
    Verify adapter and base. Returns (resolved_adapter_path, error, next_steps).
    On success: (resolved_path, None, None). On failure: (None, error_msg, next_steps).
    """
    resolved = None
    if adapter:
        ad = Path(adapter).resolve()
        if not ad.exists():
            return (
                None,
                f"Adapter path does not exist: {ad}",
                [
                    "Check the adapter path (e.g. from fetch-adapter or training output)",
                    "Run: ollama-forge retrain --base <base> --adapter <path> --name <name>",
                ],
            )
        if ad.is_file():
            if ad.suffix.lower() not in _ADAPTER_FILE_SUFFIXES:
                return (
                    None,
                    f"Adapter file must be .bin or .gguf: {ad}",
                    [
                        "Use a LoRA adapter file (e.g. from llama.cpp finetune --lora-out)",
                        "Or use a directory with adapter_config.json (PEFT) or a single .bin/.gguf",
                    ],
                )
            resolved = str(ad)
        else:
            has_config = (ad / "adapter_config.json").is_file()
            has_weights = (ad / "adapter_model.safetensors").is_file() or (ad / "adapter_model.bin").is_file()
            if has_config or has_weights:
                resolved = str(ad)
            else:
                lora_files = [f for f in ad.iterdir() if f.is_file() and f.suffix.lower() in _ADAPTER_FILE_SUFFIXES]
                if len(lora_files) == 1:
                    resolved = str(lora_files[0])
                else:
                    return (
                        None,
                        f"Adapter directory has no PEFT files or single LoRA file: {ad}",
                        [
                            "Use a LoRA/PEFT adapter directory (adapter_config.json + adapter_model.*)",
                            "Or a directory with exactly one .bin/.gguf (llama.cpp finetune output)",
                            "Or pass the .bin/.gguf file path directly",
                        ],
                    )
    base_path = Path(base)
    if ("/" in base or "\\" in base) and not base_path.exists():
        return (
            None,
            f"Base path does not exist: {base_path.resolve()}",
            [
                "Use an existing base model path or Ollama model name",
                "Run: ollama-forge retrain --base <path_or_name> --adapter <path> --name <name>",
            ],
        )
    return (resolved, None, None)


def _fetch_download_only_convert(args: argparse.Namespace) -> int:
    """Download HF safetensors and convert to GGUF (no Ollama model). Used by fetch --download-only."""
    repo_id = args.repo_id
    revision = getattr(args, "revision", "main") or "main"
    quant = getattr(args, "quant", "Q4_K_M") or "Q4_K_M"
    user_output = getattr(args, "output", None)

    # Check if we already have a converted GGUF cached
    if not user_output:
        cached = _resolve_gguf_from_forge_cache(repo_id)
        if cached:
            print(f"Found cached GGUF: {cached}", file=sys.stderr)
            print(cached)
            print(f"\nServe with: ollama-forge serve {cached}", file=sys.stderr)
            return 0

    print(f"No GGUF files in {repo_id}; downloading safetensors and converting to GGUF...", file=sys.stderr)

    # Resolve llama.cpp
    llama_cpp_dir = _resolve_llama_cpp_dir_from_arg(args)
    if not llama_cpp_dir or not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
        print_actionable_error(
            "convert_hf_to_gguf.py not found (needed to convert safetensors → GGUF)",
            next_steps=[
                "Run: ollama-forge setup-llama-cpp",
                "Or pass --llama-cpp-dir <path-to-llama.cpp-clone>",
            ],
        )
        return 1

    output_dir = Path(user_output).resolve() if user_output else _gguf_cache_dir_for_repo(repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"

    from ollama_forge.log import get_logger
    log = get_logger()
    log.info("Downloading %s (revision=%s)...", repo_id, revision)
    try:
        download_adapter(repo_id, revision=revision, local_dir=checkpoint_dir)
    except Exception as e:
        print_actionable_error(
            f"Failed to download {repo_id}",
            cause=str(e),
            next_steps=[
                "Check the repo ID is correct",
                "Ensure you are logged in: huggingface-cli login",
            ],
        )
        return 1

    gguf_path = output_dir / "model.gguf"

    # Remap architecture if needed
    from ollama_forge.model_family import remap_architecture_in_config
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        orig_arch = remap_architecture_in_config(config_path)
        if orig_arch:
            log.info("Remapped architecture %r for GGUF conversion", orig_arch)

    # Convert to GGUF
    outtype = getattr(args, "outtype", "bf16") or "bf16"
    from ollama_forge.commands.convert import _convert_gguf_checkpoint
    rc = _convert_gguf_checkpoint(
        checkpoint_dir=checkpoint_dir,
        gguf_path=gguf_path,
        llama_cpp_dir=llama_cpp_dir,
        outtype=outtype,
        quant_type=quant,
        gguf_converter="auto",
    )
    if rc != 0:
        return rc

    # Optionally quantize
    gguf_to_use = gguf_path
    quantize_bin = _which_quantize_full(llama_cpp_dir)
    if quantize_bin:
        quant_gguf = gguf_path.parent / f"{gguf_path.stem}-{quant}.gguf"
        print(f"Quantizing to {quant}...", file=sys.stderr)
        env = _llama_cpp_lib_env(quantize_bin)
        try:
            subprocess.run(
                [quantize_bin, str(gguf_path), str(quant_gguf), quant],
                check=True,
                timeout=7200,
                env=env,
            )
            gguf_to_use = quant_gguf
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("Quantization failed (%s); using unquantized GGUF", e)

    print(gguf_to_use)
    print(f"\nServe with: ollama-forge serve {gguf_to_use}", file=sys.stderr)
    return 0


def _cmd_fetch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Download a GGUF from Hugging Face and create an Ollama model (one command)."""
    download_only = getattr(args, "download_only", False)
    repo_id = getattr(args, "repo_id", None)
    name = getattr(args, "name", None)

    # --download-only only needs a repo_id
    if download_only:
        if repo_id is None:
            if sys.stdin.isatty():
                repo_id = _prompt_for_value(
                    "Repo ID [TheBloke/Llama-2-7B-GGUF]: ",
                    "TheBloke/Llama-2-7B-GGUF",
                )
            else:
                repo_id = "TheBloke/Llama-2-7B-GGUF"
        args.repo_id = repo_id
    else:
        non_interactive = getattr(args, "non_interactive", False)
        if repo_id is None or name is None:
            if non_interactive or not sys.stdin.isatty():
                if repo_id is None:
                    repo_id = "TheBloke/Llama-2-7B-GGUF"
                if name is None:
                    name = "my-model"
            elif sys.stdin.isatty():
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
                        "Or use --download-only to skip Ollama model creation",
                        "Or use --non-interactive to use defaults",
                    ],
                )
                return 1
        args.repo_id = repo_id
        args.name = name
        exit_code = require_ollama()
        if exit_code is not None:
            return exit_code
    no_hf_transfer = getattr(args, "no_hf_transfer", False)
    try:
        if args.gguf_file:
            downloaded_gguf_filename = args.gguf_file
            gguf_path = download_gguf(
                args.repo_id,
                args.gguf_file,
                revision=args.revision,
                no_hf_transfer=no_hf_transfer,
            )
        else:
            gguf_files = list_gguf_files(args.repo_id, revision=args.revision)
            if not gguf_files:
                if download_only:
                    return _fetch_download_only_convert(args)
                print_actionable_error(
                    f"no .gguf files found in {args.repo_id}",
                    next_steps=[
                        "Use a repo that already includes GGUF files",
                        f"Or convert: ollama-forge import {args.repo_id} --name <name>",
                        "Or download + convert: ollama-forge fetch --download-only " + args.repo_id,
                    ],
                )
                return 1
            chosen = pick_one_gguf(gguf_files, prefer_quant=getattr(args, "quant", None))
            downloaded_gguf_filename = chosen
            if len(gguf_files) > 1:
                print(
                    f"We auto-picked {chosen!r}; use --gguf-file <filename> to override.",
                    file=sys.stderr,
                )
            gguf_path = download_gguf(
                args.repo_id, chosen, revision=args.revision,
                no_hf_transfer=no_hf_transfer,
            )
        from ollama_forge.log import get_logger
        log = get_logger()
        log.info("Downloaded to %s", gguf_path)
        if getattr(args, "verify_checksum", False):
            try:
                verify_gguf_checksum(
                    args.repo_id,
                    downloaded_gguf_filename,
                    gguf_path,
                    revision=args.revision,
                )
                log.info("Checksum verified.")
            except ValueError as e:
                print_actionable_error(
                    "checksum verification failed",
                    cause=str(e),
                    next_steps=["Re-download or omit --verify-checksum"],
                )
                return 1
    except Exception as e:
        print_actionable_error(
            "download failed",
            cause=str(e),
            next_steps=[
                "To resume an interrupted download, re-run the same command "
                "(partial downloads are detected automatically)",
                "Or force standard downloader: add --no-hf-transfer",
                "If the repo is gated/private, run: huggingface-cli login",
                "Or set: HF_TOKEN=<your_token>",
                "Try: ollama-forge check",
            ],
        )
        return 1

    if getattr(args, "download_only", False):
        print(gguf_path)
        print(f"\nServe with: ollama-forge serve {gguf_path}", file=sys.stderr)
        return 0

    # Run convert with the downloaded path
    from ollama_forge.commands.convert import _cmd_convert
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
    import json as _json
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
    from ollama_forge.log import get_logger
    log = get_logger()
    if not getattr(args, "json", False):
        log.info("Quickstart plan:")
        log.info("  model name: %s", name)
        log.info("  repo: %s@%s", repo_id, getattr(args, "revision", "main"))
        log.info("  profile/task: %s / %s", profile, task or "none")
        print(
            f"  quant/temp/ctx/top_p/repeat: {quant} / {temperature} / {num_ctx} / {top_p} / {repeat_penalty}",
            file=sys.stderr,
        )
        log.info("  system prompt source: %s", system_source)
    if getattr(args, "plan", False):
        action = f"ollama-forge fetch {repo_id} --name {name} --quant {quant}"
        if getattr(args, "json", False):
            plan_obj = {
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
            from ollama_forge.commands.plan import _save_last_plan
            _save_last_plan("quickstart", plan_obj)
            print(_json.dumps(plan_obj))
        else:
            log.info("  action: %s", action)
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
    no_hf_transfer = getattr(args, "no_hf_transfer", False)
    adapter_dir = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="ollama-adapter-"))
    try:
        download_adapter(
            args.repo_id,
            revision=args.revision,
            local_dir=adapter_dir,
            no_hf_transfer=no_hf_transfer,
        )
        from ollama_forge.log import get_logger
        get_logger().info("Downloaded adapter to %s", adapter_dir)
    except Exception as e:
        print_actionable_error(
            "adapter download failed",
            cause=str(e),
            next_steps=[
                "To resume, re-run the same command (partial downloads are auto-detected)",
                "Or add --no-hf-transfer to force the standard downloader",
                "Confirm adapter repo id is correct on Hugging Face",
                "If gated/private, run: huggingface-cli login",
            ],
        )
        return 1
    resolved_adapter, verify_err, verify_steps = _verify_adapter_and_base(str(adapter_dir), args.base)
    if verify_err:
        print_actionable_error(
            "downloaded adapter format invalid",
            cause=verify_err,
            next_steps=verify_steps or [
                "Repo should contain PEFT files (adapter_config.json + adapter_model.*) or a single .bin/.gguf",
                "See wiki/Adapters.md for supported formats",
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
        adapter=resolved_adapter or str(adapter_dir),
        out_modelfile=args.out_modelfile,
    )
    from ollama_forge.commands.model import _cmd_create_from_base
    return _cmd_create_from_base(parser, fake)


def register(subparsers) -> None:
    """Register fetch, fetch-adapter, quickstart, start subcommands."""
    # quickstart
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
    p_quickstart.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip any prompts; use defaults (exit code 0 = success, 1 = error)",
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
    p_start.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip any prompts; use defaults (exit code 0 = success, 1 = error)",
    )
    p_start.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_start.set_defaults(handler=_cmd_start)

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
    p_fetch.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use defaults for missing repo/name (no TTY prompts); exit code 0 = success, 1 = error",
    )
    p_fetch.add_argument(
        "--verify-checksum",
        action="store_true",
        help="After download, verify file SHA256 against Hub ETag when available (LFS files)",
    )
    p_fetch.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_fetch.add_argument(
        "--download-only",
        action="store_true",
        help="Download the GGUF only (skip Ollama model creation); prints the local path. "
             "If the repo has no GGUF files, downloads safetensors and converts to GGUF automatically. "
             "Use with: ollama-forge serve <path>",
    )
    p_fetch.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for the downloaded/converted GGUF (default: ~/.cache/ollama-forge/gguf/<repo>/)",
    )
    p_fetch.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Path to llama.cpp clone (used when --download-only converts safetensors to GGUF)",
    )
    p_fetch.add_argument(
        "--no-hf-transfer",
        action="store_true",
        help="Disable hf_transfer (Rust accelerator) and use the standard downloader, "
             "which supports resuming interrupted downloads. "
             "Partial downloads are auto-detected and resumed even without this flag.",
    )
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
    p_fetch_adapter.add_argument(
        "--no-hf-transfer",
        action="store_true",
        help="Disable hf_transfer and use the standard downloader (supports resume)",
    )
    p_fetch_adapter.set_defaults(handler=_cmd_fetch_adapter)
