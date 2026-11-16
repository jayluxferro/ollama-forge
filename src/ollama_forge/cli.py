"""CLI entrypoint for ollama-forge."""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Prevent PyTorch's MPS allocator from contending with MLX on the Metal
# device. PyTorch's default high-watermark ratio lets its allocator issue
# `waitUntilCompleted` syncs that block on MLX's in-flight command buffers,
# tripping the Metal GPU watchdog (kIOGPUCommandBufferCallbackErrorTimeout)
# during mlx-vlm conversion of multimodal models whose processors touch torch
# (e.g. Gemma-3n's audio tower mel filterbank). These env vars are read
# lazily on first MPS use, so they MUST be set before anything imports torch.
# setdefault so user overrides win.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

# Import unsloth early (before transformers) so its optimizations apply.
# Optional dependency — ignore if not installed or no GPU available.
with contextlib.suppress(ImportError, NotImplementedError):
    import unsloth  # noqa: F401

from dotenv import load_dotenv

from ollama_forge.commands.abliterate import register as _register_abliterate_cmds
from ollama_forge.commands.adapters import register as _register_adapters_cmds
from ollama_forge.commands.auto import register as _register_auto_cmds
from ollama_forge.commands.cache import register as _register_cache_cmds
from ollama_forge.commands.plan import register as _register_plan_cmds
from ollama_forge.commands.quantize import register as _register_quantize_cmds
from ollama_forge.commands.security_eval import register as _register_security_eval_cmds
from ollama_forge.commands.serve import register as _register_serve_cmds
from ollama_forge.commands.study import register as _register_study_cmds
from ollama_forge.commands.train import register as _register_train_cmds
from ollama_forge.commands.turboquant import register as _register_turboquant_cmds
from ollama_forge.commands.vlm import register as _register_vlm_cmds
from ollama_forge.hf_fetch import (
    download_adapter,
    download_gguf,
    list_gguf_files,
    pick_one_gguf,
    verify_gguf_checksum,
)
from ollama_forge.log import get_logger, set_verbose
from ollama_forge.modelfile import (
    build_modelfile,
    merge_modelfile_with_reference_template,
)
from ollama_forge.recipe import load_recipe
from ollama_forge.run_helpers import (
    _gguf_cache_dir_for_repo,
    _llama_cpp_lib_env,
    _resolve_gguf_from_forge_cache,
    _resolve_llama_cpp_dir_from_arg,
    _which_quantize,
    _which_quantize_full,
    check_item,
    print_actionable_error,
    require_ollama,
    run_cmd,
    run_ollama_create,
    run_ollama_show_modelfile,
)

log = get_logger()



def _cmd_import(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Download HF safetensors → convert to GGUF → optionally quantize → create Ollama model."""
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code

    source: str = args.source
    name: str = args.name
    output_dir = Path(getattr(args, "output_dir", None) or tempfile.mkdtemp(prefix="ollama-forge-import-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = getattr(args, "revision", "main") or "main"

    gguf_converter = getattr(args, "gguf_converter", "auto") or "auto"

    # -- Resolve llama.cpp dir (optional when using unsloth-only) -----------------------
    llama_cpp_dir = getattr(args, "llama_cpp_dir", None) and Path(args.llama_cpp_dir)
    if not llama_cpp_dir:
        for candidate in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
            if (candidate / "convert_hf_to_gguf.py").is_file():
                llama_cpp_dir = candidate
                break
    if gguf_converter != "unsloth" and (not llama_cpp_dir or not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file()):
        print_actionable_error(
            "convert_hf_to_gguf.py not found",
            next_steps=[
                "Clone llama.cpp and set --llama-cpp-dir to the clone path",
                "Or run: ollama-forge setup-llama-cpp",
                "Or use --gguf-converter unsloth (requires: pip install unsloth)",
            ],
        )
        return 1

    # -- Resolve source (local dir or HF repo) ------------------------------------------
    source_path = Path(source)
    if source_path.is_dir():
        if not (source_path / "config.json").is_file():
            print_actionable_error(
                f"Local directory {source} does not contain config.json",
                next_steps=[
                    "Ensure the path is a valid HF model checkpoint with config.json",
                    "For GGUF files, use: ollama-forge convert --gguf <path> --name <name>",
                ],
            )
            return 1
        checkpoint_dir = source_path
        log.info("Using local checkpoint: %s", checkpoint_dir)
    else:
        # Treat as HF repo ID — download full snapshot
        checkpoint_dir = output_dir / "checkpoint"
        log.info("Downloading %s (revision=%s)...", source, revision)
        try:
            download_adapter(source, revision=revision, local_dir=checkpoint_dir)
        except Exception as e:
            print_actionable_error(
                f"Failed to download {source} from Hugging Face",
                cause=str(e),
                next_steps=[
                    "Check the repo ID is correct (e.g. meta-llama/Llama-3.2-1B-Instruct)",
                    "Ensure you are logged in: huggingface-cli login",
                    "Check network connectivity",
                ],
            )
            return 1

    gguf_path = output_dir / "model.gguf"

    from ollama_forge.commands.convert import _hf_checkpoint_to_ollama
    return _hf_checkpoint_to_ollama(
        checkpoint_dir=checkpoint_dir,
        gguf_path=gguf_path,
        llama_cpp_dir=llama_cpp_dir,
        name=name,
        outtype=getattr(args, "outtype", "bf16") or "bf16",
        requantize=not getattr(args, "no_requantize", False),
        quant_type=getattr(args, "quant", "Q4_K_M") or "Q4_K_M",
        template_from=getattr(args, "template_from", None),
        system=getattr(args, "system", None),
        temperature=getattr(args, "temperature", None),
        num_ctx=getattr(args, "num_ctx", None),
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        out_modelfile=getattr(args, "out_modelfile", None),
        gguf_converter=gguf_converter,
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


def _cmd_create_from_base(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    exit_code = require_ollama()
    if exit_code is not None:
        return exit_code
    adapter = getattr(args, "adapter", None)
    resolved_adapter = None
    if adapter:
        resolved_adapter, err, steps = _verify_adapter_and_base(adapter, args.base)
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
        adapter=resolved_adapter,
    )
    template_from = getattr(args, "template_from", None)
    if template_from:
        ref_content = run_ollama_show_modelfile(template_from)
        if ref_content:
            content = merge_modelfile_with_reference_template(
                content, ref_content, args.base, template_only=True
            )
            log.info("Using chat template from Ollama model %r", template_from)
        else:
            log.info("Note: no Ollama model %r found; pull it first for template.", template_from)
    return run_ollama_create(args.name, content, out_path=getattr(args, "out_modelfile", None))


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
    if getattr(args, "dry_run", False):
        out_path = getattr(args, "out_modelfile", None)
        if out_path:
            Path(out_path).write_text(merged, encoding="utf-8")
            log.info("Wrote Modelfile to %s (dry run)", out_path)
        else:
            print(merged)
        return 0
    return run_ollama_create(output_name, merged, out_path=getattr(args, "out_modelfile", None))


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
    from ollama_forge.commands.convert import _cmd_convert
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
            print(json.dumps(plan_obj))
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
        log.info("Downloaded adapter to %s", adapter_dir)
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
    return _cmd_create_from_base(parser, fake)


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
        template_from=recipe.get("template_from"),
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
        from ollama_forge.commands.convert import _cmd_convert
        return _cmd_convert(parser, _namespace_for_convert(recipe, gguf, out_modelfile))
    return _cmd_create_from_base(parser, _namespace_for_create_from_base(recipe, out_modelfile))


def _cmd_validate_recipe(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Validate a recipe file (schema and paths) without building."""
    report: dict = {"valid": False, "errors": [], "fields": {}}
    try:
        recipe = load_recipe(args.recipe)
    except FileNotFoundError as e:
        report["errors"].append(str(e))
        if getattr(args, "json", False):
            print(json.dumps(report))
            return 1
        print_actionable_error(
            str(e),
            next_steps=["Check recipe path and retry: ollama-forge validate-recipe <recipe.yaml>"],
        )
        return 1
    except (ValueError, ImportError) as e:
        report["errors"].append(str(e))
        if getattr(args, "json", False):
            print(json.dumps(report))
            return 1
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
    for key in ("name", "base", "gguf", "hf_repo", "system", "temperature", "quant", "revision", "gguf_file"):
        if key not in recipe:
            continue
        val = recipe[key]
        field_report: dict = {"value": val, "valid": True}
        if key == "gguf" and source == "gguf":
            p = Path(val).resolve()
            if not p.is_file():
                field_report["valid"] = False
                field_report["message"] = f"File not found: {p}"
                report["errors"].append(field_report["message"])
        report["fields"][key] = field_report

    if getattr(args, "validate_remote", False) and source == "hf_repo":
        repo_id = recipe.get("hf_repo")
        if repo_id:
            try:
                from huggingface_hub import HfApi
                HfApi().repo_info(repo_id=repo_id, repo_type="model")
                report["fields"].setdefault("hf_repo", {"value": repo_id, "valid": True})["remote"] = True
            except Exception as e:
                report["errors"].append(f"Remote repo check failed: {e}")
                report["fields"].setdefault("hf_repo", {"value": repo_id, "valid": True})["valid"] = False
                report["fields"]["hf_repo"]["message"] = str(e)
                report["fields"]["hf_repo"]["remote"] = False

    report["valid"] = len(report["errors"]) == 0
    report["source"] = source

    if getattr(args, "json", False):
        print(json.dumps(report))
        return 0 if report["valid"] else 1
    print(f"Recipe valid: name={recipe['name']!r}, source={source}")
    return 0


def _cmd_check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Verify ollama, HF, Python deps, and llama.cpp; print what's missing."""
    if getattr(args, "fix", False):
        fake = argparse.Namespace(
            fix=True,
            plan=False,
            fix_llama_cpp=getattr(args, "fix_llama_cpp", False),
            llama_cpp_dir=getattr(args, "llama_cpp_dir", None),
            json=getattr(args, "json", False),
        )
        return _cmd_doctor(parser, fake)
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
        "run: uv sync",
    )
    finetune = shutil.which("finetune") or shutil.which("llama-finetune")
    quantize = _which_quantize()
    check_item(
        "llama.cpp finetune",
        bool(finetune),
        "To run finetune/train-run with --base-gguf: ollama-forge setup-llama-cpp, then add build dir to PATH",
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
        # Device info
        from ollama_forge.device import get_device_name, get_memory_info

        dev_name = get_device_name()
        mem = get_memory_info()
        if mem:
            print(f"Accelerator: {dev_name} ({mem.total_gb:.1f} GB, {mem.free_gb:.1f} GB free)")
        else:
            print(f"Accelerator: {dev_name}")
        check_item(
            "abliterate deps",
            status["abliterate_deps"],
            "run: uv sync",
        )
        check_item(
            "llama.cpp finetune",
            status["finetune"],
            "For finetune/train-run with --base-gguf: ollama-forge setup-llama-cpp, then add build dir to PATH",
        )
        check_item(
            "llama.cpp quantize",
            status["quantize"],
            "run: ollama-forge setup-llama-cpp",
        )
        # Check llama.cpp staleness
        for candidate in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
            if (candidate / ".git").is_dir():
                age = _llama_cpp_git_age_days(candidate)
                if age is not None:
                    commit = _llama_cpp_current_commit(candidate)
                    if age > 30:
                        print(
                            f"llama.cpp at {candidate} is {age} days old ({commit})."
                            f" Run: ollama-forge setup-llama-cpp --dir {candidate} --update"
                        )
                    else:
                        print(f"llama.cpp at {candidate}: up to date ({commit}, {age}d old)")
                break

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
            plan_obj = {"route": "doctor-fix", "actions": planned}
            from ollama_forge.commands.plan import _save_last_plan  # noqa: PLC0415
            _save_last_plan("doctor-fix", plan_obj)
            print(json.dumps(plan_obj))
        else:
            print("\nFix plan:")
            for step in planned:
                print(f"  - {step}")
        return 0

    log.info("Applying fixes...")
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
        log.info("Applied: uv sync")

    if getattr(args, "fix_llama_cpp", False) and (not status["finetune"] or not status["quantize"]):
        code = _cmd_setup_llama_cpp(
            parser,
            argparse.Namespace(dir=getattr(args, "llama_cpp_dir", None)),
        )
        if code != 0:
            return code
    elif (not status["finetune"] or not status["quantize"]) and not getattr(args, "fix_llama_cpp", False):
        log.info("Tip: add --fix-llama-cpp to auto-install llama.cpp tools.")

    if not status["ollama"]:
        log.info("Cannot auto-install Ollama here. Install from https://ollama.com, then rerun doctor.")
        return 1

    final_status = _env_status()
    ok = final_status["ollama"] and final_status["huggingface_hub"] and final_status["pyyaml"]
    if json_mode:
        print(json.dumps(final_status))
    return 0 if ok else 1


def _build_llama_cpp(target_dir: Path) -> int:
    """Run cmake configure + build in target_dir/build. Returns exit code."""
    build_dir = target_dir / "build"
    build_dir.mkdir(exist_ok=True)
    # Remove stale CMakeCache.txt if it references a different source directory
    # (e.g. the repo was moved or cloned to a new location).
    cache_file = build_dir / "CMakeCache.txt"
    if cache_file.is_file():
        try:
            cache_text = cache_file.read_text(encoding="utf-8", errors="replace")
            stale = False
            for line in cache_text.splitlines():
                if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                    cached_src = line.split("=", 1)[1].strip()
                    if cached_src != str(target_dir.resolve()):
                        log.info("Stale CMakeCache.txt (source was %s), removing build dir...", cached_src)
                        stale = True
                        break
                if line.startswith("CMAKE_OSX_SYSROOT:"):
                    cached_sdk = line.split("=", 1)[1].strip()
                    if cached_sdk and not Path(cached_sdk).is_dir():
                        log.info("Stale CMAKE_OSX_SYSROOT (%s no longer exists), removing build dir...", cached_sdk)
                        stale = True
                        break
            if stale:
                shutil.rmtree(build_dir)
                build_dir.mkdir()
        except OSError:
            pass
    # Clean cmake artifacts that may have leaked into the source directory
    # (e.g. from a previous failed build or misconfigured cmake run).
    for stale in ["CMakeCache.txt", "CMakeFiles", "Makefile", "cmake_install.cmake"]:
        stale_path = target_dir / stale
        if stale_path.is_file():
            stale_path.unlink()
        elif stale_path.is_dir():
            shutil.rmtree(stale_path)

    log.info("Building (cmake)...")
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
        bin_dir = build_dir
    print(f'\nDone. Add to PATH: export PATH="{bin_dir}:$PATH"')
    print("Then you can use: finetune, quantize, and other llama.cpp tools.")
    return 0


def _llama_cpp_git_age_days(target_dir: Path) -> int | None:
    """Return days since the last git commit in target_dir, or None if not a git repo."""
    from datetime import datetime, timezone

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            capture_output=True, text=True, timeout=10, check=False, cwd=target_dir,
        )
        if result.returncode != 0:
            return None
        timestamp = int(result.stdout.strip())
        commit_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        age = datetime.now(timezone.utc) - commit_date
        return age.days
    except Exception:
        return None


def _llama_cpp_current_commit(target_dir: Path) -> str | None:
    """Return the short git hash of the current llama.cpp checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False, cwd=target_dir,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _cmd_setup_llama_cpp(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Clone and build llama.cpp; or update an existing clone with --update."""
    if getattr(args, "use_conda", False):
        print(
            "Use conda to install llama.cpp: conda install -c conda-forge llama-cpp\n"
            "Ensure finetune and quantize (or llama-finetune, llama-quantize) are on PATH.\n"
            "For convert/quantize you need GGUF support; see wiki or --help for CMake options.",
            file=sys.stderr,
        )
        return 0
    if getattr(args, "use_system", False):
        q = _which_quantize()
        ft = shutil.which("finetune") or shutil.which("llama-finetune")
        if q and ft:
            print("finetune and quantize are on PATH. No setup needed.")
            return 0
        print("finetune or quantize not found on PATH.", file=sys.stderr)
        print(
            "Install llama.cpp (system package or build from source) and add its bin dir to PATH.",
            file=sys.stderr,
        )
        return 1

    target_dir = Path(args.dir or "llama.cpp").resolve()
    update = getattr(args, "update", False)

    # --update on an existing clone: pull latest + rebuild
    if update:
        if not target_dir.exists() or not (target_dir / ".git").is_dir():
            log.info("No existing clone at %s; will do a fresh clone instead.", target_dir)
            # Fall through to clone path below
        else:
            old_hash = _llama_cpp_current_commit(target_dir)
            log.info("Updating llama.cpp at %s...", target_dir)
            # Unshallow if needed (setup-llama-cpp clones with --depth 1)
            code = run_cmd(
                ["git", "fetch", "--depth", "1", "origin"],
                not_found_message="Error: git not found.",
                process_error_message="Error: git fetch failed: {e}",
                cwd=target_dir,
            )
            if code != 0:
                return code
            code = run_cmd(
                ["git", "reset", "--hard", "origin/HEAD"],
                not_found_message="Error: git not found.",
                process_error_message="Error: git reset failed: {e}",
                cwd=target_dir,
            )
            if code != 0:
                return code
            new_hash = _llama_cpp_current_commit(target_dir)
            if old_hash and new_hash and old_hash == new_hash:
                print(f"Already up to date ({new_hash}).")
                return 0
            log.info("Updated %s -> %s", old_hash or "unknown", new_hash or "unknown")
            return _build_llama_cpp(target_dir)

    # Fresh clone
    if target_dir.exists() and any(target_dir.iterdir()):
        if (target_dir / ".git").is_dir():
            log.warning(
                "Directory already exists: %s. Use --update to pull latest, or --dir <other>.",
                target_dir,
            )
        else:
            log.warning(
                "Directory already exists and is non-empty: %s. Use --dir <other> or remove it.",
                target_dir,
            )
        return 1
    url = "https://github.com/ggerganov/llama.cpp"
    log.info("Cloning %s into %s...", url, target_dir)
    code = run_cmd(
        ["git", "clone", "--depth", "1", url, str(target_dir)],
        not_found_message="Error: git not found. Install git and try again.",
        process_error_message="Error: git clone failed: {e}",
    )
    if code != 0:
        return code
    return _build_llama_cpp(target_dir)


# ---------------------------------------------------------------------------
# quantize – requantize a GGUF file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# serve – spin up llama-server with a GGUF model
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) output",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # check (environment)
    p_check = subparsers.add_parser(
        "check",
        help="Verify ollama, Hugging Face, Python deps, and llama.cpp",
    )
    p_check.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable status (same shape as doctor --json)",
    )
    p_check.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes (same as doctor --fix: uv sync, optional setup-llama-cpp)",
    )
    p_check.add_argument(
        "--fix-llama-cpp",
        action="store_true",
        help="With --fix, also run setup-llama-cpp when finetune/quantize missing",
    )
    p_check.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="With --fix --fix-llama-cpp, directory for llama.cpp clone",
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
    p_plan = _register_plan_cmds(subparsers)

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

    # auto (detect source type and route)
    p_auto = _register_auto_cmds(subparsers)

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
    p_setup.add_argument(
        "--use-system",
        action="store_true",
        help="Do not clone/build; verify finetune/quantize on PATH (use system-installed llama.cpp)",
    )
    p_setup.add_argument(
        "--use-conda",
        action="store_true",
        help="Print instructions for using conda-installed llama.cpp (e.g. conda install -c conda-forge llama-cpp)",
    )
    p_setup.add_argument(
        "--update",
        action="store_true",
        help="Pull latest changes and rebuild an existing llama.cpp clone (git fetch + reset + cmake build)",
    )
    p_setup.set_defaults(handler=_cmd_setup_llama_cpp)

    # quantize (requantize a GGUF file)
    p_quantize = _register_quantize_cmds(subparsers)

    # serve / chat
    p_serve, p_chat = _register_serve_cmds(subparsers)

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
    p_create.add_argument(
        "--template-from",
        metavar="OLLAMA_MODEL",
        help="Ollama model to copy chat template from (for tool/Chat API support)",
    )
    p_create.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_create.set_defaults(handler=_cmd_create_from_base)

    # refresh-template (recreate model with base's latest chat template)
    p_refresh = subparsers.add_parser(
        "refresh-template",
        help="Recreate a model using the base model's latest chat template (fixes Chat API issues). "
             "Replaces the existing model when --output-name equals --name.",
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
        "--dry-run",
        action="store_true",
        help="Print merged Modelfile (or write to --out-modelfile) and exit without running ollama create",
    )
    p_refresh.add_argument(
        "--out-modelfile",
        help="Also write the merged Modelfile to this path",
    )
    p_refresh.set_defaults(handler=_cmd_refresh_template)

    # convert (GGUF → Ollama; use after HF→GGUF via llama.cpp)
    from ollama_forge.commands.convert import register as _register_convert_cmds
    p_convert = _register_convert_cmds(subparsers)

    # import (HF safetensors → GGUF → Ollama)
    p_import = subparsers.add_parser(
        "import",
        help="Download HF safetensors, convert to GGUF, and create an Ollama model (one command)",
    )
    p_import.add_argument(
        "source",
        help="Hugging Face repo ID (e.g. meta-llama/Llama-3.2-1B-Instruct) or local checkpoint directory",
    )
    p_import.add_argument("--name", required=True, help="Name for the new Ollama model")
    p_import.add_argument("--llama-cpp-dir", help="Path to llama.cpp clone (auto-detected if omitted)")
    p_import.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        default="bf16",
        help="GGUF output type (default: bf16)",
    )
    p_import.add_argument("--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)")
    p_import.add_argument(
        "--no-requantize",
        action="store_true",
        default=False,
        help="Skip quantization; keep full-size GGUF",
    )
    p_import.add_argument("--template-from", help="Copy chat template from an existing Ollama model")
    p_import.add_argument("--output-dir", help="Download/output directory (default: auto temp dir)")
    p_import.add_argument("--revision", default="main", help="HF repo revision (default: main)")
    p_import.add_argument("--system", help="System message (role/instructions)")
    p_import.add_argument("--temperature", type=float, help="Temperature (e.g. 0.7)")
    p_import.add_argument("--num-ctx", type=int, help="Context window size in tokens (e.g. 4096)")
    p_import.add_argument("--top-p", type=float, help="Top-p sampling (e.g. 0.9)")
    p_import.add_argument("--repeat-penalty", type=float, help="Repeat penalty (e.g. 1.1)")
    p_import.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_import.add_argument(
        "--gguf-converter",
        choices=["llama-cpp", "unsloth", "auto"],
        default="auto",
        help="GGUF converter: llama-cpp (default subprocess), unsloth (requires unsloth package), "
             "auto (try llama-cpp first, fall back to unsloth). Default: auto",
    )
    p_import.set_defaults(handler=_cmd_import)

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
    p_validate_recipe.add_argument(
        "--json",
        action="store_true",
        help="Output per-field validation result as JSON",
    )
    p_validate_recipe.add_argument(
        "--validate-remote",
        action="store_true",
        help="When recipe uses hf_repo, check that the Hugging Face repo exists",
    )
    p_validate_recipe.set_defaults(handler=_cmd_validate_recipe)

    # train group (validate-training-data, prepare-training-data, convert-training-data-format,
    # train-data, train-resolve-base, train, retrain, train-run, finetune)
    p_train_data = _register_train_cmds(subparsers)

    # abliterate (refusal removal)
    p_abliterate = _register_abliterate_cmds(subparsers)

    # adapters (search Hugging Face for adapters)
    p_adapters = _register_adapters_cmds(subparsers)

    # hf-cache / cache
    p_hf_cache, p_cache = _register_cache_cmds(subparsers)

    p_study = _register_study_cmds(subparsers)

    # security-eval
    p_security_eval = _register_security_eval_cmds(subparsers)

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

    p_vlm = _register_vlm_cmds(subparsers)

    # turboquant (TurboQuant quantization, serving, and inference)
    p_tq = _register_turboquant_cmds(subparsers)

    parsed = parser.parse_args()
    set_verbose(getattr(parsed, "verbose", False))
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
    if parsed.command == "cache" and not getattr(parsed, "cache_command", None):
        p_cache.print_help()
        return 0
    if parsed.command == "study" and not getattr(parsed, "study_command", None):
        p_study.print_help()
        return 0
    if parsed.command == "security-eval" and not getattr(parsed, "security_eval_command", None):
        p_security_eval.print_help()
        return 0
    if parsed.command == "vlm" and not getattr(parsed, "vlm_command", None):
        p_vlm.print_help()
        return 0
    if parsed.command == "turboquant" and not getattr(parsed, "turboquant_command", None):
        p_tq.print_help()
        return 0
    if parsed.command == "train-data" and not getattr(parsed, "train_data_cmd", None):
        p_train_data.print_help()
        return 0
    handler = getattr(parsed, "handler", None)
    if handler is None:
        return 0
    return handler(parser, parsed)


if __name__ == "__main__":
    sys.exit(main())
