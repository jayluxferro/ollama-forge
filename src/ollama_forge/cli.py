"""CLI entrypoint for ollama-forge."""

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from ollama_forge.commands.cache import register as _register_cache_cmds
from ollama_forge.commands.security_eval import register as _register_security_eval_cmds
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
    get_stop_tokens_from_checkpoint,
    merge_modelfile_with_reference_template,
    modelfile_append_num_predict,
    modelfile_append_renderer_parser,
    modelfile_append_stop_parameters,
    modelfile_append_template,
    template_from_hf_checkpoint,
)
from ollama_forge.recipe import load_recipe
from ollama_forge.run_helpers import (
    check_item,
    print_actionable_error,
    require_ollama,
    run_cmd,
    run_ollama_create,
    run_ollama_show_modelfile,
    stream_openai_chat_sse,
)

log = get_logger()


def _plan_file_path() -> Path:
    """Path for persisting last plan (for 'plan continue')."""
    if os.environ.get("OLLAMA_FORGE_PLAN_FILE"):
        return Path(os.environ["OLLAMA_FORGE_PLAN_FILE"]).expanduser().resolve()
    return Path.cwd() / ".ollama-forge-last-plan.json"


def _save_last_plan(plan_command: str, plan_obj: dict) -> None:
    """Persist plan JSON so 'plan continue' can show or re-run it."""
    path = _plan_file_path()
    payload = {
        "plan_command": plan_command,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **plan_obj,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        log.debug("Could not save last plan to %s: %s", path, e)


def _which_quantize() -> str | None:
    """Resolve llama.cpp quantize binary (quantize or llama-quantize)."""
    return shutil.which("quantize") or shutil.which("llama-quantize")


def _which_quantize_full(llama_cpp_dir: Path | None = None) -> str | None:
    """Resolve quantize binary: check PATH, then llama.cpp build dirs."""
    on_path = _which_quantize()
    if on_path:
        return on_path
    candidates: list[Path] = []
    if llama_cpp_dir:
        candidates.append(llama_cpp_dir / "build" / "bin" / "llama-quantize")
    for d in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
        candidates.append(d / "build" / "bin" / "llama-quantize")
        candidates.append(d / "build" / "bin" / "quantize")
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def _which_llama_server(llama_cpp_dir: Path | None = None) -> str | None:
    """Resolve llama-server binary: check PATH, then llama.cpp build dirs."""
    on_path = shutil.which("llama-server")
    if on_path:
        return on_path
    candidates: list[Path] = []
    if llama_cpp_dir:
        candidates.append(llama_cpp_dir / "build" / "bin" / "llama-server")
    for d in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
        candidates.append(d / "build" / "bin" / "llama-server")
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def _resolve_llama_cpp_dir_from_arg(args: argparse.Namespace) -> Path | None:
    """Resolve llama.cpp directory from --llama-cpp-dir or well-known locations."""
    llama_cpp_dir = getattr(args, "llama_cpp_dir", None) and Path(args.llama_cpp_dir)
    if not llama_cpp_dir:
        for candidate in [Path("llama.cpp"), Path.home() / "llama.cpp"]:
            if candidate.is_dir():
                llama_cpp_dir = candidate
                break
    return llama_cpp_dir


def _convert_gguf_with_llama_cpp(
    checkpoint_dir: Path,
    gguf_path: Path,
    llama_cpp_dir: Path,
    outtype: str = "bf16",
) -> int:
    """Convert HF checkpoint to GGUF using llama.cpp convert_hf_to_gguf.py. Returns exit code."""
    print("Converting to GGUF (llama.cpp)...", file=sys.stderr)
    convert_script = (llama_cpp_dir / "convert_hf_to_gguf.py").resolve()
    try:
        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(checkpoint_dir.resolve()),
                "--outfile",
                str(gguf_path.resolve()),
                "--outtype",
                outtype,
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
                "Re-run with --llama-cpp-dir <path>",
            ],
        )
        return 1
    except subprocess.CalledProcessError as e:
        print_actionable_error(
            "GGUF conversion failed (llama.cpp)",
            cause=str(e),
            next_steps=[
                "Ensure llama.cpp convert_hf_to_gguf.py runs in that directory",
                "Run: ollama-forge setup-llama-cpp; add build dir to PATH",
                "Or try --gguf-converter unsloth (requires: pip install unsloth)",
            ],
        )
        return 1
    if not gguf_path.is_file():
        print_actionable_error(
            "GGUF file was not produced",
            next_steps=["Check disk space and llama.cpp convert script output"],
        )
        return 1
    return 0


def _convert_gguf_checkpoint(
    *,
    checkpoint_dir: Path,
    gguf_path: Path,
    llama_cpp_dir: Path | None,
    outtype: str = "bf16",
    quant_type: str = "Q4_K_M",
    gguf_converter: str = "auto",
) -> int:
    """Dispatch GGUF conversion to llama-cpp, unsloth, or auto (try llama-cpp then unsloth). Returns exit code."""
    if gguf_converter == "unsloth":
        return _convert_gguf_with_unsloth(checkpoint_dir, gguf_path, quant_type=quant_type)

    if gguf_converter == "llama-cpp":
        if not llama_cpp_dir or not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
            print_actionable_error(
                "convert_hf_to_gguf.py not found",
                next_steps=[
                    "Clone llama.cpp and set --llama-cpp-dir to the clone path",
                    "Or run: ollama-forge setup-llama-cpp",
                ],
            )
            return 1
        return _convert_gguf_with_llama_cpp(checkpoint_dir, gguf_path, llama_cpp_dir, outtype=outtype)

    # auto: try llama-cpp first, fall back to unsloth
    if llama_cpp_dir and (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
        rc = _convert_gguf_with_llama_cpp(checkpoint_dir, gguf_path, llama_cpp_dir, outtype=outtype)
        if rc == 0:
            return 0
        log.info("llama.cpp conversion failed; trying unsloth fallback...")

    # Try unsloth
    try:
        import unsloth  # noqa: F401
    except (ImportError, NotImplementedError):
        if not llama_cpp_dir or not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
            print_actionable_error(
                "No GGUF converter available",
                next_steps=[
                    "Set up llama.cpp: ollama-forge setup-llama-cpp",
                    "Or install unsloth: pip install unsloth",
                ],
            )
        else:
            print_actionable_error(
                "llama.cpp conversion failed and unsloth is not installed for fallback",
                next_steps=[
                    "Install unsloth: pip install unsloth",
                    "Or fix the llama.cpp error above",
                ],
            )
        return 1
    return _convert_gguf_with_unsloth(checkpoint_dir, gguf_path, quant_type=quant_type)


def _convert_gguf_with_unsloth(
    checkpoint_dir: Path,
    gguf_path: Path,
    quant_type: str = "Q4_K_M",
) -> int:
    """Convert HF checkpoint to GGUF using unsloth. Returns exit code (0=success)."""
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel  # type: ignore[import-untyped]
    except (ImportError, NotImplementedError) as exc:
        print_actionable_error(
            f"unsloth is not available: {exc}",
            next_steps=[
                "Install: pip install unsloth",
                "Or use --gguf-converter llama-cpp to skip unsloth",
            ],
        )
        return 1

    log.info("Converting to GGUF with unsloth...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(str(checkpoint_dir.resolve()))
        model.save_pretrained_gguf(
            str(gguf_path.parent),
            tokenizer,
            quantization_method=quant_type,
        )
        # unsloth may produce files with varying names; find the .gguf in output dir
        produced = list(gguf_path.parent.glob("*.gguf"))
        if produced and not gguf_path.is_file():
            # Rename the first produced GGUF to the expected path
            produced[0].rename(gguf_path)
        if gguf_path.is_file():
            log.info("Unsloth GGUF conversion succeeded: %s", gguf_path)
            return 0
        print_actionable_error(
            "unsloth did not produce a GGUF file",
            next_steps=["Check unsloth output above for errors"],
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "unsloth GGUF conversion failed",
            cause=str(e),
            next_steps=[
                "Check that unsloth supports this model architecture",
                "Try --gguf-converter llama-cpp instead",
            ],
        )
        return 1


def _hf_checkpoint_to_ollama(
    *,
    checkpoint_dir: Path,
    gguf_path: Path,
    llama_cpp_dir: Path | None,
    name: str,
    outtype: str = "bf16",
    requantize: bool = True,
    quant_type: str = "Q4_K_M",
    template_from: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    out_modelfile: str | Path | None = None,
    gguf_converter: str = "auto",
) -> int:
    """Convert HF checkpoint → GGUF → (quantize) → derive template → ollama create. Returns exit code."""
    from ollama_forge.model_family import remap_architecture_in_config

    # -- 0. Remap architecture aliases in config.json before conversion --------------------
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        orig_arch = remap_architecture_in_config(config_path)
        if orig_arch:
            log.info("Remapped architecture %r for GGUF conversion", orig_arch)

    # -- 1. Convert HF checkpoint to GGUF -----------------------------------------------
    rc = _convert_gguf_checkpoint(
        checkpoint_dir=checkpoint_dir,
        gguf_path=gguf_path,
        llama_cpp_dir=llama_cpp_dir,
        outtype=outtype,
        quant_type=quant_type,
        gguf_converter=gguf_converter,
    )
    if rc != 0:
        return rc

    # -- 2. Optionally requantize (skip when unsloth already quantized) -----------------
    gguf_to_use = gguf_path
    if requantize and gguf_converter != "unsloth":
        quantize_bin = _which_quantize_full(llama_cpp_dir)
        if not quantize_bin:
            print_actionable_error(
                "requantize (default) requires llama.cpp quantize",
                next_steps=[
                    "Run: ollama-forge setup-llama-cpp; add the build dir to PATH",
                    "Or pass --no-requantize to keep full-size GGUF (no quantize step)",
                ],
            )
            return 1
        quant_gguf = gguf_path.parent / f"{gguf_path.stem}-{quant_type}.gguf"
        print(f"Quantizing to {quant_type}...", file=sys.stderr)
        env = _llama_cpp_lib_env(quantize_bin)
        try:
            subprocess.run(
                [quantize_bin, str(gguf_path), str(quant_gguf), quant_type],
                check=True,
                timeout=7200,
                env=env,
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

    # -- 3. Build Modelfile with generation params --------------------------------------
    gguf_for_modelfile = gguf_to_use.resolve()
    content = build_modelfile(
        str(gguf_for_modelfile),
        system=system,
        temperature=temperature,
        num_ctx=num_ctx,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

    # -- 4. Template selection ----------------------------------------------------------
    # Check for native Ollama RENDERER/PARSER support first — these handle chat
    # formatting, thinking, tools, and vision natively, so we skip template derivation.
    from ollama_forge.model_family import get_native_renderer_parser

    renderer, parser = get_native_renderer_parser(checkpoint_dir)
    if renderer:
        content = modelfile_append_renderer_parser(content, renderer, parser)
        log.info("Using native Ollama RENDERER %r / PARSER %r", renderer, parser)
    elif template_from:
        ref_content = run_ollama_show_modelfile(template_from)
        if ref_content:
            content = merge_modelfile_with_reference_template(
                content, ref_content, base=str(gguf_for_modelfile), template_only=True
            )
            log.info("Using chat template from Ollama model %r (for tool/Chat API support)", template_from)
        else:
            log.info("Note: no Ollama model %r found; pull it first for tool support.", template_from)

    # Detect model family for diagnostics
    if not renderer:
        try:
            from ollama_forge.model_family import get_family_name

            family_name = get_family_name(checkpoint_dir)
            if family_name:
                log.info("Detected model family: %s", family_name)
        except ImportError:
            pass

    # If still no TEMPLATE, derive from the checkpoint's HF tokenizer
    if not renderer and not re.search(r"TEMPLATE\s+\"\"\"", content, re.IGNORECASE):
        hf_template = template_from_hf_checkpoint(checkpoint_dir)
        if hf_template:
            content = modelfile_append_template(content, hf_template)
            stop_tokens = get_stop_tokens_from_checkpoint(checkpoint_dir)
            if stop_tokens:
                content = modelfile_append_stop_parameters(content, stop_tokens)
            content = modelfile_append_num_predict(content, 2048)
            log.info("Using chat template derived from checkpoint (HF format) for Ollama.")

    # -- 5. Create Ollama model ---------------------------------------------------------
    return run_ollama_create(name, content, out_path=out_modelfile)


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


def _ollama_forge_cache_dir() -> Path:
    """Return the ollama-forge cache root (``~/.cache/ollama-forge`` or ``OLLAMA_FORGE_CACHE``)."""
    return Path(os.environ.get("OLLAMA_FORGE_CACHE", Path.home() / ".cache" / "ollama-forge"))


def _gguf_cache_dir_for_repo(repo_id: str) -> Path:
    """Return ``<cache>/gguf/<owner>/<repo>/`` for storing converted GGUFs."""
    return _ollama_forge_cache_dir() / "gguf" / repo_id.replace("/", os.sep)


def _resolve_gguf_from_forge_cache(repo_id: str) -> Path | None:
    """Look for a previously converted GGUF in the ollama-forge cache.

    Prefers quantized files (smaller) over raw bf16/f16 ones.
    """
    cache = _gguf_cache_dir_for_repo(repo_id)
    if not cache.is_dir():
        return None
    gguf_files = list(cache.glob("*.gguf"))
    if not gguf_files:
        return None
    if len(gguf_files) == 1:
        return gguf_files[0]
    # Prefer quantized (smaller) files — they have quant tags like Q4_K_M in the name
    from ollama_forge.hf_fetch import pick_one_gguf
    names = [p.name for p in gguf_files]
    best = pick_one_gguf(names)
    return next(p for p in gguf_files if p.name == best)


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
    from ollama_forge.commands.adapters import _cmd_adapters_recommend

    return _cmd_adapters_recommend(parser, fake)


def _cmd_plan_continue(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Show or execute the last persisted plan (from plan ... --json)."""
    path = _plan_file_path()
    if not path.is_file():
        print_actionable_error(
            "No saved plan found",
            next_steps=[
                "Run a plan with --json first, e.g.: ollama-forge plan quickstart --json",
                f"Plan file path: {path}",
            ],
        )
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print_actionable_error("Invalid or unreadable plan file", cause=str(e), next_steps=[f"Remove or fix: {path}"])
        return 1
    if getattr(args, "json", False):
        print(json.dumps(data))
        return 0
    saved_at = data.get("saved_at", "unknown")
    plan_cmd = data.get("plan_command", "unknown")
    print(f"Last plan ({plan_cmd}, saved at {saved_at}):", file=sys.stderr)
    if "action" in data:
        action = data["action"]
        print(f"  {action}", file=sys.stderr)
        if getattr(args, "execute", False):
            # Strip optional "Run: " prefix
            cmd = action.strip()
            if cmd.lower().startswith("run:"):
                cmd = cmd[4:].strip()
            code = subprocess.run(shlex.split(cmd))
            return code.returncode
    elif "actions" in data:
        for step in data["actions"]:
            print(f"  - {step}", file=sys.stderr)
        if getattr(args, "execute", False):
            last_code = 0
            for step in data["actions"]:
                cmd = step.strip()
                if cmd.lower().startswith("run:"):
                    cmd = cmd[4:].strip()
                code = subprocess.run(shlex.split(cmd))
                last_code = code.returncode
            return last_code
    else:
        log.warning("Saved plan has no 'action' or 'actions'; nothing to run.")
    return 0


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
            plan_obj = {"route": route, "source": source, "action": detail}
            _save_last_plan("auto", plan_obj)
            print(json.dumps(plan_obj))
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
            from ollama_forge.commands.train import _cmd_retrain  # noqa: PLC0415
            return _cmd_retrain(parser, fake)
        # HF checkpoint (config.json) → import
        if (source_path / "config.json").is_file():
            fake = argparse.Namespace(
                source=str(source_path),
                name=name,
                llama_cpp_dir=None,
                outtype="bf16",
                quant=args.quant or "Q4_K_M",
                no_requantize=False,
                template_from=None,
                output_dir=None,
                revision="main",
                system=args.system,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                out_modelfile=args.out_modelfile,
            )
            if maybe_plan(
                "import",
                f"ollama-forge import {source_path} --name {name}",
            ):
                return 0
            return _cmd_import(parser, fake)
        print_actionable_error(
            f"unsupported local directory source: {source_path}",
            next_steps=[
                "Use auto with a recipe/.gguf/HF repo/base model",
                "Or provide an adapter directory (with adapter_config.json)",
                "Or provide an HF checkpoint directory (with config.json)",
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
        quantize_bin = _which_quantize_full()
        if not quantize_bin:
            print_actionable_error(
                "--quantize requires llama.cpp quantize binary",
                next_steps=[
                    "Run: ollama-forge setup-llama-cpp",
                    "Or use a pre-quantized GGUF and skip --quantize",
                ],
            )
            return 1
        out_gguf = gguf.parent / f"{gguf.stem}-{q}.gguf"
        env = _llama_cpp_lib_env(quantize_bin)
        try:
            subprocess.run(
                [quantize_bin, str(gguf), str(out_gguf), q],
                check=True, env=env, timeout=7200,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print_actionable_error(
                "quantize failed",
                cause=str(e),
                next_steps=[
                    "Run: ollama-forge setup-llama-cpp --update",
                    "Or use a pre-quantized GGUF and skip --quantize",
                ],
            )
            return 1
        log.info("Quantized to %s", out_gguf)
        gguf_to_use = str(out_gguf)
    adapter_path: str | None = None
    if getattr(args, "adapter", None):
        ap = Path(args.adapter).resolve()
        if not ap.exists():
            print_actionable_error(
                f"Adapter path not found: {ap}",
                next_steps=["Check --adapter path (directory or .bin/.gguf file)"],
            )
            return 1
        adapter_path = str(ap)
    content = build_modelfile(
        gguf_to_use,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        adapter=adapter_path,
    )
    return run_ollama_create(args.name, content, out_path=args.out_modelfile)


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


def _cmd_quantize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Quantize (or requantize) a GGUF file."""
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        # Try resolving from forge/HF cache by repo ID
        if "/" in args.input:
            cached = _resolve_gguf_from_forge_cache(args.input) or _resolve_gguf_from_hf_cache(args.input)
            if cached and cached.is_file():
                input_path = cached
                print(f"Resolved from cache: {input_path}", file=sys.stderr)
        if not input_path.is_file():
            print_actionable_error(
                f"File not found: {args.input}",
                next_steps=["Provide a path to a .gguf file or a cached repo ID (org/model)"],
            )
            return 1

    quant = getattr(args, "quant", "Q4_K_M") or "Q4_K_M"
    output = getattr(args, "output", None)
    output_path = Path(output).resolve() if output else input_path.parent / f"{input_path.stem}-{quant}.gguf"

    llama_cpp_dir = _resolve_llama_cpp_dir_from_arg(args)
    quantize_bin = _which_quantize_full(llama_cpp_dir)
    if not quantize_bin:
        print_actionable_error(
            "quantize binary not found",
            next_steps=[
                "Run: ollama-forge setup-llama-cpp; add build dir to PATH",
                "Or pass --llama-cpp-dir <path-to-llama.cpp-clone>",
            ],
        )
        return 1

    size_gb = input_path.stat().st_size / (1024 ** 3)
    print(f"Quantizing {input_path.name} ({size_gb:.1f} GiB) → {quant}...", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    env = _llama_cpp_lib_env(quantize_bin)
    try:
        subprocess.run(
            [quantize_bin, str(input_path), str(output_path), quant],
            check=True,
            timeout=7200,
            env=env,
        )
    except FileNotFoundError:
        print_actionable_error(
            f"Could not execute: {quantize_bin}",
            next_steps=["Rebuild llama.cpp: ollama-forge setup-llama-cpp --update"],
        )
        return 1
    except subprocess.TimeoutExpired:
        print_actionable_error(
            "Quantization timed out after 2 hours",
            next_steps=["Try a smaller model or a faster quant type (e.g. Q4_0)"],
        )
        return 1
    except subprocess.CalledProcessError as e:
        print_actionable_error(
            "Quantization failed",
            cause=str(e),
            next_steps=[
                "Ensure the input is a valid GGUF file",
                "Try a different quant type: Q4_K_M, Q4_0, Q8_0, Q3_K_M",
                "Rebuild llama.cpp: ollama-forge setup-llama-cpp --update",
            ],
        )
        return 1

    out_size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"\nDone: {output_path} ({out_size_gb:.1f} GiB)", file=sys.stderr)
    print(f"Serve with: ollama-forge serve {output_path}", file=sys.stderr)
    print(output_path)
    return 0


# ---------------------------------------------------------------------------
# serve – spin up llama-server with a GGUF model
# ---------------------------------------------------------------------------

def _llama_cpp_lib_env(server_bin: str) -> dict[str, str]:
    """Build environment dict so llama-server can find its shared libraries."""
    env = dict(os.environ)
    bin_dir = str(Path(server_bin).resolve().parent)
    if sys.platform == "darwin":
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
    elif sys.platform == "linux":
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
    return env


def _wait_for_server(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll GET ``url`` until it responds 200 or *timeout* seconds elapse."""
    import time
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (OSError, urllib.error.URLError, ValueError):
            pass
        time.sleep(interval)
    return False


def _resolve_gguf_from_hf_cache(repo_id: str) -> Path | None:
    """Search the HF cache for a GGUF file belonging to *repo_id*. Returns the best match or None."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None
    for repo in cache_info.repos:
        if repo.repo_id != repo_id:
            continue
        gguf_files: list[Path] = []
        for rev in repo.revisions:
            for f in rev.files:
                if f.file_name.endswith(".gguf"):
                    gguf_files.append(f.file_path)
        if not gguf_files:
            return None
        # Prefer a quantised file over the raw bf16/f16 one
        from ollama_forge.hf_fetch import pick_one_gguf
        names = [p.name for p in gguf_files]
        best = pick_one_gguf(names)
        return next(p for p in gguf_files if p.name == best)
    return None


def _hf_cache_has_repo(repo_id: str) -> bool:
    """Return True if the HF cache contains any snapshot for *repo_id*."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
    except Exception:
        return False
    return any(r.repo_id == repo_id for r in cache_info.repos)


def _cmd_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Start llama-server to serve a GGUF model via an OpenAI-compatible API."""
    model_arg = args.model
    gguf = Path(model_arg)

    # If not a local file, try resolving from caches by repo ID
    if not gguf.is_file() and "/" in model_arg:
        # 1. Check ollama-forge GGUF cache (converted models)
        forge_cached = _resolve_gguf_from_forge_cache(model_arg)
        if forge_cached and forge_cached.is_file():
            print(f"Resolved from cache: {forge_cached}", file=sys.stderr)
            gguf = forge_cached
        # 2. Check HF cache (downloaded GGUFs)
        if not gguf.is_file():
            cached = _resolve_gguf_from_hf_cache(model_arg)
            if cached and cached.is_file():
                print(f"Resolved from HF cache: {cached}", file=sys.stderr)
                gguf = cached
        if not gguf.is_file() and _hf_cache_has_repo(model_arg):
            print_actionable_error(
                f"Repo {model_arg} is in the HF cache but has no GGUF files",
                next_steps=[
                    f"Convert to GGUF first: ollama-forge import {model_arg} --name my-model",
                    "Or fetch a GGUF repo: ollama-forge fetch <GGUF_REPO> --name my-model",
                ],
            )
            return 1

    if not gguf.is_file():
        print_actionable_error(
            f"GGUF file not found: {model_arg}",
            next_steps=[
                "Provide a valid path to a .gguf file",
                "Or use a HF repo ID that has GGUF files in the cache",
                "Download one with: ollama-forge fetch <HF_REPO>",
            ],
        )
        return 1

    llama_cpp_dir = _resolve_llama_cpp_dir_from_arg(args)
    server_bin = _which_llama_server(llama_cpp_dir)
    if not server_bin:
        print_actionable_error(
            "llama-server not found",
            next_steps=[
                "Run: ollama-forge setup-llama-cpp (builds llama-server)",
                "Or install llama.cpp and add its bin dir to PATH",
                "Or pass --llama-cpp-dir <path-to-llama.cpp-clone>",
            ],
        )
        return 1

    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = getattr(args, "port", 11434) or 11434
    ctx_size = getattr(args, "ctx_size", None)
    n_gpu_layers = getattr(args, "n_gpu_layers", None)
    threads = getattr(args, "threads", None)
    parallel = getattr(args, "parallel", None)
    api_key = getattr(args, "api_key", None)
    extra_args = getattr(args, "server_args", None) or []

    cmd: list[str] = [
        server_bin,
        "-m", str(gguf.resolve()),
        "--host", host,
        "--port", str(port),
    ]
    if ctx_size is not None:
        cmd += ["-c", str(ctx_size)]
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if parallel is not None:
        cmd += ["-np", str(parallel)]
    if api_key:
        cmd += ["--api-key", api_key]
    cmd += extra_args

    env = _llama_cpp_lib_env(server_bin)
    base_url = f"http://{host}:{port}"

    print(f"Starting llama-server: {shlex.join(cmd)}", file=sys.stderr)
    print(f"Endpoint: {base_url}/v1/chat/completions", file=sys.stderr)
    if api_key:
        print(f"API key : {api_key}", file=sys.stderr)

    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        print_actionable_error(
            f"Could not execute: {server_bin}",
            next_steps=["Rebuild llama.cpp: ollama-forge setup-llama-cpp --update"],
        )
        return 1

    health_url = f"{base_url}/health"
    timeout = getattr(args, "timeout", 60) or 60
    print(f"Waiting for server to be ready (up to {timeout}s)...", file=sys.stderr)
    if _wait_for_server(health_url, timeout=timeout):
        print(f"\nServer ready at {base_url}", file=sys.stderr)
        print(f"  Chat:       {base_url}/v1/chat/completions", file=sys.stderr)
        print(f"  Completions:{base_url}/v1/completions", file=sys.stderr)
        print(f"  Health:     {health_url}", file=sys.stderr)
        print(f"\nConnect:  ollama-forge chat --base-url {base_url}", file=sys.stderr)
        print("Press Ctrl+C to stop the server.", file=sys.stderr)
    else:
        print(
            f"\nWarning: server did not respond at {health_url} within {timeout}s. "
            "It may still be loading the model — check its output above.",
            file=sys.stderr,
        )

    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down llama-server...", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


# ---------------------------------------------------------------------------
# chat – interactive chat against a running llama-server (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _cmd_chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Interactive chat session with a running llama-server (or any OpenAI-compatible endpoint)."""
    import urllib.request as _urlreq

    base_url = (getattr(args, "base_url", None) or "http://127.0.0.1:11434").rstrip("/")
    model = getattr(args, "model", None) or ""
    system = getattr(args, "system", None)
    api_key = getattr(args, "api_key", None)
    temperature = getattr(args, "temperature", None)

    # Quick health check
    health = f"{base_url}/health"
    try:
        req = _urlreq.Request(health, method="GET")
        with _urlreq.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"Warning: server at {base_url} returned {resp.status}", file=sys.stderr)
    except (OSError, ValueError):
        print(f"Warning: cannot reach {base_url}/health — is the server running?", file=sys.stderr)
        print("  Start with: ollama-forge serve <model.gguf>", file=sys.stderr)

    url = f"{base_url}/v1/chat/completions"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(f"Connected to {base_url}  (type 'quit' or Ctrl+C to exit)\n", file=sys.stderr)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
                break
            if user_input.lower() in ("/clear", "/reset"):
                messages.clear()
                if system:
                    messages.append({"role": "system", "content": system})
                print("(conversation cleared)", file=sys.stderr)
                continue

            messages.append({"role": "user", "content": user_input})

            payload: dict[str, Any] = {
                "messages": messages,
                "stream": True,
            }
            if model:
                payload["model"] = model
            if temperature is not None:
                payload["temperature"] = temperature

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = json.dumps(payload).encode("utf-8")
            req = _urlreq.Request(url, data=body, headers=headers, method="POST")

            try:
                with _urlreq.urlopen(req, timeout=300) as resp:
                    print("Assistant: ", end="", flush=True)
                    assistant_text, finish_reason = stream_openai_chat_sse(resp)
                    print()  # newline after streamed response
                    if finish_reason and finish_reason != "stop":
                        print(f"(response stopped: {finish_reason})", file=sys.stderr)
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
                # Remove the failed user message so conversation stays consistent
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                continue

            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    except KeyboardInterrupt:
        print("\n", file=sys.stderr)

    print("Bye!", file=sys.stderr)
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


def _load_env() -> None:
    """Load .env from ~/.env then cwd. Never override existing env (e.g. export in shell)."""
    load_dotenv(Path.home() / ".env")
    load_dotenv(override=False)  # do not overwrite shell exports


def _add_plan_args(subparsers) -> "argparse.ArgumentParser":
    """Register the 'plan' subcommand and all its subparsers."""
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

    p_plan_continue = plan_sub.add_parser(
        "continue",
        help="Show or run the last saved plan (save with e.g. plan quickstart --json)",
    )
    p_plan_continue.add_argument(
        "--execute",
        action="store_true",
        help="Run the planned command(s) instead of only showing them",
    )
    p_plan_continue.add_argument(
        "--json",
        action="store_true",
        help="Output saved plan as JSON only",
    )
    p_plan_continue.set_defaults(handler=_cmd_plan_continue)
    return p_plan


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
    p_plan = _add_plan_args(subparsers)

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
    p_quantize = subparsers.add_parser(
        "quantize",
        help="Quantize a GGUF model file (e.g. bf16 → Q4_K_M) to reduce size and memory usage",
    )
    p_quantize.add_argument(
        "input",
        help="Path to the .gguf file (or a cached repo ID like org/model)",
    )
    p_quantize.add_argument(
        "--quant", "-q",
        default="Q4_K_M",
        help="Quantization type (default: Q4_K_M). Common: Q4_K_M, Q4_0, Q3_K_M, Q5_K_M, Q8_0",
    )
    p_quantize.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: <input_stem>-<quant>.gguf in the same directory)",
    )
    p_quantize.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Path to llama.cpp clone (for finding the quantize binary)",
    )
    p_quantize.set_defaults(handler=_cmd_quantize)

    # serve (spin up llama-server with a GGUF model)
    p_serve = subparsers.add_parser(
        "serve",
        help="Start llama-server to serve a GGUF model via an OpenAI-compatible API",
    )
    p_serve.add_argument(
        "model",
        help="Path to a .gguf model file",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port to listen on (default: 11434)",
    )
    p_serve.add_argument(
        "-c", "--ctx-size",
        type=int,
        default=None,
        help="Context size in tokens (default: model default)",
    )
    p_serve.add_argument(
        "-ngl", "--n-gpu-layers",
        type=int,
        default=None,
        help="Number of layers to offload to GPU (-1 = all)",
    )
    p_serve.add_argument(
        "-t", "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto)",
    )
    p_serve.add_argument(
        "-np", "--parallel",
        type=int,
        default=None,
        help="Number of parallel request slots (default: auto)",
    )
    p_serve.add_argument(
        "--api-key",
        default=None,
        help="Require this API key for all requests",
    )
    p_serve.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Path to llama.cpp clone (for finding llama-server binary)",
    )
    p_serve.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Seconds to wait for server to become ready (default: 60)",
    )
    p_serve.add_argument(
        "server_args",
        nargs="*",
        metavar="-- ...",
        help="Extra arguments passed through to llama-server (place after --)",
    )
    p_serve.set_defaults(handler=_cmd_serve)

    # chat (interactive chat with a running llama-server or OpenAI-compatible endpoint)
    p_chat = subparsers.add_parser(
        "chat",
        help="Interactive chat with a running llama-server (or any OpenAI-compatible endpoint)",
    )
    p_chat.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Server base URL (default: http://127.0.0.1:11434)",
    )
    p_chat.add_argument(
        "--model",
        default=None,
        help="Model name to send in requests (optional; llama-server ignores this)",
    )
    p_chat.add_argument(
        "--system",
        default=None,
        help="System message for the conversation",
    )
    p_chat.add_argument(
        "--api-key",
        default=None,
        help="API key (Bearer token) if the server requires authentication",
    )
    p_chat.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (e.g. 0.7)",
    )
    p_chat.set_defaults(handler=_cmd_chat)

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
    p_convert.add_argument(
        "--adapter",
        help="Path to adapter to stack on the GGUF base (directory or .bin/.gguf); adds ADAPTER to Modelfile",
    )
    p_convert.add_argument("--out-modelfile", help="Also write the Modelfile to this path")
    p_convert.set_defaults(handler=_cmd_convert)

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
