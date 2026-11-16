"""Convert command group: HF checkpoint → GGUF → Ollama."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ollama_forge.log import get_logger
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
from ollama_forge.run_helpers import (
    _llama_cpp_lib_env,
    _which_quantize_full,
    print_actionable_error,
    run_ollama_create,
    run_ollama_show_modelfile,
)

log = get_logger()


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
    import re

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


def register(subparsers) -> argparse.ArgumentParser:
    """Register the convert subcommand; returns p_convert."""
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
    return p_convert
