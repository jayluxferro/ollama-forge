"""Quantize command group: re-quantize a GGUF file."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ollama_forge.run_helpers import (
    _llama_cpp_lib_env,
    _resolve_gguf_from_forge_cache,
    _resolve_gguf_from_hf_cache,
    _resolve_llama_cpp_dir_from_arg,
    _which_quantize_full,
    print_actionable_error,
)


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


def register(subparsers) -> argparse.ArgumentParser:
    """Register the quantize subcommand; returns p_quantize."""
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
    return p_quantize
