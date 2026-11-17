"""CLI entrypoint for ollama-forge."""

import argparse
import contextlib
import os
import sys
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
from ollama_forge.commands.api import register as _register_api_cmds
from ollama_forge.commands.auto import register as _register_auto_cmds
from ollama_forge.commands.cache import register as _register_cache_cmds
from ollama_forge.commands.fetch import register as _register_fetch_cmds
from ollama_forge.commands.model import register as _register_model_cmds
from ollama_forge.commands.plan import register as _register_plan_cmds
from ollama_forge.commands.quantize import register as _register_quantize_cmds
from ollama_forge.commands.security_eval import register as _register_security_eval_cmds
from ollama_forge.commands.serve import register as _register_serve_cmds
from ollama_forge.commands.setup import register as _register_setup_cmds
from ollama_forge.commands.study import register as _register_study_cmds
from ollama_forge.commands.train import register as _register_train_cmds
from ollama_forge.commands.turboquant import register as _register_turboquant_cmds
from ollama_forge.commands.vlm import register as _register_vlm_cmds
from ollama_forge.log import get_logger, set_verbose

log = get_logger()



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

    _register_setup_cmds(subparsers)

    # plan (global dry-run wrappers for key flows)
    p_plan = _register_plan_cmds(subparsers)

    _register_fetch_cmds(subparsers)

    # auto (detect source type and route)
    _register_auto_cmds(subparsers)


    # quantize (requantize a GGUF file)
    _register_quantize_cmds(subparsers)

    # serve / chat / api
    p_serve, p_chat = _register_serve_cmds(subparsers)
    _register_api_cmds(subparsers)

    _register_model_cmds(subparsers)
    from ollama_forge.commands.convert import register as _register_convert_cmds
    _register_convert_cmds(subparsers)

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
