"""Auto command group: auto-route source to the right workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ollama_forge.run_helpers import print_actionable_error


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
            from ollama_forge.commands.plan import _save_last_plan  # noqa: PLC0415
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
        from ollama_forge.commands.model import _cmd_build  # noqa: PLC0415
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
        from ollama_forge.commands.convert import _cmd_convert  # noqa: PLC0415
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
            from ollama_forge.commands.model import _cmd_import  # noqa: PLC0415
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
            from ollama_forge.commands.fetch import _cmd_fetch_adapter  # noqa: PLC0415
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
        from ollama_forge.commands.fetch import _cmd_fetch  # noqa: PLC0415
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
    from ollama_forge.commands.model import _cmd_create_from_base  # noqa: PLC0415
    return _cmd_create_from_base(parser, fake)


def register(subparsers) -> argparse.ArgumentParser:
    """Register the auto subcommand; returns p_auto."""
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
    return p_auto
