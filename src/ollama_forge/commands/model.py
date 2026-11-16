"""Model management command group: import, create-from-base, refresh-template, build, validate-recipe."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ollama_forge.hf_fetch import download_adapter
from ollama_forge.log import get_logger
from ollama_forge.modelfile import build_modelfile, merge_modelfile_with_reference_template
from ollama_forge.recipe import load_recipe
from ollama_forge.run_helpers import (
    print_actionable_error,
    require_ollama,
    run_ollama_create,
    run_ollama_show_modelfile,
)

log = get_logger()

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
        from ollama_forge.commands.fetch import _cmd_fetch
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


def register(subparsers) -> None:
    """Register model management commands: create-from-base, refresh-template, import, build, validate-recipe."""
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

    # refresh-template
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
