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
from ollama_forge.commands.fetch import (
    _cmd_fetch,
    _hf_token_available,
    _verify_adapter_and_base,
)
from ollama_forge.commands.fetch import (
    register as _register_fetch_cmds,
)
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
)
from ollama_forge.log import get_logger, set_verbose
from ollama_forge.modelfile import (
    build_modelfile,
    merge_modelfile_with_reference_template,
)
from ollama_forge.recipe import load_recipe
from ollama_forge.run_helpers import (
    _which_quantize,
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

    _register_fetch_cmds(subparsers)

    # auto (detect source type and route)
    _register_auto_cmds(subparsers)

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
    _register_quantize_cmds(subparsers)

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
    _register_convert_cmds(subparsers)

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
