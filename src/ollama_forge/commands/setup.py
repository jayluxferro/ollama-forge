"""Setup, check, and doctor command group: check, doctor, setup-llama-cpp."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ollama_forge.commands.fetch import _hf_token_available
from ollama_forge.log import get_logger
from ollama_forge.run_helpers import (
    _which_quantize,
    check_item,
    run_cmd,
)

log = get_logger()


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
    import shutil as _shutil

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
                _shutil.rmtree(build_dir)
                build_dir.mkdir()
        except OSError:
            pass
    # Clean cmake artifacts that may have leaked into the source directory
    # (e.g. from a previous failed build or misconfigured cmake run).
    for stale_name in ["CMakeCache.txt", "CMakeFiles", "Makefile", "cmake_install.cmake"]:
        stale_path = target_dir / stale_name
        if stale_path.is_file():
            stale_path.unlink()
        elif stale_path.is_dir():
            _shutil.rmtree(stale_path)

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


def register(subparsers) -> None:
    """Register setup/diagnostic commands: check, doctor, setup-llama-cpp."""
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
