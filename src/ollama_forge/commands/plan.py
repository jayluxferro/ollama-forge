"""Plan command group: guided workflow plans (dry-run wrappers for key flows)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ollama_forge.log import get_logger
from ollama_forge.run_helpers import print_actionable_error

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


def _cmd_plan_quickstart(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Plan wrapper for quickstart."""
    from ollama_forge.commands.fetch import _cmd_quickstart

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
    from ollama_forge.commands.auto import _cmd_auto

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
    from ollama_forge.cli import _cmd_doctor  # lazy: avoids circular import at module load

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


def _add_plan_args(subparsers) -> argparse.ArgumentParser:
    """Register the 'plan' subcommand and all its subparsers."""
    from ollama_forge.commands.fetch import _QUICKSTART_TASK_SYSTEMS

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


def register(subparsers) -> argparse.ArgumentParser:
    """Register the plan subcommand group; returns p_plan."""
    return _add_plan_args(subparsers)
