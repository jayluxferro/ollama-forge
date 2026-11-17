"""forge ui — unified web UI (Streamlit)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ollama_forge.run_helpers import print_actionable_error


def _cmd_ui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    import subprocess

    app_path = Path(__file__).resolve().parent.parent / "forge_ui.py"
    if not app_path.exists():
        print_actionable_error(
            f"forge UI app not found at {app_path}",
            next_steps=["Run: uv sync"],
        )
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    if args.host:
        cmd += ["--server.address", args.host]

    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print_actionable_error(
            "Streamlit not found",
            next_steps=["Run: uv sync", "Then: ollama-forge ui"],
        )
        return 1
    return 0


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "ui",
        help="Launch the unified forge web UI (Chat, Models, Train)",
    )
    p.add_argument("--host", default=None, help="Bind address (default: localhost)")
    p.add_argument("--port", type=int, default=None, help="Bind port (default: 8501)")
    p.set_defaults(handler=_cmd_ui)
    return p
