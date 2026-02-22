"""Shared helpers for CLI: ollama checks, subprocess, temp files, JSONL resolution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ollama_forge.http_util import normalize_base_url
from ollama_forge.log import get_logger
from ollama_forge.training_data import collect_jsonl_paths

log = get_logger()

OLLAMA_MISSING_MSG = "Error: ollama not found. Install Ollama and ensure it is on PATH."


def print_actionable_error(
    summary: str,
    *,
    cause: str | None = None,
    next_steps: list[str] | None = None,
) -> None:
    """Print a structured error with optional cause and next-step guidance."""
    print(f"Error: {summary}", file=sys.stderr)
    if cause:
        print(f"Cause: {cause}", file=sys.stderr)
    if next_steps:
        print("Next:", file=sys.stderr)
        for step in next_steps:
            print(f"  - {step}", file=sys.stderr)


def require_ollama() -> int | None:
    """Return exit code to use if ollama is missing; otherwise None (caller proceeds)."""
    if shutil.which("ollama"):
        return None
    print_actionable_error(
        "ollama not found on PATH",
        next_steps=[
            "Install Ollama from https://ollama.com",
            "Run: ollama-forge check",
        ],
    )
    return 1


def ping_ollama(base_url: str, timeout: float = 5.0) -> bool:
    """
    Check whether an Ollama (or Ollama-compatible) server is reachable at base_url.
    Uses GET /api/tags. Returns True if the server responds successfully, False otherwise.
    """
    url = normalize_base_url(base_url).rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 204)
    except (OSError, ValueError):
        return False


# Timeouts for subprocess calls (seconds)
OLLAMA_SHOW_TIMEOUT = 60
OLLAMA_CREATE_TIMEOUT = 300


def run_ollama_show_modelfile(model: str) -> str | None:
    """Run `ollama show --modelfile <model>` and return stdout, or None on failure."""
    if not shutil.which("ollama"):
        return None
    try:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True,
            text=True,
            check=True,
            timeout=OLLAMA_SHOW_TIMEOUT,
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _run_ollama_create_from_path(name: str, path: Path) -> int:
    """Run `ollama create -f <path>` and return exit code. Error handling only."""
    try:
        subprocess.run(
            ["ollama", "create", name, "-f", str(path)],
            check=True,
            timeout=OLLAMA_CREATE_TIMEOUT,
        )
        log.info("Created model %r. Run with: ollama run %s", name, name)
        return 0
    except FileNotFoundError:
        print_actionable_error(
            "ollama not found on PATH",
            next_steps=[
                "Install Ollama from https://ollama.com",
                "Run: ollama-forge check",
            ],
        )
        return 1
    except subprocess.TimeoutExpired:
        print_actionable_error(
            f"ollama create timed out after {OLLAMA_CREATE_TIMEOUT}s",
            next_steps=["Try again or use a smaller model"],
        )
        return 1
    except subprocess.CalledProcessError as e:
        return e.returncode


def run_ollama_create(
    name: str,
    modelfile_content: str,
    out_path: str | Path | None = None,
) -> int:
    """Write modelfile (to out_path or temp), run `ollama create`, cleanup. Returns exit code."""
    if out_path is not None:
        path = Path(out_path)
        path.write_text(modelfile_content, encoding="utf-8")
        log.info("Wrote Modelfile to %s", path)
        return _run_ollama_create_from_path(name, path)
    with temporary_text_file(modelfile_content, suffix=".Modelfile", prefix="") as path:
        return _run_ollama_create_from_path(name, path)


@contextmanager
def temporary_text_file(
    content: str,
    suffix: str = "",
    prefix: str = "",
    encoding: str = "utf-8",
) -> Iterator[Path]:
    """Create a temp file with content; yield path; delete on exit."""
    fd, raw_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        yield Path(raw_path)
    finally:
        Path(raw_path).unlink(missing_ok=True)


def write_temp_text_file(
    content: str,
    suffix: str = ".txt",
    prefix: str = "",
    encoding: str = "utf-8",
) -> Path:
    """Create a temp file with content and return its path. Caller must delete when done."""
    fd, raw_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, text=True)
    with os.fdopen(fd, "w", encoding=encoding) as f:
        f.write(content)
    return Path(raw_path)


def run_cmd(
    cmd: list[str],
    not_found_message: str,
    process_error_message: str = "Error: command failed: {e}",
    *,
    cwd: str | Path | None = None,
    not_found_next_steps: list[str] | None = None,
    process_error_next_steps: list[str] | None = None,
) -> int:
    """Run command; on FileNotFoundError print not_found_message and return 1;
    on CalledProcessError print process_error_message and return code."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return 0
    except FileNotFoundError:
        print_actionable_error(
            not_found_message.replace("Error: ", ""),
            next_steps=not_found_next_steps,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print_actionable_error(
            process_error_message.format(e=e).replace("Error: ", ""),
            next_steps=process_error_next_steps,
        )
        return e.returncode


def get_jsonl_paths_or_exit(
    data_arg: list[str | Path] | str | Path,
    error_msg: str = "Error: no .jsonl files found. Give one or more files or a directory.",
    next_steps: list[str] | None = None,
) -> list[Path] | None:
    """Resolve data_arg to .jsonl paths; if none, print error and return None (caller returns 1)."""
    paths_input = data_arg if isinstance(data_arg, list) else [data_arg]
    paths = collect_jsonl_paths(paths_input)
    if not paths:
        print_actionable_error(
            error_msg.replace("Error: ", ""),
            next_steps=next_steps,
        )
        return None
    return paths


def check_item(name: str, ok: bool, missing_hint: str) -> bool:
    """Print one check line (OK or MISSING — hint). Returns ok."""
    if ok:
        print(f"{name}: OK")
    else:
        print(f"{name}: MISSING — {missing_hint}")
    return ok
