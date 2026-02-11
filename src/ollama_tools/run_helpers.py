"""Shared helpers for CLI: ollama checks, subprocess, temp files, JSONL resolution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ollama_tools.training_data import collect_jsonl_paths

OLLAMA_MISSING_MSG = (
    "Error: ollama not found. Install Ollama and ensure it is on PATH."
)


def require_ollama() -> int | None:
    """Return exit code to use if ollama is missing; otherwise None (caller proceeds)."""
    if shutil.which("ollama"):
        return None
    print(OLLAMA_MISSING_MSG, file=sys.stderr)
    return 1


def run_ollama_create(
    name: str,
    modelfile_content: str,
    out_path: str | Path | None = None,
) -> int:
    """Write modelfile (to out_path or temp), run `ollama create`, cleanup. Returns exit code."""
    if out_path is not None:
        path = Path(out_path)
        path.write_text(modelfile_content, encoding="utf-8")
        print(f"Wrote Modelfile to {path}", file=sys.stderr)
        try:
            subprocess.run(
                ["ollama", "create", name, "-f", str(path)],
                check=True,
            )
            print(f"Created model {name!r}. Run with: ollama run {name}")
            return 0
        except FileNotFoundError:
            print(OLLAMA_MISSING_MSG, file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as e:
            return e.returncode
    with temporary_text_file(
        modelfile_content, suffix=".Modelfile", prefix=""
    ) as path:
        try:
            subprocess.run(
                ["ollama", "create", name, "-f", str(path)],
                check=True,
            )
            print(f"Created model {name!r}. Run with: ollama run {name}")
            return 0
        except FileNotFoundError:
            print(OLLAMA_MISSING_MSG, file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as e:
            return e.returncode


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
) -> int:
    """Run command; on FileNotFoundError print not_found_message and return 1;
    on CalledProcessError print process_error_message and return code."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return 0
    except FileNotFoundError:
        print(not_found_message, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(process_error_message.format(e=e), file=sys.stderr)
        return e.returncode


def get_jsonl_paths_or_exit(
    data_arg: list[str | Path] | str | Path,
    error_msg: str = "Error: no .jsonl files found. Give one or more files or a directory.",
) -> list[Path] | None:
    """Resolve data_arg to .jsonl paths; if none, print error and return None (caller returns 1)."""
    paths_input = data_arg if isinstance(data_arg, list) else [data_arg]
    paths = collect_jsonl_paths(paths_input)
    if not paths:
        print(error_msg, file=sys.stderr)
        return None
    return paths


def check_item(name: str, ok: bool, missing_hint: str) -> bool:
    """Print one check line (OK or MISSING — hint). Returns ok."""
    if ok:
        print(f"{name}: OK")
    else:
        print(f"{name}: MISSING — {missing_hint}")
    return ok
