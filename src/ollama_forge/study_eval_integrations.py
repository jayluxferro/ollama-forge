"""External evaluation integrations for study workflows."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LMEvalCommand:
    command: list[str]
    output_path: str | None = None


def build_lm_eval_command(
    *,
    model: str,
    tasks: list[str],
    model_args: str = "",
    output_path: str | Path | None = None,
    device: str | None = None,
    batch_size: str | None = None,
    limit: int | None = None,
) -> LMEvalCommand:
    command = ["lm_eval", "--model", model, "--tasks", ",".join(tasks)]
    if model_args:
        command.extend(["--model_args", model_args])
    if output_path:
        command.extend(["--output_path", str(output_path)])
    if device:
        command.extend(["--device", device])
    if batch_size:
        command.extend(["--batch_size", batch_size])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return LMEvalCommand(command=command, output_path=str(output_path) if output_path else None)


def run_lm_eval(command: LMEvalCommand) -> int:
    if shutil.which(command.command[0]) is None:
        raise FileNotFoundError("lm_eval executable not found")
    completed = subprocess.run(command.command, check=False)
    return completed.returncode


def save_lm_eval_plan(command: LMEvalCommand, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"command": command.command, "output_path": command.output_path}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
