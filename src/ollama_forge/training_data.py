"""Training data format validation (JSONL instruction format)."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = {"instruction", "output"}
OPTIONAL_KEYS = {"input"}


def validate_line(line: str, line_no: int) -> list[str]:
    """Validate a single JSONL line. Returns list of error messages (empty if valid)."""
    errors: list[str] = []
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        return [f"Line {line_no}: invalid JSON — {e}"]
    if not isinstance(obj, dict):
        errors.append(f"Line {line_no}: expected JSON object")
        return errors
    keys = set(obj.keys())
    missing = REQUIRED_KEYS - keys
    if missing:
        errors.append(f"Line {line_no}: missing required keys: {sorted(missing)}")
    extra = keys - (REQUIRED_KEYS | OPTIONAL_KEYS)
    if extra:
        errors.append(f"Line {line_no}: unknown keys: {sorted(extra)}")
    for key in REQUIRED_KEYS:
        if key in obj and not isinstance(obj[key], str):
            errors.append(f"Line {line_no}: '{key}' must be a string")
    if "input" in obj and not isinstance(obj["input"], str):
        errors.append(f"Line {line_no}: 'input' must be a string")
    return errors


def validate_training_data(path: str | Path) -> tuple[bool, list[str], int]:
    """
    Validate a JSONL file for the instruction format.
    Returns (all_valid, list of error messages, number of valid lines).
    """
    path = Path(path)
    if not path.is_file():
        return False, [f"File not found: {path}"], 0
    errors: list[str] = []
    valid_count = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            line_errors = validate_line(line, i)
            if line_errors:
                errors.extend(line_errors)
            else:
                valid_count += 1
    return len(errors) == 0, errors, valid_count


def collect_jsonl_paths(paths: list[str | Path]) -> list[Path]:
    """Expand paths to a list of .jsonl files. Directories are expanded to all .jsonl inside."""
    result: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() == ".jsonl":
                result.append(path.resolve())
        elif path.is_dir():
            for f in sorted(path.glob("*.jsonl")):
                result.append(f.resolve())
    return result


def validate_training_data_paths(paths: list[Path]) -> tuple[bool, list[str], int]:
    """Validate multiple JSONL files. Returns (all_valid, errors, total_valid_lines)."""
    all_errors: list[str] = []
    total_valid = 0
    for path in paths:
        ok, errs, count = validate_training_data(path)
        all_errors.extend(errs)
        total_valid += count
    return len(all_errors) == 0, all_errors, total_valid


def convert_jsonl_to_plain_text(
    paths: list[Path],
    output: Path,
    *,
    format_name: str = "llama.cpp",
) -> None:
    """
    Convert instruction/input/output JSONL to plain text for trainers.
    llama.cpp: Alpaca-style blocks with ### Instruction / ### Input / ### Response.
    """
    output = Path(output)
    lines_out: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                inst = obj.get("instruction", "")
                inp = obj.get("input", "")
                out_text = obj.get("output", "")
                if not inst or not out_text:
                    continue
                if format_name == "llama.cpp":
                    # One block per example; llama.cpp can use --sample-start "### Instruction"
                    block = (
                        "### Instruction:\n"
                        f"{inst}\n\n"
                        "### Input:\n"
                        f"{inp if inp else '(none)'}\n\n"
                        "### Response:\n"
                        f"{out_text}\n"
                    )
                    lines_out.append(block)
    output.write_text("\n".join(lines_out), encoding="utf-8")
