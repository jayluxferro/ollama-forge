"""Training data format validation (JSONL instruction format)."""

from __future__ import annotations

import json
from pathlib import Path

from ollama_forge.log import get_logger

log = get_logger()

REQUIRED_KEYS = {"instruction", "output"}
OPTIONAL_KEYS = {"input"}


def _message_content(msg: dict) -> str:
    """Extract text content from a message (role/content, with content as string or list of parts)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", p.get("content", "")) if isinstance(p, dict) else str(p) for p in content)
    return ""


def normalize_record(obj: dict) -> dict | None:
    """
    Normalize a JSONL record to Alpaca-style {instruction, input, output}.
    Accepts either Alpaca format (instruction, output) or messages format
    (messages: [{role, content}, ...], e.g. from TeichAI/datagen).
    Returns None if the record cannot be normalized.
    """
    if not isinstance(obj, dict):
        return None
    # Already Alpaca-style
    if "instruction" in obj and "output" in obj:
        inst = obj.get("instruction")
        out = obj.get("output")
        if isinstance(inst, str) and isinstance(out, str) and inst.strip() and out.strip():
            return {
                "instruction": inst,
                "input": obj.get("input") if isinstance(obj.get("input"), str) else "",
                "output": out,
            }
        return None
    # messages format (e.g. datagen / OpenRouter-style)
    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    system_parts: list[str] = []
    last_user = ""
    last_assistant = ""
    for m in messages:
        role = (m.get("role") or "").lower()
        content = _message_content(m)
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            last_user = content
        elif role == "assistant":
            last_assistant = content
    if not last_user.strip() or not last_assistant.strip():
        return None
    return {
        "instruction": last_user,
        "input": "\n".join(system_parts).strip() if system_parts else "",
        "output": last_assistant,
    }


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
    # Alpaca format
    if "instruction" in obj and "output" in obj:
        keys = set(obj.keys())
        extra = keys - (REQUIRED_KEYS | OPTIONAL_KEYS)
        if extra:
            errors.append(f"Line {line_no}: unknown keys: {sorted(extra)}")
        for key in REQUIRED_KEYS:
            if key in obj and not isinstance(obj[key], str):
                errors.append(f"Line {line_no}: '{key}' must be a string")
        if "input" in obj and not isinstance(obj["input"], str):
            errors.append(f"Line {line_no}: 'input' must be a string")
        return errors
    # messages format (e.g. datagen)
    if "messages" in obj:
        msgs = obj["messages"]
        if not isinstance(msgs, list):
            errors.append(f"Line {line_no}: 'messages' must be an array")
            return errors
        if normalize_record(obj) is None:
            errors.append(
                f"Line {line_no}: 'messages' must include at least one user and one assistant message with string content"  # noqa: E501
            )
        return errors
    errors.append(f"Line {line_no}: expected 'instruction' and 'output' (Alpaca) or 'messages' (e.g. datagen)")
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
) -> int:
    """
    Convert instruction/input/output JSONL to plain text for trainers.
    Accepts Alpaca format (instruction, output) or messages format (e.g. TeichAI/datagen).
    llama.cpp: Alpaca-style blocks with ### Instruction / ### Input / ### Response.
    Returns the number of samples written.
    """
    output = Path(output)
    lines_out: list[str] = []
    skipped = 0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                norm = normalize_record(obj) if isinstance(obj, dict) else None
                if norm is None:
                    skipped += 1
                    continue
                inst = norm["instruction"]
                inp = norm["input"]
                out_text = norm["output"]
                if not inst or not out_text:
                    continue
                if format_name == "llama.cpp":
                    block = (
                        "### Instruction:\n"
                        f"{inst}\n\n"
                        "### Input:\n"
                        f"{inp if inp else '(none)'}\n\n"
                        "### Response:\n"
                        f"{out_text}\n"
                    )
                    lines_out.append(block)
                elif format_name == "alpaca_plain":
                    block = f"{inst}\n{out_text}\n"
                    lines_out.append(block)
                else:
                    # fallback: same as llama.cpp
                    block = (
                        "### Instruction:\n"
                        f"{inst}\n\n"
                        "### Input:\n"
                        f"{inp if inp else '(none)'}\n\n"
                        "### Response:\n"
                        f"{out_text}\n"
                    )
                    lines_out.append(block)
    if skipped:
        log.warning(
            "convert_jsonl_to_plain_text: skipped %d record(s) with invalid JSON or unrecognised format.",
            skipped,
        )
    output.write_text("\n".join(lines_out), encoding="utf-8")
    return len(lines_out)


def convert_messages_to_alpaca_jsonl(path_in: Path, path_out: Path) -> int:
    """
    Read JSONL (Alpaca or messages format) and write Alpaca-style JSONL.
    Returns the number of lines written.
    """
    path_in = Path(path_in)
    path_out = Path(path_out)
    count = 0
    skipped = 0
    with path_in.open(encoding="utf-8") as fin, path_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            norm = normalize_record(obj) if isinstance(obj, dict) else None
            if norm is None:
                skipped += 1
                continue
            fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
            count += 1
    if skipped:
        log.warning(
            "convert_messages_to_alpaca_jsonl: skipped %d record(s) with invalid JSON or unrecognised format.",
            skipped,
        )
    return count
