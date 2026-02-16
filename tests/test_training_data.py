"""Training data validation tests."""

import tempfile
from pathlib import Path

from ollama_forge.training_data import (
    collect_jsonl_paths,
    convert_jsonl_to_plain_text,
    validate_line,
    validate_training_data,
    validate_training_data_paths,
)


def test_validate_line_valid() -> None:
    """Valid line has no errors."""
    line = '{"instruction": "Say hi.", "output": "Hi!"}'
    assert validate_line(line, 1) == []


def test_validate_line_with_input() -> None:
    """Valid line with input."""
    line = '{"instruction": "Sum.", "input": "1+1", "output": "2"}'
    assert validate_line(line, 1) == []


def test_validate_line_missing_output() -> None:
    """Missing output returns error."""
    line = '{"instruction": "Say hi."}'
    assert any("output" in e for e in validate_line(line, 1))


def test_validate_line_invalid_json() -> None:
    """Invalid JSON returns error."""
    assert len(validate_line("not json", 1)) >= 1


def test_validate_training_data_file_not_found() -> None:
    """Missing file returns error."""
    ok, errors, count = validate_training_data("/nonexistent/file.jsonl")
    assert not ok
    assert count == 0
    assert any("not found" in e.lower() for e in errors)


def test_validate_training_data_valid_file() -> None:
    """Valid JSONL file passes."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"instruction": "A", "output": "B"}\n')
        f.write('{"instruction": "C", "input": "", "output": "D"}\n')
        path = f.name
    try:
        ok, errors, count = validate_training_data(path)
        assert ok
        assert count == 2
        assert not errors
    finally:
        Path(path).unlink(missing_ok=True)


def test_validate_training_data_invalid_line() -> None:
    """Invalid line is reported."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"instruction": "A", "output": "B"}\n')
        f.write('{"wrong": "key"}\n')
        path = f.name
    try:
        ok, errors, count = validate_training_data(path)
        assert not ok
        assert count == 1
        assert any("missing" in e.lower() or "required" in e.lower() for e in errors)
    finally:
        Path(path).unlink(missing_ok=True)


def test_collect_jsonl_paths_single_file() -> None:
    """Single file returns that file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("{}")
        path = f.name
    try:
        out = collect_jsonl_paths([path])
        assert len(out) == 1
        assert out[0].suffix.lower() == ".jsonl"
    finally:
        Path(path).unlink(missing_ok=True)


def test_collect_jsonl_paths_directory() -> None:
    """Directory expands to .jsonl files."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.jsonl").write_text("{}")
        (Path(d) / "b.jsonl").write_text("{}")
        (Path(d) / "c.txt").write_text("")
        out = collect_jsonl_paths([d])
        assert len(out) == 2
        names = {p.name for p in out}
        assert names == {"a.jsonl", "b.jsonl"}


def test_validate_training_data_paths() -> None:
    """validate_training_data_paths aggregates multiple files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f1:
        f1.write('{"instruction": "X", "output": "Y"}\n')
        p1 = f1.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f2:
        f2.write('{"instruction": "A", "output": "B"}\n')
        p2 = f2.name
    try:
        ok, errs, count = validate_training_data_paths([Path(p1), Path(p2)])
        assert ok
        assert count == 2
        assert not errs
    finally:
        Path(p1).unlink(missing_ok=True)
        Path(p2).unlink(missing_ok=True)


def test_convert_jsonl_to_plain_text() -> None:
    """convert_jsonl_to_plain_text writes Alpaca-style blocks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"instruction": "Hi?", "output": "Hello!"}\n')
        p_in = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        p_out = f.name
    try:
        convert_jsonl_to_plain_text([Path(p_in)], Path(p_out), format_name="llama.cpp")
        text = Path(p_out).read_text()
        assert "### Instruction:" in text
        assert "Hi?" in text
        assert "### Response:" in text
        assert "Hello!" in text
    finally:
        Path(p_in).unlink(missing_ok=True)
        Path(p_out).unlink(missing_ok=True)
