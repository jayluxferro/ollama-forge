"""Training data validation tests."""

import json
import tempfile
from pathlib import Path

from ollama_forge.training_data import (
    collect_jsonl_paths,
    convert_jsonl_to_plain_text,
    convert_messages_to_alpaca_jsonl,
    normalize_record,
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
        assert any(
            "missing" in e.lower() or "required" in e.lower() or "expected" in e.lower()
            for e in errors
        )
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


def test_validate_line_messages_format() -> None:
    """Messages format (e.g. datagen) is valid if it has user + assistant."""
    line = '{"messages": [{"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello!"}]}'
    assert validate_line(line, 1) == []


def test_validate_line_messages_format_with_system() -> None:
    """Messages format with system message is valid."""
    line = '{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello!"}]}'  # noqa: E501
    assert validate_line(line, 1) == []


def test_normalize_record_messages_to_alpaca() -> None:
    """normalize_record converts messages format to Alpaca."""
    obj = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
        ]
    }
    out = normalize_record(obj)
    assert out is not None
    assert out["instruction"] == "What is 2+2?"
    assert out["output"] == "4."
    assert out["input"] == ""


def test_convert_jsonl_to_plain_text_messages_format() -> None:
    """convert_jsonl_to_plain_text accepts messages format (datagen)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"messages": [{"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello!"}]}\n')
        p_in = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        p_out = f.name
    try:
        convert_jsonl_to_plain_text([Path(p_in)], Path(p_out), format_name="llama.cpp")
        text = Path(p_out).read_text()
        assert "### Instruction:" in text
        assert "Hi?" in text
        assert "Hello!" in text
    finally:
        Path(p_in).unlink(missing_ok=True)
        Path(p_out).unlink(missing_ok=True)


def test_convert_messages_to_alpaca_jsonl() -> None:
    """convert_messages_to_alpaca_jsonl writes Alpaca JSONL."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]}\n')
        p_in = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        p_out = f.name
    try:
        n = convert_messages_to_alpaca_jsonl(Path(p_in), Path(p_out))
        assert n == 1
        lines = Path(p_out).read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["instruction"] == "Q?"
        assert rec["output"] == "A."
    finally:
        Path(p_in).unlink(missing_ok=True)
        Path(p_out).unlink(missing_ok=True)


def test_validate_line_extra_keys() -> None:
    """validate_line reports unknown keys in Alpaca format."""
    line = '{"instruction": "Hi", "output": "Bye", "extra_key": 1}'
    errors = validate_line(line, 1)
    assert any("unknown keys" in e for e in errors) or any("extra_key" in e for e in errors)


def test_validate_line_instruction_not_string() -> None:
    """validate_line reports when instruction is not a string."""
    line = '{"instruction": 123, "output": "x"}'
    errors = validate_line(line, 1)
    assert any("instruction" in e and "string" in e for e in errors)


def test_validate_line_messages_not_array() -> None:
    """validate_line reports when messages is not an array."""
    line = '{"messages": "not a list"}'
    errors = validate_line(line, 1)
    assert any("messages" in e and "array" in e for e in errors)


def test_validate_line_expected_instruction_or_messages() -> None:
    """validate_line reports when neither instruction+output nor valid messages."""
    line = '{"other": "key"}'
    errors = validate_line(line, 1)
    assert any("instruction" in e or "messages" in e for e in errors)


def test_normalize_record_messages_with_list_content() -> None:
    """normalize_record extracts text from message content as list of parts (e.g. multimodal)."""
    obj = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there"},
        ]
    }
    out = normalize_record(obj)
    assert out is not None
    assert out["instruction"] == "Hello"
    assert out["output"] == "Hi there"


def test_normalize_record_alpaca_with_input_non_string() -> None:
    """normalize_record uses empty string for input when not a string."""
    obj = {"instruction": "Q", "output": "A", "input": 42}
    out = normalize_record(obj)
    assert out is not None
    assert out.get("input") == ""


def test_normalize_record_not_dict_returns_none() -> None:
    """normalize_record returns None for non-dict."""
    assert normalize_record([]) is None
    assert normalize_record("x") is None


def test_validate_training_data_blank_lines_skipped() -> None:
    """validate_training_data skips blank lines and counts valid ones."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"instruction": "A", "output": "B"}\n')
        f.write("\n")
        f.write('{"instruction": "C", "output": "D"}\n')
        path = f.name
    try:
        ok, errors, count = validate_training_data(path)
        assert ok
        assert count == 2
    finally:
        Path(path).unlink(missing_ok=True)
