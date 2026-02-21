"""Tests for run_helpers: print_actionable_error, ping_ollama, require_ollama, get_jsonl_paths_or_exit."""

import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ollama_forge.run_helpers import (
    check_item,
    get_jsonl_paths_or_exit,
    ping_ollama,
    print_actionable_error,
    require_ollama,
    run_cmd,
    run_ollama_create,
    run_ollama_show_modelfile,
    temporary_text_file,
    write_temp_text_file,
)


def test_print_actionable_error_summary_only() -> None:
    """print_actionable_error with only summary prints one line."""
    buf = StringIO()
    with patch("sys.stderr", buf):
        print_actionable_error("something failed")
    out = buf.getvalue()
    assert "Error: something failed" in out
    assert "Next:" not in out
    assert "Cause:" not in out


def test_print_actionable_error_with_cause_and_next_steps() -> None:
    """print_actionable_error with cause and next_steps prints all sections."""
    buf = StringIO()
    with patch("sys.stderr", buf):
        print_actionable_error(
            "ollama not reachable",
            cause="Connection refused",
            next_steps=["Run: ollama serve", "Check OLLAMA_HOST"],
        )
    out = buf.getvalue()
    assert "Error: ollama not reachable" in out
    assert "Cause: Connection refused" in out
    assert "Next:" in out
    assert "Run: ollama serve" in out
    assert "Check OLLAMA_HOST" in out


def test_ping_ollama_unreachable_returns_false() -> None:
    """ping_ollama with unreachable or invalid URL returns False."""
    # Non-routable / invalid should yield False (no server there)
    assert ping_ollama("http://127.0.0.1:19999", timeout=0.5) is False
    # Invalid host that won't resolve or connect quickly
    assert ping_ollama("http://nonexistent.invalid.xyz.example", timeout=0.5) is False


def test_ping_ollama_normalizes_url() -> None:
    """ping_ollama normalizes URL (adds scheme, strips slash); still False for bad host."""
    # Should not raise; normalization then failed connection
    assert ping_ollama("127.0.0.1:19998", timeout=0.5) is False
    assert ping_ollama("http://127.0.0.1:19997/", timeout=0.5) is False


@patch("ollama_forge.run_helpers.shutil.which")
def test_require_ollama_returns_none_when_found(mock_which: object) -> None:
    """require_ollama returns None when ollama is on PATH."""
    mock_which.return_value = "/usr/bin/ollama"
    assert require_ollama() is None


@patch("ollama_forge.run_helpers.shutil.which")
def test_require_ollama_returns_1_and_prints_when_missing(mock_which: object) -> None:
    """require_ollama returns 1 and prints actionable error when ollama not found."""
    mock_which.return_value = None
    buf = StringIO()
    with patch("sys.stderr", buf):
        code = require_ollama()
    assert code == 1
    out = buf.getvalue()
    assert "ollama" in out.lower()
    assert "Next:" in out


def test_get_jsonl_paths_or_exit_empty_returns_none_and_prints() -> None:
    """get_jsonl_paths_or_exit with no .jsonl files returns None and prints error."""
    buf = StringIO()
    with patch("sys.stderr", buf):
        result = get_jsonl_paths_or_exit(
            [Path("/nonexistent/dir")],
            next_steps=["Add a .jsonl file"],
        )
    assert result is None
    assert "Next:" in buf.getvalue() or "no .jsonl" in buf.getvalue().lower()


def test_get_jsonl_paths_or_exit_with_valid_file(tmp_path: Path) -> None:
    """get_jsonl_paths_or_exit returns list of paths when .jsonl exists."""
    (tmp_path / "data.jsonl").write_text('{"text": "hi"}')
    result = get_jsonl_paths_or_exit([tmp_path])
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "data.jsonl"


def test_temporary_text_file(tmp_path: Path) -> None:
    """temporary_text_file creates file with content and deletes on exit."""
    with temporary_text_file("hello world", suffix=".txt") as p:
        assert p.is_file()
        assert p.read_text() == "hello world"
    assert not p.is_file()


def test_write_temp_text_file() -> None:
    """write_temp_text_file creates file and returns path; caller deletes."""
    p = write_temp_text_file("temp content", suffix=".tmp")
    try:
        assert p.is_file()
        assert p.read_text() == "temp content"
    finally:
        p.unlink(missing_ok=True)


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_cmd_success(mock_run: object) -> None:
    """run_cmd returns 0 when command succeeds."""
    mock_run.return_value = None
    assert run_cmd(["true"], "not found msg") == 0
    mock_run.assert_called_once()


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_cmd_file_not_found(mock_run: object) -> None:
    """run_cmd prints error and returns 1 on FileNotFoundError."""
    mock_run.side_effect = FileNotFoundError()
    buf = StringIO()
    with patch("sys.stderr", buf):
        code = run_cmd(["missing"], "missing command", not_found_next_steps=["Install it"])
    assert code == 1
    assert "missing command" in buf.getvalue() or "Install it" in buf.getvalue()


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_cmd_called_process_error(mock_run: object) -> None:
    """run_cmd returns exit code on CalledProcessError."""
    err = subprocess.CalledProcessError(2, "cmd")
    mock_run.side_effect = err
    buf = StringIO()
    with patch("sys.stderr", buf):
        code = run_cmd(["fail"], "not found", process_error_message="Failed: {e}")
    assert code == 2


@patch("ollama_forge.run_helpers.shutil.which")
@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_ollama_show_modelfile_success(mock_run: object, mock_which: object) -> None:
    """run_ollama_show_modelfile returns stdout when ollama show succeeds."""
    mock_which.return_value = "/usr/bin/ollama"
    mock_run.return_value = type("R", (), {"stdout": "FROM x\n"})()

    result = run_ollama_show_modelfile("mymodel")
    assert result == "FROM x\n"
    mock_run.assert_called_once()


@patch("ollama_forge.run_helpers.shutil.which")
@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_ollama_show_modelfile_timeout_returns_none(mock_run: object, mock_which: object) -> None:
    """run_ollama_show_modelfile returns None on TimeoutExpired."""
    mock_which.return_value = "/usr/bin/ollama"
    mock_run.side_effect = subprocess.TimeoutExpired("ollama", 60)
    assert run_ollama_show_modelfile("mymodel") is None


@patch("ollama_forge.run_helpers.shutil.which")
def test_run_ollama_show_modelfile_missing(mock_which: object) -> None:
    """run_ollama_show_modelfile returns None when ollama not on PATH."""
    mock_which.return_value = None
    assert run_ollama_show_modelfile("mymodel") is None


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_ollama_create_success(mock_run: object, tmp_path: Path) -> None:
    """run_ollama_create writes modelfile and runs ollama create; returns 0."""
    modelfile_path = tmp_path / "M"
    mock_run.return_value = None
    buf = StringIO()
    with patch("sys.stderr", buf):
        code = run_ollama_create("testmodel", "FROM x\n", out_path=modelfile_path)
    assert code == 0
    assert modelfile_path.read_text() == "FROM x\n"
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "ollama" in call_args and "create" in call_args and "testmodel" in call_args


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_ollama_create_temp_file_success(mock_run: object) -> None:
    """run_ollama_create with out_path=None uses temp file and returns 0."""
    mock_run.return_value = None
    buf = StringIO()
    with patch("sys.stderr", buf):
        code = run_ollama_create("testmodel", "FROM x\n", out_path=None)
    assert code == 0
    mock_run.assert_called_once()


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_ollama_create_timeout_returns_1(mock_run: object, tmp_path: Path) -> None:
    """run_ollama_create returns 1 and prints on TimeoutExpired."""
    mock_run.side_effect = subprocess.TimeoutExpired("ollama", 300)
    buf = StringIO()
    with patch("sys.stderr", buf):
        code = run_ollama_create("m", "FROM x\n", out_path=tmp_path / "M")
    assert code == 1
    assert "timed out" in buf.getvalue() or "Next:" in buf.getvalue()


@patch("ollama_forge.run_helpers.subprocess.run")
def test_run_ollama_create_called_process_error_returns_code(mock_run: object, tmp_path: Path) -> None:
    """run_ollama_create returns subprocess return code on CalledProcessError."""
    mock_run.side_effect = subprocess.CalledProcessError(3, "ollama")
    code = run_ollama_create("m", "FROM x\n", out_path=tmp_path / "M")
    assert code == 3


def test_check_item_ok() -> None:
    """check_item with ok=True prints OK and returns True."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        out = check_item("ollama", True, "install it")
    assert out is True
    assert "OK" in buf.getvalue()
    assert "MISSING" not in buf.getvalue()


def test_check_item_missing() -> None:
    """check_item with ok=False prints MISSING and hint, returns False."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        out = check_item("ollama", False, "install it")
    assert out is False
    assert "MISSING" in buf.getvalue()
    assert "install it" in buf.getvalue()
