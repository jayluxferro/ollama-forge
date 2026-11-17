"""Tests for the forge ui command."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from ollama_forge.commands.ui import _cmd_ui, register


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"host": None, "port": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_register_creates_ui_subcommand() -> None:
    p = argparse.ArgumentParser()
    subs = p.add_subparsers()
    register(subs)
    parsed = p.parse_args(["ui"])
    assert hasattr(parsed, "handler")
    assert parsed.handler is _cmd_ui


def test_cmd_ui_launches_streamlit(tmp_path: Path) -> None:
    # Create a fake forge_ui.py so the existence check passes
    fake_ui = Path(__file__).resolve().parent.parent / "src" / "ollama_forge" / "forge_ui.py"
    assert fake_ui.exists(), "forge_ui.py must exist"

    args = _make_args()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = _cmd_ui(argparse.ArgumentParser(), args)
    assert rc == 0
    call_cmd = mock_run.call_args[0][0]
    assert "streamlit" in call_cmd
    assert "run" in call_cmd
    assert str(fake_ui) in call_cmd


def test_cmd_ui_passes_port(tmp_path: Path) -> None:
    args = _make_args(port=9999)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        _cmd_ui(argparse.ArgumentParser(), args)
    call_cmd = mock_run.call_args[0][0]
    assert "--server.port" in call_cmd
    assert "9999" in call_cmd


def test_cmd_ui_passes_host() -> None:
    args = _make_args(host="0.0.0.0")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        _cmd_ui(argparse.ArgumentParser(), args)
    call_cmd = mock_run.call_args[0][0]
    assert "--server.address" in call_cmd
    assert "0.0.0.0" in call_cmd


def test_cmd_ui_streamlit_not_found() -> None:
    args = _make_args()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        rc = _cmd_ui(argparse.ArgumentParser(), args)
    assert rc == 1
