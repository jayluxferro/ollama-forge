"""Tests for the forge mlx-train command."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from ollama_forge.commands.train import (
    _MLX_TRAIN_DEFAULTS,
    _MLX_TRAIN_TEMPLATE,
    _cmd_mlx_train,
)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _make_config(tmp_path: Path, **overrides: object) -> Path:
    cfg = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "data": str(tmp_path / "data"),
        **overrides,
    }
    (tmp_path / "data").mkdir(exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# ---------------------------------------------------------------------------
# --init
# ---------------------------------------------------------------------------


def test_init_writes_template(tmp_path: Path) -> None:
    dest = tmp_path / "forge-train.yaml"
    args = argparse.Namespace(config=str(dest), init=True, dry_run=False)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    text = dest.read_text()
    assert "mlx_lm" in text or "model:" in text
    assert "data:" in text


def test_init_refuses_overwrite(tmp_path: Path) -> None:
    dest = tmp_path / "forge-train.yaml"
    dest.write_text("existing")
    args = argparse.Namespace(config=str(dest), init=True, dry_run=False)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc != 0
    assert dest.read_text() == "existing"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_missing_config_arg() -> None:
    args = argparse.Namespace(config=None, init=False, dry_run=False)
    p = argparse.ArgumentParser()
    rc = _cmd_mlx_train(p, args)
    assert rc != 0


def test_missing_config_file(tmp_path: Path) -> None:
    args = argparse.Namespace(config=str(tmp_path / "nope.yaml"), init=False, dry_run=False)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc != 0


def test_config_missing_model(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"data": "./data"}))
    args = argparse.Namespace(config=str(path), init=False, dry_run=False)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc != 0


def test_config_missing_data(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": "mlx-community/Llama-3.2-3B"}))
    args = argparse.Namespace(config=str(path), init=False, dry_run=False)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc != 0


# ---------------------------------------------------------------------------
# --dry-run (no subprocess spawned)
# ---------------------------------------------------------------------------


def test_dry_run_prints_lora_argv(tmp_path: Path, capsys) -> None:
    cfg_path = _make_config(tmp_path, iters=42)
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=True)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "mlx_lm" in out
    assert "lora" in out
    assert "--iters" in out
    assert "42" in out


def test_dry_run_fuse_step(tmp_path: Path, capsys) -> None:
    cfg_path = _make_config(tmp_path, fuse=True, fused_path=str(tmp_path / "fused"))
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=True)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "fuse" in out


def test_dry_run_export_gguf(tmp_path: Path, capsys) -> None:
    cfg_path = _make_config(tmp_path, fuse=True, export_gguf=True, gguf_path=str(tmp_path / "m.gguf"))
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=True)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "--export-gguf" in out


def test_dry_run_ollama_push(tmp_path: Path, capsys) -> None:
    cfg_path = _make_config(tmp_path, ollama_name="my-model")
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=True)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "my-model" in out


# ---------------------------------------------------------------------------
# subprocess execution (mocked)
# ---------------------------------------------------------------------------


def test_runs_lora_subprocess(tmp_path: Path) -> None:
    cfg_path = _make_config(tmp_path)
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    call_args = mock_run.call_args_list[0][0][0]  # first call, argv list
    assert "mlx_lm" in call_args
    assert "lora" in call_args
    assert "--train" in call_args


def test_lora_failure_propagates(tmp_path: Path) -> None:
    cfg_path = _make_config(tmp_path)
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        rc = _cmd_mlx_train(_parser(), args)
    assert rc == 1


def test_runs_fuse_when_requested(tmp_path: Path) -> None:
    cfg_path = _make_config(tmp_path, fuse=True)
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=False)
    calls = []
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = _cmd_mlx_train(_parser(), args)
        calls = [c[0][0] for c in mock_run.call_args_list]
    assert rc == 0
    assert any("fuse" in argv for argv in calls), "expected mlx_lm fuse to be called"


def test_no_fuse_when_not_requested(tmp_path: Path) -> None:
    cfg_path = _make_config(tmp_path, fuse=False)
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = _cmd_mlx_train(_parser(), args)
        calls = [c[0][0] for c in mock_run.call_args_list]
    assert rc == 0
    assert not any("fuse" in argv for argv in calls)


def test_defaults_applied(tmp_path: Path, capsys) -> None:
    """Config with only model+data should use all _MLX_TRAIN_DEFAULTS."""
    cfg_path = _make_config(tmp_path)
    args = argparse.Namespace(config=str(cfg_path), init=False, dry_run=True)
    rc = _cmd_mlx_train(_parser(), args)
    assert rc == 0
    out = capsys.readouterr().out
    assert str(_MLX_TRAIN_DEFAULTS["iters"]) in out
    assert str(_MLX_TRAIN_DEFAULTS["fine_tune_type"]) in out


# ---------------------------------------------------------------------------
# template content
# ---------------------------------------------------------------------------


def test_template_is_valid_yaml() -> None:
    cfg = yaml.safe_load(_MLX_TRAIN_TEMPLATE)
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "data" in cfg
