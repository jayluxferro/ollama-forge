"""Tests for config file loading (abliterate run / train-run --config)."""

import tempfile
from pathlib import Path

import pytest

from ollama_forge.config_loader import apply_config_to_args, load_config


def test_load_config_json() -> None:
    """Load JSON config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"model": "meta-llama/Llama-2-7b", "name": "my-model", "quant": "Q5_K_M"}')
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg["model"] == "meta-llama/Llama-2-7b"
        assert cfg["name"] == "my-model"
        assert cfg["quant"] == "Q5_K_M"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_config_yaml() -> None:
    """Load YAML config (requires pyyaml)."""
    pytest.importorskip("yaml")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("model: meta-llama/Llama-2-7b\nname: my-model\nnum_instructions: 64\n")
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg["model"] == "meta-llama/Llama-2-7b"
        assert cfg["num_instructions"] == 64
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_config_missing_file() -> None:
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("/nonexistent/abliterate.yaml")


def test_apply_config_to_args_only_if_default() -> None:
    """Config fills only when current value equals default; CLI overrides."""
    class Args:
        model: str | None = None
        name: str | None = None
        quant: str = "Q4_K_M"

    args = Args()
    args.quant = "Q4_K_M"  # default
    config = {"model": "hf/model", "name": "my-model", "quant": "Q8_0"}
    defaults = {"model": None, "name": None, "quant": "Q4_K_M"}
    apply_config_to_args(args, config, only_if_default=defaults)
    assert args.model == "hf/model"
    assert args.name == "my-model"
    assert args.quant == "Q8_0"

    # When user set a value (not default), config should not override
    args2 = Args()
    args2.model = "user-set-model"
    args2.name = None
    args2.quant = "Q4_K_M"
    apply_config_to_args(args2, config, only_if_default=defaults)
    assert args2.model == "user-set-model"  # unchanged
    assert args2.name == "my-model"
    assert args2.quant == "Q8_0"
