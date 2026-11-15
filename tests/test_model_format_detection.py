"""Tests for MLX-quantized and GGUF-only model detection."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ollama_forge.hf_fetch import is_gguf_only_repo
from ollama_forge.model_family import is_mlx_quantized_checkpoint, is_mlx_quantized_config

# ---------------------------------------------------------------------------
# is_mlx_quantized_checkpoint
# ---------------------------------------------------------------------------


def _write_config(d: Path, cfg: dict) -> None:
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


class TestIsMlxQuantizedConfig:
    def test_bits_and_group_size(self) -> None:
        assert is_mlx_quantized_config({"quantization": {"bits": 4, "group_size": 64}}) is True

    def test_bits_only(self) -> None:
        assert is_mlx_quantized_config({"quantization": {"bits": 2}}) is True

    def test_bnb_quantization_config_not_flagged(self) -> None:
        assert is_mlx_quantized_config({"quantization_config": {"quant_type": "nf4"}}) is False

    def test_empty_config(self) -> None:
        assert is_mlx_quantized_config({}) is False

    def test_quantization_not_a_dict(self) -> None:
        assert is_mlx_quantized_config({"quantization": "linear"}) is False

    def test_quantization_dict_without_bits_or_group_size(self) -> None:
        assert is_mlx_quantized_config({"quantization": {"type": "some_other"}}) is False


class TestIsMlxQuantizedCheckpoint:
    def test_mlx_quantized_bits_and_group_size(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"model_type": "llama", "quantization": {"bits": 4, "group_size": 64}})
        assert is_mlx_quantized_checkpoint(tmp_path) is True

    def test_mlx_quantized_bits_only(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"model_type": "gemma", "quantization": {"bits": 2}})
        assert is_mlx_quantized_checkpoint(tmp_path) is True

    def test_mlx_quantized_group_size_only(self, tmp_path: Path) -> None:
        # group_size alone (unusual but should still flag as MLX)
        _write_config(tmp_path, {"quantization": {"group_size": 64}})
        assert is_mlx_quantized_checkpoint(tmp_path) is True

    def test_bnb_quantization_config_not_flagged(self, tmp_path: Path) -> None:
        # BitsAndBytes uses "quantization_config", not "quantization"
        _write_config(
            tmp_path,
            {"model_type": "llama", "quantization_config": {"quant_type": "nf4", "load_in_4bit": True}},
        )
        assert is_mlx_quantized_checkpoint(tmp_path) is False

    def test_no_quantization_key(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"model_type": "gemma"})
        assert is_mlx_quantized_checkpoint(tmp_path) is False

    def test_quantization_not_a_dict(self, tmp_path: Path) -> None:
        # Unexpected shape — should not raise, just return False
        _write_config(tmp_path, {"quantization": "some_string"})
        assert is_mlx_quantized_checkpoint(tmp_path) is False

    def test_missing_config_json(self, tmp_path: Path) -> None:
        assert is_mlx_quantized_checkpoint(tmp_path) is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"quantization": {"bits": 4, "group_size": 64}})
        assert is_mlx_quantized_checkpoint(str(tmp_path)) is True


# ---------------------------------------------------------------------------
# is_gguf_only_repo
# ---------------------------------------------------------------------------


def _make_list_repo_files(files: list[str]):
    """Return a mock for huggingface_hub.list_repo_files that yields `files`."""
    return MagicMock(return_value=iter(files))


class TestIsGgufOnlyRepo:
    def test_gguf_only_returns_true_and_filenames(self) -> None:
        files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf", "README.md"]
        with patch("huggingface_hub.list_repo_files", _make_list_repo_files(files)):
            ok, gguf = is_gguf_only_repo("some/repo")
        assert ok is True
        assert set(gguf) == {"model-Q4_K_M.gguf", "model-Q8_0.gguf"}

    def test_mixed_repo_returns_false(self) -> None:
        files = ["model.gguf", "model.safetensors", "config.json"]
        with patch("huggingface_hub.list_repo_files", _make_list_repo_files(files)):
            ok, gguf = is_gguf_only_repo("some/repo")
        assert ok is False
        assert gguf == ["model.gguf"]

    def test_safetensors_only_no_gguf(self) -> None:
        files = ["model.safetensors", "config.json", "tokenizer.json"]
        with patch("huggingface_hub.list_repo_files", _make_list_repo_files(files)):
            ok, gguf = is_gguf_only_repo("some/repo")
        assert ok is False
        assert gguf == []

    def test_empty_repo(self) -> None:
        with patch("huggingface_hub.list_repo_files", _make_list_repo_files([])):
            ok, gguf = is_gguf_only_repo("some/repo")
        assert ok is False
        assert gguf == []

    def test_bin_weights_alongside_gguf_returns_false(self) -> None:
        files = ["pytorch_model.bin", "model-Q4_K_M.gguf"]
        with patch("huggingface_hub.list_repo_files", _make_list_repo_files(files)):
            ok, _ = is_gguf_only_repo("some/repo")
        assert ok is False

    def test_case_insensitive_gguf(self) -> None:
        files = ["MODEL.GGUF"]
        with patch("huggingface_hub.list_repo_files", _make_list_repo_files(files)):
            ok, gguf = is_gguf_only_repo("some/repo")
        assert ok is True
        assert gguf == ["MODEL.GGUF"]

    def test_revision_forwarded(self) -> None:
        files = ["model.gguf"]
        mock = _make_list_repo_files(files)
        with patch("huggingface_hub.list_repo_files", mock):
            is_gguf_only_repo("some/repo", revision="v1.0")
        mock.assert_called_once_with("some/repo", revision="v1.0")
