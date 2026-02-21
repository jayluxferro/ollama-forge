"""Tests for model_family module (model family detection)."""

import json
import tempfile
from pathlib import Path

from ollama_forge.model_family import (
    _MODEL_FAMILIES,
    _normalize,
    detect_model_family,
    get_family_name,
    get_family_stop_tokens,
    get_family_template_override,
    is_gemma_checkpoint,
    suggest_gguf_flags,
)


class TestNormalize:
    """Tests for _normalize function."""

    def test_lowercase(self) -> None:
        """Converts to lowercase."""
        assert _normalize("LlamaTokenizer") == "llamatokenizer"
        assert _normalize("GEMMA") == "gemma"

    def test_remove_dashes(self) -> None:
        """Removes dashes."""
        assert _normalize("gemma-2") == "gemma2"
        assert _normalize("phi-3") == "phi3"

    def test_remove_underscores(self) -> None:
        """Removes underscores."""
        assert _normalize("qwen2_moe") == "qwen2moe"
        assert _normalize("deepseek_v2") == "deepseekv2"

    def test_none_returns_empty(self) -> None:
        """None returns empty string."""
        assert _normalize(None) == ""

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert _normalize("") == ""


class TestModelFamilies:
    """Tests for _MODEL_FAMILIES definitions."""

    def test_known_families_exist(self) -> None:
        """All expected families are defined."""
        family_names = {f.name for f in _MODEL_FAMILIES}
        expected = {"gemma", "llama3", "mistral", "mixtral", "qwen2", "phi3", "deepseek", "cohere", "yi"}
        assert expected.issubset(family_names)

    def test_families_have_model_types(self) -> None:
        """Each family has at least one model_type."""
        for family in _MODEL_FAMILIES:
            assert len(family.model_types) > 0, f"{family.name} has no model_types"

    def test_families_have_stop_tokens(self) -> None:
        """Each family has stop tokens."""
        for family in _MODEL_FAMILIES:
            assert len(family.stop_tokens) > 0, f"{family.name} has no stop_tokens"

    def test_gemma_has_template_override(self) -> None:
        """Gemma family has template override."""
        gemma = next(f for f in _MODEL_FAMILIES if f.name == "gemma")
        assert gemma.template_override is not None
        assert "<<start_of_turn>>" in gemma.template_override


class TestDetectModelFamily:
    """Tests for detect_model_family function."""

    def test_nonexistent_dir_returns_none(self) -> None:
        """Non-existent directory returns None."""
        assert detect_model_family("/nonexistent/path") is None

    def test_detect_by_model_type_gemma(self) -> None:
        """Detect Gemma by model_type."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"]}
            (Path(d) / "config.json").write_text(json.dumps(config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "gemma"

    def test_detect_by_model_type_llama(self) -> None:
        """Detect Llama by model_type."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
            (Path(d) / "config.json").write_text(json.dumps(config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "llama3"

    def test_detect_by_model_type_qwen2(self) -> None:
        """Detect Qwen2 by model_type."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
            (Path(d) / "config.json").write_text(json.dumps(config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "qwen2"

    def test_detect_by_model_type_mistral(self) -> None:
        """Detect Mistral by model_type."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "mistral"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "mistral"

    def test_detect_by_model_type_phi(self) -> None:
        """Detect Phi by model_type."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "phi3"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "phi3"

    def test_detect_by_architecture(self) -> None:
        """Detect by architecture when model_type unknown."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "unknown", "architectures": ["LlamaForCausalLM"]}
            (Path(d) / "config.json").write_text(json.dumps(config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "llama3"

    def test_detect_by_tokenizer_class(self) -> None:
        """Detect by tokenizer_class when others fail."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "unknown", "architectures": []}
            tokenizer_config = {"tokenizer_class": "GemmaTokenizer"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            (Path(d) / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "gemma"

    def test_detect_by_tokenizer_class_substring(self) -> None:
        """Detect by tokenizer_class when family.name in tokenizer_class and tc in tokenizer_class."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "unknown", "architectures": []}
            # e.g. Qwen2TokenizerFast -> qwen2 in name, qwen2tokenizerfast contains tokenizer class
            tokenizer_config = {"tokenizer_class": "Qwen2TokenizerFast"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            (Path(d) / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "qwen2"

    def test_model_type_priority_over_tokenizer(self) -> None:
        """Model type takes priority over tokenizer class."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "mistral"}
            tokenizer_config = {"tokenizer_class": "LlamaTokenizer"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            (Path(d) / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
            family = detect_model_family(d)
            assert family is not None
            assert family.name == "mistral"

    def test_empty_config_returns_none(self) -> None:
        """Empty config.json returns None."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text("{}")
            family = detect_model_family(d)
            assert family is None

    def test_invalid_json_returns_none(self) -> None:
        """Invalid JSON returns None gracefully."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text("not json")
            family = detect_model_family(d)
            assert family is None


class TestGetFamilyStopTokens:
    """Tests for get_family_stop_tokens function."""

    def test_nonexistent_returns_empty(self) -> None:
        """Non-existent path returns empty list."""
        assert get_family_stop_tokens("/nonexistent") == []

    def test_gemma_stop_tokens(self) -> None:
        """Gemma returns expected stop tokens."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "gemma"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            tokens = get_family_stop_tokens(d)
            assert "<<end_of_turn>>" in tokens

    def test_llama_stop_tokens(self) -> None:
        """Llama returns expected stop tokens."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "llama"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            tokens = get_family_stop_tokens(d)
            assert "<|eot_id|>" in tokens


class TestGetFamilyTemplateOverride:
    """Tests for get_family_template_override function."""

    def test_nonexistent_returns_none(self) -> None:
        """Non-existent path returns None."""
        assert get_family_template_override("/nonexistent") is None

    def test_gemma_has_override(self) -> None:
        """Gemma returns template override."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "gemma"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            template = get_family_template_override(d)
            assert template is not None
            assert "<<start_of_turn>>" in template

    def test_llama_no_override(self) -> None:
        """Llama has no template override."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "llama"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            template = get_family_template_override(d)
            assert template is None


class TestIsGemmaCheckpoint:
    """Tests for is_gemma_checkpoint function."""

    def test_nonexistent_returns_false(self) -> None:
        """Non-existent path returns False."""
        assert is_gemma_checkpoint("/nonexistent") is False

    def test_gemma_config_returns_true(self) -> None:
        """Gemma model_type returns True."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "gemma2"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            assert is_gemma_checkpoint(d) is True

    def test_llama_config_returns_false(self) -> None:
        """Non-Gemma model_type returns False."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "llama"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            assert is_gemma_checkpoint(d) is False


class TestGetFamilyName:
    """Tests for get_family_name function."""

    def test_nonexistent_returns_none(self) -> None:
        """Non-existent path returns None."""
        assert get_family_name("/nonexistent") is None

    def test_returns_family_name(self) -> None:
        """Returns correct family name."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "qwen2"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            assert get_family_name(d) == "qwen2"


class TestSuggestGgufFlags:
    """Tests for suggest_gguf_flags function."""

    def test_nonexistent_returns_empty(self) -> None:
        """Non-existent path returns empty dict."""
        assert suggest_gguf_flags("/nonexistent") == {}

    def test_no_gguf_flags_returns_empty(self) -> None:
        """Family without gguf_flags returns empty dict."""
        with tempfile.TemporaryDirectory() as d:
            config = {"model_type": "llama"}
            (Path(d) / "config.json").write_text(json.dumps(config))
            flags = suggest_gguf_flags(d)
            assert flags == {}
