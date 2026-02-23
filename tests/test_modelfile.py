"""Modelfile generation tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ollama_forge.modelfile import (
    _extract_template_from_config,
    build_modelfile,
    get_stop_tokens_from_checkpoint,
    merge_modelfile_with_reference_template,
    modelfile_append_num_predict,
    modelfile_append_stop_parameters,
    modelfile_append_template,
    template_body_from_modelfile,
    template_from_hf_checkpoint_with_reason,
)


def test_build_modelfile_minimal() -> None:
    """Only FROM when no options."""
    out = build_modelfile("llama3.2")
    assert out.strip() == "FROM llama3.2"


def test_build_modelfile_with_system_and_params() -> None:
    """System and parameters are emitted."""
    out = build_modelfile(
        "llama3.2",
        system="You are helpful.",
        temperature=0.7,
        num_ctx=4096,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    assert "FROM llama3.2" in out
    assert "You are helpful." in out
    assert "PARAMETER temperature 0.7" in out
    assert "PARAMETER num_ctx 4096" in out
    assert "PARAMETER top_p 0.9" in out
    assert "PARAMETER repeat_penalty 1.1" in out


def test_build_modelfile_with_adapter() -> None:
    """ADAPTER line when adapter path given."""
    out = build_modelfile("llama3.2", adapter="/path/to/adapter")
    assert "FROM llama3.2" in out
    assert "ADAPTER /path/to/adapter" in out


def test_template_from_hf_checkpoint_with_reason_no_config(tmp_path: Path) -> None:
    """template_from_hf_checkpoint_with_reason returns (None, reason) when checkpoint has no config.json."""
    # Empty dir or dir without config.json
    template, reason = template_from_hf_checkpoint_with_reason(tmp_path)
    assert template is None
    assert reason is not None
    assert "config.json" in reason


def test_template_from_hf_checkpoint_with_reason_nonexistent_dir() -> None:
    """template_from_hf_checkpoint_with_reason returns (None, reason) for nonexistent path."""
    template, reason = template_from_hf_checkpoint_with_reason("/nonexistent/checkpoint/dir")
    assert template is None
    assert reason is not None


def test_template_body_from_modelfile() -> None:
    """template_body_from_modelfile extracts content inside TEMPLATE triple-quotes."""
    content = 'FROM x\nTEMPLATE """hello {{ .Prompt }}"""\nPARAMETER x 1'
    assert template_body_from_modelfile(content) == "hello {{ .Prompt }}"
    assert template_body_from_modelfile("FROM x") is None
    assert template_body_from_modelfile('FROM x\nTEMPLATE """\n\n"""') == ""


def test_merge_modelfile_with_reference_template_no_ref_template() -> None:
    """When reference has no TEMPLATE block, FROM is replaced (template_only=False)."""
    current = "FROM old\nPARAMETER x 1"
    ref = "FROM ref\nPARAMETER y 2"
    out = merge_modelfile_with_reference_template(current, ref, "new_base")
    assert "FROM new_base" in out
    assert "PARAMETER x 1" in out


def test_merge_modelfile_with_reference_template_template_only() -> None:
    """When template_only=True and ref has no TEMPLATE, current is unchanged."""
    current = "FROM old\nPARAMETER x 1"
    ref = "FROM ref"
    out = merge_modelfile_with_reference_template(current, ref, "new_base", template_only=True)
    assert out == current


def test_merge_modelfile_with_reference_template_replaces_template_and_from() -> None:
    """Reference TEMPLATE is merged and FROM updated."""
    current = 'FROM old\nTEMPLATE """old template"""'
    ref = 'FROM ref\nTEMPLATE """new {{ .Prompt }}"""'
    out = merge_modelfile_with_reference_template(current, ref, "new_base")
    assert "FROM new_base" in out
    assert "new {{ .Prompt }}" in out


def test_modelfile_append_template() -> None:
    """modelfile_append_template inserts TEMPLATE block after FROM."""
    content = "FROM llama3.2\n"
    body = "{{ .Prompt }}"
    out = modelfile_append_template(content, body)
    assert "FROM llama3.2" in out
    assert "TEMPLATE" in out
    assert "{{ .Prompt }}" in out


def test_modelfile_append_stop_parameters() -> None:
    """modelfile_append_stop_parameters adds PARAMETER stop lines (quoted)."""
    content = "FROM x\n"
    out = modelfile_append_stop_parameters(content, ["</s>", "<|eot|>"])
    assert 'PARAMETER stop "</s>"' in out
    assert 'PARAMETER stop "<|eot|>"' in out


def test_modelfile_append_num_predict() -> None:
    """modelfile_append_num_predict adds PARAMETER num_predict."""
    content = "FROM x\n"
    out = modelfile_append_num_predict(content, 4096)
    assert "PARAMETER num_predict 4096" in out


def test_modelfile_append_num_predict_no_duplicate() -> None:
    """modelfile_append_num_predict skips if num_predict already present."""
    content = "FROM x\nPARAMETER num_predict 1024\n"
    out = modelfile_append_num_predict(content, 4096)
    assert out.count("num_predict") == 1


def test_modelfile_append_num_predict_zero_skips() -> None:
    """modelfile_append_num_predict skips when num_predict <= 0."""
    content = "FROM x\n"
    assert modelfile_append_num_predict(content, 0) == content
    assert modelfile_append_num_predict(content, -1) == content


def test_modelfile_append_stop_parameters_deduplication() -> None:
    """modelfile_append_stop_parameters deduplicates stop tokens."""
    content = "FROM x\n"
    out = modelfile_append_stop_parameters(content, ["</s>", "</s>", "<eos>", "</s>"])
    assert out.count('PARAMETER stop "</s>"') == 1
    assert 'PARAMETER stop "<eos>"' in out


def test_modelfile_append_stop_parameters_empty() -> None:
    """modelfile_append_stop_parameters returns unchanged content when list is empty."""
    content = "FROM x\n"
    assert modelfile_append_stop_parameters(content, []) == content


def test_modelfile_append_template_replaces_existing() -> None:
    """modelfile_append_template replaces an existing TEMPLATE block."""
    content = 'FROM x\nTEMPLATE """old body"""\n'
    out = modelfile_append_template(content, "new body")
    assert "old body" not in out
    assert "new body" in out


def test_merge_modelfile_no_template_in_current_inserts_after_from() -> None:
    """When current has no TEMPLATE but ref does, template is inserted after FROM."""
    current = "FROM old\nPARAMETER x 1"
    ref = 'FROM ref\nTEMPLATE """{{ .Prompt }}"""'
    out = merge_modelfile_with_reference_template(current, ref, "new_base")
    assert "FROM new_base" in out
    assert "{{ .Prompt }}" in out
    assert "PARAMETER x 1" in out


# ---------------------------------------------------------------------------
# _extract_template_from_config
# ---------------------------------------------------------------------------


class TestExtractTemplateFromConfig:
    """Tests for _extract_template_from_config."""

    def test_string_chat_template(self) -> None:
        """Returns string chat_template directly."""
        data = {"chat_template": "{{ messages }}"}
        assert _extract_template_from_config(data) == "{{ messages }}"

    def test_dict_chat_template_with_template_key(self) -> None:
        """Returns template from dict with 'template' key."""
        data = {"chat_template": {"template": "{{ .Prompt }}"}}
        assert _extract_template_from_config(data) == "{{ .Prompt }}"

    def test_dict_chat_template_with_content_key(self) -> None:
        """Returns template from dict with 'content' key."""
        data = {"chat_template": {"content": "{{ .System }}"}}
        assert _extract_template_from_config(data) == "{{ .System }}"

    def test_list_chat_template_first_item(self) -> None:
        """Returns template from first item in list."""
        data = {"chat_template": [{"template": "first {{ .Prompt }}"}, {"template": "second"}]}
        assert _extract_template_from_config(data) == "first {{ .Prompt }}"

    def test_jinja_fallback_finds_nested(self) -> None:
        """Falls back to any Jinja-looking string nested in the config."""
        data = {"some_key": {"nested": "{{ messages | tojson }}"}}
        result = _extract_template_from_config(data)
        assert result is not None
        assert "{{" in result

    def test_no_template_returns_none(self) -> None:
        """Returns None when no chat_template present."""
        assert _extract_template_from_config({}) is None
        assert _extract_template_from_config({"model_type": "llama"}) is None

    def test_empty_string_ignored(self) -> None:
        """Empty string chat_template is ignored and returns None."""
        assert _extract_template_from_config({"chat_template": ""}) is None
        assert _extract_template_from_config({"chat_template": "   "}) is None

    def test_chat_template_jinja_key(self) -> None:
        """Also reads chat_template_jinja key."""
        data = {"chat_template_jinja": "{% for m in messages %}{{ m }}{% endfor %}"}
        result = _extract_template_from_config(data)
        assert result is not None
        assert "messages" in result


# ---------------------------------------------------------------------------
# template_from_hf_checkpoint_with_reason — mocked paths
# ---------------------------------------------------------------------------


class TestTemplateFromHfCheckpointWithReasonMocked:
    """Mocked tests for template_from_hf_checkpoint_with_reason priority paths."""

    def test_family_override_returned_first(self, tmp_path: Path) -> None:
        """Priority 1: family template_override is returned without loading tokenizer."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        override = "<|start_header_id|>user<|end_header_id|>\n{{ .Prompt }}"
        with patch("ollama_forge.model_family.get_family_template_override", return_value=override):
            template, reason = template_from_hf_checkpoint_with_reason(tmp_path)
        assert template == override
        assert reason is None

    def test_transformers_not_installed_returns_error(self, tmp_path: Path) -> None:
        """When transformers is missing, returns (None, 'transformers not installed')."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown_model"}))
        with (
            patch("ollama_forge.model_family.get_family_template_override", return_value=None),
            patch.dict("sys.modules", {"transformers": None}),
        ):
            template, reason = template_from_hf_checkpoint_with_reason(tmp_path)
        assert template is None
        assert reason is not None
        assert "transformers" in reason

    def test_gemma_fallback_when_no_chat_template(self, tmp_path: Path) -> None:
        """Priority 3: Built-in Gemma template returned when tokenizer has no chat_template."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        with (
            patch("ollama_forge.model_family.get_family_template_override", return_value=None),
            patch("transformers.AutoTokenizer") as mock_auto,
            patch("ollama_forge.model_family.is_gemma_checkpoint", return_value=True),
        ):
            mock_auto.from_pretrained.return_value = mock_tok
            template, reason = template_from_hf_checkpoint_with_reason(tmp_path)
        assert template is not None
        assert "<<start_of_turn>>" in template
        assert reason is None

    def test_no_chat_template_not_gemma_returns_error(self, tmp_path: Path) -> None:
        """When tokenizer has no chat_template and not Gemma, returns (None, reason)."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown_model"}))
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        with (
            patch("ollama_forge.model_family.get_family_template_override", return_value=None),
            patch("transformers.AutoTokenizer") as mock_auto,
            patch("ollama_forge.model_family.is_gemma_checkpoint", return_value=False),
        ):
            mock_auto.from_pretrained.return_value = mock_tok
            template, reason = template_from_hf_checkpoint_with_reason(tmp_path)
        assert template is None
        assert reason is not None

    def test_hf_tokenizer_path_with_response_placeholder(self, tmp_path: Path) -> None:
        """Priority 2: HF tokenizer path substitutes placeholders into Ollama template vars."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown_model"}))
        # Simulate a tokenizer that renders the template with our placeholders
        _SYS = "\x00ollama_sys\x00"
        _USR = "\x00ollama_user\x00"
        _RSP = "\x00ollama_resp\x00"
        rendered = f"[INST] {_SYS}\n{_USR} [/INST] {_RSP}"

        mock_tok = MagicMock()
        mock_tok.chat_template = "some jinja template"
        mock_tok.apply_chat_template.return_value = rendered

        with (
            patch("ollama_forge.model_family.get_family_template_override", return_value=None),
            patch("transformers.AutoTokenizer") as mock_auto,
        ):
            mock_auto.from_pretrained.return_value = mock_tok
            template, reason = template_from_hf_checkpoint_with_reason(tmp_path)

        assert reason is None
        assert template is not None
        assert "{{ .System }}" in template
        assert "{{ .Prompt }}" in template
        assert "{{ .Response }}" in template

    def test_hf_tokenizer_path_without_response_placeholder(self, tmp_path: Path) -> None:
        """When apply_chat_template doesn't include response placeholder, appends {{ .Response }}."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown_model"}))
        _USR = "\x00ollama_user\x00"
        rendered = f"[INST] {_USR} [/INST]"  # no response placeholder

        mock_tok = MagicMock()
        mock_tok.chat_template = "some jinja"
        mock_tok.apply_chat_template.return_value = rendered

        with (
            patch("ollama_forge.model_family.get_family_template_override", return_value=None),
            patch("transformers.AutoTokenizer") as mock_auto,
        ):
            mock_auto.from_pretrained.return_value = mock_tok
            template, reason = template_from_hf_checkpoint_with_reason(tmp_path)

        assert reason is None
        assert template is not None
        assert template.endswith("{{ .Response }}")


# ---------------------------------------------------------------------------
# get_stop_tokens_from_checkpoint
# ---------------------------------------------------------------------------


class TestGetStopTokensFromCheckpoint:
    """Tests for get_stop_tokens_from_checkpoint."""

    def test_no_config_json_returns_empty(self, tmp_path: Path) -> None:
        """Returns empty list when checkpoint has no config.json."""
        assert get_stop_tokens_from_checkpoint(tmp_path) == []

    def test_nonexistent_path_returns_empty(self) -> None:
        """Returns empty list for non-existent path."""
        assert get_stop_tokens_from_checkpoint("/nonexistent/path") == []

    def test_gemma_stop_tokens_from_family(self, tmp_path: Path) -> None:
        """Returns Gemma family stop tokens when model_type is gemma."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("no tokenizer")
            tokens = get_stop_tokens_from_checkpoint(tmp_path)
        assert "<<end_of_turn>>" in tokens

    def test_gemma_stop_tokens_from_tokenizer_config(self, tmp_path: Path) -> None:
        """Gemma: reads eos_token/pad_token from tokenizer_config.json."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma"}))
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<end_of_turn>", "pad_token": "<pad>"})
        )
        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("no tokenizer")
            tokens = get_stop_tokens_from_checkpoint(tmp_path)
        assert "<end_of_turn>" in tokens

    def test_transformers_missing_returns_family_tokens_only(self, tmp_path: Path) -> None:
        """When transformers is not installed, returns only family-based stop tokens."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
        with patch.dict("sys.modules", {"transformers": None}):
            tokens = get_stop_tokens_from_checkpoint(tmp_path)
        assert "[/INST]" in tokens
        assert "</s>" in tokens

    def test_tokenizer_eos_token_included(self, tmp_path: Path) -> None:
        """eos_token from tokenizer is included in stop tokens."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown_novel_arch"}))
        mock_tok = MagicMock()
        mock_tok.eos_token = "<|custom_eos|>"
        mock_tok.pad_token = None
        mock_tok.pad_token_id = None
        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tok
            tokens = get_stop_tokens_from_checkpoint(tmp_path)
        assert "<|custom_eos|>" in tokens
