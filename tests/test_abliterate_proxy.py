"""Tests for abliterate_proxy module (lightweight prompt proxy)."""

import json

import pytest

from ollama_forge.abliterate_proxy import (
    ProxyConfig,
    _convert_tools_to_hf,
    _get_ollama_base,
    _normalize_message,
    _parse_tool_calls,
)


class TestNormalizeMessage:
    """Tests for _normalize_message function."""

    def test_basic_user_message(self) -> None:
        """User message with content."""
        msg = _normalize_message({"role": "user", "content": "Hello"})
        assert msg == {"role": "user", "content": "Hello"}

    def test_empty_content(self) -> None:
        """Missing content defaults to empty string."""
        msg = _normalize_message({"role": "user"})
        assert msg["content"] == ""

    def test_none_content(self) -> None:
        """None content defaults to empty string."""
        msg = _normalize_message({"role": "assistant", "content": None})
        assert msg["content"] == ""

    def test_assistant_with_tool_calls(self) -> None:
        """Assistant message with tool_calls converts arguments to JSON string."""
        msg = _normalize_message({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                }
            ],
        })
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'

    def test_assistant_tool_calls_string_args(self) -> None:
        """Tool call with string arguments passes through."""
        msg = _normalize_message({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "test", "arguments": '{"x": 1}'},
                }
            ],
        })
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'

    def test_tool_message(self) -> None:
        """Tool message includes name and tool_call_id."""
        msg = _normalize_message({
            "role": "tool",
            "content": "result data",
            "name": "get_weather",
            "tool_call_id": "call_123",
        })
        assert msg["role"] == "tool"
        assert msg["name"] == "get_weather"
        assert msg["tool_call_id"] == "call_123"

    def test_tool_message_missing_name(self) -> None:
        """Tool message with missing name defaults to empty string."""
        msg = _normalize_message({"role": "tool", "content": "result"})
        assert msg["name"] == ""
        assert msg["tool_call_id"] == ""


class TestConvertToolsToHF:
    """Tests for _convert_tools_to_hf function."""

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert _convert_tools_to_hf(None) is None

    def test_empty_list_returns_none(self) -> None:
        """Empty list returns None."""
        assert _convert_tools_to_hf([]) is None

    def test_basic_tool_conversion(self) -> None:
        """Basic tool converts correctly."""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }]
        result = _convert_tools_to_hf(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"

    def test_non_function_type_skipped(self) -> None:
        """Non-function types are skipped."""
        tools = [{"type": "other", "function": {"name": "test"}}]
        result = _convert_tools_to_hf(tools)
        assert result is None

    def test_missing_fields_default(self) -> None:
        """Missing fields get defaults."""
        tools = [{"type": "function", "function": {}}]
        result = _convert_tools_to_hf(tools)
        assert result[0]["function"]["name"] == ""
        assert result[0]["function"]["description"] == ""
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}


class TestParseToolCalls:
    """Tests for _parse_tool_calls function."""

    def test_no_tool_calls(self) -> None:
        """Normal text without tool calls returns None."""
        assert _parse_tool_calls("Hello, how are you?") is None
        assert _parse_tool_calls("I don't know the answer.") is None

    def test_tool_call_tags(self) -> None:
        """Parse <tool_call> tags."""
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        result = _parse_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["arguments"] == {"city": "NYC"}

    def test_function_call_tags(self) -> None:
        """Parse <function_call> tags."""
        text = '<function_call>{"name": "search", "arguments": {"query": "test"}}</function_call>'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "search"

    def test_json_code_block(self) -> None:
        """Parse JSON in code blocks."""
        text = '```json\n{"name": "calculate", "arguments": {"x": 1, "y": 2}}\n```'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "calculate"

    def test_plain_json(self) -> None:
        """Parse plain JSON tool call."""
        text = '{"name": "test_func", "arguments": {}}'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "test_func"

    def test_arguments_as_string(self) -> None:
        """Handle arguments as JSON string."""
        text = '<tool_call>{"name": "test", "arguments": "{\\"a\\": 1}"}</tool_call>'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["arguments"] == {"a": 1}

    def test_parameters_key(self) -> None:
        """Handle 'parameters' as alternative to 'arguments'."""
        text = '{"name": "func", "parameters": {"p": "value"}}'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["arguments"] == {"p": "value"}

    def test_multiple_tool_calls(self) -> None:
        """Parse multiple tool calls."""
        text = """
        <tool_call>{"name": "func1", "arguments": {}}</tool_call>
        <tool_call>{"name": "func2", "arguments": {}}</tool_call>
        """
        result = _parse_tool_calls(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "func1"
        assert result[1]["function"]["name"] == "func2"

    def test_invalid_json_in_tags(self) -> None:
        """Invalid JSON in tags is skipped."""
        text = '<tool_call>not valid json</tool_call>'
        result = _parse_tool_calls(text)
        assert result is None

    def test_tool_call_marker(self) -> None:
        """Parse [TOOL_CALL] marker format."""
        text = '[TOOL_CALL] {"name": "action", "arguments": "none"}'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "action"

    def test_nested_braces(self) -> None:
        """Parse tool calls with nested braces in arguments."""
        text = '<tool_call>{"name": "api_call", "arguments": {"data": {"nested": "value"}}}</tool_call>'
        result = _parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "api_call"
        assert result[0]["function"]["arguments"]["data"]["nested"] == "value"


class TestProxyConfig:
    """Tests for ProxyConfig class."""

    def test_add_model(self) -> None:
        """Add model registers checkpoint."""
        config = ProxyConfig()
        config.add_model("my-model", "/path/to/checkpoint")
        assert config.get_checkpoint("my-model") == "/path/to/checkpoint"

    def test_default_checkpoint(self) -> None:
        """First added model becomes default."""
        config = ProxyConfig()
        config.add_model("model1", "/path/1")
        config.add_model("model2", "/path/2")
        assert config.default_checkpoint == "/path/1"

    def test_get_checkpoint_fallback(self) -> None:
        """Unknown model returns default checkpoint."""
        config = ProxyConfig()
        config.add_model("known", "/path/known")
        assert config.get_checkpoint("unknown") == "/path/known"

    def test_get_checkpoint_no_default(self) -> None:
        """No models registered returns None."""
        config = ProxyConfig()
        assert config.get_checkpoint("any") is None


class TestGetOllamaBase:
    """Tests for _get_ollama_base function."""

    def test_default(self, monkeypatch) -> None:
        """Default returns localhost:11434."""
        monkeypatch.delenv("OLLAMA_PROXY_TARGET", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert _get_ollama_base() == "http://127.0.0.1:11434"

    def test_proxy_target_env(self, monkeypatch) -> None:
        """OLLAMA_PROXY_TARGET takes precedence."""
        monkeypatch.setenv("OLLAMA_PROXY_TARGET", "http://custom:9999")
        monkeypatch.setenv("OLLAMA_HOST", "http://other:8888")
        assert _get_ollama_base() == "http://custom:9999"

    def test_ollama_host_env(self, monkeypatch) -> None:
        """OLLAMA_HOST used if no PROXY_TARGET."""
        monkeypatch.delenv("OLLAMA_PROXY_TARGET", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://myhost:11434")
        assert _get_ollama_base() == "http://myhost:11434"

    def test_adds_http_prefix(self, monkeypatch) -> None:
        """Adds http:// if missing."""
        monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
        monkeypatch.delenv("OLLAMA_PROXY_TARGET", raising=False)
        assert _get_ollama_base() == "http://localhost:11434"

    def test_strips_trailing_slash(self, monkeypatch) -> None:
        """Strips trailing slash."""
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434/")
        monkeypatch.delenv("OLLAMA_PROXY_TARGET", raising=False)
        assert _get_ollama_base() == "http://localhost:11434"
