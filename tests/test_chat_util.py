"""Tests for chat_util module."""

from ollama_forge.chat_util import ollama_tools_to_hf


def test_ollama_tools_to_hf_none() -> None:
    """None input returns None."""
    assert ollama_tools_to_hf(None) is None


def test_ollama_tools_to_hf_empty() -> None:
    """Empty list returns None."""
    assert ollama_tools_to_hf([]) is None


def test_ollama_tools_to_hf_basic() -> None:
    """Single function tool converts correctly."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    result = ollama_tools_to_hf(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["description"] == "Get weather for a city"


def test_ollama_tools_to_hf_non_function_skipped() -> None:
    """Non-function type is skipped; result None if none left."""
    tools = [{"type": "other", "function": {"name": "x"}}]
    assert ollama_tools_to_hf(tools) is None


def test_ollama_tools_to_hf_defaults() -> None:
    """Missing name/description/parameters get defaults."""
    tools = [{"type": "function", "function": {}}]
    result = ollama_tools_to_hf(tools)
    assert result[0]["function"]["name"] == ""
    assert result[0]["function"]["description"] == ""
    assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}
