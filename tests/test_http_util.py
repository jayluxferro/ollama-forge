"""Tests for http_util module."""

from ollama_forge.http_util import normalize_base_url


def test_normalize_adds_http() -> None:
    """URL without scheme gets http://."""
    assert normalize_base_url("localhost:11434") == "http://localhost:11434"
    assert normalize_base_url("127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_normalize_strips_trailing_slash() -> None:
    """Trailing slash is removed."""
    assert normalize_base_url("http://localhost:11434/") == "http://localhost:11434"


def test_normalize_preserves_https() -> None:
    """https is preserved."""
    assert normalize_base_url("https://example.com") == "https://example.com"


def test_normalize_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped."""
    assert normalize_base_url("  http://host/  ") == "http://host"
