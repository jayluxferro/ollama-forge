"""Shared HTTP utilities (URL normalization, etc.)."""

from __future__ import annotations


def normalize_base_url(url: str) -> str:
    """
    Normalize a base URL: strip whitespace, remove trailing slash, add http:// if no scheme.
    Used by proxy and security_eval to treat Ollama base URLs consistently.
    """
    base = url.strip().rstrip("/")
    if not base.startswith("http://") and not base.startswith("https://"):
        base = "http://" + base
    return base
