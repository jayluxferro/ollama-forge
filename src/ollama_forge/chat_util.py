"""Shared chat/tool conversion utilities for Ollama API compatibility."""

from __future__ import annotations


def ollama_tools_to_hf(tools: list[dict] | None) -> list[dict] | None:
    """
    Convert Ollama tools format to the format HF apply_chat_template expects:
    list of {type: "function", function: {name, description, parameters}}.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        out.append({
            "type": "function",
            "function": {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return out if out else None
