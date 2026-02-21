"""Call Ollama or abliterate serve: POST /api/chat or /api/generate."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def list_models(
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 5.0,
) -> list[str]:
    """
    Fetch available model names from Ollama or abliterate serve (GET /api/tags).
    Returns list of model names; empty list on connection error.
    """
    from ollama_forge.http_util import normalize_base_url

    base = normalize_base_url(base_url).rstrip("/")
    url = base + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError):
        return []
    models = data.get("models") or []
    names = []
    for m in models:
        if isinstance(m, dict) and m.get("name"):
            names.append(m["name"])
        elif isinstance(m, str):
            names.append(m)
    return sorted(names)


def query_model(
    prompt: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2",
    use_chat: bool = True,
    system: str | None = None,
    timeout: float = 120.0,
) -> tuple[str, float | None]:
    """
    Send one prompt to the model and return (response_text, duration_seconds).
    Uses POST /api/chat (messages) if use_chat else POST /api/generate (prompt).
    """
    from ollama_forge.http_util import normalize_base_url

    base = normalize_base_url(base_url)
    if use_chat:
        url = base + "/api/chat"
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    else:
        url = base + "/api/generate"
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "system": system,
        }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_b = e.read() if e.fp else b""
        try:
            err_msg = json.loads(body_b.decode("utf-8")).get("error", body_b.decode("utf-8"))
        except Exception:
            err_msg = str(e)
        raise RuntimeError(f"Model API error: {err_msg}") from e

    text = (data.get("message") or {}).get("content") or "" if use_chat else data.get("response") or ""
    duration = data.get("eval_duration")  # nanoseconds in Ollama
    duration_sec = duration / 1e9 if duration is not None else None
    return (text.strip(), duration_sec)


def query_model_multi_turn(
    messages: list[dict],
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2",
    system: str | None = None,
    timeout: float = 120.0,
) -> tuple[str, float | None]:
    """
    Send a multi-turn conversation to /api/chat and return (last_assistant_content, duration_seconds).
    messages: list of {"role": "user"|"assistant"|"system", "content": "..."}.
    """
    from ollama_forge.http_util import normalize_base_url

    base = normalize_base_url(base_url).rstrip("/")
    url = base + "/api/chat"
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    body = {"model": model, "messages": messages, "stream": False}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_b = e.read() if e.fp else b""
        try:
            err_msg = json.loads(body_b.decode("utf-8")).get("error", body_b.decode("utf-8"))
        except Exception:
            err_msg = str(e)
        raise RuntimeError(f"Model API error: {err_msg}") from e
    text = (data.get("message") or {}).get("content") or ""
    duration = data.get("eval_duration")
    duration_sec = duration / 1e9 if duration is not None else None
    return (text.strip(), duration_sec)


def query_model_multi_turn_iterative(
    turns: list[dict],
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2",
    system: str | None = None,
    timeout: float = 120.0,
) -> tuple[str, list[str], float | None]:
    """
    Run multi-turn step-by-step: send user messages one exchange at a time, collect each
    assistant response. Returns (last_response, list_of_all_responses, total_duration_sec).
    turns: list of {"role": "user"|"assistant", "content": "..."} in order (user, asst, user, ...).
    """
    from ollama_forge.http_util import normalize_base_url

    base = normalize_base_url(base_url).rstrip("/")
    url = base + "/api/chat"
    built: list[dict] = []
    if system:
        built.append({"role": "system", "content": system})
    responses: list[str] = []
    total_ns = 0
    i = 0
    while i < len(turns):
        msg = turns[i]
        if msg.get("role") != "user":
            i += 1
            continue
        built.append({"role": "user", "content": msg.get("content") or ""})
        body = {"model": model, "messages": built, "stream": False}
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg_out = data.get("message") or {}
        text = msg_out.get("content") or ""
        responses.append(text.strip())
        built.append({"role": "assistant", "content": text})
        dur = data.get("eval_duration")
        if dur is not None:
            total_ns += dur
        i += 1
    duration_sec = total_ns / 1e9 if total_ns else None
    return (responses[-1] if responses else "", responses, duration_sec)


def query_model_with_tools(
    prompt: str,
    tools: list[dict],
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2",
    system: str | None = None,
    timeout: float = 120.0,
    tool_choice: str | None = None,
) -> tuple[str, list[dict], float | None]:
    """
    POST /api/chat with tools. Returns (response_text, tool_calls_list, duration_sec).
    tools: Ollama/OpenAI-style list of {"type": "function", "function": {"name": "...", ...}}.
    """
    from ollama_forge.http_util import normalize_base_url

    base = normalize_base_url(base_url).rstrip("/")
    url = base + "/api/chat"
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    body = {"model": model, "messages": messages, "tools": tools, "stream": False}
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_b = e.read() if e.fp else b""
        try:
            err_msg = json.loads(body_b.decode("utf-8")).get("error", body_b.decode("utf-8"))
        except Exception:
            err_msg = str(e)
        raise RuntimeError(f"Model API error: {err_msg}") from e
    msg = data.get("message") or {}
    text = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    duration = data.get("eval_duration")
    duration_sec = duration / 1e9 if duration is not None else None
    return (text.strip(), tool_calls, duration_sec)


def query_model_with_image(
    prompt: str,
    image_base64: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2",
    system: str | None = None,
    timeout: float = 120.0,
) -> tuple[str, float | None]:
    """
    POST /api/chat with one image (Ollama: user message has "images": [base64]).
    Returns (response_text, duration_sec).
    """
    from ollama_forge.http_util import normalize_base_url

    base = normalize_base_url(base_url).rstrip("/")
    url = base + "/api/chat"
    messages = [{"role": "user", "content": prompt, "images": [image_base64]}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    body = {"model": model, "messages": messages, "stream": False}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_b = e.read() if e.fp else b""
        try:
            err_msg = json.loads(body_b.decode("utf-8")).get("error", body_b.decode("utf-8"))
        except Exception:
            err_msg = str(e)
        raise RuntimeError(f"Model API error: {err_msg}") from e
    text = (data.get("message") or {}).get("content") or ""
    duration = data.get("eval_duration")
    duration_sec = duration / 1e9 if duration is not None else None
    return (text.strip(), duration_sec)
