"""Call Ollama or abliterate serve: POST /api/chat or /api/generate."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


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
