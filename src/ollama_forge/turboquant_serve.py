"""TurboQuant API server.

OpenAI-compatible HTTP server for TurboQuant models.
Provides /v1/chat/completions, /v1/completions, and /v1/models endpoints
with SSE streaming support.

Usage:
    ollama-forge turboquant serve ./model.tqf --port 8000
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from ollama_forge.abliterate_proxy import _normalize_message, _parse_tool_calls
from ollama_forge.chat_util import ollama_tools_to_hf
from ollama_forge.turboquant_text import ReasoningScrubber, _boundary_markers, clean_generated_text

# ---------------------------------------------------------------------------
# Generation config — framework-independent dataclass so we don't need
# to import torch or mlx at module level.
# ---------------------------------------------------------------------------

@dataclass
class _GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_tokens: list[int] | None = None
    stop_token_sequences: list[list[int]] | None = None


def _build_gen_config(kwargs: dict, tokenizer: Any, default_max_tokens: int = 512) -> _GenConfig:
    """Build generation config from OpenAI-style request params."""
    stop_tokens = []
    if tokenizer is not None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_tokens.append(eos)

    return _GenConfig(
        max_new_tokens=kwargs.get("max_tokens", default_max_tokens),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        top_k=kwargs.get("top_k", 50),
        repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        stop_tokens=stop_tokens or None,
    )


def _build_stop_token_sequences(tokenizer: Any) -> list[list[int]]:
    """Encode common chat boundary markers as stop sequences."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok is None or not hasattr(tok, "encode"):
        return []

    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for marker in _boundary_markers(tok):
        try:
            ids = tok.encode(marker, add_special_tokens=False)
        except TypeError:
            try:
                ids = tok.encode(marker)
            except Exception:
                continue
        except Exception:
            continue
        if not ids:
            continue
        key = tuple(int(i) for i in ids)
        if key not in seen:
            seen.add(key)
            sequences.append(list(key))
    return sequences


def _infer_context_window(model: Any, tokenizer: Any = None) -> int | None:
    """Best-effort max context window for a loaded model/tokenizer."""
    candidates: list[int] = []

    cfg = getattr(getattr(model, "hf_model", model), "config", None)
    for maybe_cfg in (cfg, getattr(cfg, "text_config", None)):
        if maybe_cfg is None:
            continue
        for attr in ("max_position_embeddings", "model_max_length", "max_sequence_length", "n_positions"):
            value = getattr(maybe_cfg, attr, None)
            if isinstance(value, int) and 0 < value < 10_000_000:
                candidates.append(value)

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok is not None:
        value = getattr(tok, "model_max_length", None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)

    return min(candidates) if candidates else None


def _resolve_default_max_tokens(model: Any, tokenizer: Any, prompt_len: int, requested: int | None) -> int:
    """Choose max_new_tokens from user request or remaining context budget."""
    if requested is not None:
        return max(int(requested), 1)

    context_window = _infer_context_window(model, tokenizer)
    if context_window is None:
        return 2048

    remaining = context_window - int(prompt_len)
    return max(remaining, 1)


def _format_openai_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Convert parsed tool calls to OpenAI-compatible response objects."""
    formatted: list[dict] = []
    for idx, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function") or {}
        arguments = fn.get("arguments") or {}
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        formatted.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": fn.get("name") or "",
                    "arguments": arguments,
                },
            }
        )
    return formatted


# ---------------------------------------------------------------------------
# Server core — holds model, tokenizer, and the backend's generate function
# ---------------------------------------------------------------------------

class TurboQuantServer:
    """Holds the loaded model and tokenizer for request handling."""

    def __init__(self, model: Any, tokenizer: Any, model_name: str,
                 generate_fn: Callable, gen_config_cls: type):
        self.model = model
        self.io = tokenizer
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.model_name = model_name
        self.lock = Lock()
        self._generate_fn = generate_fn
        self._gen_config_cls = gen_config_cls

    def _has_multimodal_content(self, messages: list[dict]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"image", "image_url", "video", "audio"}:
                        return True
        return False

    def _flatten_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            kind = item.get("type")
            if kind == "text":
                parts.append(item.get("text", ""))
            elif kind in {"image", "image_url"}:
                parts.append("[image]")
            elif kind == "video":
                parts.append("[video]")
            elif kind == "audio":
                parts.append("[audio]")
        return "".join(parts)

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for msg in messages:
            clone = dict(msg)
            if not isinstance(clone.get("content"), list):
                clone["content"] = clone.get("content") or ""
            normalized.append(_normalize_message(clone))
        return normalized

    def _prompt_length(self, encoded: Any) -> int:
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
            if hasattr(input_ids, "shape"):
                return int(input_ids.shape[-1])
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                return len(input_ids[0])
            return len(input_ids or [])
        return len(encoded)

    def _supports_tools(self, tools: list[dict] | None) -> list[dict] | None:
        hf_tools = ollama_tools_to_hf(tools)
        if not hf_tools or not hasattr(self.io, "apply_chat_template"):
            return None
        try:
            self.io.apply_chat_template(
                [{"role": "user", "content": "ping"}],
                tools=hf_tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            return hf_tools
        except Exception:
            return None

    def _encode_messages(self, messages: list[dict], tools: list[dict] | None = None) -> list[int] | dict[str, Any]:
        normalized = self._normalize_messages(messages)
        hf_tools = self._supports_tools(tools)
        is_multimodal = self._has_multimodal_content(normalized)

        if hasattr(self.io, "apply_chat_template") and getattr(self.io, "chat_template", None):
            try:
                apply_kwargs: dict[str, Any] = {"add_generation_prompt": True}
                if hf_tools:
                    apply_kwargs["tools"] = hf_tools
                if hasattr(self.io, "tokenizer"):
                    encoded = self.io.apply_chat_template(
                        normalized,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        **apply_kwargs,
                    )
                    return dict(encoded)
                ids = self.io.apply_chat_template(normalized, **apply_kwargs)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                return ids
            except Exception:
                if is_multimodal:
                    raise
        # Fallback: simple concatenation
        text = ""
        for msg in normalized:
            role = msg.get("role", "user")
            content = self._flatten_content(msg.get("content", ""))
            text += f"<|{role}|>\n{content}\n"
        text += "<|assistant|>\n"
        return self.tokenizer.encode(text)

    def _make_gen_config(self, kwargs: dict, prompt_len: int):
        """Build a GenerationConfig for the active backend."""
        stop_tokens = []
        if self.tokenizer is not None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None:
                stop_tokens.append(eos)
        max_new_tokens = _resolve_default_max_tokens(
            self.model,
            self.tokenizer,
            prompt_len=prompt_len,
            requested=kwargs.get("max_tokens"),
        )
        return self._gen_config_cls(
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            stop_tokens=stop_tokens or None,
            stop_token_sequences=_build_stop_token_sequences(self.tokenizer),
        )

    def chat_completion(self, messages: list[dict], **kwargs) -> tuple[str, list[int], list[dict] | None, str]:
        input_ids = self._encode_messages(messages, kwargs.get("tools"))
        gen_cfg = self._make_gen_config(kwargs, prompt_len=self._prompt_length(input_ids))
        tokens = []
        with self.lock:
            for tok in self._generate_fn(self.model, input_ids, gen_cfg, self.tokenizer):
                tokens.append(tok)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        cleaned = clean_generated_text(text, self.tokenizer)
        tool_calls = _parse_tool_calls(cleaned) if kwargs.get("tools") else None
        if tool_calls:
            cleaned = ""
            return cleaned, tokens, tool_calls, "tool_calls"
        # Detect truncation: if we generated exactly max_new_tokens, it was likely
        # a length-limited stop rather than a natural stop token.
        finish_reason = "length" if len(tokens) >= gen_cfg.max_new_tokens else "stop"
        return cleaned, tokens, tool_calls, finish_reason

    def chat_completion_stream(self, messages: list[dict], **kwargs):
        input_ids = self._encode_messages(messages, kwargs.get("tools"))
        gen_cfg = self._make_gen_config(kwargs, prompt_len=self._prompt_length(input_ids))
        if kwargs.get("tools"):
            tokens = []
            with self.lock:
                for tok in self._generate_fn(self.model, input_ids, gen_cfg, self.tokenizer):
                    tokens.append(tok)
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            cleaned = clean_generated_text(text, self.tokenizer)
            tool_calls = _parse_tool_calls(cleaned)
            if tool_calls:
                yield {"tool_calls": _format_openai_tool_calls(tool_calls), "content": ""}, "tool_calls"
                return
            finish = "length" if len(tokens) >= gen_cfg.max_new_tokens else "stop"
            yield {"content": cleaned}, finish
            return

        scrubber = ReasoningScrubber()
        n_tokens = 0
        with self.lock:
            for tok in self._generate_fn(self.model, input_ids, gen_cfg, self.tokenizer):
                n_tokens += 1
                piece = self.tokenizer.decode([tok], skip_special_tokens=False)
                visible = scrubber.feed(piece, self.tokenizer)
                if visible:
                    yield {"content": visible}, None
        tail = scrubber.finalize(self.tokenizer)
        if tail:
            yield {"content": tail}, None
        finish = "length" if n_tokens >= gen_cfg.max_new_tokens else "stop"
        yield {"content": ""}, finish

    def completion(self, prompt: str, **kwargs) -> tuple[str, list[int]]:
        input_ids = self.tokenizer.encode(prompt)
        gen_cfg = self._make_gen_config(kwargs, prompt_len=len(input_ids))
        tokens = []
        with self.lock:
            for tok in self._generate_fn(self.model, input_ids, gen_cfg, self.tokenizer):
                tokens.append(tok)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return clean_generated_text(text, self.tokenizer), tokens


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

_server_instance: TurboQuantServer | None = None


class _Handler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP handler."""

    def log_message(self, format, *args):
        pass  # silence default logging

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, data: str):
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health" or self.path == "/":
            self._send_json({"status": "ok", "model": _server_instance.model_name})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/completions":
            self._handle_completions()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_models(self):
        srv = _server_instance
        self._send_json({
            "object": "list",
            "data": [{
                "id": srv.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "turboquant",
            }],
        })

    def _handle_chat_completions(self):
        srv = _server_instance
        body = self._read_body()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        params = {k: body[k] for k in (
            "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "tools",
        ) if k in body}

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            for delta, finish in srv.chat_completion_stream(messages, **params):
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": srv.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": delta if finish is None or finish == "tool_calls" else {},
                        "finish_reason": finish,
                    }],
                }
                self._send_sse(json.dumps(chunk))
            self._send_sse("[DONE]")
            self.wfile.flush()
        else:
            text, tokens, tool_calls, finish_reason = srv.chat_completion(messages, **params)
            message = {"role": "assistant", "content": text}
            if tool_calls:
                message["tool_calls"] = _format_openai_tool_calls(tool_calls)
            self._send_json({
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": srv.model_name,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(tokens),
                    "total_tokens": len(tokens),
                },
            })

    def _handle_completions(self):
        srv = _server_instance
        body = self._read_body()
        prompt = body.get("prompt", "")
        req_id = f"cmpl-{uuid.uuid4().hex[:8]}"

        params = {k: body[k] for k in (
            "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty",
        ) if k in body}

        text, tokens = srv.completion(prompt, **params)
        self._send_json({
            "id": req_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": srv.model_name,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
                "total_tokens": len(tokens),
            },
        })


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(
    tqf_path: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8811,
    device: str = "auto",
    dtype: str = "float16",
    model_name: str | None = None,
):
    """Start the TurboQuant API server.

    Args:
        tqf_path: path to a .tqf model directory.
        host: bind address.
        port: bind port.
        device: "auto", "cuda", "mps", or "cpu".
        dtype: "float16", "bfloat16", or "float32".
        model_name: name to report in /v1/models (default: directory name).
    """
    global _server_instance

    tqf_path = Path(tqf_path)
    if model_name is None:
        model_name = tqf_path.stem

    # Auto-select best backend. The working TurboQuant path is the
    # Transformers-based KV-cache runtime, so keep serving on the PyTorch path
    # even when MLX is otherwise available.
    from ollama_forge.device import get_turboquant_backend
    backend = get_turboquant_backend(device)
    if backend == "mlx":
        print("TurboQuant serving is using the PyTorch runtime for HF cache compatibility.")
        backend = "pytorch"

    if backend == "mlx":
        print(f"Loading TurboQuant model from {tqf_path} (MLX backend) ...")
        from ollama_forge.turboquant_engine_mlx import (
            GenerationConfigMLX,
            generate_mlx,
            load_model_mlx,
        )
        model, tokenizer = load_model_mlx(tqf_path)
        generate_fn = generate_mlx
        gen_config_cls = GenerationConfigMLX
    else:
        print(f"Loading TurboQuant model from {tqf_path} (PyTorch backend) ...")
        from ollama_forge.turboquant_engine import (
            GenerationConfig,
            generate,
            load_model,
        )
        model, tokenizer = load_model(tqf_path, device=device, dtype=dtype)
        generate_fn = generate
        gen_config_cls = GenerationConfig

    if tokenizer is None:
        raise RuntimeError(
            f"No tokenizer found in {tqf_path}. "
            "Copy tokenizer files from the original HF model."
        )

    _server_instance = TurboQuantServer(model, tokenizer, model_name, generate_fn, gen_config_cls)

    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"TurboQuant server ready: http://{host}:{port}")
    print(f"  Backend: {backend}")
    print(f"  Model: {model_name}")
    print(f"  OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
    print("  Ctrl-C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()
