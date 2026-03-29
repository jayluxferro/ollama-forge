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
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from ollama_forge.turboquant_text import ReasoningScrubber, clean_generated_text

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

    if tokenizer is not None:
        value = getattr(tokenizer, "model_max_length", None)
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


# ---------------------------------------------------------------------------
# Server core — holds model, tokenizer, and the backend's generate function
# ---------------------------------------------------------------------------

class TurboQuantServer:
    """Holds the loaded model and tokenizer for request handling."""

    def __init__(self, model: Any, tokenizer: Any, model_name: str,
                 generate_fn: Callable, gen_config_cls: type):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.lock = Lock()
        self._generate_fn = generate_fn
        self._gen_config_cls = gen_config_cls

    def _encode_messages(self, messages: list[dict]) -> list[int]:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            try:
                ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True,
                )
                # Handle dict-like BatchEncoding from newer transformers
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                return ids
            except Exception:
                pass
        # Fallback: simple concatenation
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
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
        )

    def chat_completion(self, messages: list[dict], **kwargs) -> tuple[str, list[int]]:
        input_ids = self._encode_messages(messages)
        gen_cfg = self._make_gen_config(kwargs, prompt_len=len(input_ids))
        tokens = []
        with self.lock:
            for tok in self._generate_fn(self.model, input_ids, gen_cfg, self.tokenizer):
                tokens.append(tok)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return clean_generated_text(text, self.tokenizer), tokens

    def chat_completion_stream(self, messages: list[dict], **kwargs):
        input_ids = self._encode_messages(messages)
        gen_cfg = self._make_gen_config(kwargs, prompt_len=len(input_ids))
        scrubber = ReasoningScrubber()
        with self.lock:
            for tok in self._generate_fn(self.model, input_ids, gen_cfg, self.tokenizer):
                piece = self.tokenizer.decode([tok], skip_special_tokens=False)
                visible = scrubber.feed(piece, self.tokenizer)
                if visible:
                    yield visible, None
        tail = scrubber.finalize(self.tokenizer)
        if tail:
            yield tail, None
        yield "", "stop"

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
            "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty",
        ) if k in body}

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            for piece, finish in srv.chat_completion_stream(messages, **params):
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": srv.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece} if finish is None else {},
                        "finish_reason": finish,
                    }],
                }
                self._send_sse(json.dumps(chunk))
            self._send_sse("[DONE]")
            self.wfile.flush()
        else:
            text, tokens = srv.chat_completion(messages, **params)
            self._send_json({
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": srv.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
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
