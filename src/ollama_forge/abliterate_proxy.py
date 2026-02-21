"""Lightweight prompt proxy for abliterated models.

Intercepts Ollama /api/chat, formats prompts with HF tokenizer, forwards to Ollama /api/generate.
Ollama does all inference; we only handle correct template formatting and tool support.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Iterator

# Tokenizer cache size (configurable via OLLAMA_PROXY_TOKENIZER_CACHE_SIZE for multi-model setups)
_tokenizer_cache_maxsize = max(1, int(os.environ.get("OLLAMA_PROXY_TOKENIZER_CACHE_SIZE", "4")))


def _get_ollama_base() -> str:
    """Return Ollama base URL from OLLAMA_PROXY_TARGET / OLLAMA_HOST or default."""
    from ollama_forge.http_util import normalize_base_url

    host = os.environ.get("OLLAMA_PROXY_TARGET") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    return normalize_base_url(host)


@lru_cache(maxsize=_tokenizer_cache_maxsize)
def _load_tokenizer(checkpoint_dir: str):
    """Load and cache HF tokenizer from checkpoint."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    return tokenizer


def _ensure_chat_template(tokenizer, checkpoint_dir: Path) -> bool:
    """Load chat_template from checkpoint if tokenizer doesn't have one."""
    if getattr(tokenizer, "chat_template", None):
        return True
    for fname in ("tokenizer_config.json", "chat_template.jinja"):
        path = checkpoint_dir / fname
        if not path.is_file():
            continue
        try:
            if fname.endswith(".jinja"):
                tokenizer.chat_template = path.read_text(encoding="utf-8")
                return True
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("chat_template", "chat_template_jinja"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    tokenizer.chat_template = val
                    return True
                if isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, dict):
                        for sub in ("template", "content"):
                            if isinstance(first.get(sub), str):
                                tokenizer.chat_template = first[sub]
                                return True
        except Exception:
            continue
    return False


def _normalize_message(m: dict) -> dict:
    """Convert Ollama message to HF format."""
    out = {"role": m.get("role", "user"), "content": m.get("content") or ""}
    if m.get("role") == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = []
        for tc in m["tool_calls"]:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments")
            if isinstance(args, dict):
                args = json.dumps(args)
            out["tool_calls"].append(
                {
                    "type": "function",
                    "function": {"name": name, "arguments": args or "{}"},
                }
            )
    if m.get("role") == "tool":
        out["name"] = m.get("name") or ""
        out["tool_call_id"] = m.get("tool_call_id") or ""
    return out


def _extract_json_object(text: str, start: int = 0) -> str | None:
    """Extract a balanced JSON object starting at position start."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_tool_calls(text: str) -> list[dict] | None:
    """Extract tool calls from model output. Returns list of Ollama-format tool_calls or None."""
    tool_calls = []
    # Patterns that identify tool call markers (we'll extract JSON separately)
    marker_patterns = [
        (r"<tool_call>\s*", r"\s*</tool_call>"),
        (r"<function_call>\s*", r"\s*</function_call>"),
        (r"\[TOOL_CALL\]\s*", r""),
        (r"```json\s*", r"\s*```"),
        (r"```\s*", r"\s*```"),
    ]

    # Try marker-based extraction with balanced brace parsing
    for start_pattern, _end_pattern in marker_patterns:
        for match in re.finditer(start_pattern, text, re.IGNORECASE):
            json_start = match.end()
            if json_start < len(text) and text[json_start] == "{":
                json_str = _extract_json_object(text, json_start)
                if json_str:
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict) and "name" in data:
                            name = data["name"]
                            args = data.get("arguments") or data.get("parameters") or {}
                            if isinstance(args, str):
                                with contextlib.suppress(json.JSONDecodeError):
                                    args = json.loads(args)
                            tool_calls.append(
                                {
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": args if isinstance(args, dict) else {},
                                    },
                                }
                            )
                    except json.JSONDecodeError:
                        continue

    if tool_calls:
        return tool_calls

    # Fallback: try legacy regex patterns for simpler cases
    legacy_patterns = [
        r"<tool_call>\s*(\{[^}]*\})\s*</tool_call>",
        r"<function_call>\s*(\{[^}]*\})\s*</function_call>",
    ]
    for pattern in legacy_patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    name = data["name"]
                    args = data.get("arguments") or data.get("parameters") or {}
                    if isinstance(args, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            args = json.loads(args)
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args if isinstance(args, dict) else {},
                            },
                        }
                    )
            except json.JSONDecodeError:
                continue
    if not tool_calls:
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments") or data.get("parameters") or {}
                if isinstance(args, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        args = json.loads(args)
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": args if isinstance(args, dict) else {},
                        },
                    }
                )
        except json.JSONDecodeError:
            pass
    return tool_calls if tool_calls else None


def format_prompt_with_hf(
    checkpoint_dir: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> str:
    """Format messages into a prompt string using HF tokenizer."""
    tokenizer = _load_tokenizer(checkpoint_dir)
    _ensure_chat_template(tokenizer, Path(checkpoint_dir))

    hf_messages = [_normalize_message(m) for m in messages]
    from ollama_forge.chat_util import ollama_tools_to_hf

    hf_tools = ollama_tools_to_hf(tools)

    apply_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if hf_tools and hasattr(tokenizer, "apply_chat_template"):
        try:
            test_result = tokenizer.apply_chat_template(
                hf_messages[:1],
                tools=hf_tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            if test_result:
                apply_kwargs["tools"] = hf_tools
        except Exception:
            pass

    try:
        prompt = tokenizer.apply_chat_template(hf_messages, **apply_kwargs)
        if isinstance(prompt, list):
            prompt = tokenizer.decode(prompt, skip_special_tokens=False)
        return prompt
    except Exception as e:
        print(f"Warning: apply_chat_template failed ({e}), falling back to simple format", file=sys.stderr)
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant:"


def _forward_to_ollama(
    endpoint: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict | None = None,
    stream: bool = False,
) -> tuple[int, dict, bytes | Iterator[bytes]]:
    """Forward request to Ollama and return (status, headers, body_or_iterator)."""
    base = _get_ollama_base()
    url = f"{base}{endpoint}"

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)

    try:
        resp = urllib.request.urlopen(req, timeout=300)
        resp_headers = dict(resp.headers)
        if stream:

            def stream_iter():
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    yield chunk

            return resp.status, resp_headers, stream_iter()
        return resp.status, resp_headers, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:
        error_body = json.dumps({"error": str(e)}).encode()
        return 500, {}, error_body


def _call_ollama_generate(
    model: str,
    prompt: str,
    stream: bool = False,
    options: dict | None = None,
) -> tuple[int, dict, bytes | Iterator[bytes]]:
    """Call Ollama /api/generate with pre-formatted prompt."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "raw": True,
    }
    if options:
        body["options"] = options

    return _forward_to_ollama(
        "/api/generate",
        method="POST",
        body=json.dumps(body).encode(),
        stream=stream,
    )


class ProxyConfig:
    """Configuration for the prompt proxy."""

    def __init__(self):
        self.model_checkpoints: dict[str, str] = {}
        self.default_checkpoint: str | None = None

    def add_model(self, model_name: str, checkpoint_dir: str):
        """Register a model name with its checkpoint directory."""
        self.model_checkpoints[model_name] = checkpoint_dir
        if self.default_checkpoint is None:
            self.default_checkpoint = checkpoint_dir

    def get_checkpoint(self, model_name: str) -> str | None:
        """Get checkpoint dir for model, or default if not found."""
        return self.model_checkpoints.get(model_name) or self.default_checkpoint


_proxy_config = ProxyConfig()


class PromptProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that proxies to Ollama with HF tokenizer formatting."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        print(f"[proxy] {args[0]}", file=sys.stderr)

    def _send_json(self, status: int, data: Any):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, status: int, iterator: Iterator[bytes]):
        self.send_response(status)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in iterator:
            if chunk:
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _handle_chat(self, body: dict) -> None:
        """Handle /api/chat - format with HF tokenizer and forward to /api/generate."""
        model = body.get("model", "")
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        options = body.get("options", {})

        checkpoint = _proxy_config.get_checkpoint(model)
        if not checkpoint:
            self._send_json(
                400,
                {"error": f"No checkpoint registered for model {model!r}. Use --checkpoint or register with proxy."},
            )
            return

        start_time = time.time()
        prompt = format_prompt_with_hf(checkpoint, messages, tools)
        format_time = time.time() - start_time

        status, headers, resp = _call_ollama_generate(model, prompt, stream=stream, options=options)

        if status != 200:
            if isinstance(resp, bytes):
                try:
                    error_data = json.loads(resp)
                except json.JSONDecodeError:
                    error_data = {"error": resp.decode("utf-8", errors="replace")}
            else:
                # Streaming error - consume iterator to get error message
                error_bytes = b"".join(resp)
                try:
                    error_data = json.loads(error_bytes)
                except json.JSONDecodeError:
                    error_data = {"error": error_bytes.decode("utf-8", errors="replace")}
            self._send_json(status, error_data)
            return

        if stream:
            self._handle_chat_stream(resp, model, tools, format_time)
        else:
            self._handle_chat_nonstream(resp, model, tools, format_time)

    def _handle_chat_nonstream(self, resp_body: bytes, model: str, tools: list | None, format_time: float):
        """Convert /api/generate response to /api/chat format (non-streaming)."""
        try:
            data = json.loads(resp_body)
        except json.JSONDecodeError:
            self._send_json(500, {"error": "Invalid response from Ollama"})
            return

        response_text = data.get("response", "")

        message: dict[str, Any] = {
            "role": "assistant",
            "content": response_text,
        }

        if tools:
            tool_calls = _parse_tool_calls(response_text)
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = ""

        result = {
            "model": model,
            "created_at": data.get("created_at", ""),
            "message": message,
            "done": True,
            "done_reason": data.get("done_reason", "stop"),
            "total_duration": data.get("total_duration", 0),
            "load_duration": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
        }

        self._send_json(200, result)

    def _handle_chat_stream(self, resp_iter: Iterator[bytes], model: str, tools: list | None, format_time: float):
        """Convert /api/generate stream to /api/chat stream format."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        full_response = ""
        buffer = b""

        for chunk in resp_iter:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("response", "")
                full_response += token
                done = data.get("done", False)

                chat_chunk: dict[str, Any] = {
                    "model": model,
                    "created_at": data.get("created_at", ""),
                    "message": {"role": "assistant", "content": token},
                    "done": done,
                }

                if done:
                    chat_chunk["done_reason"] = data.get("done_reason", "stop")
                    chat_chunk["total_duration"] = data.get("total_duration", 0)
                    chat_chunk["load_duration"] = data.get("load_duration", 0)
                    chat_chunk["prompt_eval_count"] = data.get("prompt_eval_count", 0)
                    chat_chunk["eval_count"] = data.get("eval_count", 0)

                    if tools:
                        tool_calls = _parse_tool_calls(full_response)
                        if tool_calls:
                            chat_chunk["message"]["tool_calls"] = tool_calls

                out = json.dumps(chat_chunk).encode() + b"\n"
                self.wfile.write(f"{len(out):x}\r\n".encode())
                self.wfile.write(out)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

        self.wfile.write(b"0\r\n\r\n")

    def _proxy_passthrough(self, method: str):
        """Forward request directly to Ollama."""
        body = self._read_body() if method in ("POST", "PUT", "PATCH") else None
        headers = dict(self.headers.items())

        status, resp_headers, resp_body = _forward_to_ollama(
            self.path,
            method=method,
            body=body,
            headers=headers,
        )

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        if isinstance(resp_body, bytes):
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        else:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in resp_body:
                if chunk:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

    def do_GET(self):
        if self.path == "/" or self.path == "/api/tags":
            # Health / tags: return 200 and list of registered models (no Ollama call)
            models = [{"name": n} for n in _proxy_config.model_checkpoints]
            self._send_json(200, {"models": models})
            return
        self._proxy_passthrough("GET")

    def do_HEAD(self):
        self._proxy_passthrough("HEAD")

    def do_DELETE(self):
        self._proxy_passthrough("DELETE")

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                body = json.loads(self._read_body())
                self._handle_chat(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
            except Exception as e:
                print(f"Error in /api/chat: {e}", file=sys.stderr)
                self._send_json(500, {"error": str(e)})
        else:
            self._proxy_passthrough("POST")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_proxy(
    checkpoint_dir: str | None = None,
    model_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 11436,
    ollama_target: str | None = None,
    models: list[tuple[str, str]] | None = None,
) -> None:
    """
    Run the prompt proxy server.

    Register one model (checkpoint_dir + model_name) or multiple via models list.

    Args:
        checkpoint_dir: Path to abliterated checkpoint (single-model mode)
        model_name: Model name to intercept (single-model mode; default derived from checkpoint dir)
        host: Bind host
        port: Bind port (default 11436 to not conflict with Ollama on 11434)
        ollama_target: Ollama base URL to forward to (default: OLLAMA_HOST or localhost:11434)
        models: Optional list of (model_name, checkpoint_dir) for multi-model; if set, checkpoint_dir/model_name ignored
    """
    if models:
        pairs = [(n, str(Path(p).resolve())) for n, p in models]
    else:
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir or models is required")
        checkpoint_dir = str(Path(checkpoint_dir).resolve())
        if not Path(checkpoint_dir).is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
        if model_name is None:
            cp = Path(checkpoint_dir)
            model_name = cp.parent.name.replace("abliterate-", "") if cp.name == "checkpoint" else cp.name
        pairs = [(model_name, checkpoint_dir)]

    for name, cdir in pairs:
        if not Path(cdir).is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {cdir}")
        _proxy_config.add_model(name, cdir)

    if ollama_target:
        os.environ["OLLAMA_PROXY_TARGET"] = ollama_target

    for name, cdir in pairs:
        print(f"Loading tokenizer for {name!r} from {cdir}...", file=sys.stderr)
        try:
            tokenizer = _load_tokenizer(cdir)
            print(f"  {name}: {type(tokenizer).__name__}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not pre-load tokenizer for {name!r}: {e}", file=sys.stderr)

    server = ThreadedHTTPServer((host, port), PromptProxyHandler)
    target = _get_ollama_base()

    print(f"Prompt proxy listening on http://{host}:{port}", file=sys.stderr)
    for name, cdir in pairs:
        print(f"  Model: {name!r} -> {cdir}", file=sys.stderr)
    print(f"  Forwarding to Ollama at: {target}", file=sys.stderr)
    print(f"  Set OLLAMA_HOST=http://{host}:{port} for agents", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy...", file=sys.stderr)
        server.shutdown()
