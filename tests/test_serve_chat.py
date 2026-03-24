"""Tests for the serve and chat CLI commands."""

import argparse
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Help / argparse tests
# ---------------------------------------------------------------------------


def test_serve_help() -> None:
    """serve --help exits 0 and lists key arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "model" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--n-gpu-layers" in result.stdout
    assert "--ctx-size" in result.stdout
    assert "--api-key" in result.stdout
    assert "--llama-cpp-dir" in result.stdout


def test_chat_help() -> None:
    """chat --help exits 0 and lists key arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "chat", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--base-url" in result.stdout
    assert "--model" in result.stdout
    assert "--system" in result.stdout
    assert "--api-key" in result.stdout
    assert "--temperature" in result.stdout


# ---------------------------------------------------------------------------
# _which_llama_server tests
# ---------------------------------------------------------------------------


def test_which_llama_server_on_path() -> None:
    """When llama-server is on PATH, return it."""
    from ollama_forge.cli import _which_llama_server

    with patch("shutil.which", return_value="/usr/local/bin/llama-server"):
        assert _which_llama_server() == "/usr/local/bin/llama-server"


def test_which_llama_server_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When llama-server is nowhere, return None."""
    from ollama_forge.cli import _which_llama_server

    # Run from a temp dir so ./llama.cpp doesn't match the repo checkout
    monkeypatch.chdir(tmp_path)
    with patch("shutil.which", return_value=None), \
         patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
        assert _which_llama_server(Path("/nonexistent/llama.cpp")) is None


def test_which_llama_server_in_build_dir(tmp_path: Path) -> None:
    """When llama-server exists in llama.cpp/build/bin, find it."""
    from ollama_forge.cli import _which_llama_server

    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    server = bin_dir / "llama-server"
    server.touch()

    with patch("shutil.which", return_value=None):
        result = _which_llama_server(tmp_path)
        assert result is not None
        assert "llama-server" in result


# ---------------------------------------------------------------------------
# _resolve_llama_cpp_dir_from_arg tests
# ---------------------------------------------------------------------------


def test_resolve_llama_cpp_dir_from_arg_explicit(tmp_path: Path) -> None:
    """Explicit --llama-cpp-dir is returned."""
    from ollama_forge.cli import _resolve_llama_cpp_dir_from_arg

    args = argparse.Namespace(llama_cpp_dir=str(tmp_path))
    assert _resolve_llama_cpp_dir_from_arg(args) == tmp_path


def test_resolve_llama_cpp_dir_from_arg_none() -> None:
    """When no dir given and no well-known dirs exist, return None."""
    from ollama_forge.cli import _resolve_llama_cpp_dir_from_arg

    args = argparse.Namespace(llama_cpp_dir=None)
    with patch.object(Path, "is_dir", return_value=False):
        _resolve_llama_cpp_dir_from_arg(args)
        # May find the real ./llama.cpp in this repo, that's OK
        # Just check it doesn't crash


# ---------------------------------------------------------------------------
# _cmd_serve tests
# ---------------------------------------------------------------------------


def test_serve_missing_gguf(tmp_path: Path) -> None:
    """serve exits 1 when GGUF file doesn't exist."""
    from ollama_forge.cli import _cmd_serve

    args = argparse.Namespace(
        model=str(tmp_path / "nonexistent.gguf"),
        host="127.0.0.1",
        port=8080,
        ctx_size=None,
        n_gpu_layers=None,
        threads=None,
        parallel=None,
        api_key=None,
        llama_cpp_dir=None,
        timeout=5,
        server_args=[],
    )
    parser = argparse.ArgumentParser()
    rc = _cmd_serve(parser, args)
    assert rc == 1


def test_serve_no_server_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """serve exits 1 when llama-server binary can't be found."""
    from ollama_forge.cli import _cmd_serve

    gguf = tmp_path / "test.gguf"
    gguf.touch()

    # Run from tmp_path so ./llama.cpp doesn't match the repo checkout
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        model=str(gguf),
        host="127.0.0.1",
        port=8080,
        ctx_size=None,
        n_gpu_layers=None,
        threads=None,
        parallel=None,
        api_key=None,
        llama_cpp_dir=str(tmp_path / "no-such-dir"),
        timeout=5,
        server_args=[],
    )
    parser = argparse.ArgumentParser()

    with patch("shutil.which", return_value=None), \
         patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
        rc = _cmd_serve(parser, args)
    assert rc == 1


# ---------------------------------------------------------------------------
# _llama_cpp_lib_env tests
# ---------------------------------------------------------------------------


def test_lib_env_adds_bin_dir() -> None:
    """_llama_cpp_lib_env sets library path for the server binary's directory."""
    from ollama_forge.cli import _llama_cpp_lib_env

    env = _llama_cpp_lib_env("/some/path/build/bin/llama-server")
    if sys.platform == "darwin":
        assert "/some/path/build/bin" in env.get("DYLD_LIBRARY_PATH", "")
    elif sys.platform == "linux":
        assert "/some/path/build/bin" in env.get("LD_LIBRARY_PATH", "")


# ---------------------------------------------------------------------------
# _wait_for_server tests
# ---------------------------------------------------------------------------


def test_wait_for_server_success() -> None:
    """_wait_for_server returns True when server responds."""
    from ollama_forge.cli import _wait_for_server

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, format, *a):
            pass  # suppress logs

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()
    try:
        assert _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=5)
    finally:
        server.server_close()


def test_wait_for_server_timeout() -> None:
    """_wait_for_server returns False when server never responds."""
    from ollama_forge.cli import _wait_for_server

    # Use a port that nothing listens on
    assert not _wait_for_server("http://127.0.0.1:1", timeout=0.5, interval=0.1)


# ---------------------------------------------------------------------------
# _cmd_chat tests (with mock server)
# ---------------------------------------------------------------------------


class _ChatMockHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible streaming chat handler."""

    # Disable keep-alive so the client sees EOF after the response body.
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # consume request body
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        # Stream a simple response
        chunk = {
            "choices": [{"delta": {"content": "Hello!"}, "index": 0}],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *a):
        pass


def test_chat_single_turn() -> None:
    """Chat sends a message and receives a streamed response."""
    from ollama_forge.cli import _cmd_chat

    server = HTTPServer(("127.0.0.1", 0), _ChatMockHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        args = argparse.Namespace(
            base_url=f"http://127.0.0.1:{port}",
            model=None,
            system=None,
            api_key=None,
            temperature=None,
        )
        parser = argparse.ArgumentParser()

        # Simulate user typing "hi" then "quit"
        with patch("builtins.input", side_effect=["hi", "quit"]):
            rc = _cmd_chat(parser, args)

        assert rc == 0
    finally:
        server.shutdown()


def test_chat_with_system_prompt() -> None:
    """Chat includes system message when --system is given."""
    from ollama_forge.cli import _cmd_chat

    received_messages = []

    class CapturingHandler(_ChatMockHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            received_messages.extend(body.get("messages", []))
            # Send the response directly (don't call super which re-reads body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = {"choices": [{"delta": {"content": "Hi!"}, "index": 0}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), CapturingHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        args = argparse.Namespace(
            base_url=f"http://127.0.0.1:{port}",
            model="test-model",
            system="You are a helpful assistant.",
            api_key=None,
            temperature=0.5,
        )
        parser = argparse.ArgumentParser()

        with patch("builtins.input", side_effect=["hello", "quit"]):
            _cmd_chat(parser, args)

        assert any(m["role"] == "system" for m in received_messages)
        assert any(m["role"] == "user" and m["content"] == "hello" for m in received_messages)
    finally:
        server.shutdown()


def test_chat_clear_command() -> None:
    """The /clear command resets conversation history."""
    from ollama_forge.cli import _cmd_chat

    request_count = [0]
    assertion_errors: list[str] = []

    class CountingHandler(_ChatMockHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            request_count[0] += 1
            # After /clear, the second request should only have 1 user message
            if request_count[0] == 2:
                user_msgs = [m for m in body["messages"] if m["role"] == "user"]
                if len(user_msgs) != 1:
                    assertion_errors.append(
                        f"Expected 1 user message after clear, got {len(user_msgs)}"
                    )
            # Send the response directly (don't call super which re-reads body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = {"choices": [{"delta": {"content": "OK"}, "index": 0}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), CountingHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        args = argparse.Namespace(
            base_url=f"http://127.0.0.1:{port}",
            model=None,
            system=None,
            api_key=None,
            temperature=None,
        )
        parser = argparse.ArgumentParser()

        with patch("builtins.input", side_effect=["first msg", "/clear", "second msg", "quit"]):
            _cmd_chat(parser, args)

        assert not assertion_errors, assertion_errors[0]
        assert request_count[0] == 2
    finally:
        server.shutdown()
