"""Serve and chat command group."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ollama_forge.run_helpers import (
    _hf_cache_has_repo,
    _llama_cpp_lib_env,
    _resolve_gguf_from_forge_cache,
    _resolve_gguf_from_hf_cache,
    _resolve_llama_cpp_dir_from_arg,
    _which_llama_server,
    print_actionable_error,
    stream_openai_chat_sse,
)


def _wait_for_server(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll GET ``url`` until it responds 200 or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (OSError, urllib.error.URLError, ValueError):
            pass
        time.sleep(interval)
    return False


def _cmd_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Start llama-server to serve a GGUF model via an OpenAI-compatible API."""
    model_arg = args.model
    gguf = Path(model_arg)

    # If not a local file, try resolving from caches by repo ID
    if not gguf.is_file() and "/" in model_arg:
        # 1. Check ollama-forge GGUF cache (converted models)
        forge_cached = _resolve_gguf_from_forge_cache(model_arg)
        if forge_cached and forge_cached.is_file():
            print(f"Resolved from cache: {forge_cached}", file=sys.stderr)
            gguf = forge_cached
        # 2. Check HF cache (downloaded GGUFs)
        if not gguf.is_file():
            cached = _resolve_gguf_from_hf_cache(model_arg)
            if cached and cached.is_file():
                print(f"Resolved from HF cache: {cached}", file=sys.stderr)
                gguf = cached
        if not gguf.is_file() and _hf_cache_has_repo(model_arg):
            print_actionable_error(
                f"Repo {model_arg} is in the HF cache but has no GGUF files",
                next_steps=[
                    f"Convert to GGUF first: ollama-forge import {model_arg} --name my-model",
                    "Or fetch a GGUF repo: ollama-forge fetch <GGUF_REPO> --name my-model",
                ],
            )
            return 1

    if not gguf.is_file():
        print_actionable_error(
            f"GGUF file not found: {model_arg}",
            next_steps=[
                "Provide a valid path to a .gguf file",
                "Or use a HF repo ID that has GGUF files in the cache",
                "Download one with: ollama-forge fetch <HF_REPO>",
            ],
        )
        return 1

    llama_cpp_dir = _resolve_llama_cpp_dir_from_arg(args)
    server_bin = _which_llama_server(llama_cpp_dir)
    if not server_bin:
        print_actionable_error(
            "llama-server not found",
            next_steps=[
                "Run: ollama-forge setup-llama-cpp (builds llama-server)",
                "Or install llama.cpp and add its bin dir to PATH",
                "Or pass --llama-cpp-dir <path-to-llama.cpp-clone>",
            ],
        )
        return 1

    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = getattr(args, "port", 11434) or 11434
    ctx_size = getattr(args, "ctx_size", None)
    n_gpu_layers = getattr(args, "n_gpu_layers", None)
    threads = getattr(args, "threads", None)
    parallel = getattr(args, "parallel", None)
    api_key = getattr(args, "api_key", None)
    extra_args = getattr(args, "server_args", None) or []

    cmd: list[str] = [
        server_bin,
        "-m", str(gguf.resolve()),
        "--host", host,
        "--port", str(port),
    ]
    if ctx_size is not None:
        cmd += ["-c", str(ctx_size)]
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if parallel is not None:
        cmd += ["-np", str(parallel)]
    if api_key:
        cmd += ["--api-key", api_key]
    cmd += extra_args

    env = _llama_cpp_lib_env(server_bin)
    base_url = f"http://{host}:{port}"

    print(f"Starting llama-server: {shlex.join(cmd)}", file=sys.stderr)
    print(f"Endpoint: {base_url}/v1/chat/completions", file=sys.stderr)
    if api_key:
        print(f"API key : {api_key}", file=sys.stderr)

    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        print_actionable_error(
            f"Could not execute: {server_bin}",
            next_steps=["Rebuild llama.cpp: ollama-forge setup-llama-cpp --update"],
        )
        return 1

    health_url = f"{base_url}/health"
    timeout = getattr(args, "timeout", 60) or 60
    print(f"Waiting for server to be ready (up to {timeout}s)...", file=sys.stderr)
    if _wait_for_server(health_url, timeout=timeout):
        print(f"\nServer ready at {base_url}", file=sys.stderr)
        print(f"  Chat:       {base_url}/v1/chat/completions", file=sys.stderr)
        print(f"  Completions:{base_url}/v1/completions", file=sys.stderr)
        print(f"  Health:     {health_url}", file=sys.stderr)
        print(f"\nConnect:  ollama-forge chat --base-url {base_url}", file=sys.stderr)
        print("Press Ctrl+C to stop the server.", file=sys.stderr)
    else:
        print(
            f"\nWarning: server did not respond at {health_url} within {timeout}s. "
            "It may still be loading the model — check its output above.",
            file=sys.stderr,
        )

    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down llama-server...", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


def _cmd_chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Interactive chat session with a running llama-server (or any OpenAI-compatible endpoint)."""
    import urllib.request as _urlreq

    base_url = (getattr(args, "base_url", None) or "http://127.0.0.1:11434").rstrip("/")
    model = getattr(args, "model", None) or ""
    system = getattr(args, "system", None)
    api_key = getattr(args, "api_key", None)
    temperature = getattr(args, "temperature", None)

    # Quick health check
    health = f"{base_url}/health"
    try:
        req = _urlreq.Request(health, method="GET")
        with _urlreq.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"Warning: server at {base_url} returned {resp.status}", file=sys.stderr)
    except (OSError, ValueError):
        print(f"Warning: cannot reach {base_url}/health — is the server running?", file=sys.stderr)
        print("  Start with: ollama-forge serve <model.gguf>", file=sys.stderr)

    url = f"{base_url}/v1/chat/completions"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(f"Connected to {base_url}  (type 'quit' or Ctrl+C to exit)\n", file=sys.stderr)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
                break
            if user_input.lower() in ("/clear", "/reset"):
                messages.clear()
                if system:
                    messages.append({"role": "system", "content": system})
                print("(conversation cleared)", file=sys.stderr)
                continue

            messages.append({"role": "user", "content": user_input})

            payload: dict[str, Any] = {
                "messages": messages,
                "stream": True,
            }
            if model:
                payload["model"] = model
            if temperature is not None:
                payload["temperature"] = temperature

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = json.dumps(payload).encode("utf-8")
            req = _urlreq.Request(url, data=body, headers=headers, method="POST")

            try:
                with _urlreq.urlopen(req, timeout=300) as resp:
                    print("Assistant: ", end="", flush=True)
                    assistant_text, finish_reason = stream_openai_chat_sse(resp)
                    print()  # newline after streamed response
                    if finish_reason and finish_reason != "stop":
                        print(f"(response stopped: {finish_reason})", file=sys.stderr)
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
                # Remove the failed user message so conversation stays consistent
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                continue

            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    except KeyboardInterrupt:
        print("\n", file=sys.stderr)

    print("Bye!", file=sys.stderr)
    return 0


def register(subparsers) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Register serve and chat subcommands; returns (p_serve, p_chat)."""
    # serve (spin up llama-server with a GGUF model)
    p_serve = subparsers.add_parser(
        "serve",
        help="Start llama-server to serve a GGUF model via an OpenAI-compatible API",
    )
    p_serve.add_argument(
        "model",
        help="Path to a .gguf model file",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port to listen on (default: 11434)",
    )
    p_serve.add_argument(
        "-c", "--ctx-size",
        type=int,
        default=None,
        help="Context size in tokens (default: model default)",
    )
    p_serve.add_argument(
        "-ngl", "--n-gpu-layers",
        type=int,
        default=None,
        help="Number of layers to offload to GPU (-1 = all)",
    )
    p_serve.add_argument(
        "-t", "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto)",
    )
    p_serve.add_argument(
        "-np", "--parallel",
        type=int,
        default=None,
        help="Number of parallel request slots (default: auto)",
    )
    p_serve.add_argument(
        "--api-key",
        default=None,
        help="Require this API key for all requests",
    )
    p_serve.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Path to llama.cpp clone (for finding llama-server binary)",
    )
    p_serve.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Seconds to wait for server to become ready (default: 60)",
    )
    p_serve.add_argument(
        "server_args",
        nargs="*",
        metavar="-- ...",
        help="Extra arguments passed through to llama-server (place after --)",
    )
    p_serve.set_defaults(handler=_cmd_serve)

    # chat (interactive chat with a running llama-server or OpenAI-compatible endpoint)
    p_chat = subparsers.add_parser(
        "chat",
        help="Interactive chat with a running llama-server (or any OpenAI-compatible endpoint)",
    )
    p_chat.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Server base URL (default: http://127.0.0.1:11434)",
    )
    p_chat.add_argument(
        "--model",
        default=None,
        help="Model name to send in requests (optional; llama-server ignores this)",
    )
    p_chat.add_argument(
        "--system",
        default=None,
        help="System message for the conversation",
    )
    p_chat.add_argument(
        "--api-key",
        default=None,
        help="API key (Bearer token) if the server requires authentication",
    )
    p_chat.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (e.g. 0.7)",
    )
    p_chat.set_defaults(handler=_cmd_chat)

    return p_serve, p_chat
