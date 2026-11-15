"""TurboQuant command group: extreme quantization, serving, and inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ollama_forge.hf_fetch import download_adapter
from ollama_forge.run_helpers import print_actionable_error, stream_openai_chat_sse

# ---------------------------------------------------------------------------
# Cache helpers (private to this module for now; shared infra follows later)
# ---------------------------------------------------------------------------


def _ollama_forge_cache_dir() -> Path:
    return Path(os.environ.get("OLLAMA_FORGE_CACHE", Path.home() / ".cache" / "ollama-forge"))


def _find_in_forge_cache(repo_id: str) -> Path | None:
    forge_cache = _ollama_forge_cache_dir()
    if not forge_cache.is_dir():
        return None
    safe_name = repo_id.replace("/", os.sep)
    for subdir in ("models", "gguf", "checkpoints", ""):
        candidate = forge_cache / subdir / safe_name if subdir else forge_cache / safe_name
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate
    return None


def _find_in_hf_cache(repo_id: str) -> Path | None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            for revision in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
                snap = Path(revision.snapshot_path)
                if snap.is_dir() and (snap / "config.json").is_file():
                    return snap
    return None


def _resolve_turboquant_model_path(model_path: str) -> str:
    """Resolve a model identifier to a local directory path.

    Resolution order:
      1. Local directory (exists with config.json or metadata.json) → use as-is
      2. ollama-forge cache (~/.cache/ollama-forge/) → check for safetensors snapshot
      3. HF cache (~/.cache/huggingface/hub/) → check for downloaded snapshot
      4. HF repo ID → download via snapshot_download → return cached path
    """
    local = Path(model_path)
    if local.is_dir() and (local / "config.json").is_file():
        print(f"Using local checkpoint: {local}")
        return str(local)
    if "/" in str(model_path) and not local.exists():
        cached = _find_in_forge_cache(model_path)
        if cached:
            print(f"Found in ollama-forge cache: {cached}")
            return str(cached)
        cached = _find_in_hf_cache(model_path)
        if cached:
            print(f"Found in HF cache: {cached}")
            return str(cached)
        print(f"Downloading {model_path} from Hugging Face ...")
        dl_dir = Path(tempfile.mkdtemp(prefix="ollama-forge-resolve-"))
        local_path = download_adapter(model_path, local_dir=dl_dir)
        print(f"Downloaded to {local_path}")
        return str(local_path)
    return str(model_path)


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------


def _tokenize_chat(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Tokenize chat messages, falling back if no chat template is set."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            return ids
        except Exception:
            pass
    text = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        text += f"<|{role}|>\n{content}\n"
    text += "<|assistant|>\n"
    return tokenizer.encode(text)


def _turboquant_chat_remote(args: argparse.Namespace, base_url: str) -> int:
    """Chat with a running TurboQuant server via its OpenAI-compatible API."""
    import urllib.request as _urlreq

    base_url = base_url.rstrip("/")
    model = getattr(args, "model", "") or ""
    system = getattr(args, "system", None)
    temperature = getattr(args, "temperature", None)
    max_tokens = getattr(args, "max_tokens", None)

    try:
        req = _urlreq.Request(f"{base_url}/health", method="GET")
        with _urlreq.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"Warning: server returned {resp.status}", file=sys.stderr)
    except (OSError, ValueError):
        print(f"Warning: cannot reach {base_url}/health — is the server running?", file=sys.stderr)
        print("  Start with: ollama-forge turboquant serve <model.tqf>", file=sys.stderr)

    url = f"{base_url}/v1/chat/completions"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(f"Connected to {base_url}  (type 'quit' or Ctrl-C to exit)\n", file=sys.stderr)

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

            payload: dict[str, Any] = {"messages": messages, "stream": True}
            if model:
                payload["model"] = model
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            body = json.dumps(payload).encode("utf-8")
            req = _urlreq.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with _urlreq.urlopen(req, timeout=300) as resp:
                    print("Assistant: ", end="", flush=True)
                    assistant_text, finish_reason = stream_openai_chat_sse(resp)
                    print()
                    if finish_reason and finish_reason != "stop":
                        print(f"(response stopped: {finish_reason})", file=sys.stderr)
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                continue

            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    except KeyboardInterrupt:
        print("\nBye.")
    return 0


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_turboquant_quantize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Quantize a HF model using TurboQuant and save as .tqf."""
    try:
        from ollama_forge.turboquant_pipeline import TurboQuantConfig, copy_tokenizer, quantize_model
    except ImportError as exc:
        print_actionable_error(f"Missing dependency: {exc}", next_steps=["Install dependencies: uv sync"])
        return 1

    model_path = args.model
    source_model = model_path
    output = getattr(args, "output", None)
    if output is None:
        output = Path(model_path).name.replace("/", "_") + ".tqf"

    model_path = _resolve_turboquant_model_path(model_path)

    config = TurboQuantConfig(
        bits=getattr(args, "bits", 3),
        outlier_channels=getattr(args, "outlier_channels", 32),
        outlier_bits=getattr(args, "outlier_bits", 4),
        use_qjl=getattr(args, "qjl", False),
        embed_bits=getattr(args, "embed_bits", 4),
        kv_bits=getattr(args, "kv_bits", 3),
    )

    def _progress(step: int, total: int, name: str) -> None:
        pct = step * 100 // total
        print(f"\r  [{pct:3d}%] {step}/{total}  {name[:60]:<60}", end="", flush=True)

    print(f"Preparing TurboQuant package for {model_path} → {output}")
    print(
        f"  KV cache={config.kv_bits}b  residual correction={config.use_qjl}"
        f"  weight bits(metadata only)={config.bits}"
    )

    import time

    t0 = time.time()
    quant_device = getattr(args, "device", "auto")
    if quant_device == "mlx":
        quant_device = "auto"
    result = quantize_model(
        model_path,
        output,
        config,
        device=quant_device,
        progress_callback=_progress,
        source_model=source_model,
    )
    elapsed = time.time() - t0
    print()

    copy_tokenizer(model_path, output)

    s = result.stats
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Parameters: {s.original_params:,}")
    print(f"  Original:   {s.original_bytes / 1e9:.2f} GB")
    print(f"  Checkpoint: {s.compressed_bytes / 1e9:.2f} GB")
    print(f"  Ratio:      {s.compression_ratio:.1f}×")
    print(f"  Avg bits:   {s.effective_bits_avg:.2f}")
    print("  Runtime:    original HF weights + TurboQuant KV cache")
    print(f"\nSaved to {output}")
    return 0


def _cmd_turboquant_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Serve a TurboQuant model via OpenAI-compatible API."""
    try:
        from ollama_forge.turboquant_serve import serve
    except ImportError as exc:
        print_actionable_error(f"Missing dependency: {exc}", next_steps=["Install dependencies: uv sync"])
        return 1

    serve(
        args.model,
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", 8811),
        device=getattr(args, "device", "auto"),
        dtype=getattr(args, "dtype", "float16"),
        model_name=getattr(args, "name", None),
    )
    return 0


def _cmd_turboquant_info(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Show info about a .tqf model."""
    tqf_dir = Path(args.model)
    meta_path = tqf_dir / "metadata.json"
    if not meta_path.exists():
        print(f"Not a valid .tqf directory: {tqf_dir}")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    qcfg = meta.get("quant_config", {})
    stats = meta.get("stats", {})
    hf_cfg = meta.get("model_config", {})

    print(f"TurboQuant Model: {tqf_dir.name}")
    print(f"  Architecture: {hf_cfg.get('model_type', '?')}")
    print(f"  Hidden size:  {hf_cfg.get('hidden_size', '?')}")
    print(f"  Layers:       {hf_cfg.get('num_hidden_layers', '?')}")
    print(f"  Vocab:        {hf_cfg.get('vocab_size', '?')}")
    print(
        f"  Quantization: {qcfg.get('bits', '?')}-bit"
        f"  outlier={qcfg.get('outlier_channels', 0)}ch@{qcfg.get('outlier_bits', 0)}b"
        f"  qjl={qcfg.get('use_qjl', False)}"
    )
    print(f"  KV cache:     {qcfg.get('kv_bits', 0)}-bit at inference")
    print(f"  Parameters:   {stats.get('original_params', 0):,}")
    print(f"  Original:     {stats.get('original_bytes', 0) / 1e9:.2f} GB")
    print(f"  Compressed:   {stats.get('compressed_bytes', 0) / 1e9:.2f} GB")
    print(f"  Ratio:        {stats.get('compression_ratio', 0):.1f}×")
    print(f"  Avg bits:     {stats.get('effective_bits_avg', 0):.2f}")

    if getattr(args, "json", False):
        print(json.dumps(meta, indent=2))
    return 0


def _cmd_turboquant_chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Interactive chat with a TurboQuant model or a running TurboQuant server."""
    base_url = getattr(args, "base_url", None)
    if base_url:
        return _turboquant_chat_remote(args, base_url)

    try:
        from ollama_forge.turboquant_engine import GenerationConfig, generate, load_model
        from ollama_forge.turboquant_serve import _build_stop_token_sequences, _resolve_default_max_tokens
        from ollama_forge.turboquant_text import ReasoningScrubber, clean_generated_text
    except ImportError as exc:
        print_actionable_error(f"Missing dependency: {exc}", next_steps=["Install dependencies: uv sync"])
        return 1

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(
        args.model,
        device=getattr(args, "device", "auto"),
        dtype=getattr(args, "dtype", "float16"),
    )
    if tokenizer is None:
        print("Error: no tokenizer found in the .tqf directory.")
        return 1
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    messages: list[dict[str, str]] = []
    system = getattr(args, "system", None)
    if system:
        messages.append({"role": "system", "content": system})

    requested_max_tokens = getattr(args, "max_tokens", None)
    gen_cfg = GenerationConfig(
        max_new_tokens=2048,
        temperature=getattr(args, "temperature", 0.7),
        top_p=getattr(args, "top_p", 0.9),
        stop_token_sequences=_build_stop_token_sequences(text_tokenizer),
    )

    print("Ready. Type your message (Ctrl-C to quit).\n")

    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break
            if not user_input.strip():
                continue
            messages.append({"role": "user", "content": user_input})

            input_ids = _tokenize_chat(text_tokenizer, messages)
            gen_cfg.max_new_tokens = _resolve_default_max_tokens(
                model,
                tokenizer,
                prompt_len=len(input_ids),
                requested=requested_max_tokens,
            )

            print("Assistant: ", end="", flush=True)
            tokens = []
            scrubber = ReasoningScrubber()
            for tok in generate(model, input_ids, gen_cfg, tokenizer):
                tokens.append(tok)
                piece = text_tokenizer.decode([tok], skip_special_tokens=False)
                visible = scrubber.feed(piece, tokenizer)
                if visible:
                    print(visible, end="", flush=True)
            tail = scrubber.finalize(tokenizer)
            if tail:
                print(tail, end="", flush=True)
            print()
            full_text = text_tokenizer.decode(tokens, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": clean_generated_text(full_text, tokenizer)})
    except KeyboardInterrupt:
        print("\nBye.")
    return 0


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Register the ``turboquant`` subcommand group and return its parser."""
    p_tq = subparsers.add_parser(
        "turboquant",
        help="TurboQuant: extreme quantization for fast inference (no llama.cpp needed)",
    )
    tq_sub = p_tq.add_subparsers(dest="turboquant_command")

    p_tq_quant = tq_sub.add_parser("quantize", help="Quantize a HF model to TurboQuant format (.tqf)")
    p_tq_quant.add_argument("model", help="HF repo id or local path to safetensors checkpoint")
    p_tq_quant.add_argument("-o", "--output", help="Output .tqf directory (default: <model>.tqf)")
    p_tq_quant.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4],
                            help="Bits per weight (default: 3)")
    p_tq_quant.add_argument("--outlier-channels", type=int, default=32,
                            help="Number of outlier channels for mixed precision (default: 32)")
    p_tq_quant.add_argument("--outlier-bits", type=int, default=4,
                            help="Bits for outlier channels (default: 4)")
    p_tq_quant.add_argument("--embed-bits", type=int, default=4,
                            help="Bits for embedding layer (default: 4)")
    p_tq_quant.add_argument("--kv-bits", type=int, default=3,
                            help="Bits for KV cache at inference (default: 3)")
    p_tq_quant.add_argument("--qjl", action="store_true", default=False,
                            help="Enable QJL residual correction for unbiased inner products")
    p_tq_quant.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    p_tq_quant.set_defaults(handler=_cmd_turboquant_quantize)

    p_tq_serve = tq_sub.add_parser("serve", help="Serve a .tqf model via OpenAI-compatible API")
    p_tq_serve.add_argument("model", help="Path to .tqf directory")
    p_tq_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_tq_serve.add_argument("--port", type=int, default=8811, help="Bind port (default: 8811)")
    p_tq_serve.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    p_tq_serve.add_argument("--dtype", default="float16",
                            choices=["float16", "bfloat16", "float32"],
                            help="Compute dtype (default: float16)")
    p_tq_serve.add_argument("--name", default=None, help="Model name for /v1/models")
    p_tq_serve.set_defaults(handler=_cmd_turboquant_serve)

    p_tq_info = tq_sub.add_parser("info", help="Show compression stats for a .tqf model")
    p_tq_info.add_argument("model", help="Path to .tqf directory")
    p_tq_info.add_argument("--json", action="store_true", help="Print full metadata as JSON")
    p_tq_info.set_defaults(handler=_cmd_turboquant_info)

    p_tq_chat = tq_sub.add_parser("chat", help="Interactive chat with a .tqf model (no server needed)")
    p_tq_chat.add_argument("model", nargs="?", default="",
                           help="Path to .tqf directory (optional with --base-url)")
    p_tq_chat.add_argument("--base-url", default=None,
                           help="Connect to a running TurboQuant server instead of loading locally"
                                " (e.g. http://localhost:8811)")
    p_tq_chat.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    p_tq_chat.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p_tq_chat.add_argument("--max-tokens", type=int, default=None)
    p_tq_chat.add_argument("--temperature", type=float, default=0.7)
    p_tq_chat.add_argument("--top-p", type=float, default=0.9)
    p_tq_chat.add_argument("--system", default=None, help="System prompt")
    p_tq_chat.set_defaults(handler=_cmd_turboquant_chat)

    return p_tq
