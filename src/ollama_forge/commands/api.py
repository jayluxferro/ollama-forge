"""forge api — unified OpenAI-compatible server (subprocess supervisor + reverse proxy)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _default_budget_mb() -> int:
    """Read iogpu.wired_limit_mb from sysctl; fall back to 75% of RAM."""
    try:
        import subprocess as _sp
        raw = _sp.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip()
        return int(raw)
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1024 / 1024 * 0.75)
    except Exception:
        return 8192


def _cmd_api(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    # Import heavy deps inside the handler so the CLI starts fast when forge api
    # is not being used
    import uvicorn

    from ollama_forge.server.app import create_app
    from ollama_forge.server.backends import (
        LlamaCppBackend,
        MlxLmBackend,
        MlxVlmBackend,
        OllamaBackend,
    )
    from ollama_forge.server.memory import MemoryTracker
    from ollama_forge.server.registry import ModelRegistry

    budget_mb = getattr(args, "budget_mb", None) or _default_budget_mb()
    kv_bits: float | None = getattr(args, "kv_bits", None)
    ollama_host = getattr(args, "ollama_host", None) or "http://127.0.0.1:11434"

    registry = ModelRegistry()
    backends: dict = {}

    for model_spec in getattr(args, "models", []) or []:
        if "=" in model_spec:
            model_id, path_str = model_spec.split("=", 1)
        else:
            p = Path(model_spec)
            model_id = p.stem
            path_str = model_spec
        path = Path(path_str)
        rec = registry.register(model_id, path)
        if rec.kind == "llama_cpp":
            backends[model_id] = LlamaCppBackend(model_id, path)
        elif rec.kind == "mlx_vlm":
            backends[model_id] = MlxVlmBackend(model_id, path, kv_bits=kv_bits)
        elif rec.kind == "mlx_lm":
            backends[model_id] = MlxLmBackend(model_id, path)

    tracker = MemoryTracker(budget_mb=budget_mb)
    ollama_backend = OllamaBackend(ollama_host)
    app = create_app(registry, tracker, backends, ollama_backend)

    print(
        f"forge api  http://{args.host}:{args.port}  budget={budget_mb}MB  ollama={ollama_host}",
        file=sys.stderr,
    )
    if backends:
        for mid, _b in backends.items():
            rec = registry.get(mid)
            print(f"  {mid}  ({rec.kind}  ~{rec.estimated_mb}MB)", file=sys.stderr)
    else:
        print("  no local models pre-registered; all requests proxy to Ollama", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the 'api' subcommand; returns its parser."""
    p = subparsers.add_parser(
        "api",
        help=(
            "Start the unified forge API server "
            "(OpenAI-compatible; routes to llama.cpp / mlx-lm / mlx-vlm / Ollama)"
        ),
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Bind port (default: 11435 — avoids Ollama's 11434)",
    )
    p.add_argument(
        "--budget-mb",
        type=int,
        default=None,
        dest="budget_mb",
        help="GPU memory budget in MB (default: iogpu.wired_limit_mb from sysctl, or 75%% of RAM)",
    )
    p.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        dest="kv_bits",
        help="KV cache quantization bits passed to mlx_vlm.server (e.g. 3.5 for TurboQuant)",
    )
    p.add_argument(
        "--ollama-host",
        default=None,
        dest="ollama_host",
        help="Ollama server base URL (default: http://127.0.0.1:11434)",
    )
    p.add_argument(
        "models",
        nargs="*",
        metavar="[NAME=]MODEL_PATH",
        help=(
            "Local models to pre-register. "
            "Format: /path/to/model.gguf or mymodel=/path/to/mlx-dir. "
            "Kind is detected automatically (.gguf → llama.cpp, MLX dir → mlx_lm/mlx_vlm)."
        ),
    )
    p.set_defaults(handler=_cmd_api)
    return p
