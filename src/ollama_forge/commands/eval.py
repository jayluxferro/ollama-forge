"""forge eval — tool-call correctness mini-eval."""
from __future__ import annotations

import argparse
import sys


def _cmd_eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.agentic.eval import format_results, run_eval

    model = args.model
    base_url = args.base_url

    # Load custom suite from YAML if provided
    cases = None
    if getattr(args, "suite", None) and args.suite != "builtins":
        from pathlib import Path

        import yaml

        from ollama_forge.agentic.eval import ToolCallCase
        raw = yaml.safe_load(Path(args.suite).read_text())
        cases = [
            ToolCallCase(
                name=c["name"],
                prompt=c["prompt"],
                tools=c["tools"],
                expected_fn=c["expected_fn"],
                expected_arg_keys=c.get("expected_arg_keys", []),
                system=c.get("system"),
            )
            for c in (raw.get("cases") or raw)
        ]

    print(f"Running tool-call eval: model={model} base_url={base_url}", file=sys.stderr)
    results = run_eval(model, base_url=base_url, cases=cases, timeout=args.timeout)
    print(format_results(results, model))

    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "eval",
        help="Run the tool-call correctness mini-eval against an OpenAI-compatible endpoint",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model ID to test (e.g. llama3.2 or llama3.2:3b)",
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:11435",
        dest="base_url",
        help="OpenAI-compatible API base URL (default: forge api at 11435)",
    )
    p.add_argument(
        "--suite",
        default="builtins",
        help="Test suite: 'builtins' (default) or path to a YAML file",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-case timeout in seconds (default: 60)",
    )
    p.set_defaults(handler=_cmd_eval)
    return p
