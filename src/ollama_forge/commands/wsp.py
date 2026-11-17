"""forge wsp — weight-space programming toolkit (Phase 1: diagnostics)."""
from __future__ import annotations

import argparse
import sys


def _cmd_diagnose(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.diagnostics import run_diagnostics

    model = args.model
    base_url = args.base_url
    quick = getattr(args, "quick", False)
    timeout = getattr(args, "timeout", 60.0)

    print(f"Diagnosing {model}  ({base_url}){'  [quick]' if quick else ''}", file=sys.stderr)
    print("", file=sys.stderr)

    steps: list[str] = []

    def progress(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    report = run_diagnostics(
        model=model,
        base_url=base_url,
        quick=quick,
        timeout=timeout,
        progress_fn=progress,
    )

    print()
    print(report.summary())

    found = report.obstructions_found()
    if found:
        print()
        print("Suggested next steps:")
        if "loops" in found:
            print(f"  forge wsp locate {model} --obstruction loops   (Phase 2: find loop-inducing weight directions)")
        if "sycophancy" in found:
            print(f"  forge wsp locate {model} --obstruction sycophancy")
        if "verbosity" in found:
            print(f"  forge wsp locate {model} --obstruction verbosity")
        if "format_fixation" in found:
            print(f"  forge wsp locate {model} --obstruction format_fixation")
        print()
        print("  Once located:")
        print(f"  forge wsp sharpen {model} --fix {'|'.join(found)} --out {model}-sharp")

    return 1 if found else 0


def _cmd_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Lightweight benchmark runner (counts correct answers on a small fixed set)."""
    from ollama_forge.wsp.bench import run_bench

    model = args.model
    base_url = args.base_url
    tasks = getattr(args, "tasks", None) or ["math", "factual", "coding"]
    timeout = getattr(args, "timeout", 60.0)

    print(f"Benchmarking {model}  ({base_url})  tasks={tasks}", file=sys.stderr)
    results = run_bench(model=model, base_url=base_url, tasks=tasks, timeout=timeout)

    print()
    print(f"Benchmark results — {model}")
    print()
    total_correct = 0
    total_n = 0
    for task, (correct, n) in results.items():
        pct = correct / n * 100 if n else 0.0
        mark = "✓" if pct >= 60 else "⚠"
        print(f"  {task:20s}  {correct}/{n}  ({pct:.0f}%)  {mark}")
        total_correct += correct
        total_n += n
    if total_n:
        print()
        print(f"  Overall: {total_correct}/{total_n}  ({total_correct/total_n*100:.0f}%)")
    return 0


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "wsp",
        help="Weight-Space Programming toolkit — diagnose and sharpen any model",
    )
    sub = p.add_subparsers(dest="wsp_command")

    # ── diagnose ──────────────────────────────────────────────────────────────
    p_diag = sub.add_parser(
        "diagnose",
        help="Measure obstruction signals: loop rate, sycophancy, verbosity, format compliance",
    )
    p_diag.add_argument("model", help="Model name (e.g. llama3.2:3b)")
    p_diag.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        dest="base_url",
        help="OpenAI-compatible endpoint (default: Ollama at 11434)",
    )
    p_diag.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller prompt banks (faster, less accurate)",
    )
    p_diag.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    p_diag.set_defaults(handler=_cmd_diagnose)

    # ── bench ─────────────────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "bench",
        help="Lightweight benchmark: math, factual, coding tasks",
    )
    p_bench.add_argument("model", help="Model name (e.g. llama3.2:3b)")
    p_bench.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        dest="base_url",
        help="OpenAI-compatible endpoint (default: Ollama at 11434)",
    )
    p_bench.add_argument(
        "--tasks",
        nargs="+",
        choices=["math", "factual", "coding", "reasoning"],
        help="Task categories to benchmark (default: all)",
    )
    p_bench.add_argument(
        "--timeout",
        type=float,
        default=60.0,
    )
    p_bench.set_defaults(handler=_cmd_bench)

    p.set_defaults(handler=lambda parser, args: (p.print_help(), 0)[1])
    return p
