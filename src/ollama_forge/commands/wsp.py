"""forge wsp — weight-space programming toolkit (Phases 1–2: diagnose + locate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def _cmd_locate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.analyzer import WeightSpaceAnalyzer, find_model_path

    model_path_str = getattr(args, "model_path", None)
    model_name = args.model
    obstruction = getattr(args, "obstruction", "loops")
    output_json = getattr(args, "output_json", None)
    proj_type_filter = getattr(args, "proj_type", None)
    n_top = getattr(args, "top", 10)

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = find_model_path(model_name)
        if model_path is None:
            print(
                f"Error: cannot find local weights for '{model_name}'.\n"
                "Use --model-path to specify the directory containing .safetensors files.",
                file=sys.stderr,
            )
            return 2

    print(f"Locating {obstruction} circuits in {model_path.name}", file=sys.stderr)
    print("", file=sys.stderr)

    analyzer = WeightSpaceAnalyzer(model_path)

    steps: list[str] = []

    def progress(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    try:
        weight_map = analyzer.scan(
            proj_types=([proj_type_filter] if proj_type_filter else None),
            progress_fn=progress,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error during scan: {exc}", file=sys.stderr)
        return 1

    print()
    print(weight_map.report(obstruction=obstruction, n=n_top))

    if output_json:
        out = Path(output_json)
        out.write_text(json.dumps(weight_map.to_json(), indent=2))
        print(f"\nWeight map saved → {out}")
        print(f"Run: forge wsp sharpen {model_name} --weight-map {out}")

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

    # ── locate ────────────────────────────────────────────────────────────────
    p_locate = sub.add_parser(
        "locate",
        help="Scan model weights (Phase 2): rank MLP layers by obstruction likelihood",
    )
    p_locate.add_argument("model", help="Model name or HF repo id (e.g. mlx-community/Qwen2.5-1.5B-Instruct-4bit)")
    p_locate.add_argument(
        "--model-path",
        dest="model_path",
        metavar="PATH",
        help="Path to directory containing .safetensors files (overrides auto-discovery)",
    )
    p_locate.add_argument(
        "--obstruction",
        choices=["loops", "sycophancy", "verbosity", "format_fixation"],
        default="loops",
        help="Which obstruction type to rank for (default: loops)",
    )
    p_locate.add_argument(
        "--proj-type",
        dest="proj_type",
        choices=["gate_proj", "up_proj", "down_proj"],
        default=None,
        help="Only scan this MLP projection type (default: all three)",
    )
    p_locate.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many candidate layers to show (default: 10)",
    )
    p_locate.add_argument(
        "--output-json",
        dest="output_json",
        metavar="PATH",
        help="Save weight map to this JSON file for use with forge wsp sharpen",
    )
    p_locate.set_defaults(handler=_cmd_locate)

    p.set_defaults(handler=lambda parser, args: (p.print_help(), 0)[1])
    return p
