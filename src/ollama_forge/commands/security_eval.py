"""Security-eval command group: LLM security evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from ollama_forge.run_helpers import ping_ollama, print_actionable_error


def _cmd_security_eval_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run security eval: load prompt set, query model, score, print KPIs and optionally write CSV/JSON."""
    if getattr(args, "schema", False):
        schema_text = """Prompt set schema:
- .txt: one prompt per line; lines starting with # are skipped. category=default.
- .jsonl: one JSON object per line. Required: "prompt" (or "text"). Optional:
  category, expected_refusal (bool), target_for_extraction (string to extract),
  context, system, turns (multi-turn), tools, dangerous_tool_names, image (path/URL/data URL).
See wiki or security_eval/loader.py for full field list."""
        print(schema_text)
        return 0
    try:
        from ollama_forge.security_eval.run import run_eval
    except ImportError as e:
        print_actionable_error(
            "security-eval failed to import",
            cause=str(e),
            next_steps=["Run: uv sync", "Then: ollama-forge security-eval run <prompt_set>"],
        )
        return 1
    prompt_set = getattr(args, "prompt_set", None)
    if not prompt_set:
        print_actionable_error(
            "prompt_set path required",
            next_steps=[
                "Run: ollama-forge security-eval run <path_to_.txt_or_.jsonl> [--model <name>]",
                "Example: ollama-forge security-eval run ./prompts.txt --model llama3.2",
            ],
        )
        return 1
    base_url = getattr(args, "base_url", None) or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = getattr(args, "model", "llama3.2")
    output_csv = getattr(args, "output_csv", None)
    output_json = getattr(args, "output_json", None)
    system = getattr(args, "system", None)
    use_chat = not getattr(args, "no_chat", False)
    timeout = getattr(args, "timeout", 120.0)
    verbose = not getattr(args, "quiet", False)
    if not getattr(args, "no_check_ollama", False) and not ping_ollama(base_url):
        print_actionable_error(
            "Ollama (or serve) is not reachable at " + base_url,
            next_steps=[
                "Start Ollama: ollama serve (or start abliterate serve and set OLLAMA_HOST)",
                "Or set --base-url to your Ollama/serve URL",
                "Or skip this check: ollama-forge security-eval run <path> --no-check-ollama",
            ],
        )
        return 1
    try:
        run_meta = run_eval(
            prompt_set,
            base_url=base_url,
            model=model,
            output_csv=output_csv,
            output_json=output_json,
            save_to_history=getattr(args, "save_history", False),
            use_chat=use_chat,
            system=system,
            timeout=timeout,
            verbose=verbose,
            retries=getattr(args, "retries", 2),
            max_prompts=getattr(args, "max_prompts", None),
            refusal_keywords_path=getattr(args, "refusal_keywords", None),
        )
    except FileNotFoundError as e:
        print_actionable_error(
            "prompt set file not found",
            cause=str(e),
            next_steps=[
                "Check the path to your .txt or .jsonl prompt set",
                "Run: ollama-forge security-eval run <path> --help",
            ],  # noqa: E501
        )
        return 1
    except ValueError as e:
        print_actionable_error(
            "invalid prompt set or options",
            cause=str(e),
            next_steps=[
                "Check prompt set format (one prompt per line or JSONL)",
                "Run: ollama-forge security-eval run --help",
            ],  # noqa: E501
        )
        return 1
    except Exception as e:
        print_actionable_error(
            "security-eval run failed",
            cause=str(e),
            next_steps=[
                "Ensure Ollama is running (ollama serve) or set --base-url",
                "Run: ollama-forge security-eval run --help",
            ],  # noqa: E501
        )
        return 1
    baseline_model = getattr(args, "baseline", None)
    if baseline_model:
        try:
            run_baseline = run_eval(
                prompt_set,
                base_url=base_url,
                model=baseline_model,
                output_csv=None,
                output_json=None,
                save_to_history=False,
                use_chat=use_chat,
                system=system,
                timeout=timeout,
                verbose=verbose,
                retries=getattr(args, "retries", 2),
                max_prompts=getattr(args, "max_prompts", None),
                refusal_keywords_path=getattr(args, "refusal_keywords", None),
            )
        except Exception as e:
            print(f"Baseline run failed: {e}", file=sys.stderr)
            run_baseline = None
        kpis_base = (run_baseline or {}).get("kpis") or {}
        print("\n--- Baseline KPIs ---", file=sys.stderr)
        print(f"  Model: {baseline_model}", file=sys.stderr)
        print(
            f"  ASR %: {kpis_base.get('asr_pct', 0):.1f}"
            f"  Refusal %: {kpis_base.get('refusal_rate_pct', 0):.1f}",
            file=sys.stderr,
        )
    kpis = run_meta.get("kpis") or {}
    print("\n--- KPIs ---", file=sys.stderr)
    print(f"  Total:        {kpis.get('total', 0)}", file=sys.stderr)
    print(f"  ASR %:        {kpis.get('asr_pct', 0):.1f}", file=sys.stderr)
    print(f"  Refusal %:    {kpis.get('refusal_rate_pct', 0):.1f}", file=sys.stderr)
    print(f"  Extraction %: {kpis.get('extraction_rate_pct', 0):.1f}", file=sys.stderr)
    print(f"  Errors:       {kpis.get('errors', 0)}", file=sys.stderr)
    if baseline_model and run_baseline:
        print("\n--- Comparison (baseline vs model) ---", file=sys.stderr)
        print(f"  ASR:      {kpis_base.get('asr_pct', 0):.1f}% → {kpis.get('asr_pct', 0):.1f}%", file=sys.stderr)
        print(
            f"  Refusal:  {kpis_base.get('refusal_rate_pct', 0):.1f}%"
            f" → {kpis.get('refusal_rate_pct', 0):.1f}%",
            file=sys.stderr,
        )
    if kpis.get("avg_latency_sec") is not None:
        print(f"  Avg latency:  {kpis['avg_latency_sec']:.2f}s", file=sys.stderr)
    if kpis.get("expected_refusal_accuracy_pct") is not None:
        print(f"  Expected-refusal accuracy: {kpis['expected_refusal_accuracy_pct']:.1f}%", file=sys.stderr)
    if kpis.get("benign_refusal_rate_pct") is not None:
        print(f"  Benign refusal rate: {kpis['benign_refusal_rate_pct']:.1f}%", file=sys.stderr)
    if kpis.get("error_counts"):
        print("  Error breakdown:", file=sys.stderr)
        for msg, count in sorted(kpis["error_counts"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {count}x {msg}", file=sys.stderr)
    by_cat = kpis.get("by_category") or {}
    if by_cat:
        print("  By category:", file=sys.stderr)
        for cat, v in by_cat.items():
            print(
                f"    {cat}: ASR={v.get('asr_pct', 0):.1f}% refusal={v.get('refusal_rate_pct', 0):.1f}%",
                file=sys.stderr,
            )
    return 0


def _cmd_security_eval_ui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Launch Streamlit UI for security evaluation."""
    app_dir = Path(__file__).resolve().parent.parent
    app_path = app_dir / "security_eval" / "app.py"
    if not app_path.exists():
        print_actionable_error(
            f"security-eval UI app not found at {app_path}",
            next_steps=[
                "Ensure the security_eval package is installed with app.py",
                "Run: uv sync",
            ],
        )
        return 1
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
            check=False,
        )
    except FileNotFoundError:
        print_actionable_error(
            "Streamlit not found",
            next_steps=[
                "Run: uv sync",
                "Then: ollama-forge security-eval ui",
            ],
        )
        return 1
    return 0


def _cmd_security_eval_compare(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Compare two security-eval run JSON files side-by-side."""
    path_a = Path(getattr(args, "run_a", ""))
    path_b = Path(getattr(args, "run_b", ""))
    if not path_a.is_file():
        print_actionable_error(
            "Run A file not found", cause=str(path_a), next_steps=["Use path from security-eval run --output-json"]
        )  # noqa: E501
        return 1
    if not path_b.is_file():
        print_actionable_error(
            "Run B file not found", cause=str(path_b), next_steps=["Use path from security-eval run --output-json"]
        )  # noqa: E501
        return 1
    try:
        run_a = json.loads(path_a.read_text(encoding="utf-8"))
        run_b = json.loads(path_b.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print_actionable_error(
            "Failed to load run JSON",
            cause=str(e),
            next_steps=["Ensure files are valid JSON from security-eval run --output-json"],
        )  # noqa: E501
        return 1
    kpis_a = run_a.get("kpis") or {}
    kpis_b = run_b.get("kpis") or {}
    label_a = run_a.get("model", "A") + " @ " + (run_a.get("timestamp_iso", "")[:10] or "?")
    label_b = run_b.get("model", "B") + " @ " + (run_b.get("timestamp_iso", "")[:10] or "?")
    print("\n--- Compare ---", file=sys.stderr)
    print(f"  {'KPI':<28} {label_a[:24]:<24} {label_b[:24]:<24}", file=sys.stderr)
    print("  " + "-" * 76, file=sys.stderr)
    for key, name in [
        ("total", "Total"),
        ("asr_pct", "ASR %"),
        ("refusal_rate_pct", "Refusal %"),
        ("extraction_rate_pct", "Extraction %"),
        ("errors", "Errors"),
        ("avg_latency_sec", "Avg latency (s)"),
        ("expected_refusal_accuracy_pct", "Expected-refusal acc %"),
        ("benign_refusal_rate_pct", "Benign refusal %"),
    ]:
        va = kpis_a.get(key)
        vb = kpis_b.get(key)
        sa = (
            f"{va:.1f}"
            if isinstance(va, (int, float)) and key.endswith("_pct")
            else (f"{va:.2f}" if isinstance(va, float) else str(va) if va is not None else "—")
        )  # noqa: E501
        sb = (
            f"{vb:.1f}"
            if isinstance(vb, (int, float)) and key.endswith("_pct")
            else (f"{vb:.2f}" if isinstance(vb, float) else str(vb) if vb is not None else "—")
        )  # noqa: E501
        print(f"  {name:<28} {sa:<24} {sb:<24}", file=sys.stderr)
    export_path = getattr(args, "export", None)
    if export_path:
        out = Path(export_path)
        if out.suffix.lower() == ".csv":
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["KPI", label_a, label_b])
                for key, name in [
                    ("total", "Total"),
                    ("asr_pct", "ASR %"),
                    ("refusal_rate_pct", "Refusal %"),
                    ("extraction_rate_pct", "Extraction %"),
                    ("errors", "Errors"),
                    ("avg_latency_sec", "Avg latency (s)"),
                    ("expected_refusal_accuracy_pct", "Expected-refusal acc %"),
                    ("benign_refusal_rate_pct", "Benign refusal %"),
                ]:
                    w.writerow([name, kpis_a.get(key, ""), kpis_b.get(key, "")])
            print(f"Exported comparison to {out}", file=sys.stderr)
        elif out.suffix.lower() in (".html", ".htm"):
            rows_html = "".join(
                f"<tr><td>{name}</td><td>{kpis_a.get(key, '')}</td><td>{kpis_b.get(key, '')}</td></tr>"
                for key, name in [
                    ("total", "Total"),
                    ("asr_pct", "ASR %"),
                    ("refusal_rate_pct", "Refusal %"),
                    ("extraction_rate_pct", "Extraction %"),
                    ("errors", "Errors"),
                    ("avg_latency_sec", "Avg latency (s)"),
                    ("expected_refusal_accuracy_pct", "Expected-refusal acc %"),
                    ("benign_refusal_rate_pct", "Benign refusal %"),
                ]
            )
            html = (  # noqa: E501
                f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Security Eval Compare</title></head><body>'
                f"<h1>Compare</h1><table border=\"1\"><tr><th>KPI</th><th>{label_a}</th><th>{label_b}</th></tr>"
                f"{rows_html}</table></body></html>"
            )
            out.write_text(html, encoding="utf-8")
            print(f"Exported comparison to {out}", file=sys.stderr)
        else:
            print(f"Unknown export format (use .csv or .html): {out}", file=sys.stderr)
    return 0


def register(subparsers) -> argparse.ArgumentParser:
    """Register the security-eval subcommand group; returns p_security_eval."""
    p_security_eval = subparsers.add_parser(
        "security-eval",
        help="LLM security evaluation: run prompt sets, score refusal/compliance, output KPIs and CSV",
    )
    se_sub = p_security_eval.add_subparsers(dest="security_eval_command")
    p_se_run = se_sub.add_parser("run", help="Run eval: load prompt set, query model, score, write CSV/JSON")
    p_se_run.add_argument(
        "prompt_set",
        nargs="?",
        default=None,
        metavar="PROMPT_SET",
        help="Path to .txt (one prompt/line) or .jsonl (omit with --schema to print schema only)",
    )
    p_se_run.add_argument(
        "--model",
        default="llama3.2",
        help="Model name (default: llama3.2)",
    )
    p_se_run.add_argument(
        "--base-url",
        default=None,
        help="Ollama or abliterate serve URL (default: OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    p_se_run.add_argument(
        "--no-check-ollama",
        action="store_true",
        help="Skip checking that Ollama/serve is reachable before running (default: check)",
    )
    p_se_run.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per prompt on transient API errors (default: 2)",
    )
    p_se_run.add_argument(
        "--output-csv",
        metavar="PATH",
        help="Write per-prompt results to CSV",
    )
    p_se_run.add_argument(
        "--output-json",
        metavar="PATH",
        help="Write full run (results + KPIs + metadata) to JSON",
    )
    p_se_run.add_argument(
        "--save-history",
        action="store_true",
        help="Append run to SQLite history (~/.ollama_forge/security_eval_runs.db) for plots over time",
    )
    p_se_run.add_argument(
        "--system",
        help="Optional system prompt to send with each request",
    )
    p_se_run.add_argument(
        "--no-chat",
        action="store_true",
        help="Use /api/generate instead of /api/chat",
    )
    p_se_run.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    p_se_run.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less progress output",
    )
    p_se_run.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N prompts (for quick smoke runs)",
    )
    p_se_run.add_argument(
        "--refusal-keywords",
        metavar="PATH",
        help="Path to file with custom refusal keywords (one per line, # comments). Default: built-in list.",
    )
    p_se_run.add_argument(
        "--baseline",
        metavar="MODEL",
        help="Also run the same prompt set against this model and print comparison (e.g. base model vs abliterated)",
    )
    p_se_run.add_argument(
        "--schema",
        action="store_true",
        help="Print prompt set schema (TXT and JSONL formats) and exit",
    )
    p_se_run.set_defaults(handler=_cmd_security_eval_run)
    p_se_ui = se_sub.add_parser(
        "ui", help="Launch Streamlit UI for security evaluation (requires: uv sync)"
    )
    p_se_ui.set_defaults(handler=_cmd_security_eval_ui)
    p_se_compare = se_sub.add_parser(
        "compare",
        help="Compare two security-eval run JSON files side-by-side (KPIs)",
    )
    p_se_compare.add_argument(
        "run_a",
        metavar="RUN_A.json",
        help="Path to first run JSON (from security-eval run --output-json)",
    )
    p_se_compare.add_argument(
        "run_b",
        metavar="RUN_B.json",
        help="Path to second run JSON",
    )
    p_se_compare.add_argument(
        "--export",
        metavar="PATH",
        help="Export comparison to CSV or HTML (suffix .csv or .html)",
    )
    p_se_compare.set_defaults(handler=_cmd_security_eval_compare)
    return p_security_eval
