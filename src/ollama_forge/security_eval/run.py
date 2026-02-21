"""Run security eval: load prompt set, query model, score, aggregate KPIs, write CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

from ollama_forge.security_eval.client import (
    query_model,
    query_model_multi_turn,
    query_model_multi_turn_iterative,
    query_model_with_image,
    query_model_with_tools,
)
from ollama_forge.security_eval.history import save_run as history_save_run
from ollama_forge.security_eval.loader import load_prompt_set
from ollama_forge.security_eval.scorers import (
    score_extraction,
    score_image_reveal,
    score_refusal,
    score_tool_misuse,
)


def _load_refusal_keywords(path: str | Path | None) -> list[str] | None:
    """Load refusal keywords from file (one per line, # skipped). Return None if path is None or file missing."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    keywords = [
        line.strip().lower()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return keywords if keywords else None


def run_eval(
    prompt_set_path: str | Path,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2",
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
    save_to_history: bool = False,
    use_chat: bool = True,
    system: str | None = None,
    timeout: float = 120.0,
    verbose: bool = True,
    retries: int = 2,
    max_prompts: int | None = None,
    refusal_keywords_path: str | Path | None = None,
    progress_callback: Callable[[int, int, list], None] | None = None,
    prompt_set_version: str | None = None,
    use_iterative_multi_turn: bool = True,
) -> dict:
    """
    Load prompt set, run each prompt against the model, score refusal/compliance/extraction,
    aggregate KPIs (ASR, refusal rate), and optionally write CSV and JSON.
    Returns a dict with keys: results (list of per-row dicts), kpis (ASR, refusal_rate, total, by_category).
    If max_prompts is set, only the first max_prompts prompts are run (for quick smoke runs).
    If refusal_keywords_path is set, load custom refusal keywords from that file (one per line).
    """
    prompts = load_prompt_set(prompt_set_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompt_set_path}")
    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]
        if verbose:
            print(f"Running first {len(prompts)} prompts (max_prompts={max_prompts}).", file=sys.stderr)

    refusal_keywords = _load_refusal_keywords(refusal_keywords_path)
    if prompt_set_version is None:
        try:
            prompt_set_version = hashlib.sha256(Path(prompt_set_path).read_bytes()).hexdigest()[:12]
        except Exception:
            prompt_set_version = None

    results: list[dict] = []
    n_prompts = len(prompts)
    for i, row in enumerate(prompts):
        prompt = row["prompt"]
        context = (row.get("context") or "").strip()
        if context:
            prompt = f"Context:\n{context}\n\nUser query:\n{prompt}"
        category = row.get("category") or "default"
        target = row.get("target_for_extraction")
        expected_refusal = row.get("expected_refusal")
        row_system = row.get("system") or row.get("system_prompt") or system
        if verbose:
            print(f"  [{i + 1}/{n_prompts}] {category}: ...", file=sys.stderr)
        tools = row.get("tools")
        dangerous_tool_names = row.get("dangerous_tool_names") or []
        image_b64 = row.get("image")
        image_reveal_keywords = row.get("image_reveal_keywords") or []
        turns = row.get("turns")
        try:
            num_attempts = max(0, retries) + 1
            response = ""
            duration = None
            tool_calls: list[dict] = []
            turns_count: int | None = None
            turns_to_success: int | None = None

            if tools and isinstance(tools, list):
                for attempt in range(num_attempts):
                    try:
                        response, tool_calls, duration = query_model_with_tools(
                            prompt,
                            tools,
                            base_url=base_url,
                            model=model,
                            system=row_system,
                            timeout=timeout,
                        )
                        break
                    except Exception as e:
                        if attempt == num_attempts - 1:
                            raise
                        if verbose:
                            print(f"    Retry {attempt + 1}/{retries} after: {e}", file=sys.stderr)
                        time.sleep(1.0 * (attempt + 1))
            elif image_b64:
                for attempt in range(num_attempts):
                    try:
                        response, duration = query_model_with_image(
                            prompt,
                            image_b64,
                            base_url=base_url,
                            model=model,
                            system=row_system,
                            timeout=timeout,
                        )
                        break
                    except Exception as e:
                        if attempt == num_attempts - 1:
                            raise
                        if verbose:
                            print(f"    Retry {attempt + 1}/{retries} after: {e}", file=sys.stderr)
                        time.sleep(1.0 * (attempt + 1))
            elif turns and isinstance(turns, list) and len(turns) >= 2 and use_iterative_multi_turn:
                for attempt in range(num_attempts):
                    try:
                        response, per_turn_responses, duration = query_model_multi_turn_iterative(
                            turns,
                            base_url=base_url,
                            model=model,
                            system=row_system,
                            timeout=timeout,
                        )
                        turns_count = len(per_turn_responses)
                        turns_to_success = None
                        for idx, r in enumerate(per_turn_responses):
                            if not score_refusal(r, keywords=refusal_keywords):
                                turns_to_success = idx + 1
                                break
                        if turns_to_success is None and turns_count:
                            turns_to_success = 0
                        break
                    except Exception as e:
                        if attempt == num_attempts - 1:
                            raise
                        if verbose:
                            print(f"    Retry {attempt + 1}/{retries} after: {e}", file=sys.stderr)
                        time.sleep(1.0 * (attempt + 1))
            elif turns and isinstance(turns, list) and len(turns) >= 2:
                for attempt in range(num_attempts):
                    try:
                        response, duration = query_model_multi_turn(
                            turns,
                            base_url=base_url,
                            model=model,
                            system=row_system,
                            timeout=timeout,
                        )
                        turns_count = max(1, len([m for m in turns if m.get("role") == "user"]))
                        break
                    except Exception as e:
                        if attempt == num_attempts - 1:
                            raise
                        if verbose:
                            print(f"    Retry {attempt + 1}/{retries} after: {e}", file=sys.stderr)
                        time.sleep(1.0 * (attempt + 1))
            else:
                for attempt in range(num_attempts):
                    try:
                        response, duration = query_model(
                            prompt,
                            base_url=base_url,
                            model=model,
                            use_chat=use_chat,
                            system=row_system,
                            timeout=timeout,
                        )
                        break
                    except Exception as e:
                        if attempt == num_attempts - 1:
                            raise
                        if verbose:
                            print(f"    Retry {attempt + 1}/{retries} after: {e}", file=sys.stderr)
                        time.sleep(1.0 * (attempt + 1))
        except Exception as e:
            err_msg = str(e)
            results.append(
                {
                    "index": i + 1,
                    "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                    "prompt_full": prompt,
                    "category": category,
                    "response": "",
                    "response_full": "",
                    "refusal": False,
                    "compliance": False,
                    "extraction": False,
                    "duration_sec": None,
                    "error": err_msg,
                    "expected_refusal": expected_refusal,
                }
            )
            if progress_callback:
                progress_callback(i + 1, n_prompts, results)
            continue
        refusal = score_refusal(response, keywords=refusal_keywords)
        extraction = score_extraction(response, target=target) if target else False
        compliance = not refusal
        tool_misuse = score_tool_misuse(tool_calls, dangerous_tool_names) if tool_calls else False
        image_reveal = (
            score_image_reveal(response, keywords=image_reveal_keywords)
            if image_reveal_keywords else False
        )
        res = {
            "index": i + 1,
            "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
            "prompt_full": prompt,
            "category": category,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "response_full": response,
            "refusal": refusal,
            "compliance": compliance,
            "extraction": extraction,
            "duration_sec": duration,
            "error": None,
            "expected_refusal": expected_refusal,
        }
        if tool_calls is not None:
            res["tool_misuse"] = tool_misuse
        if image_b64 is not None:
            res["image_reveal"] = image_reveal
        if turns_count is not None:
            res["turns_count"] = turns_count
        if turns_to_success is not None:
            res["turns_to_success"] = turns_to_success
        results.append(res)
        if progress_callback:
            progress_callback(i + 1, n_prompts, results)

    # KPIs
    total = len(results)
    refusals = sum(1 for r in results if r.get("refusal"))
    compliances = sum(1 for r in results if r.get("compliance"))
    extractions = sum(1 for r in results if r.get("extraction"))
    errors = sum(1 for r in results if r.get("error"))
    tool_misuses = sum(1 for r in results if r.get("tool_misuse"))
    image_reveals = sum(1 for r in results if r.get("image_reveal"))
    multi_turn_results = [r for r in results if r.get("turns_count") is not None]
    turns_to_success_vals = [r["turns_to_success"] for r in multi_turn_results if r.get("turns_to_success") is not None]
    asr = (compliances / total * 100.0) if total else 0.0
    refusal_rate = (refusals / total * 100.0) if total else 0.0
    extraction_rate = (extractions / total * 100.0) if total else 0.0
    tool_misuse_rate = (tool_misuses / total * 100.0) if total else 0.0
    image_reveal_rate = (image_reveals / total * 100.0) if total else 0.0
    avg_turns_to_success = (
        sum(turns_to_success_vals) / len(turns_to_success_vals)
        if turns_to_success_vals else None
    )

    # Avg latency (exclude errors)
    durations = [r.get("duration_sec") for r in results if r.get("duration_sec") is not None]
    avg_latency_sec = sum(durations) / len(durations) if durations else None

    # Error breakdown: count by error message (truncate long messages for key)
    error_counts: dict[str, int] = {}
    for r in results:
        e = r.get("error")
        if e:
            key = e[:80] + "..." if len(e) > 80 else e
            error_counts[key] = error_counts.get(key, 0) + 1

    # Expected-refusal accuracy: of rows with expected_refusal set, % where refusal matched expected
    with_expected = [r for r in results if r.get("expected_refusal") is not None]
    if with_expected:
        correct = sum(1 for r in with_expected if r.get("refusal") == r.get("expected_refusal"))
        expected_refusal_accuracy_pct = (correct / len(with_expected)) * 100.0
    else:
        expected_refusal_accuracy_pct = None

    # Benign refusal rate: of rows where expected_refusal is False (benign), % that refused
    benign_rows = [r for r in results if r.get("expected_refusal") is False]
    if benign_rows:
        benign_refusals = sum(1 for r in benign_rows if r.get("refusal"))
        benign_refusal_rate_pct = (benign_refusals / len(benign_rows)) * 100.0
    else:
        benign_refusal_rate_pct = None

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category") or "default"
        if cat not in by_category:
            by_category[cat] = {
                "total": 0, "refusal": 0, "compliance": 0, "extraction": 0,
                "tool_misuse": 0, "image_reveal": 0,
            }
        by_category[cat]["total"] += 1
        if r.get("refusal"):
            by_category[cat]["refusal"] += 1
        if r.get("compliance"):
            by_category[cat]["compliance"] += 1
        if r.get("extraction"):
            by_category[cat]["extraction"] += 1
        if r.get("tool_misuse"):
            by_category[cat]["tool_misuse"] += 1
        if r.get("image_reveal"):
            by_category[cat]["image_reveal"] += 1
    for cat in by_category:
        t = by_category[cat]["total"]
        by_category[cat]["refusal_rate_pct"] = (by_category[cat]["refusal"] / t * 100.0) if t else 0.0
        by_category[cat]["asr_pct"] = (by_category[cat]["compliance"] / t * 100.0) if t else 0.0
        by_category[cat]["extraction_rate_pct"] = (by_category[cat]["extraction"] / t * 100.0) if t else 0.0

    kpis = {
        "total": total,
        "refusals": refusals,
        "compliances": compliances,
        "extractions": extractions,
        "errors": errors,
        "asr_pct": asr,
        "refusal_rate_pct": refusal_rate,
        "extraction_rate_pct": extraction_rate,
        "tool_misuse_rate_pct": tool_misuse_rate,
        "image_reveal_rate_pct": image_reveal_rate,
        "multi_turn_prompts_count": len(multi_turn_results),
        "avg_turns_to_success": avg_turns_to_success,
        "by_category": by_category,
        "avg_latency_sec": avg_latency_sec,
        "error_counts": error_counts,
        "expected_refusal_accuracy_pct": expected_refusal_accuracy_pct,
        "benign_refusal_rate_pct": benign_refusal_rate_pct,
    }

    run_meta = {
        "model": model,
        "base_url": base_url,
        "prompt_set": str(prompt_set_path),
        "prompt_set_version": prompt_set_version,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "kpis": kpis,
    }

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "index", "category", "refusal", "compliance", "extraction",
                    "tool_misuse", "image_reveal", "turns_count", "turns_to_success",
                    "duration_sec", "error", "expected_refusal",
                ],
            )
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        if verbose:
            print(f"Wrote {out_path}", file=sys.stderr)

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        if verbose:
            print(f"Wrote {out_path}", file=sys.stderr)

    if save_to_history:
        try:
            history_save_run(run_meta)
            if verbose:
                print("Saved to run history (~/.ollama_forge/security_eval_runs.db)", file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f"Warning: could not save to history: {e}", file=sys.stderr)

    return run_meta
