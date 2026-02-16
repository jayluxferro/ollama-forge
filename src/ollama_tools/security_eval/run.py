"""Run security eval: load prompt set, query model, score, aggregate KPIs, write CSV."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from ollama_tools.security_eval.client import query_model
from ollama_tools.security_eval.history import save_run as history_save_run
from ollama_tools.security_eval.loader import load_prompt_set
from ollama_tools.security_eval.scorers import score_extraction, score_refusal


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
) -> dict:
    """
    Load prompt set, run each prompt against the model, score refusal/compliance/extraction,
    aggregate KPIs (ASR, refusal rate), and optionally write CSV and JSON.
    Returns a dict with keys: results (list of per-row dicts), kpis (ASR, refusal_rate, total, by_category).
    """
    prompts = load_prompt_set(prompt_set_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompt_set_path}")

    results: list[dict] = []
    for i, row in enumerate(prompts):
        prompt = row["prompt"]
        context = (row.get("context") or "").strip()
        if context:
            # Indirect prompt injection: model sees context then user query
            prompt = f"Context:\n{context}\n\nUser query:\n{prompt}"
        category = row.get("category") or "default"
        target = row.get("target_for_extraction")
        if verbose:
            print(f"  [{i + 1}/{len(prompts)}] {category}: ...", file=sys.stderr)
        try:
            response, duration = query_model(
                prompt,
                base_url=base_url,
                model=model,
                use_chat=use_chat,
                system=system,
                timeout=timeout,
            )
        except Exception as e:
            results.append(
                {
                    "index": i + 1,
                    "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                    "category": category,
                    "response": "",
                    "refusal": False,
                    "compliance": False,
                    "extraction": False,
                    "duration_sec": None,
                    "error": str(e),
                }
            )
            continue
        refusal = score_refusal(response)
        extraction = score_extraction(response, target=target) if target else False
        compliance = not refusal
        results.append(
            {
                "index": i + 1,
                "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "category": category,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "refusal": refusal,
                "compliance": compliance,
                "extraction": extraction,
                "duration_sec": duration,
                "error": None,
            }
        )

    # KPIs
    total = len(results)
    refusals = sum(1 for r in results if r.get("refusal"))
    compliances = sum(1 for r in results if r.get("compliance"))
    extractions = sum(1 for r in results if r.get("extraction"))
    errors = sum(1 for r in results if r.get("error"))
    # ASR = attack success rate = % that complied (for jailbreak/harmful prompts)
    asr = (compliances / total * 100.0) if total else 0.0
    refusal_rate = (refusals / total * 100.0) if total else 0.0
    extraction_rate = (extractions / total * 100.0) if total else 0.0

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category") or "default"
        if cat not in by_category:
            by_category[cat] = {"total": 0, "refusal": 0, "compliance": 0, "extraction": 0}
        by_category[cat]["total"] += 1
        if r.get("refusal"):
            by_category[cat]["refusal"] += 1
        if r.get("compliance"):
            by_category[cat]["compliance"] += 1
        if r.get("extraction"):
            by_category[cat]["extraction"] += 1
    for cat in by_category:
        t = by_category[cat]["total"]
        by_category[cat]["refusal_rate_pct"] = (by_category[cat]["refusal"] / t * 100.0) if t else 0.0
        by_category[cat]["asr_pct"] = (by_category[cat]["compliance"] / t * 100.0) if t else 0.0

    kpis = {
        "total": total,
        "refusals": refusals,
        "compliances": compliances,
        "extractions": extractions,
        "errors": errors,
        "asr_pct": asr,
        "refusal_rate_pct": refusal_rate,
        "extraction_rate_pct": extraction_rate,
        "by_category": by_category,
    }

    run_meta = {
        "model": model,
        "base_url": base_url,
        "prompt_set": str(prompt_set_path),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "kpis": kpis,
    }

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["index", "category", "refusal", "compliance", "extraction", "duration_sec", "error"]
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
                print("Saved to run history (~/.ollama_tools/security_eval_runs.db)", file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f"Warning: could not save to history: {e}", file=sys.stderr)

    return run_meta
