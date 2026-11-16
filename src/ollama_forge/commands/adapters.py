"""Adapters command group: search and recommend HF adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ollama_forge.log import get_logger
from ollama_forge.run_helpers import print_actionable_error

log = get_logger()


def _cmd_adapters_search(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Search Hugging Face for adapters and show how to use them."""
    from huggingface_hub import HfApi

    query = getattr(args, "query", None) or "lora adapter"
    limit = getattr(args, "limit", 10)
    api = HfApi()
    print(f"Searching Hugging Face for '{query}' (max {limit})...\n", file=sys.stderr)
    try:
        models = list(api.list_models(search=query, limit=limit))
    except Exception as e:
        print_actionable_error(
            "adapter search failed",
            cause=str(e),
            next_steps=["Check network and HF access", "Try: ollama-forge adapters search '<query>' --limit 10"],
        )
        return 1
    if not models:
        print("No adapters found. Try another search (e.g. 'llama lora', 'mistral adapter').")
        return 0
    print("Adapters you can use with fetch-adapter:\n")
    for m in models:
        repo = m.id
        print(f"  {repo}")
        print(f"    → ollama-forge fetch-adapter {repo} --base <BASE_MODEL> --name <NAME>")
    print("\nReplace <BASE_MODEL> with the model the adapter was trained for (e.g. llama3.2).")
    return 0


def _score_adapter_repo(repo_id: str, base: str | None) -> int:
    """Simple ranking heuristic for adapter recommendations."""
    rid = repo_id.lower()
    score = 0
    if "adapter" in rid:
        score += 5
    if "lora" in rid:
        score += 4
    if "qlora" in rid:
        score += 3
    if "gguf" in rid:
        score -= 2
    if base:
        base_tokens = [t for t in base.lower().replace("-", " ").replace("_", " ").split() if t]
        if any(tok in rid for tok in base_tokens):
            score += 6
    return score


def _adapters_recommend_cache_path(query: str, base: str | None, limit: int) -> Path:
    """Path for caching adapter recommendations (keyed by query, base, limit)."""
    key = hashlib.sha256(f"{query}|{base or ''}|{limit}".encode()).hexdigest()[:16]
    cache_dir = Path(os.environ.get("OLLAMA_FORGE_CACHE", Path.home() / ".cache" / "ollama-forge"))
    return cache_dir / "adapters-recommend" / f"{key}.json"


def _cmd_adapters_recommend(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Recommend adapter repos and optionally apply the top one."""
    from huggingface_hub import HfApi

    base = getattr(args, "base", None)
    query = getattr(args, "query", None) or (f"{base} lora adapter" if base else "lora adapter")
    limit = max(1, int(getattr(args, "limit", 5)))
    json_mode = bool(getattr(args, "json", False))
    cache_ttl = int(getattr(args, "cache_ttl", 3600))
    ranked: list[tuple[str, int]] = []
    if cache_ttl > 0:
        cache_path = _adapters_recommend_cache_path(query, base, limit)
        if cache_path.is_file():
            try:
                age = cache_path.stat().st_mtime
                if (datetime.now(timezone.utc).timestamp() - age) <= cache_ttl:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                    if data.get("query") == query and data.get("base") == base and data.get("limit") == limit:
                        ranked = [tuple(x) for x in data.get("ranked", [])]
            except (json.JSONDecodeError, OSError, TypeError) as e:
                log.debug("Could not load adapter recommendations cache: %s", e)
    if not ranked:
        api = HfApi()
        if not json_mode:
            print(f"Finding adapter recommendations for query: {query!r}", file=sys.stderr)
        try:
            candidates = list(api.list_models(search=query, limit=max(limit * 4, 20)))
        except Exception as e:
            print_actionable_error(
                "failed to search adapter recommendations",
                cause=str(e),
                next_steps=[
                    "Check internet/Hugging Face connectivity",
                    'Try a broader query: ollama-forge adapters recommend --query "lora adapter"',
                ],
            )
            return 1
        if not candidates:
            print("No adapter recommendations found.")
            return 0
        ranked = sorted(
            ((m.id, _score_adapter_repo(m.id, base)) for m in candidates),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        if cache_ttl > 0:
            cache_path = _adapters_recommend_cache_path(query, base, limit)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {"query": query, "base": base, "limit": limit, "ranked": [list(p) for p in ranked]},
                        indent=0,
                    ),
                    encoding="utf-8",
                )
            except OSError as e:
                log.debug("Could not write adapter recommendations cache: %s", e)
    if not json_mode:
        print("Recommended adapters:\n")
        for repo, score in ranked:
            print(f"  {repo}  (score={score})")
            if base:
                print(f"    -> ollama-forge fetch-adapter {repo} --base {base} --name <NAME>")
            else:
                print(f"    -> ollama-forge fetch-adapter {repo} --base <BASE_MODEL> --name <NAME>")
    elif not getattr(args, "apply", False):
        print(
            json.dumps(
                {
                    "route": "adapters-recommend",
                    "base": base,
                    "query": query,
                    "recommendations": [{"repo": repo, "score": score} for repo, score in ranked],
                }
            )
        )
        return 0
    if not getattr(args, "apply", False):
        return 0
    top_repo = ranked[0][0]
    if not base:
        print_actionable_error(
            "--apply requires --base",
            next_steps=[
                f'Re-run with: ollama-forge adapters recommend --base <BASE_MODEL> --apply --query "{query}"',
            ],
        )
        return 1
    target_name = getattr(args, "name", None) or f"{base}-adapter"
    if getattr(args, "plan", False):
        action = f"ollama-forge fetch-adapter {top_repo} --base {base} --name {target_name}"
        if getattr(args, "json", False):
            from ollama_forge.cli import _save_last_plan

            plan_obj = {
                "route": "adapters-apply",
                "top_repo": top_repo,
                "base": base,
                "name": target_name,
                "action": action,
            }
            _save_last_plan("adapters-apply", plan_obj)
            print(json.dumps(plan_obj))
        else:
            print("\nApply plan:")
            print(f"  top repo: {top_repo}")
            print(f"  base: {base}")
            print(f"  output model: {target_name}")
            print(f"  action: {action}")
        return 0
    print(f"\nApplying top recommendation: {top_repo} -> model {target_name!r}", file=sys.stderr)
    fake = argparse.Namespace(
        repo_id=top_repo,
        base=base,
        name=target_name,
        revision=getattr(args, "revision", "main"),
        output=getattr(args, "output", None),
        system=getattr(args, "system", None),
        temperature=getattr(args, "temperature", None),
        num_ctx=getattr(args, "num_ctx", None),
        top_p=getattr(args, "top_p", None),
        repeat_penalty=getattr(args, "repeat_penalty", None),
        out_modelfile=getattr(args, "out_modelfile", None),
    )
    from ollama_forge.cli import _cmd_fetch_adapter

    return _cmd_fetch_adapter(parser, fake)


def register(subparsers) -> argparse.ArgumentParser:
    """Register the adapters subcommand group; returns p_adapters."""
    p_adapters = subparsers.add_parser(
        "adapters",
        help="Find and use adapters (e.g. search Hugging Face)",
    )
    adapters_sub = p_adapters.add_subparsers(dest="adapters_command")
    p_adapters_search = adapters_sub.add_parser(
        "search",
        help="Search Hugging Face for adapters; shows fetch-adapter commands",
    )
    p_adapters_search.add_argument(
        "query",
        nargs="?",
        default="lora adapter",
        help="Search query (default: lora adapter)",
    )
    p_adapters_search.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of results (default: 10)",
    )
    p_adapters_search.set_defaults(handler=_cmd_adapters_search)

    p_adapters_recommend = adapters_sub.add_parser(
        "recommend",
        help="Recommend likely adapter repos (optionally apply top result)",
    )
    p_adapters_recommend.add_argument(
        "--base",
        default=None,
        help="Base model name/path to bias recommendations and use with --apply",
    )
    p_adapters_recommend.add_argument(
        "--query",
        default=None,
        help="Search query override (default uses --base if given)",
    )
    p_adapters_recommend.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max recommendations to show (default: 5)",
    )
    p_adapters_recommend.add_argument(
        "--cache-ttl",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Cache recommendations for this many seconds (0 = disable; default: 3600)",
    )
    p_adapters_recommend.add_argument(
        "--apply",
        action="store_true",
        help="Apply top recommendation via fetch-adapter (requires --base)",
    )
    p_adapters_recommend.add_argument(
        "--plan",
        action="store_true",
        help="When used with --apply, show the planned fetch-adapter command only",
    )
    p_adapters_recommend.add_argument(
        "--name",
        default=None,
        help="Output model name when using --apply (default: <base>-adapter)",
    )
    p_adapters_recommend.add_argument(
        "--revision",
        default="main",
        help="Repo revision when using --apply (default: main)",
    )
    p_adapters_recommend.add_argument(
        "--output",
        default=None,
        help="Adapter download directory when using --apply",
    )
    p_adapters_recommend.add_argument("--system", help="System message for --apply")
    p_adapters_recommend.add_argument(
        "--temperature",
        type=float,
        help="Temperature for --apply (e.g. 0.7)",
    )
    p_adapters_recommend.add_argument(
        "--num-ctx",
        type=int,
        help="Context window for --apply (e.g. 4096)",
    )
    p_adapters_recommend.add_argument("--top-p", type=float, help="Top-p for --apply")
    p_adapters_recommend.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty for --apply",
    )
    p_adapters_recommend.add_argument(
        "--out-modelfile",
        default=None,
        help="Also write Modelfile when using --apply",
    )
    p_adapters_recommend.set_defaults(handler=_cmd_adapters_recommend)

    return p_adapters
