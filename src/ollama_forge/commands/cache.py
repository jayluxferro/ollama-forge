"""Cache command group: HF cache and forge cache management."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from ollama_forge.run_helpers import print_actionable_error


def _ollama_forge_cache_dir() -> Path:
    """Return the ollama-forge cache root (``~/.cache/ollama-forge`` or ``OLLAMA_FORGE_CACHE``)."""
    return Path(os.environ.get("OLLAMA_FORGE_CACHE", Path.home() / ".cache" / "ollama-forge"))


def _gguf_cache_dir_for_repo(repo_id: str) -> Path:
    """Return ``<cache>/gguf/<owner>/<repo>/`` for storing converted GGUFs."""
    return _ollama_forge_cache_dir() / "gguf" / repo_id.replace("/", os.sep)


def _cmd_hf_cache_ls(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """List Hugging Face Hub cache (repos and sizes)."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print_actionable_error(
            "huggingface_hub is required for hf-cache",
            next_steps=["Run: uv sync"],
        )
        return 1
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print_actionable_error(
            "failed to scan Hugging Face cache",
            cause=str(e),
            next_steps=[
                "Check that ~/.cache/huggingface/hub exists and is readable",
                "Run: ollama-forge hf-cache ls",
            ],
        )
        return 1
    if getattr(args, "size", False):
        total = getattr(cache_info, "size_on_disk", 0) or 0
        if total >= 1024**3:
            size_str = f"{total / 1024**3:.1f} GiB"
        elif total >= 1024**2:
            size_str = f"{total / 1024**2:.1f} MiB"
        else:
            size_str = f"{total} B"
        print(f"Total cache size: {size_str}")
        return 0
    verbosity = 1 if getattr(args, "revisions", False) else 0
    print(cache_info.export_as_table(verbosity=verbosity))
    return 0


def _cmd_hf_cache_rm(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Remove one or more repos from the Hugging Face Hub cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print_actionable_error(
            "huggingface_hub is required for hf-cache",
            next_steps=["Run: uv sync"],
        )
        return 1
    repo_ids = getattr(args, "repo_id", None)
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]
    repo_ids = repo_ids or []
    if not repo_ids:
        print_actionable_error(
            "no repo_id provided",
            next_steps=[
                "Provide at least one repo id: ollama-forge hf-cache rm <repo_id> [repo_id ...]",
                "Example: ollama-forge hf-cache rm TheBloke/Llama-2-7B-GGUF",
            ],
        )
        return 1
    dry_run = getattr(args, "dry_run", False)
    yes = getattr(args, "yes", False)
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print_actionable_error(
            "failed to scan Hugging Face cache",
            cause=str(e),
            next_steps=[
                "Check that ~/.cache/huggingface/hub exists and is readable",
                "Run: ollama-forge hf-cache ls",
            ],
        )
        return 1
    revisions_to_delete: list[str] = []
    for repo in cache_info.repos:
        # repo.repo_id is e.g. "TheBloke/Llama-2-7B-GGUF" or "bert-base-cased"
        if repo.repo_id in repo_ids:
            for rev in repo.revisions:
                revisions_to_delete.append(rev.commit_hash)
    if not revisions_to_delete:
        print_actionable_error(
            "no matching repos found in cache",
            next_steps=[
                "Run: ollama-forge hf-cache ls (with --revisions) to see cached repo ids",
                "Use exact repo_id(s) from that list, e.g. TheBloke/Llama-2-7B-GGUF",
            ],
        )
        return 1
    strategy = cache_info.delete_revisions(*revisions_to_delete)
    print(f"About to free {strategy.expected_freed_size_str}.", file=sys.stderr)
    if dry_run:
        print("Dry run: no files deleted.", file=sys.stderr)
        return 0
    if not yes:
        try:
            answer = input("Proceed? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer != "y":
            print("Cancelled.", file=sys.stderr)
            return 0
    strategy.execute()
    print(f"Freed {strategy.expected_freed_size_str}.", file=sys.stderr)
    return 0


def _cmd_cache_add(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Copy (or move) a GGUF file into the ollama-forge cache."""
    gguf = Path(args.gguf).resolve()
    if not gguf.is_file():
        print_actionable_error(
            f"File not found: {gguf}",
            next_steps=["Provide a valid path to a .gguf file"],
        )
        return 1
    name: str = args.name
    if "/" not in name:
        print_actionable_error(
            f"--name must be in org/model format, got: {name!r}",
            next_steps=["Example: --name my-org/my-model"],
        )
        return 1
    dest_dir = _gguf_cache_dir_for_repo(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / gguf.name
    move = getattr(args, "move", False)
    if move:
        shutil.move(str(gguf), str(dest))
        print(f"Moved {gguf} → {dest}")
    else:
        shutil.copy2(str(gguf), str(dest))
        print(f"Copied {gguf} → {dest}")
    print(f"\nServe with: ollama-forge serve {name}", file=sys.stderr)
    return 0


def _cmd_cache_ls(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """List GGUF files in the ollama-forge cache."""
    cache_root = _ollama_forge_cache_dir() / "gguf"
    if not cache_root.is_dir():
        print("Cache is empty.")
        return 0
    found = False
    for org_dir in sorted(cache_root.iterdir()):
        if not org_dir.is_dir():
            continue
        for repo_dir in sorted(org_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            gguf_files = sorted(repo_dir.glob("*.gguf"))
            if not gguf_files:
                continue
            repo_id = f"{org_dir.name}/{repo_dir.name}"
            for gf in gguf_files:
                size_mb = gf.stat().st_size / (1024 ** 2)
                print(f"{repo_id}\t{gf.name}\t{size_mb:.0f} MiB")
                found = True
    if not found:
        print("Cache is empty.")
    return 0


def _cmd_cache_rm(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Remove a repo from the ollama-forge GGUF cache."""
    name: str = args.name
    cache_dir = _gguf_cache_dir_for_repo(name)
    if not cache_dir.is_dir():
        print_actionable_error(
            f"No cache entry for {name!r}",
            next_steps=["Run: ollama-forge cache ls"],
        )
        return 1
    gguf_files = list(cache_dir.glob("*.gguf"))
    total = sum(f.stat().st_size for f in gguf_files)
    size_str = f"{total / (1024 ** 2):.0f} MiB" if total >= 1024 ** 2 else f"{total} B"
    if not getattr(args, "yes", False):
        print(f"Will remove {len(gguf_files)} file(s) ({size_str}) from {cache_dir}")
        try:
            answer = input("Proceed? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer != "y":
            print("Cancelled.", file=sys.stderr)
            return 0
    shutil.rmtree(cache_dir)
    print(f"Removed {name} ({size_str})")
    return 0


def register(subparsers) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Register hf-cache and cache subcommand groups; returns (p_hf_cache, p_cache)."""
    # hf-cache (list / remove Hugging Face Hub cache)
    p_hf_cache = subparsers.add_parser(
        "hf-cache",
        help="List or remove Hugging Face Hub local cache (models downloaded by fetch/fetch-adapter)",
    )
    hf_cache_sub = p_hf_cache.add_subparsers(dest="hf_cache_command")
    p_hf_cache_ls = hf_cache_sub.add_parser("ls", help="List cached repos and sizes")
    p_hf_cache_ls.add_argument(
        "--revisions",
        action="store_true",
        help="Show one row per revision (default: one row per repo)",
    )
    p_hf_cache_ls.add_argument(
        "--size",
        action="store_true",
        help="Print total disk usage of the cache (human-readable)",
    )
    p_hf_cache_ls.set_defaults(handler=_cmd_hf_cache_ls)
    p_hf_cache_rm = hf_cache_sub.add_parser(
        "rm",
        help="Remove repo(s) from cache (frees disk space)",
    )
    p_hf_cache_rm.add_argument(
        "repo_id",
        nargs="+",
        help="Repo id(s) to remove (e.g. TheBloke/Llama-2-7B-GGUF)",
    )
    p_hf_cache_rm.add_argument("--dry-run", action="store_true", help="Show what would be freed, do not delete")
    p_hf_cache_rm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_hf_cache_rm.set_defaults(handler=_cmd_hf_cache_rm)

    # cache (ollama-forge GGUF cache: add, ls, rm)
    p_cache = subparsers.add_parser(
        "cache",
        help="Manage the ollama-forge GGUF cache (add, list, remove converted/imported models)",
    )
    cache_sub = p_cache.add_subparsers(dest="cache_command")
    p_cache_add = cache_sub.add_parser(
        "add",
        help="Add a GGUF file to the cache so it can be served by repo name",
    )
    p_cache_add.add_argument(
        "gguf",
        help="Path to the .gguf file to add",
    )
    p_cache_add.add_argument(
        "--name",
        required=True,
        help="Cache key in org/model format (e.g. my-org/my-model)",
    )
    p_cache_add.add_argument(
        "--move",
        action="store_true",
        help="Move the file instead of copying (deletes the original)",
    )
    p_cache_add.set_defaults(handler=_cmd_cache_add)

    p_cache_ls = cache_sub.add_parser("ls", help="List cached GGUF models")
    p_cache_ls.set_defaults(handler=_cmd_cache_ls)

    p_cache_rm = cache_sub.add_parser("rm", help="Remove a model from the GGUF cache")
    p_cache_rm.add_argument(
        "name",
        help="Cache key to remove (org/model format, from 'cache ls')",
    )
    p_cache_rm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_cache_rm.set_defaults(handler=_cmd_cache_rm)

    return p_hf_cache, p_cache
