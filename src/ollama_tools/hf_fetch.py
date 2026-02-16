"""Fetch models and adapters from Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path


# Use faster downloads when hf_transfer is installed (pip install hf-transfer)
def _enable_fast_downloads() -> None:
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except ImportError:
            pass


def list_gguf_files(repo_id: str, revision: str | None = None) -> list[str]:
    """List .gguf files in a Hugging Face repo. Returns filenames (no path)."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, revision=revision or "main")
    return [f for f in files if f.lower().endswith(".gguf")]


# Preferred quantization names (lowercase) for auto-pick when repo has multiple GGUF files.
_PREFERRED_GGUF_SUBSTRINGS = ("q4_k_m", "q4_k_s", "q5_k_m", "q4_0", "q8_0")


def pick_one_gguf(
    filenames: list[str],
    prefer_quant: str | None = None,
) -> str:
    """Pick one GGUF when multiple exist. Prefers prefer_quant (e.g. Q4_K_M) if set."""
    if not filenames:
        raise ValueError("empty filenames")
    if len(filenames) == 1:
        return filenames[0]
    lower = [f.lower() for f in filenames]
    if prefer_quant:
        q = prefer_quant.lower().replace("-", "_")
        for i, name in enumerate(lower):
            if q in name:
                return filenames[i]
    for preferred in _PREFERRED_GGUF_SUBSTRINGS:
        for i, name in enumerate(lower):
            if preferred in name:
                return filenames[i]
    return filenames[0]


def download_gguf(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> str:
    """Download a GGUF file from HF. Returns path to the local file."""
    _enable_fast_downloads()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision or "main",
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        local_dir_use_symlinks=False,
    )
    return path


def download_adapter(
    repo_id: str,
    *,
    revision: str | None = None,
    local_dir: str | Path,
) -> Path:
    """Download a full adapter repo (e.g. PEFT) to local_dir. Returns local_dir as Path."""
    _enable_fast_downloads()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision or "main",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_dir)
