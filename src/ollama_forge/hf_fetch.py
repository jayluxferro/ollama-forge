"""Fetch models and adapters from Hugging Face Hub."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from ollama_forge.log import get_logger

log = get_logger()


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


# ETag from Hub is often SHA256 for LFS files (64 hex chars, optionally wrapped in quotes)
_ETAG_SHA256_RE = re.compile(r'^"?([0-9a-fA-F]{64})"?$')


def verify_gguf_checksum(
    repo_id: str,
    filename: str,
    local_path: str | Path,
    *,
    revision: str | None = None,
) -> None:
    """
    Verify the downloaded file's SHA256 against the Hub's ETag when it is a SHA256 (e.g. LFS).
    Raises ValueError if the Hub exposes a SHA256 ETag and the local file's hash does not match.
    If the Hub does not expose a SHA256 ETag, returns without error (no-op).
    """
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(
        repo_id=repo_id,
        filename=filename,
        revision=revision or "main",
    )
    try:
        meta = get_hf_file_metadata(url)
    except Exception as e:
        log.warning("Could not fetch Hub metadata for %s/%s; skipping checksum verification: %s", repo_id, filename, e)
        return
    etag = getattr(meta, "etag", None)
    if not etag:
        return
    match = _ETAG_SHA256_RE.match(etag.strip())
    if not match:
        return  # ETag is not a SHA256 (e.g. git SHA1); skip
    expected_sha = match.group(1).lower()
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Local file not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_sha = h.hexdigest().lower()
    if actual_sha != expected_sha:
        raise ValueError(
            f"Checksum mismatch for {filename}: expected {expected_sha}, got {actual_sha}. "
            "File may be corrupted or incomplete."
        )


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
