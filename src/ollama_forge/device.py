"""Unified device abstraction for CUDA, MPS (Apple Silicon), and CPU.

Centralizes device detection, memory queries, and dtype safety checks
so callers don't need to scatter torch.cuda / torch.backends.mps checks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryInfo:
    """GPU/accelerator memory snapshot."""

    used_gb: float
    total_gb: float
    free_gb: float
    device_name: str


def is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def is_mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


def is_gpu_available() -> bool:
    return is_cuda_available() or is_mps_available()


def is_mlx_available() -> bool:
    """Check if MLX (Apple ML framework) is installed and usable."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def is_vlm_available() -> bool:
    """Check if mlx-vlm is installed for vision model support."""
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def is_triton_available() -> bool:
    """Check if Triton (GPU kernel compiler) is installed."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def get_device(preference: str = "auto") -> str:
    """Resolve a device string.

    Args:
        preference: "auto" (best GPU), "cuda", "mps", or "cpu".

    Returns:
        Resolved device string suitable for torch.device().
    """
    if preference != "auto":
        return preference
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_turboquant_backend(preference: str = "auto") -> str:
    """Select the best TurboQuant inference backend.

    Returns:
        "mlx" — MLX on Apple Silicon (fastest on Mac)
        "triton" — Triton kernels on CUDA (fastest on NVIDIA)
        "pytorch" — Pure PyTorch (universal fallback)
    """
    if preference != "auto":
        return preference
    if is_mlx_available() and is_mps_available():
        return "mlx"
    if is_triton_available() and is_cuda_available():
        return "triton"
    return "pytorch"


def get_device_name() -> str:
    """Human-readable accelerator name."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
    except Exception:
        pass
    return "CPU"


def get_device_map(device: str | None) -> str | dict:
    """Return the HuggingFace device_map for model loading.

    - None / "auto" → "auto" (let accelerate decide)
    - "mps" → {"": "mps"} (accelerate doesn't reliably handle "mps" string)
    - "cpu" → "cpu"
    - "cuda" → "auto"
    """
    if device is None or device == "auto":
        if is_mps_available():
            return {"": "mps"}
        return "auto"
    if device == "mps":
        return {"": "mps"}
    if device == "cpu":
        return "cpu"
    return "auto"


def get_memory_info(device_index: int = 0) -> MemoryInfo | None:
    """Query accelerator memory. Returns None if no GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            total = props.total_mem / (1024**3)
            used = torch.cuda.memory_allocated(device_index) / (1024**3)
            return MemoryInfo(
                used_gb=round(used, 2),
                total_gb=round(total, 2),
                free_gb=round(total - used, 2),
                device_name=props.name,
            )
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS shares system RAM; estimate ~70% usable
            try:
                import os

                total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                total = total_bytes / (1024**3) * 0.70
            except Exception:
                total = 8.0  # conservative fallback
            return MemoryInfo(
                used_gb=0.0, total_gb=round(total, 2),
                free_gb=round(total, 2), device_name="Apple Silicon (MPS)",
            )
    except Exception:
        pass
    return None


def empty_cache() -> None:
    """Free GPU memory caches (CUDA and MPS)."""
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch, "mps", None) and getattr(torch.mps, "empty_cache", None):
            torch.mps.empty_cache()
    except Exception:
        pass


def supports_bfloat16() -> bool:
    """Check if the current device supports bfloat16."""
    try:
        import torch

        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            return cc[0] >= 8  # Ampere+
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS bfloat16 support added in PyTorch 2.3
            parts = torch.__version__.split(".")
            return int(parts[0]) >= 2 and int(parts[1]) >= 3
    except Exception:
        pass
    return True  # CPU always supports bf16


def is_oom_error(exc: Exception) -> bool:
    """Check if an exception is an out-of-memory error (CUDA or generic)."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
