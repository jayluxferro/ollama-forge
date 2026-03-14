"""Tests for the device abstraction module."""

from __future__ import annotations

from ollama_forge.device import (
    empty_cache,
    get_device,
    get_device_map,
    get_device_name,
    get_memory_info,
    is_cuda_available,
    is_gpu_available,
    is_mps_available,
    is_oom_error,
    supports_bfloat16,
)


def test_get_device_cpu_fallback() -> None:
    """get_device should return a valid device string."""
    device = get_device("cpu")
    assert device == "cpu"


def test_get_device_auto_returns_string() -> None:
    device = get_device("auto")
    assert device in ("cuda", "mps", "cpu")


def test_get_device_name_returns_string() -> None:
    name = get_device_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_device_map_cpu() -> None:
    assert get_device_map("cpu") == "cpu"


def test_get_device_map_mps() -> None:
    result = get_device_map("mps")
    assert result == {"": "mps"}


def test_get_device_map_auto() -> None:
    result = get_device_map("auto")
    assert result in ("auto", {"": "mps"})


def test_get_device_map_none() -> None:
    result = get_device_map(None)
    assert result in ("auto", {"": "mps"})


def test_get_device_map_cuda() -> None:
    result = get_device_map("cuda")
    assert result == "auto"


def test_is_gpu_available_returns_bool() -> None:
    assert isinstance(is_gpu_available(), bool)


def test_is_cuda_available_returns_bool() -> None:
    assert isinstance(is_cuda_available(), bool)


def test_is_mps_available_returns_bool() -> None:
    assert isinstance(is_mps_available(), bool)


def test_get_memory_info_returns_info_or_none() -> None:
    info = get_memory_info()
    if info is not None:
        assert info.total_gb > 0
        assert info.free_gb >= 0
        assert len(info.device_name) > 0


def test_empty_cache_does_not_raise() -> None:
    empty_cache()  # Should never raise


def test_supports_bfloat16_returns_bool() -> None:
    assert isinstance(supports_bfloat16(), bool)


def test_is_oom_error_runtime_error() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert is_oom_error(RuntimeError("MPS backend out of memory"))
    assert not is_oom_error(RuntimeError("some other error"))
    assert not is_oom_error(ValueError("out of memory"))  # wrong type


def test_is_oom_error_regular_exception() -> None:
    assert not is_oom_error(Exception("out of memory"))
