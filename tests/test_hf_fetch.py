"""Tests for hf_fetch module."""

import pytest

from ollama_forge.hf_fetch import pick_one_gguf


def test_pick_one_gguf_empty_raises() -> None:
    """Empty filenames list raises ValueError."""
    with pytest.raises(ValueError, match="empty filenames"):
        pick_one_gguf([])


def test_pick_one_gguf_prefer_quant() -> None:
    """When prefer_quant is set, that quantization is chosen."""
    files = ["a-Q8_0.gguf", "b-Q4_K_M.gguf", "c-Q4_0.gguf"]
    assert pick_one_gguf(files, prefer_quant="Q4_0") == "c-Q4_0.gguf"
    assert pick_one_gguf(files, prefer_quant="Q8_0") == "a-Q8_0.gguf"


def test_pick_one_gguf_single() -> None:
    """Single file is returned as-is."""
    assert pick_one_gguf(["model.gguf"]) == "model.gguf"


def test_pick_one_gguf_prefers_q4_k_m() -> None:
    """When multiple, prefers Q4_K_M (case-insensitive)."""
    files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q4_0.gguf"]
    assert pick_one_gguf(files) == "model-Q4_K_M.gguf"


def test_pick_one_gguf_prefers_q4_k_s_over_q8() -> None:
    """Order of preference: q4_k_m, q4_k_s, q5_k_m, q4_0, q8_0."""
    files = ["a-q8_0.gguf", "b-q4_k_s.gguf"]
    assert pick_one_gguf(files) == "b-q4_k_s.gguf"


def test_pick_one_gguf_fallback_first() -> None:
    """When no preferred name, returns first."""
    files = ["other.gguf", "another.gguf"]
    assert pick_one_gguf(files) == "other.gguf"
