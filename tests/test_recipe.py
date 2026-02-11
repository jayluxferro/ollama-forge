"""Recipe loading tests."""

import tempfile
from pathlib import Path

import pytest

from ollama_tools.recipe import load_recipe


def test_load_recipe_json_minimal() -> None:
    """Load minimal JSON recipe with base."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"name": "m", "base": "llama3.2"}')
        path = f.name
    try:
        r = load_recipe(path)
        assert r["name"] == "m"
        assert r["base"] == "llama3.2"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_json_with_optional() -> None:
    """Load JSON recipe with system, temperature."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"name": "m", "base": "llama3.2", "system": "Hi", "temperature": 0.5}')
        path = f.name
    try:
        r = load_recipe(path)
        assert r["system"] == "Hi"
        assert r["temperature"] == 0.5
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_json_gguf() -> None:
    """Load JSON recipe with gguf."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"name": "m", "gguf": "/path/to/model.gguf"}')
        path = f.name
    try:
        r = load_recipe(path)
        assert r["name"] == "m"
        assert r["gguf"] == "/path/to/model.gguf"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_missing_name() -> None:
    """Recipe without name raises."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"base": "llama3.2"}')
        path = f.name
    try:
        with pytest.raises(ValueError, match="name"):
            load_recipe(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_missing_source() -> None:
    """Recipe without base, gguf, or hf_repo raises."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"name": "m"}')
        path = f.name
    try:
        with pytest.raises(ValueError, match="one of"):
            load_recipe(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_json_hf_repo() -> None:
    """Load JSON recipe with hf_repo."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"name": "m", "hf_repo": "org/repo", "gguf_file": "x.gguf"}')
        path = f.name
    try:
        r = load_recipe(path)
        assert r["name"] == "m"
        assert r["hf_repo"] == "org/repo"
        assert r["gguf_file"] == "x.gguf"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_only_one_source() -> None:
    """Recipe with both base and hf_repo raises."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"name": "m", "base": "llama3.2", "hf_repo": "org/repo"}')
        path = f.name
    try:
        with pytest.raises(ValueError, match="only one of"):
            load_recipe(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recipe_file_not_found() -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_recipe("/nonexistent/recipe.json")
