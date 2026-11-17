"""Tests for the unified forge API server (server/ package)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from ollama_forge.server.app import create_app
from ollama_forge.server.backends import OllamaBackend
from ollama_forge.server.memory import MemoryTracker
from ollama_forge.server.registry import ModelRegistry, detect_kind, estimate_mb


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------


def test_detect_kind_gguf(tmp_path: Path) -> None:
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    assert detect_kind(f) == "llama_cpp"


def test_detect_kind_mlx_lm(tmp_path: Path) -> None:
    d = tmp_path / "mlx_model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert detect_kind(d) == "mlx_lm"


def test_detect_kind_mlx_vlm(tmp_path: Path) -> None:
    d = tmp_path / "vlm_model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"vision_config": {"hidden_size": 1024}}))
    assert detect_kind(d) == "mlx_vlm"


def test_detect_kind_fallback_ollama() -> None:
    assert detect_kind(Path("/no/such/path")) == "ollama"


def test_estimate_mb_gguf(tmp_path: Path) -> None:
    f = tmp_path / "model.gguf"
    f.write_bytes(b"x" * 2 * 1024 * 1024)  # 2 MB
    assert estimate_mb(f) == 2


def test_estimate_mb_dir(tmp_path: Path) -> None:
    d = tmp_path / "mlx"
    d.mkdir()
    (d / "weights.safetensors").write_bytes(b"x" * 4 * 1024 * 1024)  # 4 MB
    assert estimate_mb(d) == 4


def test_registry_register_gguf(tmp_path: Path) -> None:
    f = tmp_path / "llama.gguf"
    f.write_bytes(b"x" * 1024 * 1024)
    reg = ModelRegistry()
    rec = reg.register("llama3", f)
    assert rec.kind == "llama_cpp"
    assert rec.model_id == "llama3"
    assert rec.path == f
    assert rec.estimated_mb >= 1


def test_registry_register_ollama() -> None:
    reg = ModelRegistry()
    rec = reg.register("remote/model")
    assert rec.kind == "ollama"
    assert reg.get("remote/model") is rec


def test_registry_all() -> None:
    reg = ModelRegistry()
    reg.register("a")
    reg.register("b")
    assert {r.model_id for r in reg.all()} == {"a", "b"}


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------


def test_memory_admit_starts_backend() -> None:
    async def _test():
        tracker = MemoryTracker(budget_mb=4096)
        mock_backend = AsyncMock()
        backends = {"m1": mock_backend}
        await tracker.admit("m1", backends, 1000)
        mock_backend.start.assert_awaited_once()
        assert "m1" in tracker._loaded
    _run(_test())


def test_memory_admit_idempotent() -> None:
    async def _test():
        tracker = MemoryTracker(budget_mb=4096)
        mock_backend = AsyncMock()
        backends = {"m1": mock_backend}
        await tracker.admit("m1", backends, 1000)
        await tracker.admit("m1", backends, 1000)
        mock_backend.start.assert_awaited_once()  # only once
    _run(_test())


def test_memory_evicts_lru() -> None:
    async def _test():
        tracker = MemoryTracker(budget_mb=1000)
        b1, b2 = AsyncMock(), AsyncMock()
        backends = {"m1": b1, "m2": b2}
        await tracker.admit("m1", backends, 600)  # uses 600 MB
        await tracker.admit("m2", backends, 600)  # triggers eviction of m1
        b1.stop.assert_awaited_once()
        assert "m1" not in tracker._loaded
        assert "m2" in tracker._loaded
    _run(_test())


def test_memory_no_evict_in_flight() -> None:
    async def _test():
        tracker = MemoryTracker(budget_mb=1000)
        b1, b2 = AsyncMock(), AsyncMock()
        backends = {"m1": b1, "m2": b2}
        await tracker.admit("m1", backends, 600)
        tracker.mark_in_flight("m1")
        # m2 admission cannot evict m1 because it has an in-flight request
        # It will try to start m2 but m1 stays
        await tracker.admit("m2", backends, 600)
        b1.stop.assert_not_awaited()
        assert "m1" in tracker._loaded
    _run(_test())


def test_memory_refcount() -> None:
    tracker = MemoryTracker(budget_mb=4096)
    tracker._loaded["m1"] = 100
    tracker.mark_in_flight("m1")
    tracker.mark_in_flight("m1")
    assert tracker._in_flight["m1"] == 2
    tracker.mark_done("m1")
    assert tracker._in_flight["m1"] == 1
    tracker.mark_done("m1")
    assert "m1" not in tracker._in_flight


# ---------------------------------------------------------------------------
# app.py — via TestClient with fake backends
# ---------------------------------------------------------------------------


def _make_fake_ollama(chat_response: dict) -> tuple[OllamaBackend, AsyncMock]:
    """Return (fake OllamaBackend, mock_client). The mock_client is what gets injected on start()."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = chat_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.get = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {"data": []}))
    mock_client.aclose = AsyncMock()

    backend = OllamaBackend()
    # patch start() to inject the mock client rather than opening a real httpx.AsyncClient
    async def _fake_start() -> None:
        backend._client = mock_client
    backend.start = _fake_start  # type: ignore[method-assign]
    backend.stop = AsyncMock()  # type: ignore[method-assign]
    return backend, mock_client


def _make_app(registry: ModelRegistry | None = None) -> tuple:
    reg = registry or ModelRegistry()
    tracker = MemoryTracker(budget_mb=4096)
    backends: dict = {}
    fake_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}],
    }
    ollama, mock_client = _make_fake_ollama(fake_response)
    app = create_app(reg, tracker, backends, ollama)
    return app, mock_client, fake_response


def test_health_endpoint() -> None:
    app, _, _ = _make_app()
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "budget_mb" in body


def test_list_models_empty() -> None:
    app, mock_client, _ = _make_app()
    # make /v1/models Ollama call fail (no Ollama running in tests)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("no server"))
    with TestClient(app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_chat_no_model_field() -> None:
    app, _, _ = _make_app()
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_chat_routes_to_ollama() -> None:
    app, _, expected = _make_app()
    payload = {
        "model": "llama3:latest",
        "messages": [{"role": "user", "content": "hi"}],
    }
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert r.json()["id"] == expected["id"]


def test_list_models_includes_local(tmp_path: Path) -> None:
    f = tmp_path / "model.gguf"
    f.write_bytes(b"x" * 1024 * 1024)
    reg = ModelRegistry()
    reg.register("local-llama", f)
    app, mock_client, _ = _make_app(registry=reg)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("no server"))
    with TestClient(app) as client:
        r = client.get("/v1/models")
    ids = {m["id"] for m in r.json()["data"]}
    assert "local-llama" in ids
