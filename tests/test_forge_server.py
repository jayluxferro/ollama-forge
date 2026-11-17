"""Tests for the unified forge API server (server/ package)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from ollama_forge.agentic.eval import BUILTIN_SUITE, ToolCallCase, run_eval
from ollama_forge.server.app import create_app
from ollama_forge.server.backends import (
    LlamaCppBackend,
    MlxLmBackend,
    MlxVlmBackend,
    OllamaBackend,
)
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


# ---------------------------------------------------------------------------
# registry.py — missing branch coverage
# ---------------------------------------------------------------------------


def test_detect_kind_vision_config(tmp_path: Path) -> None:
    d = tmp_path / "vlm"
    d.mkdir()
    import json as _json
    (d / "config.json").write_text(_json.dumps({"vision_config": {"hidden_size": 1024}}))
    assert detect_kind(d) == "mlx_vlm"


def test_estimate_mb_dir_safetensors(tmp_path: Path) -> None:
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x" * 2 * 1024 * 1024)
    mb = estimate_mb(d)
    assert mb == 2


def test_estimate_mb_dir_gguf_fallback(tmp_path: Path) -> None:
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    mb = estimate_mb(d)
    assert mb == 3


def test_estimate_mb_missing_path(tmp_path: Path) -> None:
    assert estimate_mb(tmp_path / "nonexistent") == 2000


# ---------------------------------------------------------------------------
# backends.py — _build_argv unit tests (pure functions, no subprocess)
# ---------------------------------------------------------------------------

def test_mlx_lm_build_argv_basic(tmp_path: Path) -> None:
    b = MlxLmBackend("mymodel", tmp_path / "model")
    argv = b._build_argv("127.0.0.1", 9999)
    assert "mlx_lm" in argv
    assert "server" in argv
    assert "--port" in argv
    assert "9999" in argv


def test_mlx_lm_build_argv_with_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapters"
    b = MlxLmBackend("mymodel", tmp_path / "model", adapter_path=adapter)
    argv = b._build_argv("127.0.0.1", 9999)
    assert "--adapter-path" in argv
    assert str(adapter) in argv


def test_mlx_vlm_build_argv_basic(tmp_path: Path) -> None:
    b = MlxVlmBackend("vlm", tmp_path / "vlm-model")
    argv = b._build_argv("127.0.0.1", 8888)
    assert "mlx_vlm.server" in argv
    assert "--kv-bits" not in argv


def test_mlx_vlm_build_argv_with_kv_bits(tmp_path: Path) -> None:
    b = MlxVlmBackend("vlm", tmp_path / "vlm-model", kv_bits=3.5)
    argv = b._build_argv("127.0.0.1", 8888)
    assert "--kv-bits" in argv
    assert "3.5" in argv


def test_llama_cpp_build_argv(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    b = LlamaCppBackend("llama", gguf)
    with patch("ollama_forge.run_helpers._which_llama_server", return_value="/usr/local/bin/llama-server"):
        argv = b._build_argv("127.0.0.1", 7777)
    assert argv[0] == "/usr/local/bin/llama-server"
    assert "-m" in argv
    assert str(gguf.resolve()) in argv


def test_llama_cpp_build_argv_no_server(tmp_path: Path) -> None:
    b = LlamaCppBackend("llama", tmp_path / "model.gguf")
    with patch("ollama_forge.run_helpers._which_llama_server", return_value=None):
        try:
            b._build_argv("127.0.0.1", 7777)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "llama-server" in str(e)


def test_ollama_backend_base_url() -> None:
    b = OllamaBackend("http://127.0.0.1:11434")
    assert b.base_url == "http://127.0.0.1:11434"


def test_ollama_backend_start_stop() -> None:
    b = OllamaBackend("http://127.0.0.1:11434")
    _run(b.start())
    assert b._client is not None
    _run(b.stop())
    assert b._client is None


def test_ollama_backend_healthy_no_client() -> None:
    b = OllamaBackend()
    assert _run(b.healthy()) is False


def test_subprocess_backend_stop_cleans_up(tmp_path: Path) -> None:
    """stop() terminates the proc and closes the client even if the proc has already exited."""
    b = MlxLmBackend("m", tmp_path / "model")
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    b._proc = mock_proc
    b._client = mock_client
    b._port = 1234

    _run(b.stop())

    mock_proc.terminate.assert_called_once()
    mock_client.aclose.assert_called_once()
    assert b._proc is None
    assert b._client is None
    assert b._port is None


def test_subprocess_backend_base_url(tmp_path: Path) -> None:
    b = MlxLmBackend("m", tmp_path / "model")
    b._port = 5678
    assert b.base_url == "http://127.0.0.1:5678"


# ---------------------------------------------------------------------------
# app.py — missing branch coverage
# ---------------------------------------------------------------------------


def _make_fake_local_backend(chat_response: dict) -> MagicMock:
    """Return a fake subprocess backend with a mocked httpx client."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = chat_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_response)

    backend = MagicMock()
    backend._client = mock_client
    backend.start = AsyncMock()
    backend.stop = AsyncMock()
    return backend


def test_lifespan_tears_down_backends() -> None:
    """Backend in the dict gets stop() called during lifespan shutdown."""
    reg = ModelRegistry()
    tracker = MemoryTracker(budget_mb=4096)
    fake_backend = _make_fake_local_backend({})

    ollama, _ = _make_fake_ollama({"id": "x"})
    backends = {"fake": fake_backend}
    app = create_app(reg, tracker, backends, ollama)

    with TestClient(app):
        pass  # enter + exit triggers lifespan start + stop

    fake_backend.stop.assert_called_once()


def test_chat_routes_to_local_backend(tmp_path: Path) -> None:
    """Model registered locally routes to local backend, not Ollama."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    reg = ModelRegistry()
    reg.register("local-llm", f)

    expected = {"id": "local-resp", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
    local_b = _make_fake_local_backend(expected)

    tracker = MemoryTracker(budget_mb=4096)
    ollama, mock_ollama = _make_fake_ollama({"id": "should-not-be-called"})
    backends = {"local-llm": local_b}
    # pre-mark as loaded so admit() doesn't try to start the subprocess
    tracker._loaded["local-llm"] = reg.get("local-llm").estimated_mb

    app = create_app(reg, tracker, backends, ollama)

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 200
    assert r.json()["id"] == "local-resp"


def test_chat_model_in_registry_no_backend_returns_503(tmp_path: Path) -> None:
    """Model registered but no backend instance → 503."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    reg = ModelRegistry()
    reg.register("orphan", f)

    tracker = MemoryTracker(budget_mb=4096)
    ollama, _ = _make_fake_ollama({})
    app = create_app(reg, tracker, {}, ollama)  # backends dict is empty

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "orphan",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 503


def test_list_models_merges_ollama_models(tmp_path: Path) -> None:
    """When /v1/models on Ollama returns models they are merged into the listing."""
    reg = ModelRegistry()
    tracker = MemoryTracker(budget_mb=4096)
    ollama, _ = _make_fake_ollama({})

    ollama_models = {"data": [{"id": "ollama-only", "object": "model"}]}
    app = create_app(reg, tracker, {}, ollama)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ollama_models
        mock_instance.get = AsyncMock(return_value=mock_resp)

        with TestClient(app) as client:
            r = client.get("/v1/models")

    ids = {m["id"] for m in r.json()["data"]}
    assert "ollama-only" in ids


def test_default_budget_mb_psutil_fallback() -> None:
    """_default_budget_mb uses psutil when sysctl fails."""
    from ollama_forge.server.app import _default_budget_mb
    with patch("subprocess.check_output", side_effect=Exception("no sysctl")), \
            patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.total = 32 * 1024 * 1024 * 1024  # 32 GB
        mb = _default_budget_mb()
    assert mb == int(32 * 1024 * 0.75)


def test_default_budget_mb_hardcoded_fallback() -> None:
    """_default_budget_mb falls back to 8192 when both sysctl and psutil fail."""
    from ollama_forge.server.app import _default_budget_mb
    with patch("subprocess.check_output", side_effect=Exception), \
            patch("psutil.virtual_memory", side_effect=Exception):
        mb = _default_budget_mb()
    assert mb == 8192


# ---------------------------------------------------------------------------
# agentic/eval.py — missing branch coverage
# ---------------------------------------------------------------------------

def _fake_urlopen_no_tools() -> MagicMock:
    """Response with no tool_calls."""
    import json as _json
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = _json.dumps({
        "choices": [{"message": {"role": "assistant", "content": "I cannot do that."}}]
    }).encode()
    return mock


def test_eval_case_with_system_prompt() -> None:
    case = ToolCallCase(
        name="with_system",
        prompt="What time is it?",
        tools=[{"type": "function", "function": {"name": "get_time", "description": "...",
                "parameters": {"type": "object", "properties": {}, "required": []}}}],
        expected_fn="get_time",
        system="You always call tools.",
    )
    import json as _json
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = _json.dumps({
        "choices": [{"message": {"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
        ]}}]
    }).encode()
    with patch("urllib.request.urlopen", return_value=mock):
        results = run_eval("test-model", cases=[case])
    assert results[0].fn_matched


def test_eval_no_tool_calls_in_response() -> None:
    case = BUILTIN_SUITE[0]
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_no_tools()):
        results = run_eval("test-model", cases=[case])
    assert not results[0].fn_matched
    assert not results[0].passed


def test_eval_generic_exception() -> None:
    case = BUILTIN_SUITE[0]
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        results = run_eval("test-model", cases=[case])
    assert results[0].error == "network down"
    assert not results[0].passed


def test_detect_kind_corrupt_config_json(tmp_path: Path) -> None:
    """Config.json that cannot be parsed falls through to mlx_lm."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("not valid json {{{")
    assert detect_kind(d) == "mlx_lm"


def test_default_budget_mb_sysctl_success() -> None:
    from ollama_forge.server.app import _default_budget_mb
    with patch("subprocess.check_output", return_value="36864\n"):
        mb = _default_budget_mb()
    assert mb == 36864


def test_list_models_exception_in_httpx(tmp_path: Path) -> None:
    """Exception from httpx inside list_models is swallowed; local models still returned."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    reg = ModelRegistry()
    reg.register("local-gguf", f)
    tracker = MemoryTracker(budget_mb=4096)
    ollama, _ = _make_fake_ollama({})
    app = create_app(reg, tracker, {}, ollama)

    with patch("httpx.AsyncClient") as mock_cls:
        # simulate get() raising inside the async context manager
        mock_instance = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with TestClient(app) as client:
            r = client.get("/v1/models")

    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "local-gguf" in ids


def test_chat_streaming_routes_to_ollama() -> None:
    """Streaming response returns SSE data from the Ollama backend."""
    async def _fake_stream_gen():
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_stream_ctx.aiter_bytes = _fake_stream_gen

    ollama, mock_client = _make_fake_ollama({})
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    reg = ModelRegistry()
    tracker = MemoryTracker(budget_mb=4096)
    app = create_app(reg, tracker, {}, ollama)

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "llama3:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
