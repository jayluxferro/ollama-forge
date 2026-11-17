"""FastAPI app factory for the unified forge API server."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import Backend, OllamaBackend
from .memory import MemoryTracker
from .registry import ModelRegistry


def _default_budget_mb() -> int:
    """Read iogpu.wired_limit_mb from sysctl; fall back to 75% of RAM."""
    try:
        import subprocess as _sp
        raw = _sp.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip()
        return int(raw)
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1024 / 1024 * 0.75)
    except Exception:
        return 8192


def create_app(
    registry: ModelRegistry,
    tracker: MemoryTracker,
    backends: dict[str, Backend],
    ollama_backend: OllamaBackend,
) -> FastAPI:
    """
    Build the FastAPI app.

    Why factory: tests can inject fake backends and a tiny budget without
    touching global state or spawning real subprocesses.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await ollama_backend.start()
        yield
        # shut down subprocess backends so Metal memory is returned immediately
        for b in list(backends.values()):
            await b.stop()
        await ollama_backend.stop()

    app = FastAPI(title="ollama-forge API", version="1.0", lifespan=lifespan)

    # ------------------------------------------------------------------ models

    @app.get("/v1/models")
    async def list_models() -> dict:
        data: list[dict] = []
        # locally registered models
        for rec in registry.all():
            data.append({
                "id": rec.model_id,
                "object": "model",
                "created": 0,
                "owned_by": f"ollama-forge/{rec.kind}",
            })
        # Ollama models — merged if Ollama is reachable
        try:
            async with httpx.AsyncClient(timeout=3.0) as cl:
                r = await cl.get(f"{ollama_backend.base_url}/v1/models")
                if r.status_code == 200:
                    local_ids = {r.model_id for r in registry.all()}
                    for m in r.json().get("data", []):
                        if m["id"] not in local_ids:
                            data.append(m)
        except Exception:
            pass
        return {"object": "list", "data": data}

    # --------------------------------------------------- chat completions proxy

    async def _resolve_backend(model_id: str) -> tuple[Backend, str]:
        """Return (backend, endpoint_path) for model_id."""
        rec = registry.get(model_id)
        if rec is not None:
            b = backends.get(model_id)
            if b is None:
                raise HTTPException(status_code=503, detail=f"No backend instance for {model_id!r}")
            # may start the backend and evict LRU models to stay within budget
            await tracker.admit(model_id, backends, rec.estimated_mb)
            return b, "/v1/chat/completions"
        # fall through to Ollama
        return ollama_backend, "/v1/chat/completions"

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
        payload = await request.json()
        model_id = payload.get("model", "")
        if not model_id:
            raise HTTPException(status_code=400, detail="model field is required")

        backend, path = await _resolve_backend(model_id)
        is_stream = bool(payload.get("stream"))

        if is_stream:
            async def _gen() -> AsyncIterator[bytes]:
                tracker.mark_in_flight(model_id)
                try:
                    # why own the stream context: httpx closes the connection
                    # when the context exits; callers reading a streaming SSE
                    # response need the connection alive until the last chunk
                    assert backend._client is not None  # type: ignore[attr-defined]
                    async with backend._client.stream(  # type: ignore[attr-defined]
                        "POST", path,
                        json=payload,
                        timeout=httpx.Timeout(None, connect=10.0),
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                finally:
                    tracker.mark_done(model_id)

            return StreamingResponse(_gen(), media_type="text/event-stream")

        # non-streaming
        tracker.mark_in_flight(model_id)
        try:
            assert backend._client is not None  # type: ignore[attr-defined]
            resp = await backend._client.post(  # type: ignore[attr-defined]
                path,
                json=payload,
                timeout=httpx.Timeout(None, connect=10.0),
            )
        finally:
            tracker.mark_done(model_id)

        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    # ------------------------------------------------------------------ health

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "budget_mb": tracker.budget_mb,
            "used_mb": tracker.used_mb(),
            "loaded": list(tracker._loaded.keys()),
        }

    return app
