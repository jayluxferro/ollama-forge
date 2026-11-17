"""Backend abstractions for the unified forge API server.

Each backend is either a subprocess (llama-server, mlx_lm, mlx_vlm) or an
httpx pass-through (Ollama). Subprocess backends own their Metal memory;
terminating the process is the only reliable way to reclaim it on Apple Silicon.
"""
from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import httpx


class Backend(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def healthy(self) -> bool: ...

    @property
    @abstractmethod
    def base_url(self) -> str: ...

    async def forward(self, path: str, payload: dict) -> httpx.Response:
        """Non-streaming POST, returns full response."""
        assert self._client is not None, "backend not started"  # type: ignore[attr-defined]
        return await self._client.post(  # type: ignore[attr-defined]
            path,
            json=payload,
            timeout=httpx.Timeout(None, connect=10.0),
        )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class SubprocessBackend(Backend):
    """
    Launches an external server as a subprocess, waits on /health, then proxies.

    Why subprocess: subprocess termination is the only way Metal GPU memory
    reliably returns to the system on Apple Silicon. In-process model loading
    leaks memory across model swaps.
    """

    def __init__(self, model_id: str, model_path: Path, loopback: str = "127.0.0.1") -> None:
        self.model_id = model_id
        self._model_path = model_path
        self._loopback = loopback
        self._port: int | None = None
        self._proc: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None

    def _build_argv(self, host: str, port: int) -> list[str]:
        raise NotImplementedError

    async def start(self) -> None:
        self._port = _free_port()
        argv = self._build_argv(self._loopback, self._port)
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(None, connect=10.0),
        )
        await self._wait_healthy(timeout=120)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self._proc.wait), timeout=10)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
        self._port = None

    async def healthy(self) -> bool:
        if self._client is None or self._proc is None:
            return False
        try:
            r = await self._client.get("/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    async def _wait_healthy(self, timeout: float = 120.0) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await self.healthy():
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Backend {self.model_id!r} did not become healthy in {timeout}s")

    @property
    def base_url(self) -> str:
        return f"http://{self._loopback}:{self._port}"


class LlamaCppBackend(SubprocessBackend):
    """llama-server subprocess serving a GGUF model."""

    def __init__(self, model_id: str, model_path: Path, n_gpu_layers: int = -1, **kw: object) -> None:
        super().__init__(model_id, model_path, **kw)  # type: ignore[arg-type]
        self._n_gpu_layers = n_gpu_layers

    def _build_argv(self, host: str, port: int) -> list[str]:
        from ollama_forge.run_helpers import _which_llama_server

        server_bin = _which_llama_server()
        if not server_bin:
            raise RuntimeError("llama-server not found — run: ollama-forge setup-llama-cpp")
        return [
            server_bin,
            "-m", str(self._model_path.resolve()),
            "--host", host,
            "--port", str(port),
            "-ngl", str(self._n_gpu_layers),
        ]


class MlxLmBackend(SubprocessBackend):
    """mlx_lm server subprocess for MLX language models."""

    def __init__(self, model_id: str, model_path: Path, adapter_path: Path | None = None, **kw: object) -> None:
        super().__init__(model_id, model_path, **kw)  # type: ignore[arg-type]
        self._adapter_path = adapter_path

    def _build_argv(self, host: str, port: int) -> list[str]:
        argv = [
            sys.executable, "-m", "mlx_lm", "server",
            "--model", str(self._model_path),
            "--host", host,
            "--port", str(port),
        ]
        if self._adapter_path:
            argv += ["--adapter-path", str(self._adapter_path)]
        return argv


class MlxVlmBackend(SubprocessBackend):
    """mlx_vlm server subprocess for vision-language models, supporting --kv-bits."""

    def __init__(
        self,
        model_id: str,
        model_path: Path,
        kv_bits: float | None = None,
        **kw: object,
    ) -> None:
        super().__init__(model_id, model_path, **kw)  # type: ignore[arg-type]
        self._kv_bits = kv_bits

    def _build_argv(self, host: str, port: int) -> list[str]:
        argv = [
            sys.executable, "-m", "mlx_vlm.server",
            "--model", str(self._model_path),
            "--host", host,
            "--port", str(port),
        ]
        if self._kv_bits is not None:
            argv += ["--kv-bits", str(self._kv_bits)]
        return argv


class OllamaBackend(Backend):
    """
    Thin pass-through to the local Ollama server.

    Ollama's own models sit outside the MemoryTracker's view. The /v1/models
    listing notes this — Ollama's loaded model count is not subtracted from
    the forge budget.
    """

    def __init__(self, ollama_host: str = "http://127.0.0.1:11434") -> None:
        self._base = ollama_host.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(None, connect=5.0),
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def healthy(self) -> bool:
        if self._client is None:
            return False
        try:
            r = await self._client.get("/", timeout=2.0)
            return r.status_code < 500
        except Exception:
            return False

    @property
    def base_url(self) -> str:
        return self._base
