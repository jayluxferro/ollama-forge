"""VLM (Vision Language Model) backend — mlx-vlm wrapper for Apple Silicon.

Provides multimodal (image + audio + text) inference using mlx-vlm on
Apple Silicon Macs.  Falls back gracefully if mlx-vlm is not installed.
"""

from __future__ import annotations

from typing import Any, Generator

_VLM_INSTALL_HINT = "mlx-vlm is required for VLM commands. Install with: pip install 'mlx-vlm>=0.4.3'"

_MLX_VLM_AVAILABLE = False
try:
    import mlx_vlm as _mlx_vlm  # noqa: F401

    _MLX_VLM_AVAILABLE = True
except ImportError:
    pass


def is_vlm_available() -> bool:
    """Check if mlx-vlm is installed."""
    return _MLX_VLM_AVAILABLE


def _require_vlm() -> None:
    """Raise RuntimeError if mlx-vlm is not installed."""
    if not _MLX_VLM_AVAILABLE:
        raise RuntimeError(_VLM_INSTALL_HINT)


def vlm_load(model_path: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load a VLM model. Returns (model, processor).

    Args:
        model_path: HuggingFace repo id or local path
            (e.g. ``mlx-community/Qwen2-VL-2B-Instruct-4bit``).
        adapter_path: Optional path to a LoRA adapter.

    Returns:
        Tuple of ``(model, processor)`` ready for generation.
    """
    _require_vlm()
    from mlx_vlm import load

    kwargs: dict[str, Any] = {}
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    model, processor = load(model_path, **kwargs)
    return model, processor


def vlm_apply_chat_template(
    processor: Any,
    config: Any,
    prompt: str,
    num_images: int = 0,
    num_audios: int = 0,
) -> str:
    """Apply model-specific chat template to a prompt.

    Args:
        processor: The processor returned by :func:`vlm_load`.
        config: ``model.config`` from the loaded model.
        prompt: The user text prompt.
        num_images: Number of images that will be provided.
        num_audios: Number of audio files that will be provided.

    Returns:
        Formatted prompt string ready for generation.
    """
    _require_vlm()
    from mlx_vlm.prompt_utils import apply_chat_template

    kwargs: dict[str, Any] = {}
    if num_images:
        kwargs["num_images"] = num_images
    if num_audios:
        kwargs["num_audios"] = num_audios
    return apply_chat_template(processor, config, prompt, **kwargs)


def vlm_generate(
    model: Any,
    processor: Any,
    prompt: str,
    images: list[str] | None = None,
    audio: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate text from multimodal input.

    Args:
        model: The model returned by :func:`vlm_load`.
        processor: The processor returned by :func:`vlm_load`.
        prompt: Formatted prompt (after chat template).
        images: List of image paths or URLs.
        audio: Path to an audio file.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        **kwargs: Extra args forwarded to ``mlx_vlm.generate``
            (``top_p``, ``kv_bits``, ``enable_thinking``, etc.).

    Returns:
        Dict with keys ``text``, ``prompt_tps``, ``generation_tps``,
        ``peak_memory``.
    """
    _require_vlm()
    from mlx_vlm import generate

    gen_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if images:
        gen_kwargs["image"] = images
    if audio:
        gen_kwargs["audio"] = audio
    gen_kwargs.update(kwargs)

    result = generate(model, processor, prompt, **gen_kwargs)

    # mlx_vlm.generate may return a string or an object with attributes
    if isinstance(result, str):
        return {"text": result}
    return {
        "text": getattr(result, "text", str(result)),
        "prompt_tps": getattr(result, "prompt_tps", None),
        "generation_tps": getattr(result, "generation_tps", None),
        "peak_memory": getattr(result, "peak_memory", None),
    }


def vlm_stream_generate(
    model: Any,
    processor: Any,
    prompt: str,
    images: list[str] | None = None,
    audio: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Generator[str, None, None]:
    """Streaming generation. Yields tokens as strings.

    Args:
        model: The model returned by :func:`vlm_load`.
        processor: The processor returned by :func:`vlm_load`.
        prompt: Formatted prompt (after chat template).
        images: List of image paths or URLs.
        audio: Path to an audio file.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        **kwargs: Extra args forwarded to ``mlx_vlm.stream_generate``.

    Yields:
        Token strings as they are generated.
    """
    _require_vlm()
    from mlx_vlm import stream_generate

    gen_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if images:
        gen_kwargs["image"] = images
    if audio:
        gen_kwargs["audio"] = audio
    gen_kwargs.update(kwargs)

    for token in stream_generate(model, processor, prompt, **gen_kwargs):
        yield token
