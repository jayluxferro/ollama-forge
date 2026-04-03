"""VLM (Vision Language Model) backend — mlx-vlm wrapper for Apple Silicon.

Provides multimodal (image + audio + text) inference using mlx-vlm on
Apple Silicon Macs.  Falls back gracefully if mlx-vlm is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
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


def vlm_convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    dtype: str | None = None,
    upload_repo: str | None = None,
) -> Path:
    """Convert a HuggingFace VLM to MLX format.

    Wraps ``mlx_vlm.convert()``.  Returns the output path.

    Args:
        hf_path: HuggingFace repo id or local path.
        mlx_path: Output directory for the converted model.
        quantize: Whether to quantize the model during conversion.
        q_bits: Quantization bits (default: 4).
        q_group_size: Quantization group size (default: 64).
        dtype: Output dtype (e.g. ``float16``).
        upload_repo: Optional HuggingFace repo to upload the converted model.

    Returns:
        :class:`~pathlib.Path` pointing to the output directory.
    """
    _require_vlm()
    from mlx_vlm import convert

    convert(
        hf_path,
        mlx_path=mlx_path,
        quantize=quantize,
        q_bits=q_bits,
        q_group_size=q_group_size,
        dtype=dtype,
        upload_repo=upload_repo,
    )
    return Path(mlx_path)


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


def vlm_finetune(
    model_path: str,
    dataset: str,
    output_path: str = "vlm_adapter",
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    epochs: int = 1,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    train_vision: bool = False,
    full_finetune: bool = False,
    gradient_accumulation_steps: int = 1,
    grad_checkpoint: bool = False,
    adapter_path: str | None = None,
) -> int:
    """Fine-tune a VLM using LoRA/QLoRA/full via mlx-vlm.

    Invokes ``python -m mlx_vlm.lora`` as a subprocess since the trainer
    module manages its own training loop.

    Returns the subprocess exit code.
    """
    _require_vlm()
    import subprocess

    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", model_path,
        "--dataset", dataset,
        "--output-path", output_path,
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lora-rank", str(lora_rank),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
    ]
    if train_vision:
        cmd.append("--train-vision")
    if full_finetune:
        cmd.append("--full-finetune")
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    result = subprocess.run(cmd)
    return result.returncode
