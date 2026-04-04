"""VLM (Vision Language Model) backend — mlx-vlm wrapper for Apple Silicon.

Provides multimodal (image + audio + video + text) inference using mlx-vlm on
Apple Silicon Macs.  Falls back gracefully if mlx-vlm is not installed.

Requires mlx-vlm >= 0.4.4 for TurboQuant KV cache, VisionFeatureCache,
and video generation support.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Generator

_VLM_INSTALL_HINT = "mlx-vlm is required for VLM commands. Install with: pip install 'mlx-vlm>=0.4.4'"

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


# ---------------------------------------------------------------------------
# Vision Feature Cache
# ---------------------------------------------------------------------------


def vlm_create_vision_cache(max_size: int = 20) -> Any:
    """Create a VisionFeatureCache for multi-turn image caching.

    Caches encoded image features to avoid redundant vision encoder calls
    across conversation turns (11x+ speedup in multi-turn conversations).

    Args:
        max_size: Maximum number of cached image features (default: 20).

    Returns:
        A ``VisionFeatureCache`` instance with ``.get()``, ``.put()``,
        ``.clear()`` methods.
    """
    _require_vlm()
    from mlx_vlm import VisionFeatureCache

    return VisionFeatureCache(max_size=max_size)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------


def vlm_convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    dtype: str | None = None,
    upload_repo: str | None = None,
    revision: str | None = None,
    dequantize: bool = False,
    trust_remote_code: bool = False,
    quant_predicate: str | None = None,
) -> Path:
    """Convert a HuggingFace VLM to MLX format.

    Wraps ``mlx_vlm.convert()``.  Returns the output path.

    Args:
        hf_path: HuggingFace repo id or local path.
        mlx_path: Output directory for the converted model.
        quantize: Whether to quantize the model during conversion.
        q_bits: Quantization bits (default: 4).
        q_group_size: Quantization group size (default: 64).
        q_mode: Quantization mode — ``affine``, ``mxfp4``, ``nvfp4``,
            or ``mxfp8`` (default: ``affine``).
        dtype: Output dtype (e.g. ``float16``).
        upload_repo: Optional HuggingFace repo to upload the converted model.
        revision: HuggingFace revision (branch/tag/commit) to convert from.
        dequantize: Dequantize a quantized model back to full precision.
        trust_remote_code: Trust remote code when loading from HF Hub.
        quant_predicate: Mixed-bit quantization recipe string.

    Returns:
        :class:`~pathlib.Path` pointing to the output directory.
    """
    _require_vlm()
    from mlx_vlm import convert

    kwargs: dict[str, Any] = {
        "hf_path": hf_path,
        "mlx_path": mlx_path,
        "quantize": quantize,
        "q_bits": q_bits,
        "q_group_size": q_group_size,
        "q_mode": q_mode,
        "dtype": dtype,
        "upload_repo": upload_repo,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if dequantize:
        kwargs["dequantize"] = True
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if quant_predicate is not None:
        kwargs["quant_predicate"] = quant_predicate

    # mlx_vlm.convert takes hf_path as first positional arg
    convert(**kwargs)
    return Path(mlx_path)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def vlm_load(
    model_path: str,
    adapter_path: str | None = None,
    trust_remote_code: bool = False,
    quantize_activations: bool = False,
) -> tuple[Any, Any]:
    """Load a VLM model. Returns (model, processor).

    Args:
        model_path: HuggingFace repo id or local path
            (e.g. ``mlx-community/Qwen2-VL-2B-Instruct-4bit``).
        adapter_path: Optional path to a LoRA adapter.
        trust_remote_code: Trust remote code when loading from HF Hub.
        quantize_activations: Enable activation quantization for mxfp8 models.

    Returns:
        Tuple of ``(model, processor)`` ready for generation.
    """
    _require_vlm()
    from mlx_vlm import load

    kwargs: dict[str, Any] = {}
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if quantize_activations:
        kwargs["quantize_activations"] = True
    model, processor = load(model_path, **kwargs)
    return model, processor


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


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
            (``top_p``, ``kv_bits``, ``kv_quant_scheme``,
            ``kv_group_size``, ``quantized_kv_start``, ``max_kv_size``,
            ``prefill_step_size``, ``enable_thinking``,
            ``thinking_budget``, etc.).

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
        **kwargs: Extra args forwarded to ``mlx_vlm.stream_generate``
            (``top_p``, ``kv_bits``, ``kv_quant_scheme``, etc.).

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
        yield getattr(token, "text", str(token)) if not isinstance(token, str) else token


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


def vlm_video_generate(
    model_path: str,
    video: list[str],
    prompt: str = "Describe this video.",
    system: str | None = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    max_pixels: tuple[int, int] | None = None,
    max_frames: int | None = None,
    fps: float = 1.0,
    verbose: bool = True,
) -> int:
    """Generate text from video input via ``python -m mlx_vlm.video_generate``.

    Invokes as a subprocess since the video pipeline manages its own I/O.

    Returns the subprocess exit code.
    """
    _require_vlm()
    import subprocess

    if not video:
        raise ValueError("At least one --video path is required")

    cmd = [
        sys.executable, "-m", "mlx_vlm.video_generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--fps", str(fps),
    ]
    for v in video:
        cmd += ["--video", v]
    if system is not None:
        cmd += ["--system", system]
    if max_pixels is not None:
        cmd += ["--max-pixels", str(max_pixels[0]), str(max_pixels[1])]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    if not verbose:
        cmd.append("--verbose")  # mlx-vlm flag is store_false (toggles OFF)

    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def vlm_finetune(
    model_path: str,
    dataset: str,
    output_path: str = "vlm_adapter",
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    epochs: int | None = None,
    iters: int = 1000,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    train_vision: bool = False,
    full_finetune: bool = False,
    gradient_accumulation_steps: int = 1,
    grad_checkpoint: bool = False,
    grad_clip: float | None = None,
    adapter_path: str | None = None,
    split: str = "train",
    dataset_config: str | None = None,
    image_resize_shape: tuple[int, int] | None = None,
    custom_prompt_format: str | None = None,
    steps_per_report: int = 10,
    steps_per_eval: int = 200,
    steps_per_save: int = 100,
    val_batches: int = 25,
    max_seq_length: int = 2048,
    train_on_completions: bool = False,
    train_mode: str = "sft",
    beta: float = 0.1,
    eps: float = 1e-8,
    assistant_id: int = 77091,
) -> int:
    """Fine-tune a VLM using LoRA/QLoRA/full via mlx-vlm.

    Invokes ``python -m mlx_vlm.lora`` as a subprocess since the trainer
    module manages its own training loop.

    Returns the subprocess exit code.
    """
    _require_vlm()
    import subprocess

    # Validate that path-like arguments don't start with "--" to prevent argument injection
    for name, val in [("model_path", model_path), ("dataset", dataset), ("output_path", output_path)]:
        if val.startswith("-"):
            raise ValueError(f"{name} must not start with '-': {val!r}")

    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", model_path,
        "--dataset", dataset,
        "--output-path", output_path,
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--lora-rank", str(lora_rank),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--split", split,
        "--steps-per-report", str(steps_per_report),
        "--steps-per-eval", str(steps_per_eval),
        "--steps-per-save", str(steps_per_save),
        "--val-batches", str(val_batches),
        "--max-seq-length", str(max_seq_length),
        "--train-mode", train_mode,
        "--beta", str(beta),
        "--eps", str(eps),
        "--assistant-id", str(assistant_id),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if train_vision:
        cmd.append("--train-vision")
    if full_finetune:
        cmd.append("--full-finetune")
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if grad_clip is not None:
        cmd += ["--grad-clip", str(grad_clip)]
    if train_on_completions:
        cmd.append("--train-on-completions")
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]
    if dataset_config is not None:
        cmd += ["--dataset-config", dataset_config]
    if image_resize_shape is not None:
        cmd += ["--image-resize-shape", str(image_resize_shape[0]), str(image_resize_shape[1])]
    if custom_prompt_format is not None:
        cmd += ["--custom-prompt-format", custom_prompt_format]

    result = subprocess.run(cmd)
    return result.returncode
