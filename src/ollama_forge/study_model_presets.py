"""Curated model presets and hardware-tier recommendations for study workflows."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass


@dataclass
class ModelPreset:
    name: str
    hf_id: str
    description: str
    tier: str
    params: str
    recommended_dtype: str
    recommended_quantization: str | None = None
    gated: bool = False


_PRESETS = [
    ModelPreset(
        name="distilgpt2",
        hf_id="distilgpt2",
        description="Tiny GPT-2 derivative for smoke tests and CPU-only runs.",
        tier="tiny",
        params="82M",
        recommended_dtype="float32",
    ),
    ModelPreset(
        name="TinyLlama 1.1B Chat",
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Very small instruct model suited to fast study iterations.",
        tier="small",
        params="1.1B",
        recommended_dtype="float16",
    ),
    ModelPreset(
        name="Qwen2.5 0.5B Instruct",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        description="Tiny chat model for rapid ablation sweeps.",
        tier="tiny",
        params="0.5B",
        recommended_dtype="float16",
    ),
    ModelPreset(
        name="Qwen2.5 1.5B Instruct",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        description="Small multilingual instruct model.",
        tier="small",
        params="1.5B",
        recommended_dtype="float16",
    ),
    ModelPreset(
        name="Qwen2.5 3B Instruct",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        description="Small but capable model for general ablation studies.",
        tier="small",
        params="3B",
        recommended_dtype="float16",
    ),
    ModelPreset(
        name="Qwen2.5 7B Instruct",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        description="Strong 7B instruct model for balanced study runs.",
        tier="medium",
        params="7B",
        recommended_dtype="float16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="Qwen2.5 Coder 7B Instruct",
        hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="Code-oriented 7B model.",
        tier="medium",
        params="7B",
        recommended_dtype="float16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="Qwen3 8B",
        hf_id="Qwen/Qwen3-8B",
        description="General-purpose Qwen3 reasoning model.",
        tier="medium",
        params="8B",
        recommended_dtype="float16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="Mistral 7B Instruct v0.3",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        description="Widely used instruct baseline.",
        tier="medium",
        params="7B",
        recommended_dtype="float16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="Llama 3.2 3B Instruct",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        description="Compact Llama instruct model.",
        tier="small",
        params="3B",
        recommended_dtype="float16",
        gated=True,
    ),
    ModelPreset(
        name="Llama 3.1 8B Instruct",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        description="Standard 8B instruct benchmark target.",
        tier="medium",
        params="8B",
        recommended_dtype="float16",
        recommended_quantization="4bit",
        gated=True,
    ),
    ModelPreset(
        name="Yi 1.5 6B Chat",
        hf_id="01-ai/Yi-1.5-6B-Chat",
        description="Bilingual 6B chat model.",
        tier="medium",
        params="6B",
        recommended_dtype="float16",
    ),
    ModelPreset(
        name="Qwen2.5 14B",
        hf_id="Qwen/Qwen2.5-14B",
        description="Larger dense model for richer study signals.",
        tier="large",
        params="14B",
        recommended_dtype="float16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="Qwen3 30B-A3B",
        hf_id="Qwen/Qwen3-30B-A3B",
        description="MoE model with consumer-friendly active parameter count.",
        tier="large",
        params="30B MoE",
        recommended_dtype="bfloat16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="gpt-oss 20B",
        hf_id="openai/gpt-oss-20b",
        description="Large open-weight reasoning target.",
        tier="large",
        params="20B",
        recommended_dtype="bfloat16",
        recommended_quantization="4bit",
    ),
    ModelPreset(
        name="Qwen2.5 72B",
        hf_id="Qwen/Qwen2.5-72B",
        description="Frontier-class dense model for multi-GPU studies.",
        tier="frontier",
        params="72B",
        recommended_dtype="bfloat16",
        recommended_quantization="4bit",
    ),
]

MODEL_PRESETS: dict[str, ModelPreset] = {preset.hf_id: preset for preset in _PRESETS}
_TIER_ORDER = ("tiny", "small", "medium", "large", "frontier")


def list_model_presets(*, tier: str | None = None) -> list[ModelPreset]:
    presets = list(_PRESETS)
    if tier:
        tier = tier.lower()
        presets = [preset for preset in presets if preset.tier == tier]
    return presets


def get_model_preset(hf_id: str) -> ModelPreset:
    if hf_id not in MODEL_PRESETS:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"Unknown model preset {hf_id!r}. Available: {available}")
    return MODEL_PRESETS[hf_id]


def detect_hardware_tier() -> tuple[str, dict[str, str | float | int]]:
    info: dict[str, str | float | int] = {"platform": platform.system()}
    try:
        import torch

        if torch.cuda.is_available():
            max_vram = 0.0
            gpu_count = torch.cuda.device_count()
            for idx in range(gpu_count):
                props = torch.cuda.get_device_properties(idx)
                max_vram = max(max_vram, props.total_memory / 1024**3)
            info["gpu_count"] = gpu_count
            info["max_vram_gb"] = round(max_vram, 1)
            if max_vram >= 80:
                return "frontier", info
            if max_vram >= 24:
                return "large", info
            if max_vram >= 8:
                return "medium", info
            return "small", info
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["accelerator"] = "mps"
            return "medium", info
    except Exception:
        pass
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1024**3
        info["ram_gb"] = round(ram_gb, 1)
        if ram_gb >= 32:
            return "small", info
    except Exception:
        pass
    return "tiny", info


def recommended_model_presets(*, tier: str | None = None, limit: int = 5) -> list[ModelPreset]:
    chosen_tier = tier or detect_hardware_tier()[0]
    if chosen_tier not in _TIER_ORDER:
        chosen_tier = "tiny"
    max_index = _TIER_ORDER.index(chosen_tier)
    candidates = [preset for preset in _PRESETS if _TIER_ORDER.index(preset.tier) <= max_index]
    return candidates[-limit:]


def format_hardware_info(info: dict[str, str | float | int]) -> str:
    parts = []
    if "platform" in info:
        parts.append(str(info["platform"]))
    if "gpu_count" in info:
        parts.append(f"gpus={info['gpu_count']}")
    if "max_vram_gb" in info:
        parts.append(f"max_vram_gb={info['max_vram_gb']}")
    if "accelerator" in info:
        parts.append(str(info["accelerator"]))
    if "ram_gb" in info:
        parts.append(f"ram_gb={info['ram_gb']}")
    if shutil.which("ollama"):
        parts.append("ollama=installed")
    return " ".join(parts)
