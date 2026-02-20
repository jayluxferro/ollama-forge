"""Model family detection and family-specific configuration for abliterated models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelFamily:
    """Configuration for a model family."""
    name: str
    model_types: tuple[str, ...]
    tokenizer_classes: tuple[str, ...]
    architectures: tuple[str, ...]
    template_override: str | None = None
    stop_tokens: tuple[str, ...] = ()
    gguf_flags: dict[str, Any] | None = None


# Known model families with their detection patterns and configurations
_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="gemma",
        model_types=("gemma", "gemma2", "gemma3"),
        tokenizer_classes=("gemmatokenizer", "gemma2tokenizer"),
        architectures=("gemmaforsequenceclassification", "gemma2forsequenceclassification", "gemmaforquestionanswering"),
        template_override="""<bos>{{ if .System }}{{ .System }}

{{ end }}<<start_of_turn>>user
{{ .Prompt }}<<end_of_turn>>
<<start_of_turn>>model
{{ .Response }}""",
        stop_tokens=("<<end_of_turn>>", "<<start_of_turn>>", "<eos>", "<end_of_turn>"),
    ),
    ModelFamily(
        name="llama3",
        model_types=("llama",),
        tokenizer_classes=("llamatokenizer", "llamatokenizerfast"),
        architectures=("llamaforcausallm",),
        stop_tokens=("<|eot_id|>", "<|end_of_text|>"),
    ),
    ModelFamily(
        name="mistral",
        model_types=("mistral",),
        tokenizer_classes=("mistraltokenizer", "mistraltokenizerfast"),
        architectures=("mistralforcausallm",),
        stop_tokens=("[/INST]", "</s>"),
    ),
    ModelFamily(
        name="mixtral",
        model_types=("mixtral",),
        tokenizer_classes=("llamatokenizer", "llamatokenizerfast"),
        architectures=("mixtralforsequenceclassification", "mixtralforcausallm"),
        stop_tokens=("[/INST]", "</s>"),
    ),
    ModelFamily(
        name="qwen2",
        model_types=("qwen2", "qwen2_moe"),
        tokenizer_classes=("qwen2tokenizer", "qwen2tokenizerfast"),
        architectures=("qwen2forcausallm", "qwen2moeforcausallm"),
        stop_tokens=("<|im_end|>", "<|endoftext|>"),
    ),
    ModelFamily(
        name="phi3",
        model_types=("phi3", "phi"),
        tokenizer_classes=(),  # Phi-3 uses LlamaTokenizer but we detect via model_type
        architectures=("phi3forcausallm", "phiforcausallm"),
        stop_tokens=("<|end|>", "<|endoftext|>"),
    ),
    ModelFamily(
        name="deepseek",
        model_types=("deepseek", "deepseek_v2", "deepseek_v3"),
        tokenizer_classes=("deepseekvtokenizer", "llamatokenizer"),
        architectures=("deepseekvforcausallm",),
        stop_tokens=("<|end_of_text|>", "<|end▁of▁sentence|>"),
    ),
    ModelFamily(
        name="cohere",
        model_types=("cohere", "command-r"),
        tokenizer_classes=("coheretokenizer", "coheretokenizerfast"),
        architectures=("cohereforsequenceclassification", "cohereforcausallm"),
        stop_tokens=("<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>"),
    ),
    ModelFamily(
        name="yi",
        model_types=("yi",),
        tokenizer_classes=("llamatokenizer", "llamatokenizerfast"),
        architectures=("yiforcausallm",),
        stop_tokens=("<|im_end|>", "<|endoftext|>"),
    ),
)


def _normalize(s: str | None) -> str:
    """Normalize string for matching (lowercase, remove dashes/underscores)."""
    if not s:
        return ""
    return s.lower().replace("-", "").replace("_", "")


def detect_model_family(checkpoint_dir: str | Path) -> ModelFamily | None:
    """
    Detect model family from checkpoint config.json/tokenizer_config.json.
    Returns ModelFamily or None if unknown.
    
    Detection priority: model_type > architectures > tokenizer_class
    (model_type is most specific, tokenizer_class is most generic)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    config_data: dict[str, Any] = {}
    tokenizer_data: dict[str, Any] = {}
    
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    tokenizer_path = checkpoint_dir / "tokenizer_config.json"
    if tokenizer_path.is_file():
        try:
            tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    model_type = _normalize(config_data.get("model_type"))
    tokenizer_class = _normalize(tokenizer_data.get("tokenizer_class"))
    architectures = [_normalize(a) for a in config_data.get("architectures", [])]
    
    # Priority 1: Match by model_type (most specific)
    if model_type:
        for family in _MODEL_FAMILIES:
            if any(model_type == _normalize(mt) or model_type.startswith(_normalize(mt)) for mt in family.model_types):
                return family
    
    # Priority 2: Match by architecture
    if architectures:
        for family in _MODEL_FAMILIES:
            if any(any(_normalize(fa) == arch for fa in family.architectures) for arch in architectures):
                return family
    
    # Priority 3: Match by tokenizer_class (least specific, more prone to false positives)
    if tokenizer_class:
        for family in _MODEL_FAMILIES:
            if not family.tokenizer_classes:
                continue
            # Require exact match or family-specific substring
            for tc in family.tokenizer_classes:
                tc_norm = _normalize(tc)
                if tokenizer_class == tc_norm or (family.name in tokenizer_class and tc_norm in tokenizer_class):
                    return family
    
    return None


def get_family_stop_tokens(checkpoint_dir: str | Path) -> list[str]:
    """Get stop tokens for the detected model family."""
    family = detect_model_family(checkpoint_dir)
    if family and family.stop_tokens:
        return list(family.stop_tokens)
    return []


def get_family_template_override(checkpoint_dir: str | Path) -> str | None:
    """Get template override for the detected model family (if any)."""
    family = detect_model_family(checkpoint_dir)
    if family and family.template_override:
        return family.template_override
    return None


def get_family_name(checkpoint_dir: str | Path) -> str | None:
    """Get the detected family name or None."""
    family = detect_model_family(checkpoint_dir)
    return family.name if family else None


def suggest_gguf_flags(checkpoint_dir: str | Path) -> dict[str, Any]:
    """
    Suggest flags for convert_hf_to_gguf.py based on model family.
    Returns dict of flag_name -> value.
    """
    family = detect_model_family(checkpoint_dir)
    if family and family.gguf_flags:
        return dict(family.gguf_flags)
    return {}
