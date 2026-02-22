"""Model family detection and family-specific configuration for abliterated models."""

from __future__ import annotations

import contextlib
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


# Known model families with their detection patterns and configurations.
# template_override: full Ollama Go-template string including tool-call sections.
# When set, this replaces HF-derived templates and preserves tool-calling capability
# for families whose HF Jinja2 template cannot be cleanly translated to Ollama format.
_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="gemma",
        model_types=("gemma", "gemma2", "gemma3", "gemma3_text"),
        tokenizer_classes=("gemmatokenizer", "gemma2tokenizer"),
        architectures=(
            "gemmaforsequenceclassification",
            "gemma2forsequenceclassification",
            "gemmaforquestionanswering",
        ),
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
        # Full tool-capable template matching Ollama's llama3.1/3.2 library template.
        # Supports: system prompts, tool definitions, tool calls, tool results, multi-turn.
        template_override="""{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

You have access to the following functions. To call a function, respond with JSON for a function call. Respond in the format:
{"name": "function_name", "parameters": {"key": "value"}}

Do not use variables.

{{ range .Tools }}
{{- . | toJson }}
{{ end }}
{{- end }}<|eot_id|>
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}
{{ end }}
{{- else }}
{{ .Content }}
{{- end }}{{ if not $last }}<|eot_id|>{{ end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>
{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
{{ end }}
{{- end }}
{{- end }}""",
        stop_tokens=("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"),
    ),
    ModelFamily(
        name="mistral",
        model_types=("mistral",),
        tokenizer_classes=("mistraltokenizer", "mistraltokenizerfast"),
        architectures=("mistralforcausallm",),
        # Tool-capable template for Mistral/Mistral-Nemo (v3 instruct format with [TOOL_CALLS]).
        template_override="""{{- if .System }}[INST] {{ .System }}

{{ .Prompt }}[/INST]
{{- else }}[INST] {{ .Prompt }}[/INST]
{{- end }}
{{- if .ToolCalls }}
[TOOL_CALLS] [{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{ end }}]</s>
{{- else }}
{{ .Response }}</s>
{{- end }}""",
        stop_tokens=("[/INST]", "</s>", "[TOOL_CALLS]"),
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
        model_types=("qwen2", "qwen2_moe", "qwen2_5", "qwen3"),
        tokenizer_classes=("qwen2tokenizer", "qwen2tokenizerfast"),
        architectures=("qwen2forcausallm", "qwen2moeforcausallm", "qwen25forcausallm"),
        # Tool-capable template for Qwen2/2.5 (ChatML format with tool call JSON blocks).
        template_override="""{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- if .Tools }}<|im_start|>system
You have access to the following tools:
{{ range .Tools }}
{{- . | toJson }}
{{ end }}
When you need to call a tool, respond with a JSON object inside <tool_call> tags:
<tool_call>
{"name": "tool_name", "arguments": {"key": "value"}}
</tool_call><|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .ToolCalls }}
{{ range .ToolCalls }}<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>{{ end }}
{{- else }}
{{ .Content }}
{{- end }}<|im_end|>
{{ else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- end }}<|im_start|>assistant
{{ .Response }}<|im_end|>""",
        stop_tokens=("<|im_end|>", "<|endoftext|>", "<tool_call>"),
    ),
    ModelFamily(
        name="phi3",
        model_types=("phi3", "phi", "phi4"),
        tokenizer_classes=(),  # Phi-3/4 uses LlamaTokenizer but detected via model_type
        architectures=("phi3forcausallm", "phiforcausallm", "phi3smallforcausallm"),
        stop_tokens=("<|end|>", "<|endoftext|>", "<|assistant|>"),
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


def _auto_family_from_config(
    config_data: dict[str, Any],
    tokenizer_data: dict[str, Any],
) -> ModelFamily | None:
    """Build a best-effort ModelFamily for models not in _MODEL_FAMILIES.

    Extracts stop tokens from eos_token in tokenizer_config.json. template_override
    is left None so modelfile.py falls back to deriving it from the HF chat_template.
    Returns None only if config.json has no model_type at all.
    """
    model_type = config_data.get("model_type")
    if not model_type:
        return None

    stop_tokens: list[str] = []
    eos = tokenizer_data.get("eos_token")
    if isinstance(eos, str) and eos:
        stop_tokens.append(eos)
    elif isinstance(eos, dict):
        content = eos.get("content", "")
        if content:
            stop_tokens.append(content)

    return ModelFamily(
        name=model_type,
        model_types=(model_type,),
        tokenizer_classes=(),
        architectures=tuple(config_data.get("architectures", [])),
        template_override=None,
        stop_tokens=tuple(stop_tokens),
    )


def detect_model_family(checkpoint_dir: str | Path) -> ModelFamily | None:
    """
    Detect model family from checkpoint config.json/tokenizer_config.json.

    Matching priority for the hardcoded _MODEL_FAMILIES table (which provides
    tool-calling template overrides and full stop-token lists for known families):
      model_type > architectures > tokenizer_class

    For models not in _MODEL_FAMILIES, falls back to _auto_family_from_config,
    which builds a minimal ModelFamily from the raw config files — eos_token
    becomes the stop token and template_override is None (modelfile.py derives
    the template from the HF chat_template instead). Returns None only when
    config.json is missing or has no model_type.
    """
    checkpoint_dir = Path(checkpoint_dir)

    config_data: dict[str, Any] = {}
    tokenizer_data: dict[str, Any] = {}

    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        with contextlib.suppress(Exception):
            config_data = json.loads(config_path.read_text(encoding="utf-8"))

    tokenizer_path = checkpoint_dir / "tokenizer_config.json"
    if tokenizer_path.is_file():
        with contextlib.suppress(Exception):
            tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))

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
            for tc in family.tokenizer_classes:
                tc_norm = _normalize(tc)
                if tokenizer_class == tc_norm or (family.name in tokenizer_class and tc_norm in tokenizer_class):
                    return family

    # Fallback: auto-detect a minimal family from the raw config files so that
    # any model works without needing an entry in _MODEL_FAMILIES.
    return _auto_family_from_config(config_data, tokenizer_data)


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


def is_gemma_checkpoint(checkpoint_dir: str | Path) -> bool:
    """True if the checkpoint is detected as a Gemma model (Gemma 2/3)."""
    family = detect_model_family(checkpoint_dir)
    return family is not None and family.name == "gemma"


