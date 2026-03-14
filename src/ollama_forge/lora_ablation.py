"""LoRA-based reversible ablation.

Converts the refusal-direction projection into LoRA adapter form so the
ablation can be applied and removed without modifying the base model weights.

Math:
    In-place ablation:  W' = W - s * D @ D^T @ W   (left-multiply, output proj)
                        W' = W @ (I - s * D @ D^T)  (right-multiply, input proj)

    LoRA equivalent:    W' = W + B @ A

    For left-multiply (output projections like o_proj, down_proj):
        A = D^T @ W       shape: (rank, in_features)
        B = -s * D        shape: (out_features, rank)

    For right-multiply (input projections like q_proj, gate_proj):
        A = D^T           shape: (rank, in_features)
        B = -s * W @ D    shape: (out_features, rank)

References:
    Hu et al. (2022): LoRA: Low-Rank Adaptation of Large Language Models
    Heretic (p-e-w, 2025): LoRA-mediated directional ablation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoRAAdapter:
    """A single LoRA adapter pair (lora_B, lora_A) for one weight matrix."""

    target_module: str  # e.g. "model.layers.0.self_attn.o_proj"
    lora_A: Any  # torch.Tensor, shape (rank, in_features)
    lora_B: Any  # torch.Tensor, shape (out_features, rank)
    rank: int


@dataclass
class LoRABundle:
    """Collection of LoRA adapters equivalent to a full ablation."""

    adapters: list[LoRAAdapter]
    model_name: str
    strength: float
    rank: int


def compute_lora_adapters(
    model: Any,
    direction: Any,
    *,
    strength: float = 1.0,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 1,
    output_only: bool = False,
) -> LoRABundle:
    """Compute LoRA adapters equivalent to the refusal-direction projection.

    Args:
        model: The loaded HuggingFace model.
        direction: Refusal direction tensor, shape (H,) or (H, 1) or (H, k).
        strength: Ablation strength.
        skip_begin_layers: Layers to skip at start.
        skip_end_layers: Layers to skip at end.
        output_only: Only compute adapters for output projections.

    Returns:
        LoRABundle with all adapter pairs.
    """

    from ollama_forge.abliterate import get_layers

    D = direction.float()
    if D.dim() == 1:
        D = D.unsqueeze(1)
    if D.dim() == 2 and D.shape[1] == 1:
        D = D.squeeze(1)
    rank = D.shape[1] if D.dim() == 2 else 1
    if D.dim() == 1:
        D = D.unsqueeze(1)  # (H, 1)

    layers = get_layers(model)
    n_layers = len(layers)
    start_idx = skip_begin_layers
    end_idx = n_layers - skip_end_layers
    hidden_size = D.shape[0]

    adapters: list[LoRAAdapter] = []

    for layer_idx, layer in enumerate(layers):
        if layer_idx < start_idx or layer_idx >= end_idx:
            continue

        prefix = f"model.layers.{layer_idx}"

        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attention", None)
            or getattr(layer, "attn", None)
        )
        if attn is not None:
            # Output projections (left-multiply): A = D^T @ W, B = -s * D
            for name in ("o_proj", "out_proj", "dense", "c_proj"):
                proj = getattr(attn, name, None)
                if proj is None:
                    continue
                w = proj.weight.data.float()
                if w.shape[0] != hidden_size:
                    continue
                lora_A = D.T @ w  # (rank, in_features)
                lora_B = -strength * D  # (hidden_size, rank)
                adapters.append(LoRAAdapter(
                    target_module=f"{prefix}.self_attn.{name}",
                    lora_A=lora_A.cpu(), lora_B=lora_B.cpu(), rank=rank,
                ))

            # Input projections (right-multiply): A = D^T, B = -s * W @ D
            if not output_only:
                for name in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn, name, None)
                    if proj is None:
                        continue
                    w = proj.weight.data.float()
                    if w.shape[1] != hidden_size:
                        continue
                    lora_A = D.T  # (rank, hidden_size)
                    lora_B = -strength * (w @ D)  # (out_features, rank)
                    adapters.append(LoRAAdapter(
                        target_module=f"{prefix}.self_attn.{name}",
                        lora_A=lora_A.cpu(), lora_B=lora_B.cpu(), rank=rank,
                    ))

        # MLP
        for mlp_attr in ("mlp", "ffn", "feed_forward"):
            mlp = getattr(layer, mlp_attr, None)
            if mlp is None:
                continue
            # Output (left-multiply)
            for name in ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2", "out_proj"):
                proj = getattr(mlp, name, None)
                if proj is None:
                    continue
                w = proj.weight.data.float()
                if w.shape[0] != hidden_size:
                    continue
                lora_A = D.T @ w
                lora_B = -strength * D
                adapters.append(LoRAAdapter(
                    target_module=f"{prefix}.{mlp_attr}.{name}",
                    lora_A=lora_A.cpu(), lora_B=lora_B.cpu(), rank=rank,
                ))
            # Input (right-multiply)
            if not output_only:
                for name in ("gate_proj", "up_proj", "c_fc", "fc1", "dense_h_to_4h", "w1", "w3"):
                    proj = getattr(mlp, name, None)
                    if proj is None:
                        continue
                    w = proj.weight.data.float()
                    if w.shape[1] != hidden_size:
                        continue
                    lora_A = D.T
                    lora_B = -strength * (w @ D)
                    adapters.append(LoRAAdapter(
                        target_module=f"{prefix}.{mlp_attr}.{name}",
                        lora_A=lora_A.cpu(), lora_B=lora_B.cpu(), rank=rank,
                    ))

    model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
    return LoRABundle(adapters=adapters, model_name=model_name, strength=strength, rank=rank)


def save_lora_adapter(bundle: LoRABundle, output_dir: str | Path) -> Path:
    """Save LoRA adapters in PEFT-compatible format.

    Creates adapter_config.json and adapter_model.bin (state dict).
    """
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build state dict
    state_dict: dict[str, Any] = {}
    target_modules: list[str] = []
    for adapter in bundle.adapters:
        # PEFT naming: base_model.model.<module>.lora_A.weight
        key_prefix = f"base_model.model.{adapter.target_module}"
        state_dict[f"{key_prefix}.lora_A.weight"] = adapter.lora_A
        state_dict[f"{key_prefix}.lora_B.weight"] = adapter.lora_B
        # Extract module name (last part) for target_modules
        parts = adapter.target_module.split(".")
        mod_name = parts[-1]
        if mod_name not in target_modules:
            target_modules.append(mod_name)

    torch.save(state_dict, str(out / "adapter_model.bin"))

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": bundle.model_name,
        "r": bundle.rank,
        "lora_alpha": bundle.rank,
        "target_modules": target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "ablation_strength": bundle.strength,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return out
