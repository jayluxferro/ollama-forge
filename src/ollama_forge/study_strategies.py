"""Strategy registry and built-in study strategies."""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AblationSpec:
    strategy_name: str
    component: str
    description: str
    metadata: dict[str, Any] | None = None


class AblationStrategy(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def enumerate(self, handle: Any, **kwargs: Any) -> list[AblationSpec]:
        """Return the ablation operations supported for the current model."""

    @abc.abstractmethod
    def apply(self, handle: Any, spec: AblationSpec) -> None:
        """Apply an ablation to the provided handle."""


_STRATEGY_REGISTRY: dict[str, type[AblationStrategy]] = {}


def register_strategy(cls: type[AblationStrategy]) -> type[AblationStrategy]:
    _STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> AblationStrategy:
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown strategy {name!r}. Available: {available}")
    return _STRATEGY_REGISTRY[name]()


def list_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGY_REGISTRY))


def _handle_value(handle: Any, attr: str, default: int | None = None) -> int:
    if hasattr(handle, attr):
        value = getattr(handle, attr)
        if isinstance(value, int):
            return value
    if hasattr(handle, "summary") and callable(handle.summary):
        summary = handle.summary()
        value = summary.get(attr)
        if isinstance(value, int):
            return value
    if default is None:
        raise ValueError(f"Handle is missing required attribute {attr!r}")
    return default


def _apply_handle_method(handle: Any, method_name: str, *args: Any) -> None:
    fn: Callable[..., Any] | None = getattr(handle, method_name, None)
    if fn is None or not callable(fn):
        raise NotImplementedError(f"Handle does not implement {method_name}()")
    try:
        params = list(inspect.signature(fn).parameters.values())
        positional = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
            fn(*args)
            return
        fn(*args[: len(positional)])
    except (TypeError, ValueError):
        fn(*args)


@register_strategy
class LayerRemovalStrategy(AblationStrategy):
    name = "layer_removal"

    def enumerate(self, handle: Any, **kwargs: Any) -> list[AblationSpec]:
        count = _handle_value(handle, "num_layers")
        strength = float(kwargs.get("strength", 1.0))
        return [
            AblationSpec(
                strategy_name=self.name,
                component=f"layer_{idx}",
                description=f"Remove transformer layer {idx}",
                metadata={"layer_index": idx, "strength": strength},
            )
            for idx in range(count)
        ]

    def apply(self, handle: Any, spec: AblationSpec) -> None:
        metadata = spec.metadata or {}
        _apply_handle_method(
            handle,
            "remove_layer",
            int(metadata["layer_index"]),
            float(metadata.get("strength", 1.0)),
        )


@register_strategy
class HeadPruningStrategy(AblationStrategy):
    name = "head_pruning"

    def enumerate(self, handle: Any, **kwargs: Any) -> list[AblationSpec]:
        layer_count = _handle_value(handle, "num_layers")
        head_count = _handle_value(handle, "num_heads")
        strength = float(kwargs.get("strength", 1.0))
        specs: list[AblationSpec] = []
        for layer_idx in range(layer_count):
            for head_idx in range(head_count):
                specs.append(
                    AblationSpec(
                        strategy_name=self.name,
                        component=f"head_{layer_idx}_{head_idx}",
                        description=f"Prune attention head {head_idx} in layer {layer_idx}",
                        metadata={"layer_index": layer_idx, "head_index": head_idx, "strength": strength},
                    )
                )
        return specs

    def apply(self, handle: Any, spec: AblationSpec) -> None:
        metadata = spec.metadata or {}
        _apply_handle_method(
            handle,
            "prune_head",
            int(metadata["layer_index"]),
            int(metadata["head_index"]),
            float(metadata.get("strength", 1.0)),
        )


@register_strategy
class FFNAblationStrategy(AblationStrategy):
    name = "ffn_ablation"

    def enumerate(self, handle: Any, **kwargs: Any) -> list[AblationSpec]:
        count = _handle_value(handle, "num_layers")
        strength = float(kwargs.get("strength", 1.0))
        return [
            AblationSpec(
                strategy_name=self.name,
                component=f"ffn_{idx}",
                description=f"Ablate FFN block in layer {idx}",
                metadata={"layer_index": idx, "strength": strength},
            )
            for idx in range(count)
        ]

    def apply(self, handle: Any, spec: AblationSpec) -> None:
        metadata = spec.metadata or {}
        _apply_handle_method(
            handle,
            "ablate_ffn",
            int(metadata["layer_index"]),
            float(metadata.get("strength", 1.0)),
        )


@register_strategy
class EmbeddingAblationStrategy(AblationStrategy):
    name = "embedding_ablation"

    def enumerate(self, handle: Any, **kwargs: Any) -> list[AblationSpec]:
        hidden_size = _handle_value(handle, "hidden_size")
        chunk_size = int(kwargs.get("chunk_size", 32))
        strength = float(kwargs.get("strength", 1.0))
        specs: list[AblationSpec] = []
        for start in range(0, hidden_size, chunk_size):
            end = min(start + chunk_size, hidden_size)
            specs.append(
                AblationSpec(
                    strategy_name=self.name,
                    component=f"embedding_{start}_{end}",
                    description=f"Ablate embedding dimensions {start}:{end}",
                    metadata={"start": start, "end": end, "chunk_size": chunk_size, "strength": strength},
                )
            )
        return specs

    def apply(self, handle: Any, spec: AblationSpec) -> None:
        metadata = spec.metadata or {}
        _apply_handle_method(
            handle,
            "ablate_embedding_range",
            int(metadata["start"]),
            int(metadata["end"]),
            float(metadata.get("strength", 1.0)),
        )


@register_strategy
class LoraAblationStrategy(AblationStrategy):
    name = "lora_ablation"

    def enumerate(self, handle: Any, **kwargs: Any) -> list[AblationSpec]:
        count = _handle_value(handle, "num_layers")
        target = str(kwargs.get("target", "mlp"))
        rank = int(kwargs.get("rank", 1))
        strength = float(kwargs.get("strength", 1.0))
        return [
            AblationSpec(
                strategy_name=self.name,
                component=f"lora_{target}_{idx}",
                description=f"Apply reversible low-rank ablation to {target} output projection in layer {idx}",
                metadata={"layer_index": idx, "target": target, "rank": rank, "strength": strength},
            )
            for idx in range(count)
        ]

    def apply(self, handle: Any, spec: AblationSpec) -> None:
        metadata = spec.metadata or {}
        _apply_handle_method(
            handle,
            "apply_low_rank_adapter",
            int(metadata["layer_index"]),
            str(metadata.get("target", "mlp")),
            int(metadata.get("rank", 1)),
            float(metadata.get("strength", 1.0)),
        )
