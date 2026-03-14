"""Tests for built-in study strategies."""

from __future__ import annotations

from ollama_forge.study_strategies import get_strategy, list_strategies


class _FakeHandle:
    num_layers = 3
    num_heads = 2
    hidden_size = 10

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def remove_layer(self, idx: int) -> None:
        self.calls.append(("remove_layer", idx))

    def prune_head(self, layer_idx: int, head_idx: int) -> None:
        self.calls.append(("prune_head", layer_idx, head_idx))

    def ablate_ffn(self, layer_idx: int) -> None:
        self.calls.append(("ablate_ffn", layer_idx))

    def ablate_embedding_range(self, start: int, end: int) -> None:
        self.calls.append(("ablate_embedding_range", start, end))


def test_list_strategies_contains_builtins() -> None:
    strategies = set(list_strategies())
    assert {"layer_removal", "head_pruning", "ffn_ablation", "embedding_ablation", "lora_ablation"} <= strategies


def test_layer_removal_strategy_enumerates_and_applies() -> None:
    handle = _FakeHandle()
    strategy = get_strategy("layer_removal")
    specs = strategy.enumerate(handle)
    assert len(specs) == 3
    strategy.apply(handle, specs[1])
    assert handle.calls == [("remove_layer", 1)]


def test_head_pruning_strategy_enumerates_all_heads() -> None:
    handle = _FakeHandle()
    strategy = get_strategy("head_pruning")
    specs = strategy.enumerate(handle)
    assert len(specs) == 6
    strategy.apply(handle, specs[-1])
    assert handle.calls == [("prune_head", 2, 1)]


def test_embedding_strategy_respects_chunk_size() -> None:
    handle = _FakeHandle()
    strategy = get_strategy("embedding_ablation")
    specs = strategy.enumerate(handle, chunk_size=4)
    assert [spec.component for spec in specs] == [
        "embedding_0_4",
        "embedding_4_8",
        "embedding_8_10",
    ]


def test_lora_ablation_strategy_enumerates_and_applies() -> None:
    handle = _FakeHandle()
    handle.apply_low_rank_adapter = lambda layer_idx, target="mlp", rank=1, strength=1.0: handle.calls.append(
        ("apply_low_rank_adapter", layer_idx, target, rank, strength)
    )
    strategy = get_strategy("lora_ablation")
    specs = strategy.enumerate(handle, target="attention", rank=2, strength=0.4)
    assert len(specs) == 3
    strategy.apply(handle, specs[0])
    assert handle.calls == [("apply_low_rank_adapter", 0, "attention", 2, 0.4)]
