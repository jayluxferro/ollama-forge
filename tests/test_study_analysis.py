"""Tests for baseline study analysis modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ollama_forge.study_analysis import (
    analyze_activation_probe,
    analyze_activation_patching,
    analyze_causal_patching,
    analyze_concept_geometry,
    analyze_conditional_similarity,
    analyze_cross_layer_similarity,
    analyze_defense_robustness,
    analyze_logit_lens,
    analyze_residual_stream,
    analyze_steering_vectors,
    trace_causal_layers,
    available_analysis_modules,
)


def test_available_analysis_modules_lists_expected_entries() -> None:
    modules = set(available_analysis_modules())
    assert {
        "activation_probe",
        "cross_layer_similarity",
        "logit_lens",
        "residual_stream",
        "causal_tracing",
        "conditional_similarity",
        "activation_patching",
        "causal_patching",
        "steering_vectors",
        "concept_geometry",
        "architecture_profile",
        "defense_robustness",
    } <= modules


def test_analyze_activation_probe_returns_ranked_layers() -> None:
    vectors = {
        0: [torch.ones(4), torch.ones(4) * 2],
        1: [torch.ones(4) * 5, torch.ones(4) * 4],
    }
    result = analyze_activation_probe(vectors)
    assert len(result.layers) == 2
    assert result.top_layers_by_norm[0] == 1


def test_analyze_cross_layer_similarity_returns_cosine_matrix() -> None:
    vectors = {
        0: [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        1: [torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0])],
        2: [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])],
    }
    result = analyze_cross_layer_similarity(vectors)
    assert len(result.cosine_matrix) == 3
    assert result.most_similar_pair == (0, 1)
    assert result.mean_adjacent_cosine < 1.0


class _FakeLMHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 0.0],
                    ]
                )
            )

    def forward(self, x):
        return self.proj(x)


class _FakeModel:
    def __init__(self) -> None:
        self.lm_head = _FakeLMHead()
        self.config = type("Cfg", (), {"hidden_size": 2, "num_attention_heads": 1})()
        self.model = type("Inner", (), {"layers": []})()

    def get_output_embeddings(self):
        return self.lm_head

    def __call__(self, **kwargs):
        logits = torch.tensor([[[1.0, 0.0, 0.5, -1.0]]], dtype=torch.float32)
        return type("Out", (), {"logits": logits})()


class _FakeTokenizer:
    def decode(self, ids):
        mapping = {0: "A", 1: "B", 2: "AB", 3: "NEG"}
        return mapping[ids[0]]


class _FakeHandle:
    task = "causal_lm"
    model = _FakeModel()
    tokenizer = _FakeTokenizer()


def test_analyze_logit_lens_returns_top_tokens() -> None:
    vectors = {
        0: [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        1: [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])],
    }
    result = analyze_logit_lens(_FakeHandle(), vectors, top_k=2)
    assert len(result.layers) == 2
    assert result.layers[0].top_tokens[0] in {"A", "AB"}
    assert result.final_layer_top_token in {"B", "AB"}


def test_analyze_residual_stream_returns_deltas() -> None:
    vectors = {
        0: [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        1: [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])],
        2: [torch.tensor([3.0, 1.0]), torch.tensor([3.0, 1.0])],
    }
    result = analyze_residual_stream(vectors)
    assert len(result.layers) == 3
    assert result.largest_delta_layer == 2
    assert result.mean_delta > 0.0


def test_analyze_conditional_similarity_returns_lowest_similarity_layer() -> None:
    grouped = {
        "safe": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([1.0, 1.0])]},
        "harmful": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([0.0, 1.0])]},
    }
    result = analyze_conditional_similarity(grouped)
    assert result.groups == ["harmful", "safe"]
    assert result.lowest_similarity_layer == 1


class _TraceHandle:
    task = "causal_lm"
    device = "cpu"
    num_layers = 2

    def __init__(self) -> None:
        self.removed = None
        self.tokenizer = lambda prompt, return_tensors="pt", truncation=True, max_length=256: {"input_ids": torch.tensor([[1, 2]])}
        self.model = self

    def __call__(self, **kwargs):
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        if self.removed == 1:
            logits = torch.tensor([[[1.0, 0.0], [2.0, -1.0]]], dtype=torch.float32)
        return type("Out", (), {"logits": logits})()

    def remove_layer(self, idx: int, strength: float = 1.0) -> None:
        self.removed = idx

    def restore(self) -> None:
        self.removed = None


def test_trace_causal_layers_finds_most_critical_layer() -> None:
    result = trace_causal_layers(_TraceHandle(), "hello")
    assert result.most_critical_layer == 1


class _PatchHandle(_TraceHandle):
    def capture_layer_last_token(self, prompt: str, *, layer_index: int, max_length: int = 256):
        return torch.tensor([10.0, 0.0]) if layer_index == 1 else torch.tensor([0.0, 0.0])

    def patch_layer_last_token(self, layer_index: int, replacement, *, strength: float = 1.0) -> None:
        self.removed = layer_index


def test_analyze_causal_patching_finds_strongest_patch_layer() -> None:
    result = analyze_causal_patching(_PatchHandle(), source_prompt="source", target_prompt="target")
    assert result.strongest_patch_layer == 1


def test_analyze_activation_patching_returns_strongest_layer() -> None:
    grouped = {
        "safe": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([1.0, 1.0])]},
        "harmful": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([3.0, 1.0])]},
    }
    result = analyze_activation_patching(grouped, source_group="safe", target_group="harmful")
    assert result.strongest_patch_layer == 1


def test_analyze_steering_vectors_returns_strongest_layer() -> None:
    grouped = {
        "safe": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([1.0, 1.0])]},
        "harmful": {0: [torch.tensor([2.0, 0.0])], 1: [torch.tensor([4.0, 1.0])]},
    }
    result = analyze_steering_vectors(grouped)
    assert result.strongest_layer == 1


def test_analyze_concept_geometry_returns_polyhedral_layer() -> None:
    grouped = {
        "safe": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([1.0, 1.0])]},
        "harmful": {0: [torch.tensor([0.0, 1.0])], 1: [torch.tensor([1.0, -1.0])]},
        "fraud": {0: [torch.tensor([1.0, 1.0])], 1: [torch.tensor([-1.0, 1.0])]},
    }
    result = analyze_concept_geometry(grouped)
    assert result.most_polyhedral_layer is not None


def test_analyze_defense_robustness_returns_entangled_layers() -> None:
    grouped = {
        "safe": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([1.0, 1.0])]},
        "harmful": {0: [torch.tensor([1.0, 0.0])], 1: [torch.tensor([1.1, 1.0])]},
    }
    result = analyze_defense_robustness(grouped)
    assert 0.0 <= result.self_repair_risk <= 1.0
    assert result.entanglement_score >= 0.0
