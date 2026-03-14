"""Tests for the transformer-backed study runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from ollama_forge.study_runtime import StudyEvaluator, StudyModelHandle, _macro_f1  # noqa: E402


class _FakeAttn(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttn(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class _FakeInner(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList([_FakeLayer(hidden_size) for _ in range(num_layers)])


class _FakeModel(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 2, num_heads: int = 2) -> None:
        super().__init__()
        self.model = _FakeInner(hidden_size, num_layers)
        self.config = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=num_heads)


def _make_handle() -> StudyModelHandle:
    return StudyModelHandle(
        model=_FakeModel(),
        tokenizer=None,
        task="causal_lm",
        architecture="FakeModel",
        device="cpu",
        num_layers=2,
        num_heads=2,
        hidden_size=8,
    )


def test_prune_head_is_reversible() -> None:
    handle = _make_handle()
    attention = handle.model.model.layers[0].self_attn
    original = attention.q_proj.weight.detach().clone()
    handle.prune_head(0, 1)
    assert torch.count_nonzero(attention.q_proj.weight[4:8]) == 0
    handle.restore()
    assert torch.equal(attention.q_proj.weight, original)


def test_embedding_ablation_hook_is_reversible() -> None:
    handle = _make_handle()
    embedding = handle.model.model.embed_tokens
    input_ids = torch.tensor([[1, 2]])
    baseline = embedding(input_ids)
    handle.ablate_embedding_range(1, 3)
    ablated = embedding(input_ids)
    assert torch.count_nonzero(ablated[..., 1:3]) == 0
    handle.restore()
    restored = embedding(input_ids)
    assert torch.equal(restored, baseline)


def test_embedding_ablation_strength_scales_values() -> None:
    handle = _make_handle()
    embedding = handle.model.model.embed_tokens
    input_ids = torch.tensor([[1, 2]])
    baseline = embedding(input_ids)
    handle.ablate_embedding_range(1, 3, strength=0.5)
    ablated = embedding(input_ids)
    assert torch.allclose(ablated[..., 1:3], baseline[..., 1:3] * 0.5)


def test_low_rank_adapter_is_reversible() -> None:
    handle = _make_handle()
    weight = handle.model.model.layers[0].mlp.weight
    original = weight.detach().clone()
    handle.apply_low_rank_adapter(0, target="mlp", rank=1, strength=0.5)
    assert not torch.equal(weight, original)
    handle.restore()
    assert torch.equal(weight, original)


def test_macro_f1_handles_simple_case() -> None:
    score = _macro_f1([0, 1, 1, 0], [0, 1, 0, 0])
    assert 0.0 <= score <= 1.0
    assert round(score, 4) == 0.7333


class _FakeTokenizer:
    def __call__(self, texts, return_tensors="pt", truncation=True, max_length=32, padding=True):
        ids = torch.tensor([[1, 2, 3] for _ in texts])
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


class _FakeClassifier(nn.Module):
    def forward(self, input_ids=None, attention_mask=None):
        batch = input_ids.shape[0]
        logits = torch.tensor([[2.0, 0.5], [0.5, 2.0]], dtype=torch.float32)[:batch]
        return SimpleNamespace(logits=logits)


def test_study_evaluator_classification_metrics() -> None:
    handle = StudyModelHandle(
        model=_FakeClassifier(),
        tokenizer=_FakeTokenizer(),
        task="classification",
        architecture="FakeClassifier",
        device="cpu",
        num_layers=1,
        num_heads=1,
        hidden_size=4,
    )
    dataset = [
        {"text": "one", "label": 0},
        {"text": "two", "label": 1},
    ]
    metrics = StudyEvaluator(handle=handle, dataset=dataset, metrics=["accuracy", "f1"], batch_size=2).evaluate()
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
