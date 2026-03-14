"""Tests for advanced study metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ollama_forge.study_metrics import (
    classification_logit_margin,
    effective_rank,
    kl_divergence_from_logits,
    mean_token_entropy,
)


def test_kl_divergence_from_logits_is_non_negative() -> None:
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.5, 0.5]])
    assert kl_divergence_from_logits(a, b) >= 0.0


def test_mean_token_entropy_positive() -> None:
    logits = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    assert mean_token_entropy(logits) > 0.0


def test_effective_rank_of_identity_is_dimension() -> None:
    matrix = torch.eye(3)
    assert round(effective_rank(matrix), 4) == 3.0


def test_classification_logit_margin_positive() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.5, 0.4]])
    assert classification_logit_margin(logits) > 0.0
