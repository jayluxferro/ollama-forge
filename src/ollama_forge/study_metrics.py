"""Advanced metrics for study and benchmark workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def kl_divergence_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    log_p = torch.log_softmax(logits_a.float(), dim=-1)
    q = torch.softmax(logits_b.float(), dim=-1)
    return float(torch.sum(q * (torch.log(q.clamp(min=1e-8)) - log_p), dim=-1).mean().item())


def mean_token_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
    return float(entropy.mean().item())


def effective_rank(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix.float())
    probs = singular_values / singular_values.sum().clamp(min=1e-8)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum()
    return float(torch.exp(entropy).item())


def classification_logit_margin(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), k=min(2, logits.shape[-1]), dim=-1).values
    if top2.shape[-1] < 2:
        return float(top2.mean().item())
    return float((top2[..., 0] - top2[..., 1]).mean().item())


@dataclass
class MetricSummary:
    name: str
    value: float
    metadata: dict[str, Any] | None = None


def summarize_metric(name: str, value: float, **metadata: Any) -> MetricSummary:
    return MetricSummary(name=name, value=float(value), metadata=metadata or None)
