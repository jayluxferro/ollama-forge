"""Baseline analysis modules for generic transformer study workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LayerActivationStats:
    layer_index: int
    mean_norm: float
    std_norm: float
    sample_count: int


@dataclass
class ActivationProbeResult:
    layers: list[LayerActivationStats]
    top_layers_by_norm: list[int]


@dataclass
class CrossLayerSimilarityResult:
    layer_indices: list[int]
    cosine_matrix: list[list[float]]
    mean_adjacent_cosine: float
    most_similar_pair: tuple[int, int] | None


@dataclass
class LogitLensLayerResult:
    layer_index: int
    top_tokens: list[str]
    top_token_ids: list[int]
    top_scores: list[float]


@dataclass
class LogitLensResult:
    layers: list[LogitLensLayerResult]
    final_layer_top_token: str | None


@dataclass
class ResidualStreamLayerResult:
    layer_index: int
    mean_norm: float
    delta_from_previous: float


@dataclass
class ResidualStreamResult:
    layers: list[ResidualStreamLayerResult]
    largest_delta_layer: int | None
    mean_delta: float


@dataclass
class CausalTraceLayerResult:
    layer_index: int
    kl_divergence: float
    changed_top_token: bool


@dataclass
class CausalTraceResult:
    prompt: str
    layers: list[CausalTraceLayerResult]
    most_critical_layer: int | None


@dataclass
class ConditionalSimilarityLayerResult:
    layer_index: int
    group_pair: tuple[str, str]
    cosine_similarity: float


@dataclass
class ConditionalSimilarityResult:
    groups: list[str]
    layers: list[ConditionalSimilarityLayerResult]
    lowest_similarity_layer: int | None


@dataclass
class ActivationPatchLayerResult:
    layer_index: int
    patch_delta_norm: float
    source_norm: float
    target_norm: float


@dataclass
class ActivationPatchResult:
    source_group: str
    target_group: str
    layers: list[ActivationPatchLayerResult]
    strongest_patch_layer: int | None


@dataclass
class CausalPatchLayerResult:
    layer_index: int
    kl_divergence: float
    changed_top_token: bool


@dataclass
class CausalPatchResult:
    source_prompt: str
    target_prompt: str
    layers: list[CausalPatchLayerResult]
    strongest_patch_layer: int | None


@dataclass
class SteeringVectorLayerResult:
    layer_index: int
    norm: float
    top_alignment_group_pair: tuple[str, str] | None


@dataclass
class SteeringVectorResult:
    layers: list[SteeringVectorLayerResult]
    strongest_layer: int | None


@dataclass
class ConceptGeometryLayerResult:
    layer_index: int
    group_count: int
    mean_pairwise_cosine: float
    cone_dimensionality: float
    is_polyhedral: bool


@dataclass
class ConceptGeometryResult:
    layers: list[ConceptGeometryLayerResult]
    most_polyhedral_layer: int | None


@dataclass
class DefenseRobustnessLayerResult:
    layer_index: int
    mean_pairwise_cosine: float
    mean_delta_norm: float
    entangled: bool


@dataclass
class DefenseRobustnessResult:
    layers: list[DefenseRobustnessLayerResult]
    entanglement_score: float
    self_repair_risk: float
    entangled_layers: list[int]
    clean_layers: list[int]


def available_analysis_modules() -> tuple[str, ...]:
    return (
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
    )


def _stack_norms(values: list[torch.Tensor]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    stacked = torch.stack([value.float() for value in values])
    norms = torch.norm(stacked, dim=-1)
    mean = float(norms.mean().item())
    std = float(norms.std(correction=1).item()) if len(values) > 1 else 0.0
    return mean, std


def analyze_activation_probe(layer_vectors: dict[int, list[torch.Tensor]]) -> ActivationProbeResult:
    layers: list[LayerActivationStats] = []
    for layer_idx in sorted(layer_vectors):
        mean, std = _stack_norms(layer_vectors[layer_idx])
        layers.append(
            LayerActivationStats(
                layer_index=layer_idx,
                mean_norm=mean,
                std_norm=std,
                sample_count=len(layer_vectors[layer_idx]),
            )
        )
    top_layers = [item.layer_index for item in sorted(layers, key=lambda item: item.mean_norm, reverse=True)[:5]]
    return ActivationProbeResult(layers=layers, top_layers_by_norm=top_layers)


def analyze_cross_layer_similarity(layer_vectors: dict[int, list[torch.Tensor]]) -> CrossLayerSimilarityResult:
    layer_indices = sorted(layer_vectors)
    if not layer_indices:
        return CrossLayerSimilarityResult([], [], 0.0, None)
    means = []
    for layer_idx in layer_indices:
        samples = layer_vectors[layer_idx]
        stacked = torch.stack([value.float() for value in samples])
        mean_vector = stacked.mean(dim=0)
        means.append(mean_vector / mean_vector.norm().clamp(min=1e-8))
    matrix = (torch.stack(means) @ torch.stack(means).T).clamp(min=-1.0, max=1.0)
    cosine_matrix = [[float(value) for value in row] for row in matrix.tolist()]
    if len(layer_indices) > 1:
        adjacent = [cosine_matrix[idx][idx + 1] for idx in range(len(layer_indices) - 1)]
        mean_adjacent = float(sum(adjacent) / len(adjacent))
        best_pair: tuple[int, int] | None = None
        best_value = -2.0
        for i in range(len(layer_indices)):
            for j in range(i + 1, len(layer_indices)):
                if cosine_matrix[i][j] > best_value:
                    best_value = cosine_matrix[i][j]
                    best_pair = (layer_indices[i], layer_indices[j])
    else:
        mean_adjacent = 1.0
        best_pair = None
    return CrossLayerSimilarityResult(
        layer_indices=layer_indices,
        cosine_matrix=cosine_matrix,
        mean_adjacent_cosine=mean_adjacent,
        most_similar_pair=best_pair,
    )


def analyze_logit_lens(handle: Any, layer_vectors: dict[int, list[torch.Tensor]], *, top_k: int = 5) -> LogitLensResult:
    if handle.task != "causal_lm":
        raise ValueError("logit_lens currently supports causal_lm tasks only")
    get_output_embeddings = getattr(handle.model, "get_output_embeddings", None)
    lm_head = get_output_embeddings() if callable(get_output_embeddings) else getattr(handle.model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Model has no output embedding head for logit-lens analysis")

    layers: list[LogitLensLayerResult] = []
    for layer_idx in sorted(layer_vectors):
        samples = layer_vectors[layer_idx]
        if not samples:
            continue
        stacked = torch.stack([value.float() for value in samples])
        mean_vector = stacked.mean(dim=0)
        logits = lm_head(mean_vector.to(next(lm_head.parameters()).device)).detach().cpu()
        scores, token_ids = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        token_ids_list = [int(token_id) for token_id in token_ids.tolist()]
        top_tokens = []
        for token_id in token_ids_list:
            try:
                token = handle.tokenizer.decode([token_id]).strip()
            except Exception:
                token = str(token_id)
            top_tokens.append(token or str(token_id))
        layers.append(
            LogitLensLayerResult(
                layer_index=layer_idx,
                top_tokens=top_tokens,
                top_token_ids=token_ids_list,
                top_scores=[float(score) for score in scores.tolist()],
            )
        )
    final_token = layers[-1].top_tokens[0] if layers else None
    return LogitLensResult(layers=layers, final_layer_top_token=final_token)


def analyze_residual_stream(layer_vectors: dict[int, list[torch.Tensor]]) -> ResidualStreamResult:
    rows: list[ResidualStreamLayerResult] = []
    previous_mean: torch.Tensor | None = None
    deltas: list[tuple[int, float]] = []
    for layer_idx in sorted(layer_vectors):
        samples = layer_vectors[layer_idx]
        if not samples:
            continue
        stacked = torch.stack([value.float() for value in samples])
        mean_vector = stacked.mean(dim=0)
        mean_norm = float(mean_vector.norm().item())
        if previous_mean is None:
            delta = 0.0
        else:
            delta = float((mean_vector - previous_mean).norm().item())
        rows.append(
            ResidualStreamLayerResult(
                layer_index=layer_idx,
                mean_norm=mean_norm,
                delta_from_previous=delta,
            )
        )
        deltas.append((layer_idx, delta))
        previous_mean = mean_vector
    if not rows:
        return ResidualStreamResult([], None, 0.0)
    largest_delta_layer = max(deltas, key=lambda item: item[1])[0] if deltas else None
    mean_delta = float(sum(item[1] for item in deltas) / max(len(deltas), 1))
    return ResidualStreamResult(
        layers=rows,
        largest_delta_layer=largest_delta_layer,
        mean_delta=mean_delta,
    )


def trace_causal_layers(handle: Any, prompt: str, *, max_length: int = 256) -> CausalTraceResult:
    from ollama_forge.study_metrics import kl_divergence_from_logits

    encodings = handle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    encodings = {key: value.to(handle.device) for key, value in encodings.items()}
    with torch.no_grad():
        baseline = handle.model(**encodings).logits[:, -1, :].detach().cpu()
    baseline_top = int(baseline.argmax(dim=-1).item())
    rows: list[CausalTraceLayerResult] = []
    for layer_idx in range(handle.num_layers):
        handle.remove_layer(layer_idx, strength=1.0)
        try:
            with torch.no_grad():
                perturbed = handle.model(**encodings).logits[:, -1, :].detach().cpu()
        finally:
            handle.restore()
        kl = kl_divergence_from_logits(baseline, perturbed)
        changed = int(perturbed.argmax(dim=-1).item()) != baseline_top
        rows.append(CausalTraceLayerResult(layer_index=layer_idx, kl_divergence=kl, changed_top_token=changed))
    most_critical = max(rows, key=lambda item: item.kl_divergence).layer_index if rows else None
    return CausalTraceResult(prompt=prompt, layers=rows, most_critical_layer=most_critical)


def collect_grouped_layer_vectors(
    handle: Any,
    dataset: Any,
    *,
    group_column: str,
    text_column: str = "text",
    max_samples: int | None = None,
    batch_size: int = 4,
    max_length: int = 256,
) -> dict[str, dict[int, list[torch.Tensor]]]:
    if handle.task != "causal_lm":
        raise ValueError("Grouped layer collection currently supports causal_lm tasks only")
    groups: dict[str, dict[int, list[torch.Tensor]]] = {}
    limit = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    for start in range(0, limit, batch_size):
        rows = dataset[start : min(start + batch_size, limit)]
        if isinstance(rows, dict):
            texts = [str(value) for value in rows[text_column]]
            labels = [str(value) for value in rows[group_column]]
        else:
            texts = [str(row[text_column]) for row in rows]
            labels = [str(row[group_column]) for row in rows]
        encodings = handle.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encodings = {key: value.to(handle.device) for key, value in encodings.items()}
        with torch.no_grad():
            outputs = handle.model(**encodings, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden states")
        attention_mask = encodings["attention_mask"].unsqueeze(-1)
        denom = attention_mask.sum(dim=1).clamp(min=1)
        for layer_idx, layer_state in enumerate(hidden_states[1:]):
            pooled = ((layer_state * attention_mask).sum(dim=1) / denom).detach().cpu()
            for row_idx, label in enumerate(labels):
                groups.setdefault(label, {}).setdefault(layer_idx, []).append(pooled[row_idx])
    return groups


def analyze_conditional_similarity(grouped_vectors: dict[str, dict[int, list[torch.Tensor]]]) -> ConditionalSimilarityResult:
    groups = sorted(grouped_vectors)
    layers: list[ConditionalSimilarityLayerResult] = []
    if len(groups) < 2:
        return ConditionalSimilarityResult(groups=groups, layers=[], lowest_similarity_layer=None)
    all_layers = sorted({layer_idx for per_group in grouped_vectors.values() for layer_idx in per_group})
    for layer_idx in all_layers:
        means: dict[str, torch.Tensor] = {}
        for group in groups:
            samples = grouped_vectors[group].get(layer_idx, [])
            if not samples:
                continue
            stacked = torch.stack([sample.float() for sample in samples])
            mean = stacked.mean(dim=0)
            means[group] = mean / mean.norm().clamp(min=1e-8)
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                if left not in means or right not in means:
                    continue
                cosine = float(torch.dot(means[left], means[right]).item())
                layers.append(
                    ConditionalSimilarityLayerResult(
                        layer_index=layer_idx,
                        group_pair=(left, right),
                        cosine_similarity=cosine,
                    )
                )
    lowest = min(layers, key=lambda item: item.cosine_similarity).layer_index if layers else None
    return ConditionalSimilarityResult(groups=groups, layers=layers, lowest_similarity_layer=lowest)


def analyze_activation_patching(
    grouped_vectors: dict[str, dict[int, list[torch.Tensor]]],
    *,
    source_group: str,
    target_group: str,
) -> ActivationPatchResult:
    if source_group not in grouped_vectors:
        raise ValueError(f"Unknown source_group {source_group!r}")
    if target_group not in grouped_vectors:
        raise ValueError(f"Unknown target_group {target_group!r}")

    source_layers = grouped_vectors[source_group]
    target_layers = grouped_vectors[target_group]
    rows: list[ActivationPatchLayerResult] = []
    shared_layers = sorted(set(source_layers) & set(target_layers))
    for layer_idx in shared_layers:
        source_samples = source_layers.get(layer_idx, [])
        target_samples = target_layers.get(layer_idx, [])
        if not source_samples or not target_samples:
            continue
        source_mean = torch.stack([sample.float() for sample in source_samples]).mean(dim=0)
        target_mean = torch.stack([sample.float() for sample in target_samples]).mean(dim=0)
        delta = float((target_mean - source_mean).norm().item())
        rows.append(
            ActivationPatchLayerResult(
                layer_index=layer_idx,
                patch_delta_norm=delta,
                source_norm=float(source_mean.norm().item()),
                target_norm=float(target_mean.norm().item()),
            )
        )
    strongest = max(rows, key=lambda item: item.patch_delta_norm).layer_index if rows else None
    return ActivationPatchResult(
        source_group=source_group,
        target_group=target_group,
        layers=rows,
        strongest_patch_layer=strongest,
    )


def analyze_causal_patching(
    handle: Any,
    *,
    source_prompt: str,
    target_prompt: str,
    max_length: int = 256,
) -> CausalPatchResult:
    from ollama_forge.study_metrics import kl_divergence_from_logits

    import torch

    target_enc = handle.tokenizer(target_prompt, return_tensors="pt", truncation=True, max_length=max_length)
    target_enc = {key: value.to(handle.device) for key, value in target_enc.items()}
    with torch.no_grad():
        baseline = handle.model(**target_enc).logits[:, -1, :].detach().cpu()
    baseline_top = int(baseline.argmax(dim=-1).item())

    rows: list[CausalPatchLayerResult] = []
    for layer_idx in range(handle.num_layers):
        source_vec = handle.capture_layer_last_token(source_prompt, layer_index=layer_idx, max_length=max_length)
        handle.patch_layer_last_token(layer_idx, source_vec, strength=1.0)
        try:
            with torch.no_grad():
                patched = handle.model(**target_enc).logits[:, -1, :].detach().cpu()
        finally:
            handle.restore()
        rows.append(
            CausalPatchLayerResult(
                layer_index=layer_idx,
                kl_divergence=kl_divergence_from_logits(baseline, patched),
                changed_top_token=int(patched.argmax(dim=-1).item()) != baseline_top,
            )
        )
    strongest = max(rows, key=lambda item: item.kl_divergence).layer_index if rows else None
    return CausalPatchResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        layers=rows,
        strongest_patch_layer=strongest,
    )


def analyze_steering_vectors(grouped_vectors: dict[str, dict[int, list[torch.Tensor]]]) -> SteeringVectorResult:
    groups = sorted(grouped_vectors)
    layers: list[SteeringVectorLayerResult] = []
    if len(groups) < 2:
        return SteeringVectorResult(layers=[], strongest_layer=None)
    all_layers = sorted({layer_idx for per_group in grouped_vectors.values() for layer_idx in per_group})
    for layer_idx in all_layers:
        means: dict[str, torch.Tensor] = {}
        for group in groups:
            samples = grouped_vectors[group].get(layer_idx, [])
            if not samples:
                continue
            means[group] = torch.stack([sample.float() for sample in samples]).mean(dim=0)
        best_pair = None
        best_norm = -1.0
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                if left not in means or right not in means:
                    continue
                diff = means[left] - means[right]
                norm = float(diff.norm().item())
                if norm > best_norm:
                    best_norm = norm
                    best_pair = (left, right)
        if best_pair is not None:
            layers.append(
                SteeringVectorLayerResult(
                    layer_index=layer_idx,
                    norm=max(best_norm, 0.0),
                    top_alignment_group_pair=best_pair,
                )
            )
    strongest = max(layers, key=lambda item: item.norm).layer_index if layers else None
    return SteeringVectorResult(layers=layers, strongest_layer=strongest)


def analyze_concept_geometry(grouped_vectors: dict[str, dict[int, list[torch.Tensor]]]) -> ConceptGeometryResult:
    groups = sorted(grouped_vectors)
    layers: list[ConceptGeometryLayerResult] = []
    all_layers = sorted({layer_idx for per_group in grouped_vectors.values() for layer_idx in per_group})
    for layer_idx in all_layers:
        direction_list: list[torch.Tensor] = []
        for group in groups:
            samples = grouped_vectors[group].get(layer_idx, [])
            if not samples:
                continue
            mean = torch.stack([sample.float() for sample in samples]).mean(dim=0)
            direction_list.append(mean / mean.norm().clamp(min=1e-8))
        if not direction_list:
            continue
        matrix = torch.stack(direction_list)
        cosine = (matrix @ matrix.T).abs()
        if len(direction_list) > 1:
            off_diag = cosine[~torch.eye(len(direction_list), dtype=torch.bool)]
            mean_pairwise = float(off_diag.mean().item()) if off_diag.numel() else 1.0
        else:
            mean_pairwise = 1.0
        singular_values = torch.linalg.svdvals(matrix)
        probs = singular_values / singular_values.sum().clamp(min=1e-8)
        dimensionality = float(torch.exp(-(probs * torch.log(probs.clamp(min=1e-8))).sum()).item())
        layers.append(
            ConceptGeometryLayerResult(
                layer_index=layer_idx,
                group_count=len(direction_list),
                mean_pairwise_cosine=mean_pairwise,
                cone_dimensionality=dimensionality,
                is_polyhedral=dimensionality > 1.5 and mean_pairwise < 0.95,
            )
        )
    poly_layer = max(layers, key=lambda item: item.cone_dimensionality).layer_index if layers else None
    return ConceptGeometryResult(layers=layers, most_polyhedral_layer=poly_layer)


def analyze_defense_robustness(grouped_vectors: dict[str, dict[int, list[torch.Tensor]]]) -> DefenseRobustnessResult:
    groups = sorted(grouped_vectors)
    rows: list[DefenseRobustnessLayerResult] = []
    if len(groups) < 2:
        return DefenseRobustnessResult([], 0.0, 0.0, [], [])

    all_layers = sorted({layer_idx for per_group in grouped_vectors.values() for layer_idx in per_group})
    for layer_idx in all_layers:
        means: list[torch.Tensor] = []
        for group in groups:
            samples = grouped_vectors[group].get(layer_idx, [])
            if not samples:
                continue
            stacked = torch.stack([sample.float() for sample in samples])
            means.append(stacked.mean(dim=0))
        if len(means) < 2:
            continue
        normalized = [mean / mean.norm().clamp(min=1e-8) for mean in means]
        cosines: list[float] = []
        deltas: list[float] = []
        for i, left in enumerate(normalized):
            for right in normalized[i + 1 :]:
                cosines.append(float(torch.dot(left, right).abs().item()))
        for i, left in enumerate(means):
            for right in means[i + 1 :]:
                deltas.append(float((left - right).norm().item()))
        mean_cos = sum(cosines) / len(cosines)
        mean_delta = sum(deltas) / len(deltas)
        entangled = mean_cos >= 0.85
        rows.append(
            DefenseRobustnessLayerResult(
                layer_index=layer_idx,
                mean_pairwise_cosine=mean_cos,
                mean_delta_norm=mean_delta,
                entangled=entangled,
            )
        )

    if not rows:
        return DefenseRobustnessResult([], 0.0, 0.0, [], [])

    entanglement_score = float(sum(row.mean_pairwise_cosine for row in rows) / len(rows))
    delta_mean = float(sum(row.mean_delta_norm for row in rows) / len(rows))
    # Distributed small deltas plus high cosine imply the signal is spread and may self-repair.
    self_repair_risk = min(1.0, 0.6 * entanglement_score + 0.4 * (1.0 / (1.0 + delta_mean)))
    entangled_layers = [row.layer_index for row in rows if row.entangled]
    clean_layers = [row.layer_index for row in rows if not row.entangled]
    return DefenseRobustnessResult(
        layers=rows,
        entanglement_score=entanglement_score,
        self_repair_risk=self_repair_risk,
        entangled_layers=entangled_layers,
        clean_layers=clean_layers,
    )


def collect_layer_vectors(
    handle: Any,
    dataset: Any,
    *,
    text_column: str = "text",
    max_samples: int | None = None,
    batch_size: int = 4,
    max_length: int = 256,
) -> dict[int, list[torch.Tensor]]:
    if handle.task != "causal_lm":
        raise ValueError("Analysis modules currently support causal_lm tasks only")
    vectors: dict[int, list[torch.Tensor]] = {}
    limit = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    for start in range(0, limit, batch_size):
        rows = dataset[start : min(start + batch_size, limit)]
        if isinstance(rows, dict):
            texts = [str(value) for value in rows[text_column]]
        else:
            texts = [str(row[text_column]) for row in rows]
        encodings = handle.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encodings = {key: value.to(handle.device) for key, value in encodings.items()}
        with torch.no_grad():
            outputs = handle.model(**encodings, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden states")
        attention_mask = encodings["attention_mask"].unsqueeze(-1)
        denom = attention_mask.sum(dim=1).clamp(min=1)
        for layer_idx, layer_state in enumerate(hidden_states[1:]):
            pooled = (layer_state * attention_mask).sum(dim=1) / denom
            vectors.setdefault(layer_idx, []).extend([row.detach().cpu() for row in pooled])
    return vectors


def save_analysis_result(result: Any, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return target
