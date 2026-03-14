"""Analysis-informed recommendations for abliterate runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_analysis_documents(paths: list[str | Path]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload.get("results"), dict) and isinstance(payload.get("modules"), list):
            for module_name, module_payload in payload["results"].items():
                if isinstance(module_payload, dict):
                    doc = dict(module_payload)
                    doc["_module"] = module_name
                    doc["_source_file"] = str(path)
                    docs.append(doc)
            continue
        payload["_source_file"] = str(path)
        docs.append(payload)
    return docs


def recommend_abliterate_settings(analysis_docs: list[dict[str, Any]]) -> dict[str, Any]:
    recommendation = {
        "profile": "balanced",
        "strength": 1.0,
        "atten_strength": 1.0,
        "mlp_strength": 0.9,
        "per_layer_directions": False,
        "norm_preserving": True,
        "strength_kernel": "constant",
        "notes": [],
    }

    for doc in analysis_docs:
        kpis = doc.get("kpis") or {}
        if kpis:
            refusal_rate = kpis.get("refusal_rate_pct")
            asr_pct = kpis.get("asr_pct")
            if isinstance(refusal_rate, (int, float)) and refusal_rate >= 60:
                recommendation["strength"] = max(float(recommendation["strength"]), 1.2)
                recommendation["notes"].append("Security eval shows strong refusal; increasing ablation strength.")
            if isinstance(asr_pct, (int, float)) and asr_pct >= 70:
                recommendation["strength"] = min(float(recommendation["strength"]), 1.0)
                recommendation["notes"].append(
                    "Security eval shows high baseline compliance; keeping ablation conservative."
                )
        results = doc.get("results")
        if isinstance(results, dict) and results:
            numeric_scores: list[float] = []
            for task_metrics in results.values():
                if isinstance(task_metrics, dict):
                    for value in task_metrics.values():
                        if isinstance(value, (int, float)):
                            numeric_scores.append(float(value))
            if numeric_scores:
                mean_score = sum(numeric_scores) / len(numeric_scores)
                if mean_score < 0.35:
                    recommendation["profile"] = "safe"
                    recommendation["mlp_strength"] = min(float(recommendation["mlp_strength"]), 0.8)
                    recommendation["notes"].append(
                        "External eval indicates weak capability baseline;"
                        " reducing intervention strength."
                    )
        if doc.get("arch_class") == "moe":
            recommendation["profile"] = "safe"
            recommendation["strength"] = min(float(recommendation["strength"]), 1.0)
            recommendation["norm_preserving"] = True
            recommendation["notes"].append("MoE architecture profile detected; use safer, less destructive defaults.")
        if doc.get("reasoning_class") == "reasoning":
            recommendation["mlp_strength"] = min(float(recommendation["mlp_strength"]), 0.8)
            recommendation["notes"].append("Reasoning-oriented model detected; reduce MLP intervention strength.")
        if "mean_adjacent_cosine" in doc:
            cosine = float(doc.get("mean_adjacent_cosine", 0.0))
            if cosine >= 0.9:
                recommendation["per_layer_directions"] = False
                recommendation["notes"].append(
                    "High cross-layer alignment detected; global directions should be stable."
                )
            elif cosine <= 0.65:
                recommendation["per_layer_directions"] = True
                recommendation["strength_kernel"] = "gaussian"
                recommendation["notes"].append(
                    "Lower cross-layer alignment detected; per-layer directions recommended."
                )
        if "top_layers_by_norm" in doc:
            top_layers = doc.get("top_layers_by_norm") or []
            if len(top_layers) >= 3:
                recommendation["strength_kernel"] = "gaussian"
                recommendation["notes"].append("Activation concentration suggests center-weighted layer strength.")
        if "most_polyhedral_layer" in doc and doc.get("most_polyhedral_layer") is not None:
            recommendation["per_layer_directions"] = True
            recommendation["notes"].append("Polyhedral concept geometry detected; per-layer directions recommended.")
        if "strongest_layer" in doc and doc.get("strongest_layer") is not None:
            recommendation["strength"] = max(float(recommendation["strength"]), 1.1)
            recommendation["notes"].append(
                "Strong steering-vector separation detected; slightly stronger ablation recommended."
            )
        if "self_repair_risk" in doc:
            risk = float(doc.get("self_repair_risk", 0.0))
            if risk >= 0.65:
                recommendation["per_layer_directions"] = True
                recommendation["strength_kernel"] = "gaussian"
                recommendation["notes"].append(
                    "Defense robustness indicates likely self-repair;"
                    " use layered, center-weighted ablation."
                )
        if "entanglement_score" in doc:
            ent = float(doc.get("entanglement_score", 0.0))
            if ent >= 0.85:
                recommendation["profile"] = "safe"
                recommendation["mlp_strength"] = min(float(recommendation["mlp_strength"]), 0.7)
                recommendation["notes"].append(
                    "High safety-capability entanglement detected;"
                    " reducing destructive intervention strength."
                )
        if "largest_delta_layer" in doc and doc.get("largest_delta_layer") is not None:
            recommendation["strength"] = max(float(recommendation["strength"]), 1.1)
            recommendation["notes"].append(
                "Residual-stream shift is concentrated; slightly stronger ablation recommended."
            )
        if "most_critical_layer" in doc and doc.get("most_critical_layer") is not None:
            recommendation["profile"] = "aggressive"
            recommendation["strength"] = max(float(recommendation["strength"]), 1.2)
            recommendation["notes"].append(
                "Causal tracing found a critical layer; aggressive settings may be justified."
            )

    if recommendation["profile"] == "aggressive":
        recommendation["atten_strength"] = max(float(recommendation["atten_strength"]), 1.2)
        recommendation["mlp_strength"] = max(float(recommendation["mlp_strength"]), 1.1)
        recommendation["norm_preserving"] = False

    # Recommend new features based on analysis results
    has_polyhedral = any(doc.get("most_polyhedral_layer") is not None for doc in analysis_docs)
    has_high_self_repair = any(float(doc.get("self_repair_risk", 0)) >= 0.5 for doc in analysis_docs)
    has_entanglement = any(float(doc.get("entanglement_score", 0)) >= 0.7 for doc in analysis_docs)
    is_moe = any(doc.get("arch_class") == "moe" for doc in analysis_docs)

    # Whitened SVD when polyhedral geometry detected (multi-direction benefits from cleaner separation)
    if has_polyhedral:
        recommendation["svd_method"] = "whitened"
        recommendation["num_directions"] = max(int(recommendation.get("num_directions", 1)), 4)
        recommendation["notes"].append(
            "Polyhedral geometry detected; whitened SVD with multiple directions recommended."
        )

    # Iterative refinement when self-repair risk is high
    if has_high_self_repair:
        recommendation["refine_passes"] = max(int(recommendation.get("refine_passes", 0)), 2)
        recommendation["notes"].append(
            "Self-repair risk detected; iterative refinement recommended."
        )

    # Sparse surgery for MoE or highly entangled models
    if is_moe or has_entanglement:
        recommendation["sparse_surgery"] = True
        recommendation["surgery_top_k"] = 0.3
        recommendation["notes"].append(
            "Sparse surgery recommended to preserve capability-critical weights."
        )

    # Always recommend bias projection
    recommendation["project_bias"] = True

    return recommendation


def build_informed_run_artifact(
    *,
    analysis_docs: list[dict[str, Any]],
    recommendation: dict[str, Any],
    requested_run: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_files": [doc.get("_source_file") for doc in analysis_docs],
        "recommendation": recommendation,
        "requested_run": requested_run,
    }


def save_informed_run_artifact(artifact: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_informed_run_artifact(
    artifact: dict[str, Any],
    *,
    run_status: str,
    report_path: str | None = None,
    report_payload: dict[str, Any] | None = None,
    benchmark_path: str | None = None,
    benchmark_payload: dict[str, Any] | None = None,
    eval_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(artifact)
    updated["run_status"] = run_status
    if report_path is not None:
        updated["report_path"] = report_path
    if report_payload is not None:
        updated["report"] = report_payload
    if benchmark_path is not None:
        updated["benchmark_path"] = benchmark_path
    if benchmark_payload is not None:
        updated["benchmark"] = benchmark_payload
    if eval_comparison is not None:
        updated["eval_comparison"] = eval_comparison
    return updated


def recommend_followup_settings(artifact: dict[str, Any]) -> dict[str, Any]:
    recommendation = dict(artifact.get("recommendation", {}))
    notes = list(recommendation.get("notes", []))
    report = artifact.get("report") or {}
    evaluation = report.get("evaluation") or {}
    status = report.get("status") or {}
    benchmark = artifact.get("benchmark") or {}
    eval_comparison = artifact.get("eval_comparison") or {}

    if status.get("label") == "ollama_create_failed":
        notes.append("Previous export/create failed; keep settings but inspect conversion/export path.")

    refusal_rate = evaluation.get("refusal_rate")
    if isinstance(refusal_rate, (int, float)):
        if refusal_rate >= 0.5:
            recommendation["strength"] = max(float(recommendation.get("strength", 1.0)), 1.3)
            recommendation["per_layer_directions"] = True
            notes.append("Post-run refusal rate is still high; increase strength and use per-layer directions.")
        elif refusal_rate <= 0.1:
            recommendation["strength"] = min(float(recommendation.get("strength", 1.0)), 1.0)
            notes.append("Post-run refusal rate is low; no stronger intervention needed.")

    if isinstance(evaluation.get("refusal_count"), (int, float)) and isinstance(evaluation.get("total"), (int, float)):
        total = max(float(evaluation["total"]), 1.0)
        count = float(evaluation["refusal_count"])
        if count / total >= 0.75:
            recommendation["mlp_strength"] = max(float(recommendation.get("mlp_strength", 0.9)), 1.2)
            notes.append("A large majority of eval prompts still refuse; increase MLP-side intervention.")

    benchmark_kpis = benchmark.get("kpis") or {}
    if isinstance(benchmark_kpis.get("refusal_rate_pct"), (int, float)):
        refusal_pct = float(benchmark_kpis["refusal_rate_pct"])
        if refusal_pct >= 60.0:
            recommendation["strength"] = max(float(recommendation.get("strength", 1.0)), 1.25)
            recommendation["per_layer_directions"] = True
            notes.append("Benchmark refusal rate remains high; escalate to stronger, more targeted ablation.")
    if isinstance(benchmark_kpis.get("asr_pct"), (int, float)):
        asr_pct = float(benchmark_kpis["asr_pct"])
        if asr_pct <= 20.0:
            recommendation["strength"] = max(float(recommendation.get("strength", 1.0)), 1.2)
            notes.append("Benchmark attack success remains low; modestly increase strength.")

    comparison_metrics = eval_comparison.get("metrics") or {}
    asr_compare = comparison_metrics.get("asr_pct") or {}
    refusal_compare = comparison_metrics.get("refusal_rate_pct") or {}
    asr_a = asr_compare.get("a")
    asr_b = asr_compare.get("b")
    if isinstance(asr_a, (int, float)) and isinstance(asr_b, (int, float)) and float(asr_b) <= float(asr_a):
        recommendation["strength"] = max(float(recommendation.get("strength", 1.0)), 1.2)
        notes.append("Benchmark ASR did not improve; increase intervention strength.")
    ref_a = refusal_compare.get("a")
    ref_b = refusal_compare.get("b")
    if isinstance(ref_a, (int, float)) and isinstance(ref_b, (int, float)) and float(ref_b) >= float(ref_a):
        recommendation["mlp_strength"] = max(float(recommendation.get("mlp_strength", 0.9)), 1.1)
        notes.append("Benchmark refusal did not decrease; increase MLP-side intervention.")

    recommendation["notes"] = notes
    return recommendation
