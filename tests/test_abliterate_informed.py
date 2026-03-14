"""Tests for analysis-informed abliterate recommendations."""

from ollama_forge.abliterate_informed import recommend_abliterate_settings, recommend_followup_settings


def test_recommend_abliterate_settings_uses_alignment_and_tracing_signals() -> None:
    recommendation = recommend_abliterate_settings(
        [
            {"mean_adjacent_cosine": 0.55},
            {"largest_delta_layer": 12},
            {"most_critical_layer": 18},
        ]
    )
    assert recommendation["profile"] == "aggressive"
    assert recommendation["per_layer_directions"] is True
    assert recommendation["strength"] >= 1.2


def test_recommend_abliterate_settings_uses_architecture_and_geometry_signals() -> None:
    recommendation = recommend_abliterate_settings(
        [
            {"arch_class": "moe", "reasoning_class": "reasoning"},
            {"most_polyhedral_layer": 10},
            {"strongest_layer": 12},
        ]
    )
    assert recommendation["profile"] == "safe"
    assert recommendation["per_layer_directions"] is True
    assert recommendation["norm_preserving"] is True


def test_recommend_abliterate_settings_uses_external_eval_docs() -> None:
    recommendation = recommend_abliterate_settings(
        [
            {"kpis": {"refusal_rate_pct": 80.0, "asr_pct": 10.0}},
            {"results": {"hellaswag": {"acc_norm,none": 0.2}}},
        ]
    )
    assert recommendation["profile"] == "safe"
    assert recommendation["strength"] >= 1.2


def test_recommend_abliterate_settings_uses_defense_robustness() -> None:
    recommendation = recommend_abliterate_settings(
        [
            {"self_repair_risk": 0.8, "entanglement_score": 0.9},
        ]
    )
    assert recommendation["per_layer_directions"] is True
    assert recommendation["profile"] == "safe"
    assert recommendation["mlp_strength"] <= 0.7


def test_recommend_followup_settings_raises_strength_when_refusal_remains_high() -> None:
    refined = recommend_followup_settings(
        {
            "recommendation": {"profile": "balanced", "strength": 1.0, "mlp_strength": 0.9, "notes": []},
            "report": {
                "status": {"label": "ollama_created"},
                "evaluation": {"refusal_rate": 0.8, "refusal_count": 8, "total": 10},
            },
        }
    )
    assert refined["strength"] >= 1.3
    assert refined["per_layer_directions"] is True


def test_recommend_followup_settings_uses_benchmark_and_eval_comparison() -> None:
    refined = recommend_followup_settings(
        {
            "recommendation": {"profile": "balanced", "strength": 1.0, "mlp_strength": 0.9, "notes": []},
            "benchmark": {"kpis": {"refusal_rate_pct": 80.0, "asr_pct": 15.0}},
            "eval_comparison": {
                "metrics": {
                    "asr_pct": {"a": 30.0, "b": 15.0},
                    "refusal_rate_pct": {"a": 40.0, "b": 80.0},
                }
            },
        }
    )
    assert refined["strength"] >= 1.25
    assert refined["mlp_strength"] >= 1.1
