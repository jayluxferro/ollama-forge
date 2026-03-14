"""Optimization helpers for study interventions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ollama_forge.study_config import StrategyConfig, StudyConfig
from ollama_forge.study_runner import run_study


@dataclass
class StudyOptimizationResult:
    metric: str
    objective: str
    best_strength: float
    best_score: float
    tried: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "objective": self.objective,
            "best_strength": self.best_strength,
            "best_score": self.best_score,
            "tried": self.tried,
        }


def _clone_with_strength(config: StudyConfig, strength: float, output_dir: str) -> StudyConfig:
    copied = StudyConfig.from_dict(config.to_dict())
    copied.output_dir = output_dir
    copied.strategies = [
        StrategyConfig(name=item.name, params={**item.params, "strength": strength}) for item in copied.strategies
    ]
    return copied


def optimize_study_strength(
    config: StudyConfig,
    *,
    strengths: list[float],
    metric: str,
    objective: str,
    model_loader: Callable[[Any], Any],
    dataset_loader: Callable[[Any], Any],
    evaluator_factory: Callable[..., Any],
    output_dir: str | Path,
) -> StudyOptimizationResult:
    maximize = objective == "max"
    trials: list[dict[str, float]] = []
    best_strength = strengths[0]
    best_score = float("-inf") if maximize else float("inf")
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for strength in strengths:
        trial_dir = base_dir / f"strength-{strength:.2f}".replace(".", "_")
        trial_cfg = _clone_with_strength(config, strength, str(trial_dir))
        report = run_study(
            trial_cfg,
            model_loader=model_loader,
            dataset_loader=dataset_loader,
            evaluator_factory=evaluator_factory,
            output_dir=trial_dir,
        )
        if not report.results:
            continue
        values = [item.metrics.get(metric) for item in report.results if metric in item.metrics]
        if not values:
            continue
        score = float(sum(values) / len(values))
        trials.append({"strength": strength, "score": score})
        if maximize:
            if score > best_score:
                best_score = score
                best_strength = strength
        else:
            if score < best_score:
                best_score = score
                best_strength = strength

    result = StudyOptimizationResult(
        metric=metric,
        objective=objective,
        best_strength=best_strength,
        best_score=best_score,
        tried=trials,
    )
    (base_dir / "study-optimize.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
