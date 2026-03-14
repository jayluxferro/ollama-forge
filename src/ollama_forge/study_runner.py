"""Planning and execution for generic ablation studies."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ollama_forge.study_config import StudyConfig
from ollama_forge.study_manifest import build_study_manifest, save_study_manifest
from ollama_forge.study_reports import StudyReport, StudyResult
from ollama_forge.study_strategies import get_strategy


@dataclass
class StudyPlanItem:
    strategy: str
    params: dict[str, Any]


@dataclass
class StudyPlan:
    model_name: str
    dataset_name: str
    metrics: list[str]
    output_dir: str
    strategies: list[StudyPlanItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
            "output_dir": self.output_dir,
            "strategies": [
                {"strategy": item.strategy, "params": item.params} for item in self.strategies
            ],
        }


def plan_study(config: StudyConfig) -> StudyPlan:
    return StudyPlan(
        model_name=config.model.name,
        dataset_name=config.dataset.name,
        metrics=list(config.metrics),
        output_dir=config.output_dir,
        strategies=[
            StudyPlanItem(strategy=item.name, params=dict(item.params)) for item in config.strategies
        ],
    )


def run_study(
    config: StudyConfig,
    *,
    model_loader: Callable[[Any], Any],
    dataset_loader: Callable[[Any], Any],
    evaluator_factory: Callable[..., Any],
    output_dir: str | Path | None = None,
) -> StudyReport:
    report = StudyReport(model_name=config.model.name, config=config.to_dict())
    output_root = Path(output_dir or config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    handle = model_loader(config.model)
    dataset = dataset_loader(config.dataset)

    baseline_evaluator = evaluator_factory(
        handle=handle,
        dataset=dataset,
        metrics=config.metrics,
        batch_size=config.batch_size,
        max_length=config.max_length,
        max_samples=config.dataset.max_samples,
        text_column=config.dataset.text_column,
        label_column=config.dataset.label_column,
    )
    baseline_metrics = baseline_evaluator.evaluate()
    report.add_baseline(baseline_metrics)

    for strategy_cfg in config.strategies:
        strategy = get_strategy(strategy_cfg.name)
        specs = strategy.enumerate(handle, **strategy_cfg.params)
        for spec in specs:
            strategy.apply(handle, spec)
            evaluator = evaluator_factory(
                handle=handle,
                dataset=dataset,
                metrics=config.metrics,
                batch_size=config.batch_size,
                max_length=config.max_length,
                max_samples=config.dataset.max_samples,
                text_column=config.dataset.text_column,
                label_column=config.dataset.label_column,
            )
            metrics = evaluator.evaluate()
            report.add_result(
                StudyResult(
                    strategy=spec.strategy_name,
                    component=spec.component,
                    description=spec.description,
                    metrics=metrics,
                    metadata=spec.metadata,
                )
            )
            restore = getattr(handle, "restore", None)
            if callable(restore):
                restore()

    report.save_json(output_root / "study-results.json")
    report.save_csv(output_root / "study-results.csv")
    report.save_summary(output_root / "study-summary.txt")
    report.save_markdown(output_root / "study-report.md")
    report.save_html(output_root / "study-report.html")
    with contextlib.suppress(ImportError, ValueError):
        report.plot_impact(output_root / "study-impact.png")
    manifest = build_study_manifest(
        config=config.to_dict(),
        artifacts={
            "study_results_json": str(output_root / "study-results.json"),
            "study_results_csv": str(output_root / "study-results.csv"),
            "study_summary": str(output_root / "study-summary.txt"),
            "study_report_markdown": str(output_root / "study-report.md"),
            "study_report_html": str(output_root / "study-report.html"),
            "study_impact_plot": str(output_root / "study-impact.png"),
        },
    )
    save_study_manifest(manifest, output_root / "study-manifest.json")
    return report
