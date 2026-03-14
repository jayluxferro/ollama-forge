"""Reporting helpers for generic ablation studies."""

from __future__ import annotations

import csv
import json
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StudyResult:
    strategy: str
    component: str
    description: str
    metrics: dict[str, float]
    metadata: dict[str, Any] | None = None


@dataclass
class StudyReport:
    model_name: str
    config: dict[str, Any]
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    results: list[StudyResult] = field(default_factory=list)

    def add_baseline(self, metrics: dict[str, float]) -> None:
        self.baseline_metrics = dict(metrics)

    def add_result(self, result: StudyResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config,
            "baseline_metrics": self.baseline_metrics,
            "results": [
                {
                    "strategy": item.strategy,
                    "component": item.component,
                    "description": item.description,
                    "metrics": item.metrics,
                    "metadata": item.metadata,
                }
                for item in self.results
            ],
        }

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    def save_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metric_names = sorted({metric for item in self.results for metric in item.metrics})
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["strategy", "component", "description", *metric_names])
            for item in self.results:
                writer.writerow(
                    [
                        item.strategy,
                        item.component,
                        item.description,
                        *[item.metrics.get(metric, "") for metric in metric_names],
                    ]
                )
        return target

    def summary_lines(self) -> list[str]:
        lines = [f"Model: {self.model_name}", f"Results: {len(self.results)}", f"Baseline: {self.baseline_metrics}"]
        metric_names = sorted({metric for item in self.results for metric in item.metrics})
        for metric in metric_names:
            values = [item.metrics.get(metric) for item in self.results if metric in item.metrics]
            if not values:
                continue
            best = min(values) if metric == "perplexity" else max(values)
            lines.append(f"{metric}: best={best:.4f}")
        return lines

    def save_summary(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(self.summary_lines()) + "\n", encoding="utf-8")
        return target

    def plot_impact(self, output_path: str | Path, metric: str | None = None) -> Path:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("study plot generation requires matplotlib") from exc

        metric_names = sorted({name for item in self.results for name in item.metrics})
        if not metric_names:
            raise ValueError("No metrics available to plot")
        metric = metric or metric_names[0]
        labels = [item.component for item in self.results if metric in item.metrics]
        values = [item.metrics[metric] for item in self.results if metric in item.metrics]
        if not labels:
            raise ValueError(f"No results for metric {metric!r}")
        figure, axis = plt.subplots(figsize=(12, max(4, len(labels) * 0.3)))
        axis.barh(labels, values)
        axis.set_title(f"{self.model_name} - {metric}")
        axis.set_xlabel(metric)
        axis.set_ylabel("Component")
        figure.tight_layout()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return target

    def to_markdown(self) -> str:
        lines = [f"# Study Report: {self.model_name}", ""]
        lines.append("## Baseline")
        for key, value in sorted(self.baseline_metrics.items()):
            lines.append(f"- {key}: {value:.4f}")
        lines.append("")
        lines.append("## Results")
        lines.append("| Strategy | Component | Metrics |")
        lines.append("|---|---|---|")
        for item in self.results:
            metrics = ", ".join(f"{key}={value:.4f}" for key, value in sorted(item.metrics.items()))
            lines.append(f"| {item.strategy} | {item.component} | {metrics} |")
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        rows = []
        for item in self.results:
            metrics = ", ".join(f"{key}={value:.4f}" for key, value in sorted(item.metrics.items()))
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(item.strategy),
                    html.escape(item.component),
                    html.escape(metrics),
                )
            )
        baseline = "".join(
            f"<li>{html.escape(key)}: {value:.4f}</li>" for key, value in sorted(self.baseline_metrics.items())
        )
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Study Report</title></head><body>"
            f"<h1>Study Report: {html.escape(self.model_name)}</h1>"
            "<h2>Baseline</h2><ul>"
            f"{baseline}</ul>"
            "<h2>Results</h2>"
            "<table border='1'><tr><th>Strategy</th><th>Component</th><th>Metrics</th></tr>"
            f"{''.join(rows)}</table></body></html>"
        )

    def save_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    def save_html(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_html(), encoding="utf-8")
        return target


def load_study_report(path: str | Path) -> StudyReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = StudyReport(
        model_name=payload["model_name"],
        config=payload.get("config", {}),
        baseline_metrics=payload.get("baseline_metrics", {}),
    )
    for item in payload.get("results", []):
        report.add_result(
            StudyResult(
                strategy=item["strategy"],
                component=item["component"],
                description=item["description"],
                metrics=item.get("metrics", {}),
                metadata=item.get("metadata"),
            )
        )
    return report


def compare_study_reports(report_a: StudyReport, report_b: StudyReport) -> dict[str, Any]:
    metrics = sorted(set(report_a.baseline_metrics) | set(report_b.baseline_metrics))
    return {
        "model_a": report_a.model_name,
        "model_b": report_b.model_name,
        "baseline_metrics": {
            metric: {"a": report_a.baseline_metrics.get(metric), "b": report_b.baseline_metrics.get(metric)}
            for metric in metrics
        },
        "result_count": {"a": len(report_a.results), "b": len(report_b.results)},
    }


def compare_study_reports_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Study Report Comparison", ""]
    lines.append(f"A: {payload['model_a']}")
    lines.append(f"B: {payload['model_b']}")
    lines.append("")
    lines.append("| Metric | A | B |")
    lines.append("|---|---:|---:|")
    for metric, values in sorted(payload["baseline_metrics"].items()):
        lines.append(f"| {metric} | {values.get('a')} | {values.get('b')} |")
    lines.append("")
    lines.append(f"- result_count.a: {payload['result_count']['a']}")
    lines.append(f"- result_count.b: {payload['result_count']['b']}")
    return "\n".join(lines) + "\n"


def compare_study_reports_html(payload: dict[str, Any]) -> str:
    rows = []
    for metric, values in sorted(payload["baseline_metrics"].items()):
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(metric),
                html.escape(str(values.get("a"))),
                html.escape(str(values.get("b"))),
            )
        )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Study Comparison</title></head><body>"
        "<h1>Study Report Comparison</h1>"
        f"<p><strong>A:</strong> {html.escape(payload['model_a'])}</p>"
        f"<p><strong>B:</strong> {html.escape(payload['model_b'])}</p>"
        "<table border='1'><tr><th>Metric</th><th>A</th><th>B</th></tr>"
        f"{''.join(rows)}</table></body></html>"
    )


def save_study_report_comparison(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        path.write_text(compare_study_reports_markdown(payload), encoding="utf-8")
    elif suffix in (".html", ".htm"):
        path.write_text(compare_study_reports_html(payload), encoding="utf-8")
    elif suffix == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    elif suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "a", "b"])
            for metric, values in sorted(payload["baseline_metrics"].items()):
                writer.writerow([metric, values.get("a"), values.get("b")])
            writer.writerow(["result_count", payload["result_count"]["a"], payload["result_count"]["b"]])
    else:
        raise ValueError(f"Unsupported comparison export format: {path}")
    return path
