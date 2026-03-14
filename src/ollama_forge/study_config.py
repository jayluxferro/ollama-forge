"""YAML-backed configuration for generic ablation studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    task: str = "causal_lm"
    dtype: str = "float32"
    device: str = "auto"
    trust_remote_code: bool = False
    num_labels: int = 2


@dataclass
class DatasetConfig:
    name: str
    split: str = "test"
    subset: str | None = None
    text_column: str = "text"
    label_column: str = "label"
    max_samples: int | None = None


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyConfig:
    model: ModelConfig
    dataset: DatasetConfig
    strategies: list[StrategyConfig]
    metrics: list[str] = field(default_factory=lambda: ["perplexity"])
    batch_size: int = 8
    max_length: int = 512
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StudyConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudyConfig":
        data = dict(raw)
        if "preset" in data and "study_preset" not in data:
            data["study_preset"] = data["preset"]
        if "study_preset" in data:
            from ollama_forge.study_presets import get_study_preset

            preset = get_study_preset(data["study_preset"])
            if "strategies" not in data:
                data["strategies"] = preset.strategies
            if "metrics" not in data:
                data["metrics"] = preset.metrics
            if "batch_size" not in data:
                data["batch_size"] = preset.batch_size
            if "max_length" not in data:
                data["max_length"] = preset.max_length
            dataset = dict(data.get("dataset", {}))
            if dataset and "max_samples" not in dataset:
                dataset["max_samples"] = preset.max_samples
                data["dataset"] = dataset

        model = ModelConfig(**data["model"])
        dataset = DatasetConfig(**data["dataset"])
        strategies = [StrategyConfig(**item) for item in data["strategies"]]
        return cls(
            model=model,
            dataset=dataset,
            strategies=strategies,
            metrics=data.get("metrics", ["perplexity"]),
            batch_size=data.get("batch_size", 8),
            max_length=data.get("max_length", 512),
            output_dir=data.get("output_dir", "results"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.model.name,
                "task": self.model.task,
                "dtype": self.model.dtype,
                "device": self.model.device,
                "trust_remote_code": self.model.trust_remote_code,
                "num_labels": self.model.num_labels,
            },
            "dataset": {
                "name": self.dataset.name,
                "split": self.dataset.split,
                "subset": self.dataset.subset,
                "text_column": self.dataset.text_column,
                "label_column": self.dataset.label_column,
                "max_samples": self.dataset.max_samples,
            },
            "strategies": [{"name": item.name, "params": item.params} for item in self.strategies],
            "metrics": self.metrics,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "output_dir": self.output_dir,
        }
