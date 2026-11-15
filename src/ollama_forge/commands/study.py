"""Study command group: ablation study planning, running, analysis, and reporting."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from ollama_forge.run_helpers import print_actionable_error


def _prompt_with_default(prompt: str, default: str) -> str:
    """Prompt in interactive terminals; return default when blank or non-interactive."""
    if not sys.stdin.isatty():
        return default
    try:
        value = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return value or default


def _cmd_study_ui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Launch Streamlit UI for study workflows."""
    import subprocess

    app_dir = Path(__file__).resolve().parent.parent
    app_path = app_dir / "study_app.py"
    if not app_path.exists():
        print_actionable_error(
            f"study UI app not found at {app_path}",
            next_steps=[
                "Ensure the study_app module is installed",
                "Run: uv sync",
            ],
        )
        return 1
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
            check=False,
        )
    except FileNotFoundError:
        print_actionable_error(
            "Streamlit not found",
            next_steps=[
                "Run: uv sync",
                "Then: ollama-forge study ui",
            ],
        )
        return 1
    return 0


def _cmd_study_presets(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_presets import list_study_presets

    presets = list_study_presets()
    if getattr(args, "json", False):
        print(
            json.dumps(
                [
                    {
                        "key": preset.key,
                        "name": preset.name,
                        "description": preset.description,
                        "strategies": preset.strategies,
                        "metrics": preset.metrics,
                        "max_samples": preset.max_samples,
                        "batch_size": preset.batch_size,
                        "max_length": preset.max_length,
                        "tags": preset.tags,
                    }
                    for preset in presets
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for preset in presets:
        print(f"{preset.key}: {preset.name}")
        print(f"  {preset.description}")
        print(
            "  strategies={0} metrics={1} max_samples={2} batch_size={3} max_length={4}".format(
                ",".join(item["name"] for item in preset.strategies),
                ",".join(preset.metrics),
                preset.max_samples,
                preset.batch_size,
                preset.max_length,
            )
        )
    return 0


def _cmd_study_strategies(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_strategies import list_strategies

    strategies = list_strategies()
    if getattr(args, "json", False):
        print(json.dumps(list(strategies), indent=2))
    else:
        for name in strategies:
            print(name)
    return 0


def _cmd_study_validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_config import StudyConfig

    try:
        config = StudyConfig.from_yaml(args.config)
    except Exception as e:
        print_actionable_error(
            "study config validation failed",
            cause=str(e),
            next_steps=["Check the YAML/JSON format and preset names", "Run: ollama-forge study presets"],
        )
        return 1
    payload = config.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Model: {config.model.name}")
        print(f"Dataset: {config.dataset.name}:{config.dataset.split}")
        print(f"Strategies: {', '.join(item.name for item in config.strategies)}")
        print(f"Output dir: {config.output_dir}")
    return 0


def _cmd_study_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_config import StudyConfig
    from ollama_forge.study_runner import plan_study

    try:
        config = StudyConfig.from_yaml(args.config)
        plan = plan_study(config)
    except Exception as e:
        print_actionable_error(
            "study planning failed",
            cause=str(e),
            next_steps=["Check the config file", "Run: ollama-forge study validate <config>"],
        )
        return 1
    payload = plan.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Model: {payload['model_name']}")
        print(f"Dataset: {payload['dataset_name']}")
        print(f"Metrics: {', '.join(payload['metrics'])}")
        print(f"Output dir: {payload['output_dir']}")
        print("Strategies:")
        for item in payload["strategies"]:
            print(f"  - {item['strategy']}: {item['params']}")
    return 0


def _cmd_study_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_config import StudyConfig
    from ollama_forge.study_runner import run_study

    try:
        from ollama_forge.study_runtime import StudyEvaluator, load_study_dataset, load_study_model
    except ImportError as e:
        print_actionable_error(
            "study run requires optional study dependencies",
            cause=str(e),
            next_steps=["Run: uv sync", "Then: ollama-forge study run <config>"],
        )
        return 1

    try:
        config = StudyConfig.from_yaml(args.config)
        if getattr(args, "output_dir", None):
            config.output_dir = args.output_dir
        report = run_study(
            config,
            model_loader=load_study_model,
            dataset_loader=load_study_dataset,
            evaluator_factory=StudyEvaluator,
        )
    except Exception as e:
        print_actionable_error(
            "study run failed",
            cause=str(e),
            next_steps=[
                "Check the model id, dataset config, and strategy list",
                "Run: ollama-forge study validate <config>",
                "Ensure dependencies are installed: uv sync",
            ],
        )
        return 1

    output_root = Path(config.output_dir)
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Model: {report.model_name}")
        print(f"Baseline: {report.baseline_metrics}")
        print(f"Results: {len(report.results)}")
        print(f"Saved: {output_root / 'study-results.json'}")
        print(f"Saved: {output_root / 'study-results.csv'}")
        print(f"Saved: {output_root / 'study-summary.txt'}")
        impact_plot = output_root / "study-impact.png"
        if impact_plot.is_file():
            print(f"Saved: {impact_plot}")
    return 0


def _cmd_study_models(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_model_presets import (
        detect_hardware_tier,
        format_hardware_info,
        list_model_presets,
        recommended_model_presets,
    )

    tier = getattr(args, "tier", None)
    recommend = getattr(args, "recommend", False)
    if recommend:
        detected_tier, info = detect_hardware_tier()
        tier = tier or detected_tier
        presets = recommended_model_presets(tier=tier, limit=getattr(args, "limit", 5))
    else:
        detected_tier, info = detect_hardware_tier()
        presets = list_model_presets(tier=tier)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "requested_tier": tier,
                    "detected_tier": detected_tier,
                    "hardware": info,
                    "models": [preset.__dict__ for preset in presets],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"Detected tier: {detected_tier} ({format_hardware_info(info)})")
    if tier:
        print(f"Filter tier: {tier}")
    for preset in presets:
        quant = f" quant={preset.recommended_quantization}" if preset.recommended_quantization else ""
        gated = " gated" if preset.gated else ""
        print(f"{preset.hf_id} [{preset.tier}] {preset.params} dtype={preset.recommended_dtype}{quant}{gated}")
        print(f"  {preset.description}")
    return 0


def _cmd_study_analysis_modules(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_analysis import available_analysis_modules

    modules = list(available_analysis_modules())
    if getattr(args, "json", False):
        print(json.dumps(modules, indent=2))
    else:
        for module in modules:
            print(module)
    return 0


def _run_study_analysis_module(config, handle, dataset, module_name: str, args: argparse.Namespace):
    from ollama_forge.study_analysis import (
        analyze_activation_patching,
        analyze_activation_probe,
        analyze_causal_patching,
        analyze_concept_geometry,
        analyze_conditional_similarity,
        analyze_cross_layer_similarity,
        analyze_defense_robustness,
        analyze_logit_lens,
        analyze_residual_stream,
        analyze_steering_vectors,
        collect_grouped_layer_vectors,
        collect_layer_vectors,
        trace_causal_layers,
    )
    from ollama_forge.study_architecture import detect_architecture_profile

    if module_name == "causal_tracing":
        prompt = getattr(args, "prompt", None)
        if not prompt:
            limit = min(len(dataset), 1)
            if limit == 0:
                raise ValueError("Dataset is empty; pass --prompt for causal tracing")
            row = dataset[0]
            prompt = row.get(config.dataset.text_column) if isinstance(row, dict) else row[config.dataset.text_column]
        max_len = getattr(args, "max_length", None) or config.max_length
        return trace_causal_layers(handle, str(prompt), max_length=max_len)

    if module_name == "causal_patching":
        source_prompt = getattr(args, "source_prompt", None)
        target_prompt = getattr(args, "target_prompt", None)
        if not source_prompt or not target_prompt:
            if len(dataset) < 2:
                raise ValueError(
                    "causal_patching requires --source-prompt and --target-prompt "
                    "when dataset has fewer than 2 rows"
                )
            row0 = dataset[0]
            row1 = dataset[1]
            tc = config.dataset.text_column
            if not source_prompt:
                source_prompt = row0.get(tc) if isinstance(row0, dict) else row0[tc]
            if not target_prompt:
                target_prompt = row1.get(tc) if isinstance(row1, dict) else row1[tc]
        return analyze_causal_patching(
            handle,
            source_prompt=str(source_prompt),
            target_prompt=str(target_prompt),
            max_length=getattr(args, "max_length", None) or config.max_length,
        )

    if module_name == "architecture_profile":
        return detect_architecture_profile(handle, model_name=config.model.name)

    if module_name in {
        "conditional_similarity",
        "activation_patching",
        "steering_vectors",
        "concept_geometry",
        "defense_robustness",
    }:
        group_column = getattr(args, "group_column", None) or config.dataset.label_column
        grouped = collect_grouped_layer_vectors(
            handle,
            dataset,
            group_column=group_column,
            text_column=config.dataset.text_column,
            max_samples=getattr(args, "max_samples", None) or config.dataset.max_samples,
            batch_size=getattr(args, "batch_size", None) or config.batch_size,
            max_length=getattr(args, "max_length", None) or config.max_length,
        )
        if module_name == "conditional_similarity":
            return analyze_conditional_similarity(grouped)
        if module_name == "activation_patching":
            source_group = getattr(args, "source_group", None)
            target_group = getattr(args, "target_group", None)
            if not source_group or not target_group:
                raise ValueError("activation_patching requires --source-group and --target-group")
            return analyze_activation_patching(grouped, source_group=source_group, target_group=target_group)
        if module_name == "steering_vectors":
            return analyze_steering_vectors(grouped)
        if module_name == "concept_geometry":
            return analyze_concept_geometry(grouped)
        return analyze_defense_robustness(grouped)

    vectors = collect_layer_vectors(
        handle,
        dataset,
        text_column=config.dataset.text_column,
        max_samples=getattr(args, "max_samples", None) or config.dataset.max_samples,
        batch_size=getattr(args, "batch_size", None) or config.batch_size,
        max_length=getattr(args, "max_length", None) or config.max_length,
    )
    if module_name == "activation_probe":
        return analyze_activation_probe(vectors)
    if module_name == "cross_layer_similarity":
        return analyze_cross_layer_similarity(vectors)
    if module_name == "logit_lens":
        return analyze_logit_lens(handle, vectors, top_k=getattr(args, "top_k", None) or 5)
    return analyze_residual_stream(vectors)


def _cmd_study_analyze_bundle(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from ollama_forge.study_analysis import available_analysis_modules
    from ollama_forge.study_analysis_bundle import build_analysis_bundle, save_analysis_bundle
    from ollama_forge.study_config import StudyConfig

    try:
        from ollama_forge.study_runtime import load_study_dataset, load_study_model
    except ImportError as e:
        print_actionable_error(
            "study analyze-bundle requires optional study dependencies",
            cause=str(e),
            next_steps=["Run: uv sync", "Then: ollama-forge study analyze-bundle <config>"],
        )
        return 1

    module_list = getattr(args, "modules", None)
    if module_list:
        modules = [item.strip() for item in module_list.split(",") if item.strip()]
    else:
        modules = list(available_analysis_modules())
    try:
        config = StudyConfig.from_yaml(args.config)
        handle = load_study_model(config.model)
        dataset = load_study_dataset(config.dataset)
        results = {}
        for module_name in modules:
            result = _run_study_analysis_module(config, handle, dataset, module_name, args)
            results[module_name] = asdict(result)
        bundle = build_analysis_bundle(config_path=str(args.config), modules=modules, results=results)
        output_path = Path(getattr(args, "output_file", None) or (Path(config.output_dir) / "analysis-bundle.json"))
        save_analysis_bundle(bundle, output_path)
    except Exception as e:
        print_actionable_error(
            "study analyze-bundle failed",
            cause=str(e),
            next_steps=["Check the config and optional prompt/group arguments"],
        )
        return 1
    if getattr(args, "json", False):
        print(json.dumps(bundle, indent=2, sort_keys=True))
    else:
        print(f"Saved bundle: {output_path}")
        print(f"Modules: {', '.join(modules)}")
    return 0


def _cmd_study_analyze(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_config import StudyConfig

    try:
        from ollama_forge.study_analysis import save_analysis_result
        from ollama_forge.study_runtime import load_study_dataset, load_study_model
    except ImportError as e:
        print_actionable_error(
            "study analyze requires optional study dependencies",
            cause=str(e),
            next_steps=["Run: uv sync", "Then: ollama-forge study analyze <config> --module <name>"],
        )
        return 1

    module_name = getattr(args, "module", None)
    if module_name not in (
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
    ):
        print_actionable_error(
            "unknown analysis module",
            cause=str(module_name),
            next_steps=["Run: ollama-forge study analysis-modules"],
        )
        return 1
    try:
        config = StudyConfig.from_yaml(args.config)
        handle = load_study_model(config.model)
        dataset = load_study_dataset(config.dataset)
        result = _run_study_analysis_module(config, handle, dataset, module_name, args)
        output_dir = Path(getattr(args, "output_dir", None) or config.output_dir)
        output_path = Path(getattr(args, "output_file", None) or (output_dir / f"{module_name}.json"))
        save_analysis_result(result, output_path)
    except Exception as e:
        print_actionable_error(
            "study analysis failed",
            cause=str(e),
            next_steps=[
                "Check the model and dataset configuration",
                "Run: ollama-forge study validate <config>",
                "Ensure dependencies are installed: uv sync",
            ],
        )
        return 1
    if getattr(args, "json", False):
        from dataclasses import asdict

        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print(f"Module: {module_name}")
        print(f"Saved: {output_path}")
    return 0


def _cmd_study_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_reports import load_study_report

    path = Path(getattr(args, "path", ""))
    if not path.is_file():
        print_actionable_error(
            "study report file not found",
            cause=str(path),
            next_steps=["Pass the study-results.json file produced by `ollama-forge study run`"],
        )
        return 1
    try:
        report = load_study_report(path)
    except Exception as e:
        print_actionable_error(
            "failed to load study report", cause=str(e),
            next_steps=["Ensure the file is valid JSON"],
        )
        return 1
    export_path = getattr(args, "export", None)
    if export_path:
        export_target = Path(export_path)
        if export_target.suffix.lower() in (".md", ".markdown"):
            report.save_markdown(export_target)
        elif export_target.suffix.lower() in (".html", ".htm"):
            report.save_html(export_target)
        elif export_target.suffix.lower() == ".json":
            report.save_json(export_target)
        elif export_target.suffix.lower() == ".csv":
            report.save_csv(export_target)
        else:
            print_actionable_error(
                "unsupported export format", cause=str(export_target),
                next_steps=["Use .md, .html, .json, or .csv"],
            )
            return 1
        print(f"Exported {export_target}")
        if not getattr(args, "json", False):
            return 0
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    for line in report.summary_lines():
        print(line)
    if report.results:
        print("Top components:")
        for item in report.results[: min(10, len(report.results))]:
            print(f"  {item.strategy} {item.component} {item.metrics}")
    return 0


def _cmd_study_compare(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_reports import (
        compare_study_reports,
        load_study_report,
        save_study_report_comparison,
    )

    path_a = Path(getattr(args, "report_a", ""))
    path_b = Path(getattr(args, "report_b", ""))
    if not path_a.is_file() or not path_b.is_file():
        missing = path_a if not path_a.is_file() else path_b
        print_actionable_error(
            "study report file not found",
            cause=str(missing),
            next_steps=["Pass two study-results.json files produced by `ollama-forge study run`"],
        )
        return 1
    try:
        report_a = load_study_report(path_a)
        report_b = load_study_report(path_b)
    except Exception as e:
        print_actionable_error(
            "failed to load study reports", cause=str(e),
            next_steps=["Ensure both files are valid JSON"],
        )
        return 1
    payload = compare_study_reports(report_a, report_b)
    export = getattr(args, "export", None)
    if export:
        try:
            save_study_report_comparison(payload, export)
        except Exception as e:
            print_actionable_error("failed to export study comparison", cause=str(e))
            return 1
        print(f"Exported {export}")
        if not getattr(args, "json", False):
            return 0
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"A: {payload['model_a']}")
    print(f"B: {payload['model_b']}")
    for metric, values in payload["baseline_metrics"].items():
        print(f"{metric}: A={values.get('a')} B={values.get('b')}")
    print(f"results: A={len(report_a.results)} B={len(report_b.results)}")
    return 0


def _cmd_study_contribute(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_community import save_study_contribution
    from ollama_forge.study_reports import load_study_report

    path = Path(getattr(args, "report", ""))
    if not path.is_file():
        print_actionable_error("study report file not found", cause=str(path))
        return 1
    try:
        report = load_study_report(path)
        saved = save_study_contribution(
            report,
            source_report=str(path),
            output_dir=getattr(args, "dir", None) or "study_results_community",
            notes=getattr(args, "notes", "") or "",
        )
    except Exception as e:
        print_actionable_error("study contribute failed", cause=str(e))
        return 1
    print(f"Saved {saved}")
    return 0


def _cmd_study_aggregate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_community import aggregate_study_contributions, load_study_contributions

    records = load_study_contributions(getattr(args, "dir", None) or "study_results_community")
    if not records:
        print("No study contributions found.")
        return 0
    payload = aggregate_study_contributions(records)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for model_name, summary in sorted(payload.items()):
        print(model_name)
        print(f"  n_reports={summary['n_reports']}")
        for key, value in sorted(summary.items()):
            if key == "n_reports":
                continue
            print(f"  {key}: mean={value['mean']:.4f} std={value['std']:.4f} n={value['n']}")
    return 0


def _cmd_study_regenerate_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_reports import load_study_report

    path = Path(getattr(args, "path", ""))
    if not path.is_file():
        print_actionable_error("study report file not found", cause=str(path))
        return 1
    try:
        report = load_study_report(path)
        output_dir = Path(getattr(args, "output_dir", None) or path.parent)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.save_json(output_dir / "study-results.json")
        report.save_csv(output_dir / "study-results.csv")
        report.save_summary(output_dir / "study-summary.txt")
        report.save_markdown(output_dir / "study-report.md")
        report.save_html(output_dir / "study-report.html")
        with contextlib.suppress(ImportError, ValueError):
            report.plot_impact(output_dir / "study-impact.png")
    except Exception as e:
        print_actionable_error("study regenerate-report failed", cause=str(e))
        return 1
    print(f"Regenerated exports in {output_dir}")
    return 0


def _study_default_model_for_tier(tier: str) -> str:
    from ollama_forge.study_model_presets import recommended_model_presets

    presets = recommended_model_presets(tier=tier, limit=1)
    return presets[0].hf_id if presets else "distilgpt2"


def _cmd_study_init(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_presets import get_study_preset

    try:
        import yaml
    except ImportError as e:
        print_actionable_error("study init requires pyyaml", cause=str(e), next_steps=["Install PyYAML"])
        return 1

    preset_name = getattr(args, "preset", None) or "quick"
    try:
        preset = get_study_preset(preset_name)
    except KeyError as e:
        print_actionable_error("unknown study preset", cause=str(e), next_steps=["Run: ollama-forge study presets"])
        return 1

    model_name = getattr(args, "model", None) or _study_default_model_for_tier(getattr(args, "tier", None) or "tiny")
    dataset_name = getattr(args, "dataset", None) or "wikitext"
    dataset_subset = getattr(args, "dataset_subset", None) or "wikitext-2-raw-v1"
    dataset_split = getattr(args, "dataset_split", None) or "test"
    output_dir = getattr(args, "output_dir", None) or "study-results"
    config = {
        "preset": preset.key,
        "model": {
            "name": model_name,
            "task": getattr(args, "task", None) or "causal_lm",
            "dtype": getattr(args, "dtype", None) or "float16",
            "device": getattr(args, "device", None) or "auto",
        },
        "dataset": {
            "name": dataset_name,
            "subset": dataset_subset,
            "split": dataset_split,
            "text_column": getattr(args, "text_column", None) or "text",
            "label_column": getattr(args, "label_column", None) or "label",
        },
        "output_dir": output_dir,
    }
    out_path = Path(getattr(args, "out", None) or "study.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def _cmd_study_interactive(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_model_presets import detect_hardware_tier, recommended_model_presets
    from ollama_forge.study_presets import list_study_presets

    non_interactive = getattr(args, "non_interactive", False) or not sys.stdin.isatty()
    detected_tier, _info = detect_hardware_tier()
    tier = getattr(args, "tier", None) or (
        detected_tier if non_interactive else _prompt_with_default("Hardware tier", detected_tier)
    )
    tier = tier.lower()

    recommended = recommended_model_presets(tier=tier, limit=3)
    default_model = getattr(args, "model", None) or (
        recommended[-1].hf_id if recommended else _study_default_model_for_tier(tier)
    )
    if non_interactive:
        model_name = default_model
    else:
        print("Recommended models:")
        for preset in recommended:
            print(f"  {preset.hf_id} [{preset.tier}] {preset.params}")
        model_name = _prompt_with_default("Model HF id", default_model)

    presets = list_study_presets()
    preset_keys = {preset.key for preset in presets}
    default_preset = getattr(args, "preset", None) or "quick"
    if non_interactive:
        preset_key = default_preset
    else:
        print("Available presets:")
        for preset in presets:
            print(f"  {preset.key}: {preset.name}")
        preset_key = _prompt_with_default("Study preset", default_preset)
    if preset_key not in preset_keys:
        print_actionable_error("unknown study preset", cause=preset_key, next_steps=["Run: ollama-forge study presets"])
        return 1

    dataset_name = getattr(args, "dataset", None) or (
        "wikitext" if non_interactive else _prompt_with_default("Dataset", "wikitext")
    )
    dataset_subset = getattr(args, "dataset_subset", None) or (
        "wikitext-2-raw-v1" if non_interactive else _prompt_with_default("Dataset subset", "wikitext-2-raw-v1")
    )
    dataset_split = getattr(args, "dataset_split", None) or (
        "test" if non_interactive else _prompt_with_default("Dataset split", "test")
    )
    output_dir = getattr(args, "output_dir", None) or (
        "study-results" if non_interactive else _prompt_with_default("Output dir", "study-results")
    )
    out_path = Path(getattr(args, "out", None) or "study.yaml")

    init_args = argparse.Namespace(
        preset=preset_key,
        model=model_name,
        tier=tier,
        dataset=dataset_name,
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
        output_dir=output_dir,
        task=getattr(args, "task", None) or "causal_lm",
        dtype=getattr(args, "dtype", None) or "float16",
        device=getattr(args, "device", None) or "auto",
        text_column=getattr(args, "text_column", None) or "text",
        label_column=getattr(args, "label_column", None) or "label",
        out=str(out_path),
    )
    rc = _cmd_study_init(parser, init_args)
    if rc != 0:
        return rc
    print(f"Plan: ollama-forge study plan {out_path}")
    if getattr(args, "run", False):
        run_args = argparse.Namespace(config=str(out_path), output_dir=output_dir, json=getattr(args, "json", False))
        return _cmd_study_run(parser, run_args)
    return 0


def _cmd_study_benchmarks(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_benchmarks import list_benchmark_presets

    presets = list_benchmark_presets(kind=getattr(args, "kind", None))
    if getattr(args, "json", False):
        print(json.dumps([preset.__dict__ for preset in presets], indent=2, sort_keys=True))
        return 0
    for preset in presets:
        print(f"{preset.key} [{preset.kind}]")
        print(f"  {preset.name}")
        print(f"  path: {preset.path}")
        print(f"  {preset.description}")
    return 0


def _cmd_study_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_config import StudyConfig
    from ollama_forge.study_optimize import optimize_study_strength

    try:
        from ollama_forge.study_runtime import StudyEvaluator, load_study_dataset, load_study_model
    except ImportError as e:
        print_actionable_error(
            "study optimize requires optional study dependencies",
            cause=str(e),
            next_steps=["Run: uv sync", "Then: ollama-forge study optimize <config>"],
        )
        return 1
    try:
        config = StudyConfig.from_yaml(args.config)
        strengths = [float(part) for part in (getattr(args, "strengths", None) or "0.25,0.5,0.75,1.0").split(",")]
        output_dir = Path(getattr(args, "output_dir", None) or (Path(config.output_dir) / "optimize"))
        result = optimize_study_strength(
            config,
            strengths=strengths,
            metric=getattr(args, "metric", None) or config.metrics[0],
            objective=getattr(args, "objective", None) or (
                "min" if (getattr(args, "metric", None) or config.metrics[0]) == "perplexity" else "max"
            ),
            model_loader=load_study_model,
            dataset_loader=load_study_dataset,
            evaluator_factory=StudyEvaluator,
            output_dir=output_dir,
        )
    except Exception as e:
        print_actionable_error(
            "study optimize failed",
            cause=str(e),
            next_steps=[
                "Check the config file and metric name",
                "Run: ollama-forge study validate <config>",
                "Ensure dependencies are installed: uv sync",
            ],
        )
        return 1
    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Best strength: {result.best_strength}")
        print(f"Best {result.metric}: {result.best_score}")
        print(f"Saved: {Path(output_dir) / 'study-optimize.json'}")
    return 0


def _cmd_study_benchmark_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_benchmarks import compare_benchmark_runs, get_benchmark_preset

    preset_key = getattr(args, "preset", None)
    if not preset_key:
        print_actionable_error("pass --preset <key>", next_steps=["Run: ollama-forge study benchmarks"])
        return 1
    try:
        preset = get_benchmark_preset(preset_key)
    except KeyError as e:
        print_actionable_error(
            "unknown benchmark preset", cause=str(e),
            next_steps=["Run: ollama-forge study benchmarks"],
        )
        return 1
    try:
        if preset.kind == "security_eval":
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(
                preset.path,
                base_url=getattr(args, "base_url", None) or "http://127.0.0.1:11434",
                model=getattr(args, "model", None) or "llama3.2",
                output_json=getattr(args, "output_json", None),
                output_csv=getattr(args, "output_csv", None),
                save_to_history=getattr(args, "save_history", False),
                max_prompts=getattr(args, "max_prompts", None),
                timeout=float(getattr(args, "timeout", 120.0)),
                verbose=not getattr(args, "quiet", False),
            )
            compare_model = getattr(args, "compare_model", None)
            compare_meta = None
            if compare_model:
                compare_meta = run_eval(
                    preset.path,
                    base_url=getattr(args, "compare_base_url", None) or getattr(args, "base_url", None) or "http://127.0.0.1:11434",
                    model=compare_model,
                    output_json=getattr(args, "compare_output_json", None),
                    output_csv=None,
                    save_to_history=getattr(args, "save_history", False),
                    max_prompts=getattr(args, "max_prompts", None),
                    timeout=float(getattr(args, "timeout", 120.0)),
                    verbose=not getattr(args, "quiet", False),
                )
            payload = {"primary": run_meta}
            if compare_meta:
                payload["secondary"] = compare_meta
                payload["comparison"] = compare_benchmark_runs(run_meta, compare_meta)
            if getattr(args, "json", False):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                kpis = run_meta.get("kpis") or {}
                print(f"preset: {preset.key}")
                print(f"model: {run_meta.get('model')}")
                print(f"ASR %: {kpis.get('asr_pct', 0):.1f}")
                print(f"Refusal %: {kpis.get('refusal_rate_pct', 0):.1f}")
                if compare_meta:
                    compare_kpis = compare_meta.get("kpis") or {}
                    print(f"compare_model: {compare_meta.get('model')}")
                    print(f"compare_ASR %: {compare_kpis.get('asr_pct', 0):.1f}")
                    print(f"compare_Refusal %: {compare_kpis.get('refusal_rate_pct', 0):.1f}")
            return 0

        # dataset preset: run baseline-only study evaluation
        from ollama_forge.study_config import StudyConfig
        from ollama_forge.study_runner import run_study
        from ollama_forge.study_runtime import StudyEvaluator, load_study_dataset, load_study_model

        dataset_name, dataset_subset, dataset_split = (preset.path.split(":", 2) + ["", ""])[:3]

        def _build_cfg(model_name: str, output_dir: str) -> StudyConfig:
            return StudyConfig.from_dict(
                {
                    "model": {
                        "name": model_name,
                        "task": "causal_lm",
                        "dtype": getattr(args, "dtype", None) or "float16",
                        "device": getattr(args, "device", None) or "auto",
                    },
                    "dataset": {
                        "name": dataset_name,
                        "subset": dataset_subset or None,
                        "split": dataset_split or "test",
                        "text_column": getattr(args, "text_column", None) or "text",
                    },
                    "strategies": [],
                    "metrics": [getattr(args, "metric", None) or "perplexity"],
                    "output_dir": output_dir,
                }
            )

        primary_output = getattr(args, "output_dir", None) or f"study-results/benchmark-{preset.key}"
        primary_report = run_study(
            _build_cfg(getattr(args, "model", None), primary_output),
            model_loader=load_study_model,
            dataset_loader=load_study_dataset,
            evaluator_factory=StudyEvaluator,
        )
        payload = {"primary": primary_report.to_dict()}
        compare_model = getattr(args, "compare_model", None)
        if compare_model:
            secondary_output = getattr(args, "compare_output_dir", None) or f"{primary_output}-compare"
            secondary_report = run_study(
                _build_cfg(compare_model, secondary_output),
                model_loader=load_study_model,
                dataset_loader=load_study_dataset,
                evaluator_factory=StudyEvaluator,
            )
            payload["secondary"] = secondary_report.to_dict()
            payload["comparison"] = {
                "metric": getattr(args, "metric", None) or "perplexity",
                "primary": primary_report.baseline_metrics,
                "secondary": secondary_report.baseline_metrics,
            }
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"preset: {preset.key}")
            print(f"model: {payload['primary']['model_name']}")
            print(f"baseline: {payload['primary']['baseline_metrics']}")
            if "secondary" in payload:
                print(f"compare_model: {payload['secondary']['model_name']}")
                print(f"compare_baseline: {payload['secondary']['baseline_metrics']}")
        return 0
    except Exception as e:
        print_actionable_error(
            "study benchmark-run failed",
            cause=str(e),
            next_steps=["Check the preset, model, and runtime dependencies"],
        )
        return 1


def _cmd_study_lm_eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_eval_integrations import build_lm_eval_command, run_lm_eval, save_lm_eval_plan

    tasks = [item.strip() for item in (getattr(args, "tasks", None) or "").split(",") if item.strip()]
    if not tasks:
        print_actionable_error("pass --tasks <task1,task2,...>", next_steps=["Example: --tasks hellaswag,arc_easy"])
        return 1
    command = build_lm_eval_command(
        model=getattr(args, "model", None) or "hf",
        tasks=tasks,
        model_args=getattr(args, "model_args", None) or "",
        output_path=getattr(args, "output_path", None),
        device=getattr(args, "device", None),
        batch_size=getattr(args, "batch_size", None),
        limit=getattr(args, "limit", None),
    )
    if getattr(args, "plan", False):
        plan_path = getattr(args, "plan_file", None)
        if plan_path:
            save_lm_eval_plan(command, plan_path)
        if getattr(args, "json", False):
            payload = {"command": command.command, "output_path": command.output_path}
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(" ".join(command.command))
        return 0
    try:
        rc = run_lm_eval(command)
    except FileNotFoundError as e:
        print_actionable_error(
            "lm_eval executable not found",
            cause=str(e),
            next_steps=["Install lm-evaluation-harness", "Or re-run with --plan to print the command only"],
        )
        return 1
    return rc


def _cmd_study_eval_compare(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.study_eval_reports import compare_eval_reports, load_eval_report

    path_a = Path(getattr(args, "report_a", ""))
    path_b = Path(getattr(args, "report_b", ""))
    if not path_a.is_file() or not path_b.is_file():
        missing = path_a if not path_a.is_file() else path_b
        print_actionable_error("eval report file not found", cause=str(missing))
        return 1
    try:
        report_a = load_eval_report(path_a)
        report_b = load_eval_report(path_b)
        payload = compare_eval_reports(report_a, report_b)
    except Exception as e:
        print_actionable_error(
            "study eval-compare failed",
            cause=str(e),
            next_steps=["Pass a security-eval JSON or lm-eval JSON report"],
        )
        return 1
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"A: {payload['source_a']} ({payload['kind_a']})")
        print(f"B: {payload['source_b']} ({payload['kind_b']})")
        for name, values in payload["metrics"].items():
            print(f"{name}: A={values.get('a')} B={values.get('b')}")
    return 0


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Register the ``study`` subcommand group and return its parser."""
    p_study = subparsers.add_parser(
        "study",
        help="Generic ablation study helpers: presets, strategies, validation, and planning",
    )
    study_sub = p_study.add_subparsers(dest="study_command")
    p_study_presets = study_sub.add_parser("presets", help="List available study presets")
    p_study_presets.add_argument("--json", action="store_true", help="Print presets as JSON")
    p_study_presets.set_defaults(handler=_cmd_study_presets)

    p_study_models = study_sub.add_parser("models", help="List curated model presets and hardware-tier recommendations")
    p_study_models.add_argument(
        "--tier",
        choices=("tiny", "small", "medium", "large", "frontier"),
        help="Filter models by hardware tier",
    )
    p_study_models.add_argument(
        "--recommend",
        action="store_true",
        help="Show a small recommended set based on detected hardware",
    )
    p_study_models.add_argument(
        "--limit",
        type=int,
        default=5,
        metavar="N",
        help="Max recommendations when using --recommend (default: 5)",
    )
    p_study_models.add_argument("--json", action="store_true", help="Print models as JSON")
    p_study_models.set_defaults(handler=_cmd_study_models)

    p_study_benchmarks = study_sub.add_parser("benchmarks", help="List curated benchmark presets")
    p_study_benchmarks.add_argument("--kind", choices=("dataset", "security_eval"), help="Filter by benchmark kind")
    p_study_benchmarks.add_argument("--json", action="store_true", help="Print benchmarks as JSON")
    p_study_benchmarks.set_defaults(handler=_cmd_study_benchmarks)

    p_study_benchmark_run = study_sub.add_parser("benchmark-run", help="Run a curated security benchmark preset")
    p_study_benchmark_run.add_argument("--preset", required=True, help="Benchmark preset key from `study benchmarks`")
    p_study_benchmark_run.add_argument("--model", required=True, help="Model name to query")
    p_study_benchmark_run.add_argument(
        "--base-url", default="http://127.0.0.1:11434", help="Base URL for the model API"
    )
    p_study_benchmark_run.add_argument("--compare-model", help="Optional comparison model")
    p_study_benchmark_run.add_argument("--compare-base-url", help="Optional comparison base URL")
    p_study_benchmark_run.add_argument("--compare-output-json", help="Optional comparison output JSON path")
    p_study_benchmark_run.add_argument(
        "--compare-output-dir", help="Optional comparison output dir for dataset presets"
    )
    p_study_benchmark_run.add_argument("--output-json", help="Optional output JSON path")
    p_study_benchmark_run.add_argument("--output-csv", help="Optional output CSV path")
    p_study_benchmark_run.add_argument("--max-prompts", type=int, help="Optional prompt limit")
    p_study_benchmark_run.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout")
    p_study_benchmark_run.add_argument("--save-history", action="store_true", help="Save to security-eval history")
    p_study_benchmark_run.add_argument("--quiet", action="store_true", help="Reduce per-prompt logging")
    p_study_benchmark_run.add_argument("--json", action="store_true", help="Print full run metadata as JSON")
    p_study_benchmark_run.set_defaults(handler=_cmd_study_benchmark_run)

    p_study_lm_eval = study_sub.add_parser("lm-eval", help="Plan or run lm-evaluation-harness against a model")
    p_study_lm_eval.add_argument("--model", default="hf", help="lm_eval model backend (default: hf)")
    p_study_lm_eval.add_argument("--tasks", required=True, help="Comma-separated task list")
    p_study_lm_eval.add_argument("--model-args", default="", help="lm_eval --model_args string")
    p_study_lm_eval.add_argument("--output-path", help="lm_eval output path")
    p_study_lm_eval.add_argument("--device", help="lm_eval device")
    p_study_lm_eval.add_argument("--batch-size", help="lm_eval batch size")
    p_study_lm_eval.add_argument("--limit", type=int, help="Optional sample limit")
    p_study_lm_eval.add_argument("--plan", action="store_true", help="Print the lm_eval command without executing")
    p_study_lm_eval.add_argument("--plan-file", help="Optional JSON file to save the command plan")
    p_study_lm_eval.add_argument("--json", action="store_true", help="Print the plan as JSON when used with --plan")
    p_study_lm_eval.set_defaults(handler=_cmd_study_lm_eval)

    p_study_eval_compare = study_sub.add_parser("eval-compare", help="Compare two external eval reports")
    p_study_eval_compare.add_argument("report_a", help="First eval report JSON")
    p_study_eval_compare.add_argument("report_b", help="Second eval report JSON")
    p_study_eval_compare.add_argument("--json", action="store_true", help="Print the comparison as JSON")
    p_study_eval_compare.set_defaults(handler=_cmd_study_eval_compare)

    p_study_modules = study_sub.add_parser("analysis-modules", help="List available baseline analysis modules")
    p_study_modules.add_argument("--json", action="store_true", help="Print module names as JSON")
    p_study_modules.set_defaults(handler=_cmd_study_analysis_modules)

    p_study_strategies = study_sub.add_parser("strategies", help="List built-in study strategies")
    p_study_strategies.add_argument("--json", action="store_true", help="Print strategy names as JSON")
    p_study_strategies.set_defaults(handler=_cmd_study_strategies)

    p_study_validate = study_sub.add_parser("validate", help="Validate a study config file")
    p_study_validate.add_argument("config", help="Path to a study YAML/JSON config")
    p_study_validate.add_argument("--json", action="store_true", help="Print normalized config as JSON")
    p_study_validate.set_defaults(handler=_cmd_study_validate)

    p_study_plan = study_sub.add_parser("plan", help="Expand a study config into an execution plan")
    p_study_plan.add_argument("config", help="Path to a study YAML/JSON config")
    p_study_plan.add_argument("--json", action="store_true", help="Print the plan as JSON")
    p_study_plan.set_defaults(handler=_cmd_study_plan)

    p_study_init = study_sub.add_parser("init", help="Write a starter study config")
    p_study_init.add_argument("--out", default="study.yaml", help="Output YAML path (default: study.yaml)")
    p_study_init.add_argument("--preset", default="quick", help="Study preset key (default: quick)")
    p_study_init.add_argument(
        "--tier", choices=("tiny", "small", "medium", "large", "frontier"), help="Hardware tier hint"
    )
    p_study_init.add_argument("--model", help="Model HF id")
    p_study_init.add_argument("--dataset", help="Dataset name or local file path")
    p_study_init.add_argument("--dataset-subset", help="Dataset subset/config name")
    p_study_init.add_argument("--dataset-split", default="test", help="Dataset split (default: test)")
    p_study_init.add_argument("--output-dir", default="study-results", help="Study output directory")
    p_study_init.add_argument("--task", default="causal_lm", help="Task type (default: causal_lm)")
    p_study_init.add_argument("--dtype", default="float16", help="Model dtype (default: float16)")
    p_study_init.add_argument("--device", default="auto", help="Model device (default: auto)")
    p_study_init.add_argument("--text-column", default="text", help="Dataset text column (default: text)")
    p_study_init.add_argument("--label-column", default="label", help="Dataset label column (default: label)")
    p_study_init.set_defaults(handler=_cmd_study_init)

    p_study_interactive = study_sub.add_parser("interactive", help="Guided study setup flow")
    p_study_interactive.add_argument("--out", default="study.yaml", help="Output YAML path (default: study.yaml)")
    p_study_interactive.add_argument("--preset", help="Default preset key")
    p_study_interactive.add_argument(
        "--tier", choices=("tiny", "small", "medium", "large", "frontier"), help="Default hardware tier"
    )
    p_study_interactive.add_argument("--model", help="Default model HF id")
    p_study_interactive.add_argument("--dataset", help="Default dataset name or path")
    p_study_interactive.add_argument("--dataset-subset", help="Default dataset subset/config name")
    p_study_interactive.add_argument("--dataset-split", help="Default dataset split")
    p_study_interactive.add_argument("--output-dir", help="Default output directory")
    p_study_interactive.add_argument("--task", default="causal_lm", help="Task type (default: causal_lm)")
    p_study_interactive.add_argument("--dtype", default="float16", help="Model dtype (default: float16)")
    p_study_interactive.add_argument("--device", default="auto", help="Model device (default: auto)")
    p_study_interactive.add_argument("--text-column", default="text", help="Dataset text column (default: text)")
    p_study_interactive.add_argument("--label-column", default="label", help="Dataset label column (default: label)")
    p_study_interactive.add_argument(
        "--non-interactive", action="store_true", help="Use detected/default values without prompts"
    )
    p_study_interactive.add_argument("--run", action="store_true", help="Run the generated config immediately")
    p_study_interactive.add_argument(
        "--json", action="store_true", help="When used with --run, print the report as JSON"
    )
    p_study_interactive.set_defaults(handler=_cmd_study_interactive)

    p_study_ui = study_sub.add_parser("ui", help="Launch the local Streamlit UI for study workflows")
    p_study_ui.set_defaults(handler=_cmd_study_ui)

    p_study_run = study_sub.add_parser("run", help="Execute a study config against a transformer model")
    p_study_run.add_argument("config", help="Path to a study YAML/JSON config")
    p_study_run.add_argument("--output-dir", help="Override the config output directory")
    p_study_run.add_argument("--json", action="store_true", help="Print the final report as JSON")
    p_study_run.set_defaults(handler=_cmd_study_run)

    p_study_optimize = study_sub.add_parser("optimize", help="Grid-search intervention strength for a study config")
    p_study_optimize.add_argument("config", help="Path to a study YAML/JSON config")
    p_study_optimize.add_argument("--metric", help="Metric to optimize (default: first metric in config)")
    p_study_optimize.add_argument("--objective", choices=("min", "max"), help="Optimization direction")
    p_study_optimize.add_argument(
        "--strengths",
        help="Comma-separated strength values to try (default: 0.25,0.5,0.75,1.0)",
    )
    p_study_optimize.add_argument("--output-dir", help="Override output directory for optimization artifacts")
    p_study_optimize.add_argument("--json", action="store_true", help="Print the optimization report as JSON")
    p_study_optimize.set_defaults(handler=_cmd_study_optimize)

    p_study_analyze = study_sub.add_parser("analyze", help="Run a baseline analysis module from a study config")
    p_study_analyze.add_argument("config", help="Path to a study YAML/JSON config")
    p_study_analyze.add_argument(
        "--module",
        required=True,
        choices=(
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
        ),
        help="Analysis module to run",
    )
    p_study_analyze.add_argument("--output-dir", help="Override the output directory")
    p_study_analyze.add_argument("--output-file", help="Explicit JSON output path")
    p_study_analyze.add_argument("--max-samples", type=int, help="Override max samples for analysis")
    p_study_analyze.add_argument("--batch-size", type=int, help="Override batch size for analysis")
    p_study_analyze.add_argument("--max-length", type=int, help="Override tokenization max length for analysis")
    p_study_analyze.add_argument("--top-k", type=int, help="Top-K tokens for logit_lens (default: 5)")
    p_study_analyze.add_argument("--prompt", help="Prompt text for causal_tracing (defaults to first dataset row)")
    p_study_analyze.add_argument("--source-prompt", help="Source prompt for causal_patching")
    p_study_analyze.add_argument("--target-prompt", help="Target prompt for causal_patching")
    p_study_analyze.add_argument("--group-column", help="Grouping column for conditional_similarity")
    p_study_analyze.add_argument("--source-group", help="Source group label for activation_patching")
    p_study_analyze.add_argument("--target-group", help="Target group label for activation_patching")
    p_study_analyze.add_argument("--json", action="store_true", help="Print the analysis result as JSON")
    p_study_analyze.set_defaults(handler=_cmd_study_analyze)

    p_study_analyze_bundle = study_sub.add_parser(
        "analyze-bundle", help="Run multiple analysis modules and save one bundle"
    )
    p_study_analyze_bundle.add_argument("config", help="Path to a study YAML/JSON config")
    p_study_analyze_bundle.add_argument("--modules", help="Comma-separated module list (default: all)")
    p_study_analyze_bundle.add_argument("--output-file", help="Explicit output bundle JSON path")
    p_study_analyze_bundle.add_argument("--max-samples", type=int, help="Override max samples for analysis")
    p_study_analyze_bundle.add_argument("--batch-size", type=int, help="Override batch size for analysis")
    p_study_analyze_bundle.add_argument("--max-length", type=int, help="Override tokenization max length for analysis")
    p_study_analyze_bundle.add_argument("--top-k", type=int, help="Top-K tokens for logit_lens (default: 5)")
    p_study_analyze_bundle.add_argument("--prompt", help="Prompt text for causal_tracing")
    p_study_analyze_bundle.add_argument("--source-prompt", help="Source prompt for causal_patching")
    p_study_analyze_bundle.add_argument("--target-prompt", help="Target prompt for causal_patching")
    p_study_analyze_bundle.add_argument("--group-column", help="Grouping column for grouped modules")
    p_study_analyze_bundle.add_argument("--source-group", help="Source group label for activation_patching")
    p_study_analyze_bundle.add_argument("--target-group", help="Target group label for activation_patching")
    p_study_analyze_bundle.add_argument("--json", action="store_true", help="Print the bundle as JSON")
    p_study_analyze_bundle.set_defaults(handler=_cmd_study_analyze_bundle)

    p_study_report = study_sub.add_parser("report", help="Show a saved study-results.json report")
    p_study_report.add_argument("path", help="Path to study-results.json")
    p_study_report.add_argument("--export", help="Optional export path (.md, .html, .json, or .csv)")
    p_study_report.add_argument("--json", action="store_true", help="Print the full report as JSON")
    p_study_report.set_defaults(handler=_cmd_study_report)

    p_study_regenerate = study_sub.add_parser(
        "regenerate-report",
        help="Regenerate study exports (json/csv/summary/md/html/plot) from a study-results.json file",
    )
    p_study_regenerate.add_argument("path", help="Path to study-results.json")
    p_study_regenerate.add_argument("--output-dir", help="Directory to write regenerated exports")
    p_study_regenerate.set_defaults(handler=_cmd_study_regenerate_report)

    p_study_compare = study_sub.add_parser("compare", help="Compare two study-results.json reports")
    p_study_compare.add_argument("report_a", help="First study-results.json path")
    p_study_compare.add_argument("report_b", help="Second study-results.json path")
    p_study_compare.add_argument("--export", help="Optional export path (.md, .html, .json, or .csv)")
    p_study_compare.add_argument("--json", action="store_true", help="Print the comparison as JSON")
    p_study_compare.set_defaults(handler=_cmd_study_compare)

    p_study_contribute = study_sub.add_parser(
        "contribute", help="Save a study report into the local contribution store"
    )
    p_study_contribute.add_argument("report", help="Path to study-results.json")
    p_study_contribute.add_argument("--dir", default="study_results_community", help="Contribution directory")
    p_study_contribute.add_argument("--notes", default="", help="Optional notes to include with the contribution")
    p_study_contribute.set_defaults(handler=_cmd_study_contribute)

    p_study_aggregate = study_sub.add_parser("aggregate", help="Aggregate local study contributions")
    p_study_aggregate.add_argument("--dir", default="study_results_community", help="Contribution directory")
    p_study_aggregate.add_argument("--json", action="store_true", help="Print aggregation as JSON")
    p_study_aggregate.set_defaults(handler=_cmd_study_aggregate)

    return p_study
