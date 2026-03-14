"""Streamlit UI for study workflows.

Run with:
  streamlit run ollama_forge.study_app
Or:
  uv run ollama-forge study ui
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

try:
    from ollama_forge.study_analysis import available_analysis_modules, save_analysis_result
    from ollama_forge.abliterate_pipeline import compare_pipeline_results, load_informed_pipeline_result
    from ollama_forge.study_config import StudyConfig
    from ollama_forge.study_reports import compare_study_reports
    from ollama_forge.study_model_presets import detect_hardware_tier, recommended_model_presets
    from ollama_forge.study_presets import list_study_presets
    from ollama_forge.study_reports import load_study_report
    from ollama_forge.study_runner import plan_study, run_study
    from ollama_forge.study_runtime import (
        StudyEvaluator,
        load_study_dataset,
        load_study_model,
    )
    from ollama_forge.study_analysis import (
        analyze_activation_probe,
        analyze_activation_patching,
        analyze_conditional_similarity,
        analyze_cross_layer_similarity,
        analyze_logit_lens,
        analyze_residual_stream,
        collect_grouped_layer_vectors,
        collect_layer_vectors,
        trace_causal_layers,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ollama_forge.study_analysis import available_analysis_modules, save_analysis_result
    from ollama_forge.abliterate_pipeline import compare_pipeline_results, load_informed_pipeline_result
    from ollama_forge.study_config import StudyConfig
    from ollama_forge.study_reports import compare_study_reports
    from ollama_forge.study_model_presets import detect_hardware_tier, recommended_model_presets
    from ollama_forge.study_presets import list_study_presets
    from ollama_forge.study_reports import load_study_report
    from ollama_forge.study_runner import plan_study, run_study
    from ollama_forge.study_runtime import (
        StudyEvaluator,
        load_study_dataset,
        load_study_model,
    )
    from ollama_forge.study_analysis import (
        analyze_activation_probe,
        analyze_activation_patching,
        analyze_conditional_similarity,
        analyze_cross_layer_similarity,
        analyze_logit_lens,
        analyze_residual_stream,
        collect_grouped_layer_vectors,
        collect_layer_vectors,
        trace_causal_layers,
    )


def main() -> None:
    st.set_page_config(page_title="Study UI", layout="wide")
    st.title("Transformer Study UI")
    st.caption("Build, run, analyze, and compare study configs without leaving the local workspace.")
    tab_build, tab_run, tab_reports = st.tabs(["Config Builder", "Run & Analyze", "Reports"])
    with tab_build:
        _render_build_tab(st)
    with tab_run:
        _render_run_tab(st)
    with tab_reports:
        _render_reports_tab(st)


def _render_build_tab(st_module) -> None:
    tier, _info = detect_hardware_tier()
    presets = list_study_presets()
    preset_labels = {preset.name: preset for preset in presets}
    rec_models = recommended_model_presets(tier=tier, limit=5)

    col1, col2 = st_module.columns(2)
    with col1:
        preset_name = st_module.selectbox("Preset", options=[preset.name for preset in presets], index=0)
        model_name = st_module.selectbox(
            "Recommended model",
            options=[preset.hf_id for preset in rec_models],
            index=max(len(rec_models) - 1, 0) if rec_models else 0,
        ) if rec_models else st_module.text_input("Model HF id", value="distilgpt2")
        custom_model = st_module.text_input("Or custom model HF id", value="")
    with col2:
        dataset_name = st_module.text_input("Dataset name or path", value="wikitext")
        dataset_subset = st_module.text_input("Dataset subset", value="wikitext-2-raw-v1")
        dataset_split = st_module.text_input("Dataset split", value="test")
        output_dir = st_module.text_input("Output dir", value="study-results/ui")

    chosen_model = custom_model.strip() or model_name
    preset = preset_labels[preset_name]
    config = {
        "preset": preset.key,
        "model": {"name": chosen_model, "task": "causal_lm", "dtype": "float16", "device": "auto"},
        "dataset": {
            "name": dataset_name,
            "subset": dataset_subset or None,
            "split": dataset_split,
            "text_column": "text",
            "label_column": "label",
        },
        "output_dir": output_dir,
    }
    st_module.code(json.dumps(config, indent=2), language="json")
    if st_module.button("Save config"):
        out_path = Path(output_dir) / "study-ui-config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        st_module.success(f"Saved {out_path}")


def _render_run_tab(st_module) -> None:
    config_path = st_module.text_input("Study config path", value="study.yaml")
    action = st_module.radio("Action", options=["plan", "run", "analyze"], horizontal=True)
    if action == "analyze":
        module_name = st_module.selectbox("Analysis module", options=list(available_analysis_modules()))
        group_column = st_module.text_input("Group column", value="label")
        source_group = st_module.text_input("Source group", value="safe")
        target_group = st_module.text_input("Target group", value="harmful")
        trace_prompt = st_module.text_input("Trace prompt", value="")
    else:
        module_name = None
        group_column = source_group = target_group = trace_prompt = None
    if st_module.button("Execute"):
        path = Path(config_path)
        if not path.exists():
            st_module.error(f"Config not found: {path}")
            return
        try:
            config = StudyConfig.from_yaml(path)
            if action == "plan":
                st_module.json(plan_study(config).to_dict())
                return
            if action == "run":
                with st_module.spinner("Running study..."):
                    report = run_study(
                        config,
                        model_loader=load_study_model,
                        dataset_loader=load_study_dataset,
                        evaluator_factory=StudyEvaluator,
                    )
                st_module.success("Study completed")
                st_module.json(report.to_dict())
                return
            handle = load_study_model(config.model)
            dataset = load_study_dataset(config.dataset)
            vectors = collect_layer_vectors(
                handle,
                dataset,
                text_column=config.dataset.text_column,
                max_samples=config.dataset.max_samples,
                batch_size=config.batch_size,
                max_length=config.max_length,
            )
            if module_name == "causal_tracing":
                prompt = trace_prompt.strip() or "Hello"
                result = trace_causal_layers(handle, prompt)
            elif module_name in {"conditional_similarity", "activation_patching"}:
                grouped = collect_grouped_layer_vectors(
                    handle,
                    dataset,
                    group_column=group_column or config.dataset.label_column,
                    text_column=config.dataset.text_column,
                    max_samples=config.dataset.max_samples,
                    batch_size=config.batch_size,
                    max_length=config.max_length,
                )
                if module_name == "conditional_similarity":
                    result = analyze_conditional_similarity(grouped)
                else:
                    result = analyze_activation_patching(
                        grouped,
                        source_group=source_group or "safe",
                        target_group=target_group or "harmful",
                    )
            else:
                vectors = collect_layer_vectors(
                    handle,
                    dataset,
                    text_column=config.dataset.text_column,
                    max_samples=config.dataset.max_samples,
                    batch_size=config.batch_size,
                    max_length=config.max_length,
                )
                if module_name == "activation_probe":
                    result = analyze_activation_probe(vectors)
                elif module_name == "cross_layer_similarity":
                    result = analyze_cross_layer_similarity(vectors)
                elif module_name == "logit_lens":
                    result = analyze_logit_lens(handle, vectors, top_k=5)
                else:
                    result = analyze_residual_stream(vectors)
            output_path = Path(config.output_dir) / f"{module_name}.json"
            save_analysis_result(result, output_path)
            st_module.success(f"Saved {output_path}")
            st_module.json(asdict(result))
        except Exception as exc:
            st_module.exception(exc)


def _render_reports_tab(st_module) -> None:
    mode = st_module.radio("Mode", options=["Study report", "Study compare", "Pipeline report", "Pipeline compare"], horizontal=True)
    if mode == "Study report":
        report_a = st_module.text_input("Study report path", value="study-results/study-results.json")
        if st_module.button("Load study report"):
            path = Path(report_a)
            if not path.exists():
                st_module.error(f"Report not found: {path}")
                return
            try:
                report = load_study_report(path)
                st_module.json(report.to_dict())
            except Exception as exc:
                st_module.exception(exc)
        return

    if mode == "Study compare":
        report_a = st_module.text_input("Study report A", value="study-results/study-results.json", key="study_compare_a")
        report_b = st_module.text_input("Study report B", value="study-results-2/study-results.json", key="study_compare_b")
        if st_module.button("Compare study reports"):
            try:
                payload = compare_study_reports(load_study_report(report_a), load_study_report(report_b))
                st_module.json(payload)
            except Exception as exc:
                st_module.exception(exc)
        return

    if mode == "Pipeline report":
        path_text = st_module.text_input("Pipeline report path", value="abliterate-demo/informed-pipeline.json")
        if st_module.button("Load pipeline report"):
            try:
                payload = load_informed_pipeline_result(path_text).to_dict()
                st_module.json(payload)
            except Exception as exc:
                st_module.exception(exc)
        return

    report_a = st_module.text_input("Pipeline A", value="abliterate-demo/informed-pipeline.json", key="pipe_compare_a")
    report_b = st_module.text_input("Pipeline B", value="abliterate-demo-2/informed-pipeline.json", key="pipe_compare_b")
    if st_module.button("Compare pipelines"):
        try:
            payload = compare_pipeline_results(
                load_informed_pipeline_result(report_a),
                load_informed_pipeline_result(report_b),
            )
            st_module.json(payload)
        except Exception as exc:
            st_module.exception(exc)


if __name__ == "__main__":
    main()
