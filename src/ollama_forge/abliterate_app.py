"""Streamlit UI for abliterate and informed pipeline workflows.

Run with:
  streamlit run ollama_forge.abliterate_app
Or:
  uv run ollama-forge abliterate ui
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

try:
    from ollama_forge.abliterate_informed_reports import compare_informed_artifacts, load_informed_artifact
    from ollama_forge.abliterate_pipeline import compare_pipeline_results, load_informed_pipeline_result
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ollama_forge.abliterate_informed_reports import compare_informed_artifacts, load_informed_artifact
    from ollama_forge.abliterate_pipeline import compare_pipeline_results, load_informed_pipeline_result


def main() -> None:
    st.set_page_config(page_title="Abliterate UI", layout="wide")
    st.title("Abliterate UI")
    st.caption("Inspect informed artifacts and pipeline outputs, and scaffold informed runs.")
    tab_informed, tab_pipeline = st.tabs(["Informed Artifacts", "Pipelines"])
    with tab_informed:
        _render_informed_tab(st)
    with tab_pipeline:
        _render_pipeline_tab(st)


def _render_informed_tab(st_module) -> None:
    mode = st_module.radio("Mode", options=["Single", "Compare"], horizontal=True, key="informed_mode")
    if mode == "Single":
        path_text = st_module.text_input("Informed artifact path", value="abliterate-demo/informed-run.json")
        if st_module.button("Load informed artifact"):
            try:
                payload = load_informed_artifact(path_text)
                st_module.json(payload)
            except Exception as exc:
                st_module.exception(exc)
        return

    a = st_module.text_input("Artifact A", value="abliterate-demo/informed-run.json", key="artifact_a")
    b = st_module.text_input("Artifact B", value="abliterate-demo-2/informed-run.json", key="artifact_b")
    if st_module.button("Compare informed artifacts"):
        try:
            payload = compare_informed_artifacts(load_informed_artifact(a), load_informed_artifact(b))
            st_module.json(payload)
        except Exception as exc:
            st_module.exception(exc)


def _render_pipeline_tab(st_module) -> None:
    mode = st_module.radio("Pipeline mode", options=["Single", "Compare"], horizontal=True, key="pipe_mode")
    if mode == "Single":
        path_text = st_module.text_input("Pipeline path", value="abliterate-demo/informed-pipeline.json")
        if st_module.button("Load pipeline"):
            try:
                payload = load_informed_pipeline_result(path_text).to_dict()
                st_module.json(payload)
            except Exception as exc:
                st_module.exception(exc)
        return

    a = st_module.text_input("Pipeline A", value="abliterate-demo/informed-pipeline.json", key="pipe_a")
    b = st_module.text_input("Pipeline B", value="abliterate-demo-2/informed-pipeline.json", key="pipe_b")
    if st_module.button("Compare pipelines"):
        try:
            payload = compare_pipeline_results(
                load_informed_pipeline_result(a),
                load_informed_pipeline_result(b),
            )
            st_module.json(payload)
        except Exception as exc:
            st_module.exception(exc)


if __name__ == "__main__":
    main()
