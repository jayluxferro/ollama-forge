"""
Streamlit UI for LLM security evaluation.
Run with: streamlit run ollama_forge.security_eval.app
Or: uv run ollama-forge security-eval ui  (after `uv sync`)
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

try:
    from ollama_forge.security_eval.client import (
        list_models,
        query_model,
        query_model_with_image,
    )
    from ollama_forge.security_eval.history import load_runs
    from ollama_forge.security_eval.run import run_eval
    from ollama_forge.security_eval.scorers import score_extraction, score_refusal
except ImportError:
    # When run as streamlit app from repo root without install
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ollama_forge.security_eval.client import (
        list_models,
        query_model,
        query_model_with_image,
    )
    from ollama_forge.security_eval.history import load_runs
    from ollama_forge.security_eval.run import run_eval
    from ollama_forge.security_eval.scorers import score_extraction, score_refusal


class _EvalAborted(Exception):
    """Raised when user clicks the Stop button during evaluation."""
    pass


def main() -> None:
    st.set_page_config(page_title="LLM Security Eval", layout="wide")
    st.title("LLM Security Evaluation")
    st.markdown("Run prompt sets against Ollama or abliterate serve, view KPIs and per-category results.")

    tab_quick, tab_run, tab_compare, tab_history = st.tabs(
        ["Quick test", "Run evaluation", "Compare runs", "Run history"],
    )

    with tab_quick:
        _render_quick_test_tab(st)
    with tab_run:
        _render_run_tab(st)
    with tab_compare:
        _render_compare_runs_tab(st)
    with tab_history:
        _render_history_tab(st)


def _render_quick_test_tab(st) -> None:
    """Single-prompt test with optional image — instant feedback loop."""
    import base64

    st.subheader("Quick test")
    st.caption("Send one prompt, see the raw response and refusal/extraction scoring. Attach an image for VLM testing.")

    col_conn, col_model = st.columns(2)
    with col_conn:
        qt_url = st.text_input(
            "Base URL",
            value="http://127.0.0.1:11434",
            key="qt_url",
            help="Ollama, abliterate serve, or any OpenAI-compatible endpoint",
        )
    with col_model:
        qt_model = st.text_input("Model", value="llama3.2", key="qt_model")

    qt_system = st.text_input("System prompt (optional)", value="", key="qt_system")
    qt_prompt = st.text_area(
        "Prompt",
        height=120,
        placeholder="Type a prompt to test...",
        key="qt_prompt",
    )

    # Image upload
    qt_image_file = st.file_uploader(
        "Attach image (optional — for vision models)",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        key="qt_image",
    )
    qt_image_b64: str | None = None
    if qt_image_file is not None:
        raw_bytes = qt_image_file.read()
        qt_image_b64 = base64.b64encode(raw_bytes).decode("ascii")
        st.image(raw_bytes, caption=qt_image_file.name, width=300)

    # Extraction target (optional)
    qt_target = st.text_input(
        "Extraction target (optional — checks if this string leaks in the response)",
        value="",
        key="qt_target",
        help="e.g. a secret system prompt phrase you expect the model to NOT reveal",
    )

    if st.button("Send", key="qt_send", type="primary"):
        if not qt_prompt.strip():
            st.warning("Enter a prompt first.")
            return
        base = (qt_url.strip() or "http://127.0.0.1:11434").rstrip("/")
        if not base.startswith("http"):
            base = "http://" + base

        with st.spinner("Querying model..."):
            try:
                if qt_image_b64:
                    response_text, duration = query_model_with_image(
                        qt_prompt.strip(),
                        qt_image_b64,
                        base_url=base,
                        model=qt_model.strip(),
                        system=qt_system.strip() or None,
                        timeout=120.0,
                    )
                else:
                    response_text, duration = query_model(
                        qt_prompt.strip(),
                        base_url=base,
                        model=qt_model.strip(),
                        system=qt_system.strip() or None,
                        timeout=120.0,
                    )
            except Exception as e:
                st.error(f"Request failed: {e}")
                return

        # Scoring
        refusal = score_refusal(response_text)
        extraction = (
            score_extraction(response_text, target=qt_target.strip())
            if qt_target.strip()
            else None
        )

        # Display results
        st.divider()

        # Score badges
        col_ref, col_ext, col_lat = st.columns(3)
        with col_ref:
            if refusal:
                st.success("REFUSAL — model declined")
            else:
                st.error("COMPLIANCE — model complied")
        with col_ext:
            if extraction is None:
                st.info("No extraction target set")
            elif extraction:
                st.error("EXTRACTED — target string found in response")
            else:
                st.success("SAFE — target string not leaked")
        with col_lat:
            if duration is not None:
                st.metric("Latency", f"{duration:.2f}s")
            else:
                st.metric("Latency", "—")

        # Raw response
        st.subheader("Response")
        st.text_area(
            "Model output",
            value=response_text,
            height=250,
            disabled=True,
            key="qt_response_display",
        )

        # Token-level detail
        word_count = len(response_text.split())
        st.caption(f"{word_count} words  ·  {len(response_text)} chars")


def _render_run_tab(st) -> None:
    base_url = st.text_input(
        "Base URL (Ollama or abliterate serve)",
        value="http://127.0.0.1:11434",
        help="e.g. http://127.0.0.1:11434 or http://127.0.0.1:11435 for abliterate serve",
    )
    base_url_normalized = (base_url.strip() or "http://127.0.0.1:11434").rstrip("/")
    if not base_url_normalized.startswith("http"):
        base_url_normalized = "http://" + base_url_normalized

    if "available_models" not in st.session_state:
        st.session_state["available_models"] = []
    if "models_base_url" not in st.session_state:
        st.session_state["models_base_url"] = None

    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        if st.button("Refresh models"):
            with st.spinner("Fetching models..."):
                st.session_state["available_models"] = list_models(base_url_normalized, timeout=5.0)
                st.session_state["models_base_url"] = base_url_normalized
            if st.session_state["available_models"]:
                st.success(f"Found {len(st.session_state['available_models'])} model(s).")
            else:
                st.warning("No models found or server unreachable. Use custom name below.")

    models = st.session_state["available_models"] if st.session_state["models_base_url"] == base_url_normalized else []
    run_multi = st.checkbox("Compare multiple models (run same set on each)", value=False, key="run_multi")
    if models:
        options = [""] + models + ["Custom..."]
        if run_multi:
            model_options = [m for m in options if m and m != "(Select model)"]
            selected_models = st.multiselect(
                "Models to compare",
                options=model_options,
                default=model_options[:1] if model_options else [],
                help="Select one or more; 'Run all' will run eval for each.",
            )
            model = selected_models[0] if selected_models else (model_options[0] if model_options else "llama3.2")
        else:
            selected = st.selectbox(
                "Model",
                options=options,
                format_func=lambda x: "(Select model)" if x == "" else x,
                help="Choose from server or select Custom... to type a name",
            )
            if selected == "Custom..." or selected == "":
                model = st.text_input("Custom model name", value="llama3.2", key="model_custom")
            else:
                model = selected
            selected_models = [model]
    else:
        model = st.text_input("Model name", value="llama3.2")
        selected_models = [model]
        run_multi = False
    _data_dir = Path(__file__).resolve().parent / "data"
    bundled_sets = [
        ("(Custom path below)", None),
        ("sample_prompts.txt", _data_dir / "sample_prompts.txt"),
        ("sample_prompts.jsonl", _data_dir / "sample_prompts.jsonl"),
        ("sample_indirect.jsonl", _data_dir / "sample_indirect.jsonl"),
        ("system_prompt_extraction.jsonl", _data_dir / "system_prompt_extraction.jsonl"),
    ]
    bundled_labels = [x[0] for x in bundled_sets]
    bundled_paths = {x[0]: x[1] for x in bundled_sets}
    prompt_set_choice = st.selectbox(
        "Prompt set",
        options=bundled_labels,
        help="Bundled sample sets or use custom path below",
    )
    prompt_set_path = st.text_input(
        "Custom prompt set path (if not using a bundled set)",
        value="",
        placeholder="/path/to/prompts.txt or .jsonl",
        help=".txt: one prompt per line; .jsonl: prompt, category, target_for_extraction",
    )
    # --- #6: Drag-and-drop prompt set upload ---
    uploaded_prompt_file = st.file_uploader(
        "Or upload a prompt set",
        type=["txt", "jsonl"],
        key="upload_prompt_set",
    )
    chosen_path = bundled_paths.get(prompt_set_choice)
    if uploaded_prompt_file is not None:
        # Write uploaded file to a temp location so the eval runner can read it
        suffix = "." + (uploaded_prompt_file.name.rsplit(".", 1)[-1] if "." in uploaded_prompt_file.name else "txt")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_prompt_file.getvalue())
            effective_prompt_set_path = tmp.name
        st.caption(f"Using uploaded file: {uploaded_prompt_file.name}")
    elif chosen_path is not None and chosen_path.exists():
        effective_prompt_set_path = str(chosen_path)
    else:
        effective_prompt_set_path = prompt_set_path.strip()
    system_prompt = st.text_area("System prompt (optional)", value="", height=80)

    col_timeout, col_retries, _ = st.columns(3)
    with col_timeout:
        timeout_sec = st.number_input("Request timeout (s)", min_value=5, max_value=600, value=120, step=5)
    with col_retries:
        retries = st.number_input("Retries per prompt", min_value=0, max_value=10, value=2, step=1)

    col1, col2 = st.columns(2)
    with col1:
        output_csv = st.text_input("Output CSV path (optional)", value="")
    with col2:
        output_json = st.text_input("Output JSON path (optional)", value="")
    save_history = st.checkbox("Save run to history (for plots over time)", value=True)

    with st.sidebar:
        st.caption("Theme: Use the app menu (⋮) → Settings → Theme to switch to dark mode.")

    if st.button("Run evaluation"):
        if not effective_prompt_set_path or not Path(effective_prompt_set_path).exists():
            st.error("Please select a bundled prompt set or provide a valid custom path.")
            return
        try:
            from ollama_forge.security_eval.loader import load_prompt_set

            preview = load_prompt_set(effective_prompt_set_path)
            cats = ", ".join(sorted({p.get("category", "default") for p in preview}))
            st.info(f"Loaded {len(preview)} prompts. Categories: {cats}.")
        except Exception as e:
            st.error(f"Prompt set validation failed: {e}")
            return
        base = base_url.strip() or "http://127.0.0.1:11434"
        models_to_run = selected_models if run_multi and len(selected_models) > 0 else [model.strip() or "llama3.2"]
        multi_metas: list[dict] = []

        # --- #5: Abort button ---
        st.session_state["eval_abort"] = False
        stop_container = st.empty()
        stop_container.button(
            "Stop evaluation",
            key="eval_stop_btn",
            on_click=lambda: st.session_state.update({"eval_abort": True}),
            type="secondary",
        )

        def progress_cb(current: int, total: int, results_so_far: list) -> None:
            # Check abort flag
            if st.session_state.get("eval_abort"):
                raise _EvalAborted("Evaluation stopped by user.")
            progress_bar.progress(current / total if total else 0, text=f"Running prompt {current}/{total}...")
            if results_so_far:
                import pandas as pd

                df = pd.DataFrame(results_so_far)
                cols = [
                    c for c in ["index", "category", "refusal", "compliance", "extraction", "error"] if c in df.columns
                ]  # noqa: E501
                if cols:
                    results_placeholder.dataframe(df[cols], use_container_width=True)

        progress_bar = st.progress(0.0, text="Running evaluation...")
        results_placeholder = st.empty()
        aborted = False
        try:
            for mi, m in enumerate(models_to_run):
                if run_multi and len(models_to_run) > 1:
                    progress_bar.progress(
                        (mi + 0.5) / len(models_to_run), text=f"Model {mi + 1}/{len(models_to_run)}: {m}..."
                    )  # noqa: E501
                run_meta = run_eval(
                    effective_prompt_set_path,
                    base_url=base,
                    model=m,
                    output_csv=output_csv.strip() or None if not run_multi else None,
                    output_json=output_json.strip() or None if not run_multi else None,
                    save_to_history=save_history,
                    system=system_prompt.strip() or None,
                    verbose=False,
                    timeout=float(timeout_sec),
                    retries=int(retries),
                    progress_callback=progress_cb if len(models_to_run) == 1 else None,
                )
                multi_metas.append(run_meta)
            if save_history:
                st.caption("Run saved to history.")
        except _EvalAborted:
            aborted = True
            st.warning("Evaluation was stopped by user. Showing partial results.")
        except Exception as e:
            st.exception(e)
            return
        finally:
            progress_bar.empty()
            results_placeholder.empty()
            stop_container.empty()
        if multi_metas:
            st.session_state["last_run_meta"] = multi_metas[-1]
        elif aborted:
            # No complete run_meta available; clear previous results
            st.session_state.pop("last_run_meta", None)
        else:
            st.session_state["last_run_meta"] = {}
        st.session_state["multi_run_metas"] = multi_metas if run_multi and len(multi_metas) > 1 else []

    if "last_run_meta" in st.session_state:
        run_meta = st.session_state["last_run_meta"]
        kpis = run_meta.get("kpis") or {}
        results = run_meta.get("results") or []

        st.divider()
        st.subheader("Results")
        st.success("Evaluation complete.")
        multi_metas = st.session_state.get("multi_run_metas") or []
        if len(multi_metas) > 1:
            import pandas as pd

            comp = [
                {
                    "model": m.get("model", ""),
                    "ASR %": (m.get("kpis") or {}).get("asr_pct"),
                    "Refusal %": (m.get("kpis") or {}).get("refusal_rate_pct"),
                    "Extraction %": (m.get("kpis") or {}).get("extraction_rate_pct"),
                    "Tool misuse %": (m.get("kpis") or {}).get("tool_misuse_rate_pct"),
                    "Errors": (m.get("kpis") or {}).get("errors"),
                    "Avg turns to success": (m.get("kpis") or {}).get("avg_turns_to_success"),
                }
                for m in multi_metas
            ]
            st.subheader("Compare runs")
            st.dataframe(pd.DataFrame(comp), use_container_width=True)
            try:
                import plotly.express as px

                df_c = pd.DataFrame(comp)
                if not df_c.empty and "ASR %" in df_c.columns:
                    fig = px.bar(df_c, x="model", y="ASR %", title="ASR % by model")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        st.subheader("KPIs")

        # --- #2c: KPI deltas from history ---
        prev_kpis: dict = {}
        try:
            recent_runs = load_runs(limit=2)
            if len(recent_runs) >= 2:
                prev_kpis = recent_runs[1].get("kpis") or {}
        except Exception:
            pass

        asr = kpis.get("asr_pct", 0)
        refusal_pct = kpis.get("refusal_rate_pct", 0)
        extraction_pct = kpis.get("extraction_rate_pct", 0)
        total = kpis.get("total", 0)
        errors = kpis.get("errors", 0)

        c1, c2, c3, c4, c5 = st.columns(5)
        if prev_kpis:
            prev_total = prev_kpis.get("total", 0)
            prev_asr = prev_kpis.get("asr_pct", 0)
            prev_refusal = prev_kpis.get("refusal_rate_pct", 0)
            prev_extraction = prev_kpis.get("extraction_rate_pct", 0)
            prev_errors = prev_kpis.get("errors", 0)
            c1.metric("Total", total, delta=f"{total - prev_total:+d}" if total != prev_total else None)
            c2.metric("ASR %", f"{asr:.1f}", delta=f"{asr - prev_asr:+.1f}", delta_color="inverse")
            c3.metric("Refusal %", f"{refusal_pct:.1f}", delta=f"{refusal_pct - prev_refusal:+.1f}")
            c4.metric(
                "Extraction %", f"{extraction_pct:.1f}",
                delta=f"{extraction_pct - prev_extraction:+.1f}", delta_color="inverse",
            )
            c5.metric("Errors", errors, delta=f"{errors - prev_errors:+d}" if errors != prev_errors else None,
                       delta_color="inverse")
        else:
            c1.metric("Total", total)
            c2.metric("ASR %", f"{asr:.1f}")
            c3.metric("Refusal %", f"{refusal_pct:.1f}")
            c4.metric("Extraction %", f"{extraction_pct:.1f}")
            c5.metric("Errors", errors)

        # --- #2b: Severity colors on ASR KPI ---
        if asr < 10:
            st.success("Low risk")
        elif asr <= 50:
            st.warning("Medium risk")
        else:
            st.error("High risk")
        if kpis.get("tool_misuse_rate_pct") is not None:
            st.metric("Tool misuse %", f"{kpis.get('tool_misuse_rate_pct', 0):.1f}")
        if kpis.get("avg_turns_to_success") is not None:
            st.metric("Avg turns to success", f"{kpis.get('avg_turns_to_success', 0):.1f}")
        if (
            kpis.get("avg_latency_sec") is not None
            or kpis.get("expected_refusal_accuracy_pct") is not None
            or kpis.get("benign_refusal_rate_pct") is not None
        ):  # noqa: E501
            c6, c7, c8, c9, c10 = st.columns(5)
            if kpis.get("avg_latency_sec") is not None:
                c6.metric("Avg latency", f"{kpis['avg_latency_sec']:.2f}s")
            if kpis.get("expected_refusal_accuracy_pct") is not None:
                c7.metric("Expected-refusal accuracy %", f"{kpis['expected_refusal_accuracy_pct']:.1f}")
            if kpis.get("benign_refusal_rate_pct") is not None:
                c8.metric("Benign refusal %", f"{kpis['benign_refusal_rate_pct']:.1f}")
        if kpis.get("error_counts"):
            with st.expander("Error breakdown"):
                for msg, count in sorted(kpis["error_counts"].items(), key=lambda x: -x[1]):
                    st.caption(f"**{count}×** {msg}")

        by_cat = kpis.get("by_category") or {}
        if by_cat:
            st.subheader("By category")
            import pandas as pd

            df_cat = pd.DataFrame(
                [
                    {
                        "category": cat,
                        "ASR %": v.get("asr_pct", 0),
                        "Refusal %": v.get("refusal_rate_pct", 0),
                        "extraction_rate_pct": v.get("extraction_rate_pct", 0),
                        "total": v.get("total", 0),
                    }
                    for cat, v in by_cat.items()
                ]
            )
            st.dataframe(df_cat, use_container_width=True)
            try:
                import plotly.express as px

                fig = px.bar(
                    df_cat,
                    x="category",
                    y=["ASR %", "Refusal %"],
                    barmode="group",
                    title="ASR and Refusal rate by category",
                )
                st.plotly_chart(fig, use_container_width=True)
                fig_refusal = px.bar(
                    df_cat,
                    x="category",
                    y="Refusal %",
                    title="Refusal % by category",
                )
                st.plotly_chart(fig_refusal, use_container_width=True)
                if "extraction_rate_pct" in df_cat.columns:
                    fig_ext = px.bar(
                        df_cat,
                        x="category",
                        y="extraction_rate_pct",
                        title="Extraction % by category",
                    )
                    st.plotly_chart(fig_ext, use_container_width=True)
            except Exception:
                pass

            # --- #3b: Radar chart (ASR % by category, multi-model overlay) ---
            try:
                import plotly.graph_objects as go

                categories_list = df_cat["category"].tolist()
                asr_values = df_cat["ASR %"].tolist()
                if categories_list:
                    fig_radar = go.Figure()
                    # Current run trace
                    fig_radar.add_trace(go.Scatterpolar(
                        r=asr_values + [asr_values[0]],
                        theta=categories_list + [categories_list[0]],
                        fill="toself",
                        name=run_meta.get("model", "Current"),
                    ))
                    # Overlay multi-model traces if available
                    if multi_metas and len(multi_metas) > 1:
                        for mm in multi_metas:
                            mm_by_cat = (mm.get("kpis") or {}).get("by_category") or {}
                            mm_vals = [mm_by_cat.get(c, {}).get("asr_pct", 0) for c in categories_list]
                            fig_radar.add_trace(go.Scatterpolar(
                                r=mm_vals + [mm_vals[0]],
                                theta=categories_list + [categories_list[0]],
                                fill="toself",
                                name=mm.get("model", "?"),
                            ))
                    fig_radar.update_layout(
                        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
                        title="ASR % by category (radar)",
                        showlegend=True,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            except Exception:
                pass

            # --- #3a: Confusion matrix ---
            try:
                import pandas as _pd_cm

                rows_with_expected = [
                    r for r in results if r.get("expected_refusal") is not None
                ]
                if rows_with_expected:
                    # 2x2: rows = Expected (Refusal, Compliance), cols = Actual (Refusal, Compliance)
                    tp = sum(
                        1 for r in rows_with_expected
                        if r.get("expected_refusal") is True and r.get("refusal") is True
                    )
                    fn = sum(
                        1 for r in rows_with_expected
                        if r.get("expected_refusal") is True and r.get("refusal") is not True
                    )
                    fp = sum(
                        1 for r in rows_with_expected
                        if r.get("expected_refusal") is not True and r.get("refusal") is True
                    )
                    tn = sum(
                        1 for r in rows_with_expected
                        if r.get("expected_refusal") is not True and r.get("refusal") is not True
                    )
                    z = [[tp, fn], [fp, tn]]
                    x_labels = ["Actual Refusal", "Actual Compliance"]
                    y_labels = ["Expected Refusal", "Expected Compliance"]
                    try:
                        import plotly.figure_factory as ff

                        fig_cm = ff.create_annotated_heatmap(
                            z, x=x_labels, y=y_labels,
                            colorscale="Blues", showscale=True,
                        )
                        fig_cm.update_layout(title="Confusion Matrix (Expected vs Actual)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    except Exception:
                        import plotly.express as _px_cm

                        df_cm = _pd_cm.DataFrame(z, index=y_labels, columns=x_labels)
                        fig_cm = _px_cm.imshow(
                            df_cm, text_auto=True, color_continuous_scale="Blues",
                            title="Confusion Matrix (Expected vs Actual)",
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
            except Exception:
                pass

            durations = [r.get("duration_sec") for r in results if r.get("duration_sec") is not None]
            if durations:
                try:
                    import plotly.express as px

                    df_dur = pd.DataFrame({"duration_sec": durations})
                    fig_hist = px.histogram(
                        df_dur,
                        x="duration_sec",
                        title="Latency (s) distribution",
                        nbins=min(30, max(5, len(durations) // 3)),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception:
                    pass

        st.subheader("Per-prompt results")
        import pandas as pd

        df = pd.DataFrame(results)
        if not df.empty:
            display_cols = ["index", "category", "refusal", "compliance", "extraction", "duration_sec", "error"]
            available = [c for c in display_cols if c in df.columns]
            filter_col = st.columns(3)[0]
            with filter_col:
                filter_category = st.selectbox(
                    "Filter by category",
                    options=["(all)"] + sorted(df["category"].dropna().unique().tolist())
                    if "category" in df.columns
                    else ["(all)"],  # noqa: E501
                )
                filter_refusal = st.selectbox("Filter by refusal", options=["(all)", "Refusal", "Compliance"])
                filter_error = st.selectbox("Filter by error", options=["(all)", "Has error", "No error"])
            df_filtered = df.copy()
            if "category" in df.columns and filter_category != "(all)":
                df_filtered = df_filtered[df_filtered["category"] == filter_category]
            if filter_refusal == "Refusal":
                df_filtered = df_filtered[df_filtered["refusal"] is True]
            elif filter_refusal == "Compliance":
                df_filtered = df_filtered[df_filtered["refusal"] is False]
            if "error" in df_filtered.columns:
                has_error = df_filtered["error"].notna() & (df_filtered["error"].astype(str).str.len() > 0)
                if filter_error == "Has error":
                    df_filtered = df_filtered[has_error]
                elif filter_error == "No error":
                    df_filtered = df_filtered[~has_error]
            # --- #2a: Color-coded results table ---
            def _color_result_rows(row):
                """Return background colors per row based on refusal/compliance/error."""
                if row.get("error") and str(row.get("error")).strip():
                    return ["background-color: #e0e0e0"] * len(row)
                if row.get("refusal") is True:
                    return ["background-color: #c8e6c9"] * len(row)
                if row.get("compliance") is True:
                    return ["background-color: #ffcdd2"] * len(row)
                return [""] * len(row)

            styled_df = df_filtered[available].style.apply(_color_result_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            row_options = [f"Row {r['index']} ({r.get('category', '')})" for _, r in df.iterrows()]
            view_row = st.selectbox("View full prompt/response for row", options=["(none)"] + row_options)
            if view_row != "(none)" and row_options:
                idx = row_options.index(view_row)
                row = results[idx]
                with st.expander("Full prompt and response", expanded=True):
                    st.text_area(
                        "Prompt", value=row.get("prompt_full", row.get("prompt", "")), height=120, disabled=True
                    )  # noqa: E501
                    st.text_area(
                        "Response", value=row.get("response_full", row.get("response", "")), height=120, disabled=True
                    )  # noqa: E501
                    if row.get("duration_sec") is not None:
                        st.caption(f"Duration: {row['duration_sec']:.2f}s")
                    if row.get("error"):
                        st.caption(f"Error: {row['error']}")
            st.download_button(
                "Download results CSV",
                data=pd.DataFrame(results).to_csv(index=False),
                file_name="security_eval_results.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download run JSON",
                data=json.dumps(run_meta, indent=2),
                file_name="security_eval_run.json",
                mime="application/json",
            )
            report_lines = [
                "# Security Eval Report",
                "",
                f"- **Model:** {run_meta.get('model', '')}",
                f"- **Prompt set:** {run_meta.get('prompt_set', '')}",
                f"- **Timestamp:** {run_meta.get('timestamp_iso', '')}",
                "",
                "## KPIs",
                f"- Total: {kpis.get('total', 0)}",
                f"- ASR %: {kpis.get('asr_pct', 0):.1f}",
                f"- Refusal %: {kpis.get('refusal_rate_pct', 0):.1f}",
                f"- Extraction %: {kpis.get('extraction_rate_pct', 0):.1f}",
                f"- Errors: {kpis.get('errors', 0)}",
                "",
                "## Top failures (refusal or error)",
                "",
            ]
            failures = [r for r in results if r.get("refusal") or r.get("error")]
            for r in failures[:20]:
                report_lines.append(f"- **Row {r.get('index')}** [{r.get('category', '')}]")
                report_lines.append(f"  - Prompt: {(r.get('prompt_full') or r.get('prompt', ''))[:200]}...")
                report_lines.append(f"  - Response: {(r.get('response_full') or r.get('response', ''))[:200]}...")
                if r.get("error"):
                    report_lines.append(f"  - Error: {r.get('error')}")
                report_lines.append("")
            report_md = "\n".join(report_lines)
            st.download_button(
                "Download report (Markdown)",
                data=report_md,
                file_name="security_eval_report.md",
                mime="text/markdown",
            )
        st.json(
            {
                "model": run_meta.get("model"),
                "prompt_set": run_meta.get("prompt_set"),
                "timestamp_iso": run_meta.get("timestamp_iso"),
            }
        )


def _render_history_tab(st) -> None:
    st.subheader("Run history")
    try:
        runs = load_runs(limit=50)
        if runs:
            import pandas as pd

            df_runs = pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "model": r["model"],
                        "prompt_set": r["prompt_set"],
                        "timestamp": r["timestamp_iso"],
                        "ASR %": r["kpis"].get("asr_pct"),
                        "Refusal %": r["kpis"].get("refusal_rate_pct"),
                    }
                    for r in runs
                ]
            )
            st.dataframe(df_runs, use_container_width=True)
            try:
                import plotly.express as px

                df_runs["timestamp"] = pd.to_datetime(df_runs["timestamp"], errors="coerce")
                df_plot = df_runs.dropna(subset=["timestamp"]).sort_values("timestamp")
                if not df_plot.empty:
                    fig = px.line(df_plot, x="timestamp", y="ASR %", color="model", title="ASR % over time (by model)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        else:
            st.info("No runs in history yet. Run an evaluation and check 'Save run to history'.")
    except Exception as e:
        st.info("Run history is not available. Save a run with «Save run to history» to enable history and plots.")
        st.caption(f"Details: {e}")


def _render_compare_runs_tab(st) -> None:
    st.subheader("Compare two runs")
    try:
        runs_compare = load_runs(limit=50)
        if runs_compare and len(runs_compare) >= 2:
            options = [f"Run #{r['id']}: {r['model']} @ {r.get('timestamp_iso', '')[:16]}" for r in runs_compare]
            col_a, col_b = st.columns(2)
            with col_a:
                sel_a = st.selectbox("Run A", options=["(select)"] + options, key="compare_a")
            with col_b:
                sel_b = st.selectbox("Run B", options=["(select)"] + options, key="compare_b")
            if sel_a != "(select)" and sel_b != "(select)" and sel_a != sel_b:
                idx_a = options.index(sel_a)
                idx_b = options.index(sel_b)
                r_a = runs_compare[idx_a]
                r_b = runs_compare[idx_b]
                kpis_a = r_a.get("kpis") or {}
                kpis_b = r_b.get("kpis") or {}
                st.write("")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("", "KPI", "")
                with c2:
                    st.metric("Run A", r_a.get("model", "—"), r_a.get("timestamp_iso", "")[:16])
                with c3:
                    st.metric("Run B", r_b.get("model", "—"), r_b.get("timestamp_iso", "")[:16])
                kpi_rows = [
                    ("total", "Total"),
                    ("asr_pct", "ASR %"),
                    ("refusal_rate_pct", "Refusal %"),
                    ("extraction_rate_pct", "Extraction %"),
                    ("tool_misuse_rate_pct", "Tool misuse %"),
                    ("errors", "Errors"),
                    ("avg_latency_sec", "Avg latency (s)"),
                    ("avg_turns_to_success", "Avg turns to success"),
                    ("expected_refusal_accuracy_pct", "Expected-refusal acc %"),
                    ("benign_refusal_rate_pct", "Benign refusal %"),
                ]
                for key, label in kpi_rows:
                    va = kpis_a.get(key)
                    vb = kpis_b.get(key)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(label)
                    with c2:
                        st.write(f"{va:.1f}" if isinstance(va, float) else str(va) if va is not None else "—")
                    with c3:
                        st.write(f"{vb:.1f}" if isinstance(vb, float) else str(vb) if vb is not None else "—")
                # Export comparison (same format as CLI security-eval compare --export)
                label_a = r_a.get("model", "A") + " @ " + (r_a.get("timestamp_iso", "")[:16] or "?")
                label_b = r_b.get("model", "B") + " @ " + (r_b.get("timestamp_iso", "")[:16] or "?")
                export_rows = [
                    ("total", "Total"),
                    ("asr_pct", "ASR %"),
                    ("refusal_rate_pct", "Refusal %"),
                    ("extraction_rate_pct", "Extraction %"),
                    ("errors", "Errors"),
                    ("avg_latency_sec", "Avg latency (s)"),
                    ("expected_refusal_accuracy_pct", "Expected-refusal acc %"),
                    ("benign_refusal_rate_pct", "Benign refusal %"),
                ]
                buf_csv = io.StringIO()
                w = csv.writer(buf_csv)
                w.writerow(["KPI", label_a, label_b])
                for key, name in export_rows:
                    w.writerow([name, kpis_a.get(key, ""), kpis_b.get(key, "")])
                csv_data = buf_csv.getvalue()
                rows_html = "".join(
                    f"<tr><td>{name}</td><td>{kpis_a.get(key, '')}</td><td>{kpis_b.get(key, '')}</td></tr>"
                    for key, name in export_rows
                )
                html_data = (
                    '<!DOCTYPE html><html><head>'
                    '<meta charset="utf-8">'
                    '<title>Security Eval Compare</title></head><body>'
                    '<h1>Compare</h1><table border="1">'
                    f"<tr><th>KPI</th><th>{label_a}</th><th>{label_b}</th></tr>"
                    f"{rows_html}</table></body></html>"
                )
                st.download_button(
                    "Download comparison (CSV)", data=csv_data,
                    file_name="security_eval_compare.csv", mime="text/csv", key="compare_dl_csv",
                )
                st.download_button(
                    "Download comparison (HTML)", data=html_data,
                    file_name="security_eval_compare.html", mime="text/html", key="compare_dl_html",
                )

                # --- #4: Per-prompt response diff ---
                results_a = r_a.get("results") or []
                results_b = r_b.get("results") or []
                if results_a and results_b:
                    st.subheader("Per-prompt response diff")
                    max_idx = max(len(results_a), len(results_b))
                    diff_options = [
                        f"Prompt {i + 1}"
                        + (f" ({results_a[i].get('category', '')})" if i < len(results_a) else "")
                        for i in range(max_idx)
                    ]
                    diff_sel = st.selectbox(
                        "Select prompt index",
                        options=diff_options,
                        key="compare_diff_prompt",
                    )
                    if diff_sel:
                        diff_idx = diff_options.index(diff_sel)
                        col_da, col_db = st.columns(2)
                        with col_da:
                            st.caption(f"**Run A** — {r_a.get('model', '?')}")
                            if diff_idx < len(results_a):
                                ra = results_a[diff_idx]
                                st.text_area(
                                    "Response A",
                                    value=ra.get("response_full", ra.get("response", "")),
                                    height=200,
                                    disabled=True,
                                    key="diff_resp_a",
                                )
                                if ra.get("refusal"):
                                    st.success("REFUSAL")
                                elif ra.get("compliance"):
                                    st.error("COMPLIANCE")
                            else:
                                st.info("No result at this index for Run A.")
                        with col_db:
                            st.caption(f"**Run B** — {r_b.get('model', '?')}")
                            if diff_idx < len(results_b):
                                rb = results_b[diff_idx]
                                st.text_area(
                                    "Response B",
                                    value=rb.get("response_full", rb.get("response", "")),
                                    height=200,
                                    disabled=True,
                                    key="diff_resp_b",
                                )
                                if rb.get("refusal"):
                                    st.success("REFUSAL")
                                elif rb.get("compliance"):
                                    st.error("COMPLIANCE")
                            else:
                                st.info("No result at this index for Run B.")
        else:
            st.caption("Save at least two runs to history to compare them here.")
    except Exception:
        st.caption("Compare unavailable (history not loaded).")


if __name__ == "__main__":
    main()
