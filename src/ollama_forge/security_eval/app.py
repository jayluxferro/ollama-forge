"""
Streamlit UI for LLM security evaluation.
Run with: streamlit run ollama_forge.security_eval.app
Or: uv run ollama-forge security-eval ui  (with uv sync --extra security-eval-ui)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

try:
    from ollama_forge.security_eval.client import list_models
    from ollama_forge.security_eval.history import load_runs
    from ollama_forge.security_eval.run import run_eval
except ImportError:
    # When run as streamlit app from repo root without install
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ollama_forge.security_eval.client import list_models
    from ollama_forge.security_eval.history import load_runs
    from ollama_forge.security_eval.run import run_eval


def main() -> None:
    st.set_page_config(page_title="LLM Security Eval", layout="wide")
    st.title("LLM Security Evaluation")
    st.markdown("Run prompt sets against Ollama or abliterate serve, view KPIs and per-category results.")

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
    chosen_path = bundled_paths.get(prompt_set_choice)
    if chosen_path is not None and chosen_path.exists():
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

        def progress_cb(current: int, total: int, results_so_far: list) -> None:
            progress_bar.progress(current / total if total else 0, text=f"Running prompt {current}/{total}...")
            if results_so_far:
                import pandas as pd
                df = pd.DataFrame(results_so_far)
                cols = [c for c in ["index", "category", "refusal", "compliance", "extraction", "error"] if c in df.columns]  # noqa: E501
                if cols:
                    results_placeholder.dataframe(df[cols], use_container_width=True)

        progress_bar = st.progress(0.0, text="Running evaluation...")
        results_placeholder = st.empty()
        try:
            for mi, m in enumerate(models_to_run):
                if run_multi and len(models_to_run) > 1:
                    progress_bar.progress((mi + 0.5) / len(models_to_run), text=f"Model {mi + 1}/{len(models_to_run)}: {m}...")  # noqa: E501
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
        except Exception as e:
            st.exception(e)
            return
        finally:
            progress_bar.empty()
            results_placeholder.empty()
        st.session_state["last_run_meta"] = multi_metas[-1] if multi_metas else {}
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
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", kpis.get("total", 0))
        c2.metric("ASR %", f"{kpis.get('asr_pct', 0):.1f}")
        c3.metric("Refusal %", f"{kpis.get('refusal_rate_pct', 0):.1f}")
        c4.metric("Extraction %", f"{kpis.get('extraction_rate_pct', 0):.1f}")
        c5.metric("Errors", kpis.get("errors", 0))
        if kpis.get("tool_misuse_rate_pct") is not None:
            st.metric("Tool misuse %", f"{kpis.get('tool_misuse_rate_pct', 0):.1f}")
        if kpis.get("avg_turns_to_success") is not None:
            st.metric("Avg turns to success", f"{kpis.get('avg_turns_to_success', 0):.1f}")
        if kpis.get("avg_latency_sec") is not None or kpis.get("expected_refusal_accuracy_pct") is not None or kpis.get("benign_refusal_rate_pct") is not None:  # noqa: E501
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
                    df_cat, x="category", y="Refusal %",
                    title="Refusal % by category",
                )
                st.plotly_chart(fig_refusal, use_container_width=True)
                if "extraction_rate_pct" in df_cat.columns:
                    fig_ext = px.bar(
                        df_cat, x="category", y="extraction_rate_pct",
                        title="Extraction % by category",
                    )
                    st.plotly_chart(fig_ext, use_container_width=True)
            except Exception:
                pass
            durations = [r.get("duration_sec") for r in results if r.get("duration_sec") is not None]
            if durations:
                try:
                    import plotly.express as px
                    df_dur = pd.DataFrame({"duration_sec": durations})
                    fig_hist = px.histogram(
                        df_dur, x="duration_sec",
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
                    options=["(all)"] + sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else ["(all)"],  # noqa: E501
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
            st.dataframe(df_filtered[available], use_container_width=True)
            row_options = [f"Row {r['index']} ({r.get('category', '')})" for _, r in df.iterrows()]
            view_row = st.selectbox("View full prompt/response for row", options=["(none)"] + row_options)
            if view_row != "(none)" and row_options:
                idx = row_options.index(view_row)
                row = results[idx]
                with st.expander("Full prompt and response", expanded=True):
                    st.text_area("Prompt", value=row.get("prompt_full", row.get("prompt", "")), height=120, disabled=True)  # noqa: E501
                    st.text_area("Response", value=row.get("response_full", row.get("response", "")), height=120, disabled=True)  # noqa: E501
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

    st.divider()
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
                for key, label in [
                    ("total", "Total"),
                    ("asr_pct", "ASR %"),
                    ("refusal_rate_pct", "Refusal %"),
                    ("extraction_rate_pct", "Extraction %"),
                    ("tool_misuse_rate_pct", "Tool misuse %"),
                    ("errors", "Errors"),
                    ("avg_latency_sec", "Avg latency (s)"),
                    ("avg_turns_to_success", "Avg turns to success"),
                ]:
                    va = kpis_a.get(key)
                    vb = kpis_b.get(key)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(label)
                    with c2:
                        st.write(f"{va:.1f}" if isinstance(va, float) else str(va) if va is not None else "—")
                    with c3:
                        st.write(f"{vb:.1f}" if isinstance(vb, float) else str(vb) if vb is not None else "—")
        else:
            st.caption("Save at least two runs to history to compare them here.")
    except Exception:
        st.caption("Compare unavailable (history not loaded).")


if __name__ == "__main__":
    main()
