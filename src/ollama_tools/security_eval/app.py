"""
Streamlit UI for LLM security evaluation.
Run with: streamlit run ollama_tools.security_eval.app
Or: uv run ollama-forge security-eval ui  (with uv sync --extra security-eval-ui)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:
    from ollama_tools.security_eval.history import load_runs
    from ollama_tools.security_eval.run import run_eval
except ImportError:
    # When run as streamlit app from repo root without install
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ollama_tools.security_eval.history import load_runs
    from ollama_tools.security_eval.run import run_eval


def main() -> None:
    st.set_page_config(page_title="LLM Security Eval", layout="wide")
    st.title("LLM Security Evaluation")
    st.markdown("Run prompt sets against Ollama or abliterate serve, view KPIs and per-category results.")

    base_url = st.text_input(
        "Base URL (Ollama or abliterate serve)",
        value="http://127.0.0.1:11434",
        help="e.g. http://127.0.0.1:11434 or http://127.0.0.1:11435 for abliterate serve",
    )
    model = st.text_input("Model name", value="llama3.2")
    prompt_set_path = st.text_input(
        "Prompt set path",
        value="",
        placeholder="/path/to/prompts.txt or .jsonl",
        help=".txt: one prompt per line; .jsonl: prompt, category, target_for_extraction",
    )
    system_prompt = st.text_area("System prompt (optional)", value="", height=80)

    col1, col2 = st.columns(2)
    with col1:
        output_csv = st.text_input("Output CSV path (optional)", value="")
    with col2:
        output_json = st.text_input("Output JSON path (optional)", value="")
    save_history = st.checkbox("Save run to history (for plots over time)", value=False)

    if st.button("Run evaluation"):
        if not prompt_set_path or not Path(prompt_set_path).exists():
            st.error("Please provide a valid prompt set file path.")
            return
        with st.spinner("Running evaluation..."):
            try:
                run_meta = run_eval(
                    prompt_set_path,
                    base_url=base_url.strip() or "http://127.0.0.1:11434",
                    model=model.strip() or "llama3.2",
                    output_csv=output_csv.strip() or None,
                    output_json=output_json.strip() or None,
                    save_to_history=save_history,
                    system=system_prompt.strip() or None,
                    verbose=False,
                )
            except Exception as e:
                st.exception(e)
                return
        kpis = run_meta.get("kpis") or {}
        results = run_meta.get("results") or []

        st.success("Evaluation complete.")
        st.subheader("KPIs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", kpis.get("total", 0))
        c2.metric("ASR %", f"{kpis.get('asr_pct', 0):.1f}")
        c3.metric("Refusal %", f"{kpis.get('refusal_rate_pct', 0):.1f}")
        c4.metric("Errors", kpis.get("errors", 0))

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
            except Exception:
                pass

        st.subheader("Per-prompt results")
        import pandas as pd

        df = pd.DataFrame(results)
        if not df.empty:
            st.dataframe(
                df[["index", "category", "refusal", "compliance", "extraction", "duration_sec", "error"]],
                use_container_width=True,
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
        st.caption(f"History unavailable: {e}")


if __name__ == "__main__":
    main()
