"""Unified forge web UI.

Run via: ollama-forge ui
Or directly: streamlit run ollama_forge.forge_ui

Tabs:
  Chat  — stream responses from forge api (11435) or Ollama (11434)
  Models — list available models and forge api budget
  Train  — configure and launch forge mlx-train
"""
from __future__ import annotations

import json
import subprocess
import tempfile

import requests
import streamlit as st
import yaml

st.set_page_config(
    page_title="ollama-forge",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ollama-forge")
    base_url = st.text_input("API base URL", value="http://127.0.0.1:11435")
    base_url = base_url.rstrip("/")
    st.caption(
        "Points to `forge api` (port 11435) by default. "
        "Change to `http://127.0.0.1:11434` to use Ollama directly."
    )
    st.markdown("---")
    st.markdown("**Other UIs**")
    st.markdown("- `ollama-forge security-eval ui`")
    st.markdown("- `ollama-forge abliterate ui`")
    st.markdown("- `ollama-forge study ui`")


def _get_models(url: str) -> list[dict]:
    try:
        r = requests.get(f"{url}/v1/models", timeout=5)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


def _get_health(url: str) -> dict | None:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_chat, tab_models, tab_train = st.tabs(["💬 Chat", "🤖 Models", "🏋️ Train"])


# ── Chat ─────────────────────────────────────────────────────────────────────

with tab_chat:
    models = _get_models(base_url)
    model_ids = [m["id"] for m in models] if models else []

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Model",
            options=model_ids if model_ids else ["(no models — start forge api first)"],
            key="chat_model",
        )
    with col2:
        temperature = st.slider("Temp", 0.0, 2.0, 0.7, 0.05, key="chat_temp")

    system_prompt = st.text_input("System prompt (optional)", key="chat_system")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Message…", key="chat_input"):
        if not model_ids:
            st.warning("No models found. Start `ollama-forge api` first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.extend(st.session_state.messages)

            payload = {
                "model": selected_model,
                "messages": msgs,
                "temperature": temperature,
                "stream": True,
            }

            with st.chat_message("assistant"):
                def _stream():
                    with requests.post(
                        f"{base_url}/v1/chat/completions",
                        json=payload,
                        stream=True,
                        timeout=None,
                    ) as resp:
                        for line in resp.iter_lines():
                            if not line or line == b"data: [DONE]":
                                continue
                            raw = line.decode().removeprefix("data: ")
                            try:
                                chunk = json.loads(raw)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                continue

                reply = st.write_stream(_stream())

            st.session_state.messages.append({"role": "assistant", "content": reply or ""})

    if st.session_state.messages:  # noqa: SIM102
        if st.button("Clear chat", key="chat_clear"):
            st.session_state.messages = []
            st.rerun()


# ── Models ───────────────────────────────────────────────────────────────────

with tab_models:
    health = _get_health(base_url)
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", "✅ running")
        col2.metric("Used", f"{health.get('used_mb', 0):,} MB")
        col3.metric("Budget", f"{health.get('budget_mb', 0):,} MB")
        if loaded := health.get("loaded"):
            st.caption(f"Loaded: {', '.join(loaded)}")
    else:
        st.warning(f"forge api not reachable at `{base_url}`. Run: `ollama-forge api`")

    models = _get_models(base_url)
    if models:
        st.subheader(f"{len(models)} models")
        rows = []
        health_loaded = set(health.get("loaded", [])) if health else set()
        for m in models:
            rows.append({
                "ID": m["id"],
                "Owner": m.get("owned_by", "ollama"),
                "Loaded": "✅" if m["id"] in health_loaded else "",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No models found.")

    if st.button("🔄 Refresh", key="models_refresh"):
        st.rerun()


# ── Train ─────────────────────────────────────────────────────────────────────

with tab_train:
    st.markdown(
        "Configure and run `forge mlx-train` (mlx-lm LoRA on Apple Silicon). "
        "Requires an MLX model and training data in mlx_lm format "
        "(`train.jsonl` + `valid.jsonl` in the data directory)."
    )

    with st.form("train_form"):
        st.subheader("Required")
        model = st.text_input("Model (HF repo or local MLX dir)", value="mlx-community/Llama-3.2-3B-Instruct-4bit")
        data = st.text_input("Data directory (with train.jsonl, valid.jsonl)", value="./data")

        st.subheader("Training")
        col1, col2, col3 = st.columns(3)
        fine_tune_type = col1.selectbox("Fine-tune type", ["lora", "dora", "full"])
        num_layers = col2.number_input("Layers", min_value=-1, value=16)
        batch_size = col3.number_input("Batch size", min_value=1, value=4)

        col4, col5, col6 = st.columns(3)
        iters = col4.number_input("Iterations", min_value=1, value=500)
        lr = col5.number_input("Learning rate", value=1e-4, format="%e")
        max_seq = col6.number_input("Max seq length", min_value=128, value=2048, step=128)

        adapter_path = st.text_input("Adapter output dir", value="./adapters")
        grad_checkpoint = st.checkbox("Gradient checkpointing")

        st.subheader("Post-training (optional)")
        col7, col8 = st.columns(2)
        do_fuse = col7.checkbox("Fuse adapter into model")
        do_gguf = col8.checkbox("Export GGUF (requires fuse)", disabled=not do_fuse)

        ollama_name = st.text_input("Ollama model name (leave blank to skip push)", value="")

        submitted = st.form_submit_button("Run training")

    cfg = {
        "model": model,
        "data": data,
        "fine_tune_type": fine_tune_type,
        "num_layers": int(num_layers),
        "batch_size": int(batch_size),
        "iters": int(iters),
        "learning_rate": float(lr),
        "max_seq_length": int(max_seq),
        "adapter_path": adapter_path,
        "grad_checkpoint": grad_checkpoint,
        "fuse": bool(do_fuse),
        "export_gguf": bool(do_gguf if do_fuse else False),
    }
    if ollama_name:
        cfg["ollama_name"] = ollama_name

    with st.expander("Generated config.yaml (copy to run manually)"):
        st.code(yaml.dump(cfg, default_flow_style=False), language="yaml")

    if submitted:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="forge-train-"
        ) as tmp:
            yaml.dump(cfg, tmp, default_flow_style=False)
            tmp_path = tmp.name

        st.info(f"Config written to `{tmp_path}`. Running training…")
        cli = ["ollama-forge", "mlx-train", tmp_path]
        st.code(" ".join(cli), language="bash")

        with st.spinner("Training…"):
            result = subprocess.run(cli, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("Training complete!")
        else:
            st.error(f"Training failed (exit {result.returncode})")

        if result.stdout:
            with st.expander("stdout"):
                st.code(result.stdout)
        if result.stderr:
            with st.expander("stderr"):
                st.code(result.stderr)
