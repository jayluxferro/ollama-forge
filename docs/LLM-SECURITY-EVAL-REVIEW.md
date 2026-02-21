# LLM Security Evaluation Platform — Feasibility Review

A review of what’s possible for building an LLM security evaluation platform for researchers: attack coverage, KPIs, UI, and integration with ollama-tools.

---

## 1. What exists today (ollama-tools)

| Capability | Status | Notes |
|------------|--------|--------|
| **Local models (Ollama / HF)** | ✅ | fetch, convert, abliterate run, serve; no data leaves the machine |
| **Refusal ablation** | ✅ | abliterate compute-dir/run; uses harmful/harmless lists for “refusal direction” |
| **Harmful/harmless lists** | ✅ | Bundled defaults; download-lists (AdvBench, HarmBench, JailbreakBench, refusal_direction, JBB benign); custom `--harmful`/`--harmless` |
| **Ollama-compatible API** | ✅ | abliterate serve: /api/chat, /api/generate, stream, tools, images, format, think, logprobs |
| **Structured evaluation** | ❌ | No runnable “eval” that takes a prompt set + model and returns metrics |
| **KPIs / metrics** | ❌ | No ASR, refusal rate, or extraction rate computed or stored |
| **UI** | ❌ | CLI only (chat, serve); no dashboard or plots |
| **Reproducibility** | ✅ | Recipes, plan, checkpoint paths; no eval-run versioning |

So: **foundation is strong for local, reproducible models and red-team data; missing are evaluation harness, metrics, and UI.**

---

## 2. Attack categories to support (feasibility)

| Attack type | Description | Feasible in-platform? | Notes |
|-------------|-------------|------------------------|--------|
| **Jailbreak / refusal override** | Adversarial prompts to get refusals lifted | ✅ Yes | You already have harmful lists (AdvBench, HarmBench, JBB, etc.); need to run prompts → model and score refusals vs. complies. |
| **Prompt injection (direct)** | User prompt contains hidden instructions | ✅ Yes | Send crafted prompts to /api/chat or /api/generate; check if model follows “attacker” goal (e.g. ignore system prompt). |
| **Prompt injection (indirect)** | Malicious content in RAG/context | ⚠️ Partial | Needs “documents” + user query; can run via chat with injected content in messages; full RAG pipeline is out of scope unless you add a small RAG harness. |
| **System prompt extraction** | Recover system prompt via queries | ✅ Yes | Run extraction prompts, compare output to known system prompt; define “leak” (exact match, overlap, regex). |
| **PII / training data extraction** | Extract memorized PII or training snippets | ⚠️ Partial | Needs canonical PII/extraction benchmarks (e.g. PII-Scope style); you can run queries and score with regex/NER; no built-in “training data” oracle. |
| **Tool misuse / function calling** | Malicious tool use (e.g. delete, send data) | ✅ Yes | abliterate serve supports tools; run chats with tool definitions and malicious user requests; score if model calls dangerous tools with bad args. |
| **Multimodal (image)** | Malicious or misleading images | ✅ Yes | Serve supports images; add image-based test cases and pass via API; scoring is prompt-defined (e.g. “did model reveal content of image?”). |
| **Excessive agency / autonomy** | Model does too much without user confirmation | ⚠️ Design | Define scenarios (e.g. “only suggest, don’t execute”); score via rules on tool calls or response content. |

**Summary:** Jailbreak, direct injection, system-prompt extraction, tool misuse, and image-based tests are **directly feasible** against your existing serve/chat API. Indirect injection and PII extraction are **partially feasible** (custom prompts + simple scoring). Full reproduction of every academic benchmark is **not** required; you can define a **core set** of attack types and KPIs and extend later.

---

## 3. KPIs and metrics (what to implement)

| KPI | Meaning | How to compute | Difficulty |
|-----|--------|----------------|------------|
| **ASR (Attack Success Rate)** | % of attack prompts that “succeed” | For each attack: run prompt → get response → classifier or rule says success/fail; ASR = successes / total. | Low (need success criteria per attack type). |
| **Refusal rate** | % of (e.g. harmful) prompts that get refused | Run harmful (or mixed) list; detect refusal (keyword, list, or small classifier). | Low; refusal detection is simple (keywords, length, optional model). |
| **Compliance rate** | % where model follows the (malicious) instruction | Same as ASR with “compliance” as success. | Low. |
| **Extraction rate** | % of extraction prompts that leak target info | Run extraction prompts; check if response contains target string/regex or PII pattern. | Low for system-prompt; medium for PII (need target list). |
| **Query efficiency** | Mean number of queries to first success | For multi-query attacks, record first success; average over runs. | Medium (need multi-step harness). |
| **Latency / throughput** | Time per request, requests/sec | Already available from serve (eval_duration, etc.); aggregate in eval. | Low. |
| **Per-category breakdown** | ASR (or refusal rate) by category (e.g. jailbreak, extraction) | Tag prompts with category; compute ASR per tag; show table or plot. | Low. |
| **Model comparison** | Same eval across models | Run same prompt set against multiple models (Ollama or serve); one row per model, columns = KPIs. | Low if models share same API (Ollama/serve). |

**Recommendation:** Start with **ASR**, **refusal rate**, and **per-category breakdown**; add **extraction rate** and **query efficiency** for relevant attack types. Store **run metadata** (model, timestamp, prompt set version) for reproducibility and plotting over time.

---

## 4. UI and visualization (what’s possible)

| Feature | Feasibility | Options |
|---------|-------------|--------|
| **Web UI** | ✅ Yes | **Streamlit** or **Gradio** (Python); minimal deps, good for internal/research tools. Streamlit is common in security/red-team UIs (e.g. Vigil-LLM); Gradio is good for leaderboards and model comparison. |
| **Run evals from UI** | ✅ Yes | UI calls backend (Python) that runs prompts against Ollama or abliterate serve (same /api/chat, /api/generate). Backend can be same process (Streamlit/Gradio script) or a small FastAPI/Flask service. |
| **KPI tables** | ✅ Yes | Pandas DataFrame → table (Streamlit/Gradio); export CSV/JSON. |
| **Plots** | ✅ Yes | **Plotly** or **Matplotlib**: ASR by category (bar), ASR over time (line), model comparison (radar/bar), refusal rate by prompt set. |
| **Export** | ✅ Yes | CSV, JSON, or SQLite for runs; optional PDF report (e.g. WeasyPrint or reportlab). |
| **Multi-model comparison** | ✅ Yes | Same eval config, different `model` (or serve URL); aggregate results in one table and charts. |
| **Real-time progress** | ✅ Yes | Streamlit/Gradio can show “Running prompt 42/200…” and live-update tables; stream logs if needed. |
| **User management / auth** | ⚠️ Optional | For lab use, optional login (e.g. Streamlit auth, OAuth); not required for single-user research. |

**Technical note:** The evaluator does **not** need to load the model itself; it only needs to call **Ollama or abliterate serve** over HTTP. So the UI process can be light (no GPU); the heavy work stays in Ollama/serve.

---

## 5. What’s possible vs. not (summary)

### Possible and aligned with ollama-tools

- **Eval harness:** Prompt sets (from files or built-in) → send to Ollama/serve → collect responses → score with rules or small classifier → store results (JSON/CSV/DB).
- **Core KPIs:** ASR, refusal rate, extraction rate, per-category breakdown; optional query efficiency and latency.
- **UI:** Streamlit or Gradio app: select model(s), select prompt set / attack type, run eval, view tables and plots, export.
- **Reuse existing data:** Harmful/harmless lists and download-lists (JailbreakBench, HarmBench, etc.) as prompt sets for jailbreak/refusal evals.
- **Reproducibility:** Save eval config (model name, prompt set path/version, seed) with each run; optional versioning of prompt sets in repo or artifact store.

### Possible with more work

- **Indirect prompt injection:** Add a small “RAG” path (e.g. inject document + user query into messages) and define extraction/success criteria.
- **PII extraction benchmarks:** Integrate public PII/extraction datasets and target lists; score with regex or NER.
- **Multi-step / adaptive attacks:** Harness that supports multiple turns and stops on first success for query-efficiency metrics.
- **Custom detectors:** Use a small classifier or NER model to score “refusal” vs “compliance” instead of keyword rules.

### Not in scope (or out of platform)

- **Reimplementing every published attack:** Use a subset of attack types and prompt sets; cite benchmarks (JailbreakBench, HarmBench, etc.) and align where useful.
- **Proprietary model APIs:** Focus on local (Ollama/serve) and optional OpenAI/Anthropic adapters later if needed.
- **Full red-team automation:** Platform = eval + KPIs + UI; human defines or curates prompt sets and interprets results.

---

## 6. Suggested architecture (high level)

```
┌─────────────────────────────────────────────────────────────────┐
│  UI (Streamlit or Gradio)                                        │
│  - Select model(s) / serve URL                                   │
│  - Select prompt set & attack type                              │
│  - Run eval → progress + results                                │
│  - Tables, plots, export CSV/JSON                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Eval engine (Python)                                           │
│  - Load prompt set (JSON/JSONL/TXT)                             │
│  - For each prompt: POST /api/chat or /api/generate to Ollama   │
│    or abliterate serve                                          │
│  - Score response (refusal, compliance, extraction, etc.)       │
│  - Aggregate → KPIs (ASR, refusal rate, by category)            │
│  - Save run (config + results) to JSON/CSV/SQLite               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Ollama or abliterate serve (existing)                          │
│  - One model per serve; or Ollama with multiple models           │
└─────────────────────────────────────────────────────────────────┘
```

- **Prompt sets:** Directory or artifact store of JSON/JSONL (and/or TXT) with fields like `prompt`, `category`, `expected_refusal`, `target_for_extraction` (optional).
- **Scorers:** One module per attack type (e.g. jailbreak, extraction); each returns success/fail (and optional score); evaluator aggregates.
- **Runs:** One run = one model + one prompt set + timestamp; results = per-prompt outcome + KPIs; store and optionally version.

---

## 7. Phased plan (if you build it)

| Phase | Deliverable | Effort (rough) |
|-------|-------------|----------------|
| **1** | Eval harness (CLI): load prompt set, call Ollama/serve, score with simple rules (refusal keywords, extraction regex), output ASR + refusal rate + CSV. | Small |
| **2** | Add prompt set format (e.g. JSONL with category), per-category KPIs, and run metadata (model, prompt set, timestamp). | Small |
| **3** | Streamlit (or Gradio) UI: run eval from UI, view table and one or two plots (e.g. ASR by category, model comparison). | Medium |
| **4** | Export (CSV/JSON), optional SQLite for history, and basic plots (over time, radar for multi-model). | Small |
| **5** | Optional: indirect injection harness, PII extraction benchmarks, or custom refusal/compliance classifier. | Medium–Large |

---

## 8. Conclusion

- **Yes, you can build** an LLM security evaluation platform that lets researchers run attack prompt sets against models (including abliterated ones via serve), compute KPIs (ASR, refusal rate, extraction rate, by category), and use a **UI with tables and plots** and export.
- **Reuse:** Ollama and abliterate serve as the model backend; existing harmful/harmless and download-lists as prompt sources; Python + Streamlit or Gradio for the UI.
- **Scope:** Start with jailbreak/refusal and direct injection + system-prompt extraction; add indirect injection and PII extraction in a second step if needed.
- **Not required:** Full reproduction of every benchmark; proprietary APIs; or building new model runtimes — the value is in **consistent eval harness + KPIs + UI** on top of what you already have.

If you want to proceed, the most impactful first step is **Phase 1: eval harness CLI** (prompt set → Ollama/serve → scoring → ASR/refusal rate + CSV), then add the UI and extra attack types on top.
