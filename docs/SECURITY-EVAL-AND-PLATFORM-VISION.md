# Security Eval: Research Perspective & Platform Vision

Analysis of the **security-eval** feature from a security-research perspective (what to add/improve, especially UI), followed by a vision for making **ollama-forge** uniquely valuable for LLM security, distillation, and abliteration.

---

## Part 1: Security Eval — Security Research Perspective

### 1.1 What Exists Today (Strengths)

| Area | Current state |
|------|----------------|
| **Attack coverage** | Jailbreak/refusal (harmful prompts → compliance/refusal), indirect prompt injection (context + user query), system-prompt / PII extraction (target string or regex). |
| **Scorers** | Refusal (keyword list + min length), compliance (= not refusal), extraction (literal, regex, or PII patterns). |
| **Prompt sets** | .txt (one per line), .jsonl (prompt, category, target_for_extraction, context). Reuse of abliterate harmful lists via `abliterate download-lists`. |
| **KPIs** | ASR %, Refusal %, Extraction %, Errors; per-category breakdown. |
| **Backend** | Any Ollama or abliterate serve (same API); no model loaded in eval process. |
| **History** | SQLite runs (model, prompt_set, timestamp, kpis); UI shows table + ASR-over-time plot. |
| **UI** | Streamlit: base URL, model name, prompt set path (text), system prompt, optional CSV/JSON out, save history, run → KPIs + by-category table + per-prompt table + run history + line chart. |

These give researchers a **reproducible, local-first** way to run attack prompt sets and compare models (e.g. base vs abliterated) without sending data off-machine.

---

### 1.2 Gaps for Identifying LLM Issues (What to Add)

#### A. Attack coverage and harness

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| **No expected-outcome validation** | Can’t measure “false refusals” (benign prompts refused) or “false compliance” (harmful that should refuse). | Use `expected_refusal` in JSONL (already in loader); in UI and reports show **accuracy**: % where outcome matched expected. Add KPI: **Benign refusal rate** (lower is better for UX). |
| **Single-turn only** | Multi-step jailbreaks and conversational extraction aren’t measured. | Add optional **multi-turn** mode: sequence of user/assistant turns per row; stop on first refusal or after N turns; score final state; record turns-to-success for query efficiency. |
| **Tool misuse not scored** | abliterate serve supports tools; no eval path that sends tool definitions and scores “model called dangerous tool with bad args”. | Add **tool-use** prompt set format (messages + tools) and scorer: did the model invoke a given tool with given (e.g. exfil) args? KPI: **Tool misuse rate**. |
| **No image / multimodal** | Serve supports images; no eval for “model revealed image content” or “followed hidden instruction in image”. | Add image URL/path in JSONL and pass through /api/chat; scorer: keyword or regex on response (or optional small judge). |
| **System prompt not varied** | Only one global system prompt per run. | Support **per-prompt system override** in JSONL (e.g. for “ignore system” tests) and/or **system prompt extraction** benchmark (known system string as target_for_extraction). |

#### B. Scorers and metrics

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| **Refusal = keywords only** | Misses paraphrased refusals; no confidence. | Allow **custom keyword file** (like abliterate refusal_markers); optional **refusal regex** per category; later: tiny classifier or embedding similarity for “refusal-like” score. |
| **Extraction is binary** | No partial credit (e.g. 50% of secret leaked). | Optional **extraction score**: overlap ratio or count of target fragments; or severity band (none / partial / full). |
| **No latency/throughput in UI** | Hard to see if abliteration or model choice hurt performance. | Add **duration_sec** to per-prompt table and **avg latency** to KPIs; optional throughput (prompts/min) in run summary. |
| **Errors not broken down** | Only total error count. | In UI and CSV: error type or message summary (e.g. timeout vs 4xx vs connection); filter results by “has error”. |

#### C. Prompt sets and reproducibility

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| **No built-in benchmark references** | Hard to cite or compare to AdvBench, HarmBench, etc. | Document **which bundled/downloaded lists map to which benchmarks**; add optional **prompt_set_version** or **source** in run metadata and in history. |
| **Path-only in UI** | Typing paths is error-prone. | **File picker** or **dropdown of recent paths**; optional “bundled sets” (e.g. sample_prompts.txt, sample_indirect.jsonl) selectable by name. |
| **No sampling** | Large sets slow; no quick “smoke” run. | CLI/UI: `--max-prompts N` or “Sample: first N / random N” for quick iteration. |

---

### 1.3 UI Improvements for Smooth Operation

The UI is critical for daily use. Below are changes that make runs smoother and results easier to interpret.

#### Must-have for flow

| Improvement | Why |
|-------------|-----|
| **File picker for prompt set** | Avoid typos and “file not found”; support “recent files” or workspace-relative paths. |
| **Model selection from API** | Call `GET /api/tags` (or equivalent) and show **dropdown of available models** instead of free-text model name. |
| **Live progress during run** | Show “Running prompt 42/200…” and **live-update** the results table (e.g. every 5–10 prompts) so researchers see progress without waiting for the full run. |
| **Clear “Run” vs “Results”** | Separate “Configure and run” from “Results”; after run, show results in an expandable section and keep config visible so users can tweak and re-run. |
| **One-click “Save to history”** | Default to checked for research (or remember last choice); show short confirmation “Saved run #N”. |

#### Results and analysis

| Improvement | Why |
|-------------|-----|
| **Drill-down on per-prompt table** | Click a row to expand **full prompt + full response** (and duration, error); essential for debugging and writing up findings. |
| **Filter by category / refusal / error** | Allow filtering the results table by category, refusal (y/n), compliance, extraction, or “has error”; export filtered view. |
| **Compare two runs side-by-side** | Select run A and run B from history; show KPIs and per-category metrics **side-by-side** (e.g. base vs abliterated). |
| **Extraction % in main KPI strip** | Today extraction is in by_category and per-prompt; add **Extraction %** next to ASR % and Refusal % in the top metrics. |
| **Export from UI** | Buttons: “Download CSV”, “Download JSON” for **current run** (and optional “Download all history as CSV”). |

#### Reliability and polish

| Improvement | Why |
|-------------|-----|
| **Validate prompt set before run** | If path is set, on “Run” do a quick load and show “Loaded N prompts, categories: …”; on failure show which line or file is bad. |
| **Timeouts and retries visible** | Show timeout and retry count in config; optional “Last run: X s, Y errors” under the button. |
| **Graceful history failure** | If DB is missing or locked, show a clear message and hide the history section instead of a generic exception. |
| **Dark mode / theme** | Streamlit theme selector or default dark for long sessions. |

#### Optional (higher effort)

- **Multi-model run**: Select multiple models, run same prompt set on each, show comparison table and plots (ASR/refusal by model).
- **Charts**: Refusal rate by category (bar), extraction rate by category, latency distribution (histogram).
- **Report**: Generate a one-page PDF or Markdown summary (model, prompt set, KPIs, top failures).

---

### 1.4 Summary: Security Eval Additions (Prioritized)

1. **High impact, lower effort**  
   - Model dropdown from `GET /api/tags`.  
   - File picker or bundled-set selector for prompt set.  
   - Live progress and incremental results table.  
   - Drill-down: click row → full prompt + response.  
   - Use `expected_refusal` and report accuracy / benign refusal rate.  
   - Extraction % in top KPI strip; filter results by category/refusal/error.

2. **Medium effort**  
   - Multi-turn harness and query-efficiency metric.  
   - Tool-use prompt sets and tool-misuse scorer.  
   - Compare two runs side-by-side in UI.  
   - Custom refusal keyword file and optional regex.  
   - `--max-prompts` and optional sampling.

3. **Larger / research**  
   - Image/multimodal cases and scorer.  
   - Per-prompt system prompt and system-prompt-extraction benchmark.  
   - Optional refusal classifier or embedding-based score.  
   - Prompt set versioning and benchmark references in metadata.

---

## Part 2: Making the Tool Amazing and Unique

### 2.1 Where ollama-forge Already Stands Out

- **Local-first**: No data leaves the machine; Ollama and abliterate serve are local.
- **Refusal ablation + serve in one stack**: compute-dir → run → serve (or proxy) with the same toolchain.
- **Ollama-native**: Same API as Ollama; abliterate serve and proxy drop into existing workflows.
- **Security eval on the same API**: One harness for both “vanilla” and “abliterated” models.
- **Reproducibility**: Recipes, checkpoint paths, prompt sets, and run history (SQLite) support reproducible experiments.
- **Optional optimization**: Optuna over ablation strength/skip layers to minimize refusal rate (Heretic-style).

So the **unique angle** is: **refusal removal (abliteration) + evaluation + optional optimization**, all against the **same local API**, with **no cloud dependency**.

---

### 2.2 Integration That Would Make It Unique

| Integration | Description | Why it’s unique |
|-------------|-------------|------------------|
| **Baseline vs abliterated in one flow** | CLI or UI: “Run security-eval on model A (base) and model B (abliterated) with the same prompt set; show comparison.” | Standard workflow today is manual A then B; first-class A/B reduces friction and highlights impact of abliteration. |
| **Eval inside abliterate optimize loop** | Optuna trials already run harmful prompts and count refusals; optionally **also** run security-eval (e.g. a small prompt set) and use ASR/refusal as an objective or constraint. | Connects “refusal rate on harmful list” with “broader security eval” in one optimization. |
| **Proxy + security-eval** | Use abliterate proxy (with tools) as the backend for security-eval; add tool-misuse prompt sets. | Only stack that evaluates “abliterated model behind proxy with tools” out of the box. |
| **download-lists → security-eval** | One command: download HarmBench/JBB/etc. → run security-eval on that list and save to history. | Tight coupling between “known benchmarks” and “eval + history” for citations and comparison. |

### 2.3 The “Full Pipeline” Story

A single narrative that no other tool offers end-to-end:

1. **Fetch** a model (e.g. HF) and optionally **convert** to GGUF / Ollama.  
2. **Abliterate** (compute refusal direction, run ablation, optionally optimize with Optuna).  
3. **Serve** the abliterated model (serve or proxy) on a known port.  
4. **Evaluate** with security-eval (same prompt set for base and abliterated).  
5. **Compare** in UI or CLI: ASR/refusal/extraction for base vs abliterated; over time if you iterate.

Optional: **Distillation** (downsize) then abliterate then eval, to study “small abliterated” vs “large abliterated”.

Documenting this as a **“Security and abliteration pipeline”** in the wiki (with one-page diagram and commands) would make the value proposition obvious.

### 2.4 Research and Ecosystem

| Idea | Description |
|------|-------------|
| **Run metadata** | Every eval run stores: model, base_url, prompt_set path (and optional hash/version), timestamp, and optionally system_prompt, timeout. Makes “exact repro” and papers easy. |
| **Prompt set registry** | Curated list (in wiki or README): bundled samples, download-lists (AdvBench, HarmBench, JBB, etc.), and how to cite them. |
| **Report template** | Optional: “Export report” → Markdown or PDF with KPIs, by-category table, and “method: ollama-forge security-eval, model X, prompt set Y, date”. |
| **Community prompt sets** | Accept contributions (e.g. PRs) of .jsonl/.txt under a `security_eval/data/` or `prompt_sets/` tree with clear license and category tags. |

### 2.5 Distillation + Abliteration + Eval

Today: **downsize** (distillation), **abliterate** (refusal removal), **security-eval** (run prompt sets). They are separate commands. A **unified story** could be:

- “Small model (distilled) + abliterated” vs “Large model + abliterated” vs “Large base”: compare ASR, refusal rate, and latency in one table.
- Recipe or script: `build distilled-abliterated → serve → security-eval → output comparison`.

That positions ollama-forge as the place for **efficient, safe, local models** with a single toolchain.

---

## Part 3: Concrete Next Steps (and todo list)

### Security eval — done in first pass

- [x] **UI**: Model dropdown from API (`list_models`, “Refresh models” button).
- [x] **UI**: Bundled prompt set selector (sample_prompts.txt, sample_prompts.jsonl, sample_indirect.jsonl) + custom path.
- [x] **UI**: Clear Run vs Results (results in session_state, shown after run).
- [x] **UI**: Extraction % in main KPI strip.
- [x] **UI**: Drill-down: select row → full prompt + response (prompt_full/response_full in run_meta).
- [x] **UI**: Filter results by category / refusal / error.
- [x] **UI**: Export buttons: Download CSV, Download JSON (current run).
- [x] **UI**: Validate prompt set before run (“Loaded N prompts, categories: …”).
- [x] **UI**: Save to history default True; graceful history failure (message + caption).
- [x] **CLI**: `--max-prompts N` (run_eval + parser).

### Security eval — still to do

- [x] **UI**: Live progress during run + incremental results table (progress_callback, st.progress, results placeholder).
- [x] **Backend**: `expected_refusal` accuracy + benign refusal rate KPI.
- [x] **CLI/run**: Custom refusal keyword file option (`--refusal-keywords`).
- [x] **KPIs**: Avg latency in summary; error type breakdown in CSV/UI.
- [x] **UI**: Compare two runs side-by-side (Run history → Compare two runs section).
- [x] **UI**: Timeout/retry visible in config (number inputs); dark theme note in sidebar.
- [x] **JSONL**: Per-prompt system override (`system`/`system_prompt`); prompt_set_version in run metadata (hash).
- [x] Multi-turn harness: JSONL `turns` (list of {role, content}), `query_model_multi_turn`, scored on last response.
- [ ] **Tool-use**: Prompt set format with `tools` + messages; scorer for tool misuse (design only).
- [ ] **Image/multimodal**: JSONL field `image`; scorer for revealed content (design only).

### Platform

- [x] **Wiki**: “Security and abliteration pipeline” page + diagram ([Security-and-Abliteration-Pipeline](../wiki/Security-and-Abliteration-Pipeline.md)).
- [x] **CLI/UI**: Compare model A vs B: `security-eval compare run_a.json run_b.json`; UI “Compare two runs” section.
- [x] **Doc**: Prompt set registry ([wiki/Prompt-Set-Registry.md](../wiki/Prompt-Set-Registry.md)) with benchmark refs.
- [ ] Optional: Eval step inside abliterate optimize.

This document can be used as a roadmap for security-eval improvements and for positioning ollama-forge as the go-to platform for local LLM security research, abliteration, and distillation.
