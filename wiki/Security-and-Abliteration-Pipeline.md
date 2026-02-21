# Security and abliteration pipeline

End-to-end flow: **fetch a model → abliterate (refusal removal) → serve → security-eval → compare** base vs abliterated. All steps run locally; no data leaves your machine.

---

## Pipeline overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Fetch     │     │  Abliterate  │     │   Serve     │     │  Security-eval  │     │  Compare    │
│  (HF/Ollama)│ ──► │ compute-dir  │ ──► │ (or proxy)  │ ──► │  same prompts   │ ──► │  A vs B     │
│             │     │ run → GGUF   │     │ port 11435  │     │  ASR / refusal  │     │  in UI/CLI  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────────┘     └─────────────┘
```

1. **Fetch** — Get a base model (Hugging Face or existing Ollama).
2. **Abliterate** — Compute refusal direction, apply ablation, optionally optimize (Optuna), export to GGUF and create an Ollama model.
3. **Serve** — Run the abliterated model via `abliterate serve` (or `abliterate proxy` if you use tools) on a dedicated port.
4. **Security-eval** — Run the same prompt set against the base model (Ollama) and the abliterated model (serve); get KPIs (ASR %, refusal %, extraction %, etc.).
5. **Compare** — View base vs abliterated side-by-side (run history in UI, or two CSV/JSON runs in CLI).

---

## Step-by-step commands

### 1. Fetch a base model

**Option A — From Hugging Face (then abliterate in HF space):**

```bash
# Abliterate expects HF model id or path; compute-dir downloads if needed
uv sync --extra abliterate
uv run ollama-forge abliterate compute-dir --model <hf_id> --output refusal.pt
uv run ollama-forge abliterate run --model <hf_id> --name my-abliterated --refusal-dir refusal.pt
# Then export to GGUF and create Ollama model (see Abliterate doc)
```

**Option B — Already have Ollama base:**

Use the base model name in step 4 for “base” eval. For abliteration you still need the HF checkpoint (e.g. fetch from HF, abliterate, then serve).

### 2. Abliterate (compute direction + run)

```bash
uv sync --extra abliterate
# Compute refusal direction (optional: --harmful / --harmless, or use built-in lists)
uv run ollama-forge abliterate compute-dir --model <hf_id> --output refusal.pt
# Apply ablation and create Ollama model
uv run ollama-forge abliterate run --model <hf_id> --name my-abliterated --refusal-dir refusal.pt
```

See [Abliterate](Abliterate) and [Heretic-Integration](Heretic-Integration) for options (per-layer directions, Optuna optimize, etc.).

### 3. Serve the abliterated model

```bash
# Default port 11435 (so it doesn’t clash with Ollama on 11434)
uv run ollama-forge abliterate serve --name my-abliterated --port 11435
```

Or use **abliterate proxy** if you need tool/function-calling and already have the model loaded in Ollama.

### 4. Security-eval (base and abliterated)

**Eval base model (Ollama on 11434):**

```bash
uv run ollama-forge security-eval run ./prompts.jsonl --model <base_model> --output-csv base.csv --save-history
```

**Eval abliterated model (serve on 11435):**

```bash
uv run ollama-forge security-eval run ./prompts.jsonl --base-url http://127.0.0.1:11435 --model my-abliterated --output-csv abliterated.csv --save-history
```

Use the **same prompt set** (e.g. jailbreak/harmful list) so results are comparable.

**Optional — use harmful list from abliterate:**

```bash
uv run ollama-forge abliterate download-lists --output-dir ./eval_lists
uv run ollama-forge security-eval run ./eval_lists/harmful.txt --model <base> --output-csv base.csv
uv run ollama-forge security-eval run ./eval_lists/harmful.txt --base-url http://127.0.0.1:11435 --model my-abliterated --output-csv abliterated.csv
```

### 5. Compare

- **UI:** Run both evals with “Save run to history” checked. Open `ollama-forge security-eval ui` and use **Run history** to compare ASR % / Refusal % across runs (and “Compare two runs” when that feature is available).
- **CLI:** Compare the two CSV files (e.g. ASR, refusal rate, errors) or the printed KPIs.

---

## Optional: distillation then abliterate

To study “small abliterated” vs “large abliterated”:

1. **Downsize** (distillation): `ollama-forge downsize --teacher <large> --student <small> --name small-model`
2. Abliterate the **student** (HF path for that small model), then serve.
3. Run security-eval on base large, base small, abliterated large, abliterated small and compare.

---

## See also

- [Security-Eval](Security-Eval) — Prompt set format, KPIs, UI, history.
- [Abliterate](Abliterate) — compute-dir, run, serve, proxy, evaluate, optimize.
- [Heretic-Integration](Heretic-Integration) — Per-layer directions, Optuna, strength kernel.
- [docs/SECURITY-EVAL-AND-PLATFORM-VISION.md](../docs/SECURITY-EVAL-AND-PLATFORM-VISION.md) — Research perspective and roadmap.
