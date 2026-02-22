# Heretic-inspired integration

**Status:** Implemented (Phases 1–5).  
**Prior art:** [Heretic (Uncensored-AI)](https://github.com/0xSojalSec/Uncensored-AI) — automatic censorship removal with directional ablation and Optuna. We do not reuse Heretic code (AGPL-3.0); this page documents the ideas we integrated into ollama-forge abliteration.

See also: [Abliterate](Abliterate) (main refusal-removal doc), [Prior art: Heretic](Abliterate#prior-art-heretic-uncensored-ai) in Abliterate.

---

## On this page

- [Overview](#overview)
- [Prior art (summary)](#prior-art-summary)
- [Phase 1: Attention vs MLP strength](#phase-1-attention-vs-mlp-strength)
- [Phase 2: Per-layer refusal directions](#phase-2-per-layer-refusal-directions)
- [Phase 3: Layer-dependent strength kernel](#phase-3-layer-dependent-strength-kernel)
- [Phase 4: Evaluation](#phase-4-evaluation)
- [Phase 5: Parameter optimization (Optuna)](#phase-5-parameter-optimization-optuna)
- [Implementation status](#implementation-status)
- [Prompt proxy (hybrid approach)](#prompt-proxy-hybrid-approach)
- [See also](#see-also)

---

## Overview

Abliteration in ollama-forge removes refusal behavior by computing a “refusal direction” (difference of harmful vs harmless activations) and ablating it in transformer weights. The **Heretic integration** adds:

1. **Separate strengths** for attention vs MLP (Heretic finds MLP ablation more damaging).
2. **Per-layer directions** — one direction per layer, with optional interpolation.
3. **Layer-dependent strength** — kernel over layers (e.g. peak in the middle).
4. **Evaluation** — run harmful prompts and count refusals via configurable markers.
5. **Optimization** — Optuna search over strength and skip layers to minimize refusal rate.

All phases are backward-compatible: existing `.pt` files and CLI usage still work; new flags and commands are optional.

---

## Prior art (summary)

| Aspect | Heretic | Ollama-forge (after integration) |
|--------|--------|-----------------------------------|
| Direction | Per-layer; optional interpolation | Single or [per-layer](Heretic-Integration#phase-2-per-layer-refusal-directions); optional `--direction-index` |
| Ablation shape | Layer-dependent kernel; separate attn/MLP | [Strength kernel](Heretic-Integration#phase-3-layer-dependent-strength-kernel) + [atten/mlp strength](Heretic-Integration#phase-1-attention-vs-mlp-strength) |
| Optimization | Optuna TPE, refusal + KL | [abliterate optimize](Heretic-Integration#phase-5-parameter-optimization-optuna) (refusal rate) |
| Eval | Refusal count + KL | [abliterate evaluate](Heretic-Integration#phase-4-evaluation) (refusal markers) |

Full comparison: [Abliterate § Prior art: Heretic](Abliterate#prior-art-heretic-uncensored-ai).

---

## Phase 1: Attention vs MLP strength

**Goal:** Apply different ablation strength to attention layers vs MLP layers. Softer MLP ablation can preserve coherence while still reducing refusals.

**CLI (run):**

- `--atten-strength ALPHA` — strength for attention (q, k, v, o). Default: same as `--strength`.
- `--mlp-strength ALPHA` — strength for MLP (gate_proj, up_proj, down_proj). Default: same as `--strength`.

**Example:**

```bash
uv run ollama-forge abliterate run --model <hf_id> --name my-abliterated --strength 1 --mlp-strength 0.5
```

**Behavior:** Attention linears use `--atten-strength` (or `--strength`); MLP linears use `--mlp-strength`. Norm-preserving projection is applied in both cases.

---

## Phase 2: Per-layer refusal directions

**Goal:** One refusal direction per layer (difference-of-means at that layer). Optionally use a single layer’s direction (or a blend of two) for all layers.

**CLI:**

- **compute-dir / run:** `--per-layer-directions` — compute one direction per layer; save format `{"per_layer": True, "directions": (num_layers, hidden_size)}`.
- **run:** `--direction-index IDX` — with per-layer directions: integer = use that layer’s direction for every layer; float = blend two adjacent layers; omit = use each layer’s own direction.

**Examples:**

```bash
# Compute per-layer directions
uv run ollama-forge abliterate compute-dir --model <hf_id> --harmful harmful.txt --harmless harmless.txt --output refusal_per_layer.pt --per-layer-directions

# Apply using each layer’s own direction (default when .pt is per-layer)
uv run ollama-forge abliterate run --model <hf_id> --name my-abliterated  # use refusal_per_layer.pt as produced by run

# Apply using direction from layer 20 only, for all layers
uv run ollama-forge abliterate run ... --direction-index 20
```

**Behavior:** Apply step detects per-layer `.pt`; for each layer it uses `directions[layer_idx]` or the result of `--direction-index` (int or float blend). Single-direction `.pt` files (no `per_layer` key) are unchanged.

---

## Phase 3: Layer-dependent strength kernel

**Goal:** Ablation strength varies by layer (e.g. peak in the middle, weaker at the ends) instead of a constant.

**CLI (run):**

- `--strength-kernel {constant|linear_peak|gaussian}` — default: `constant`.
- `--kernel-center-frac F` — center of the kernel as a fraction of layer index (default: 0.5).
- `--kernel-width-frac F` — width (default: 0.4).

**Behavior:** For each layer, a scale in (0, 1] is computed from the kernel; `atten_strength` and `mlp_strength` are multiplied by this scale for that layer. `constant` gives scale 1.0; `linear_peak` and `gaussian` peak near the center.

---

## Phase 4: Evaluation

**Goal:** Measure refusal rate on an abliterated checkpoint by running harmful prompts and counting responses that contain any refusal marker (substring).

**CLI:**

```bash
uv run ollama-forge abliterate evaluate --checkpoint <dir> --harmful harmful.txt [--refusal-markers FILE] [--num-prompts N] [--max-new-tokens N]
```

- `--checkpoint` — path to abliterated checkpoint (e.g. `./abliterate-<name>/checkpoint`).
- `--harmful` — file with one harmful prompt per line (lines starting with `#` skipped).
- `--refusal-markers` — file with one substring per line (default: bundled `src/ollama_forge/data/refusal_markers.txt`).
- `--num-prompts` — max prompts to run (default: 50).
- `--max-new-tokens` — max tokens per response (default: 256).

**Output:** Prints refusals / total and refusal rate (e.g. `Refusals: 3 / 50 (6.0%)`).

---

## Phase 5: Parameter optimization (Optuna)

**Goal:** Search over ablation parameters to minimize refusal rate (each trial: apply with suggested params, then evaluate).

**CLI:**

```bash
uv run ollama-forge abliterate optimize --model <hf_id> --refusal-pt refusal.pt --harmful harmful.txt [--output-dir .] [--n-trials 20] [--num-eval-prompts 30] [--refusal-markers FILE]
```

- `--model` — same model id/path as used for compute-dir.
- `--refusal-pt` — path to refusal direction `.pt` (from compute-dir).
- `--harmful` — harmful prompts file for evaluation.
- `--output-dir` — directory for `optimize_best_params.json` (default: current dir).
- `--n-trials` — number of Optuna trials (default: 20).
- `--max-evals` — overrides `--n-trials` when set.
- `--timeout` — stop after this many seconds (optional).
- `--max-parallel` — run up to N trials in parallel (default: 1; use only if enough CPU/memory).
- `--num-eval-prompts` — prompts per evaluation (default: 30).

**Output:** Best params (strength, atten_strength, mlp_strength, skip_begin_layers, skip_end_layers, refusal_rate) printed and written to `optimize_best_params.json`. Requires the `abliterate` extra (includes `optuna`).

---

## Implementation status

| Phase | Status | Notes |
|-------|--------|--------|
| 1 | Done | `--atten-strength`, `--mlp-strength`; MLP gate/up/down ablation |
| 2 | Done | `--per-layer-directions`, `--direction-index`; dict `.pt` format |
| 3 | Done | `--strength-kernel`, `--kernel-center-frac`, `--kernel-width-frac` |
| 4 | Done | `abliterate evaluate`; bundled `refusal_markers.txt` |
| 5 | Done | `abliterate optimize`; Optuna in abliterate extra |

---

---

## Ports (Ollama vs abliterate)

| Service           | Default port | Purpose                          |
|-------------------|-------------|-----------------------------------|
| Ollama            | 11434       | Main Ollama server               |
| abliterate serve  | 11435       | Full-model Ollama-API server     |
| abliterate proxy  | 11436       | Tokenizer-only proxy → Ollama    |

Use `--port` to change serve or proxy port when running multiple services.

---

## Prompt proxy (hybrid approach)

When Ollama's GGUF tokenization produces garbled or incorrect output, the **lightweight prompt proxy** lets agents use Ollama for inference while formatting prompts with the original HF tokenizer.

**Architecture:**
1. Proxy listens on port 11436 (by default)
2. Intercepts `/api/chat` requests, formats prompts with HF tokenizer (from the checkpoint)
3. Forwards to Ollama `/api/generate` with `raw: true` (bypasses Ollama's templating)
4. Parses tool calls from response and returns in Ollama `/api/chat` format

**CLI:**

```bash
uv run ollama-forge abliterate proxy --name <name> [--port 11436] [--ollama-target http://localhost:11434]
```

**With agents:** Set `OLLAMA_HOST=http://127.0.0.1:11436` so agents call the proxy instead of Ollama directly. The proxy handles template formatting and tool parsing, then forwards inference to Ollama.

**Why proxy instead of serve?**
- `abliterate serve` loads the full model (GPU memory, slower startup)
- `abliterate proxy` only loads the tokenizer; Ollama does inference
- Proxy is lightweight, compatible with any agent expecting the Ollama API, and supports tool/function calling

---

## See also

- [Abliterate](Abliterate) — setup, lists, run, chat, serve, proxy, tools, prior art.
- **Data lists:** `src/ollama_forge/data/README.md` — harmful/harmless lists, curated vs merged.
- `ollama-forge abliterate --help`, `abliterate proxy --help`, `abliterate evaluate --help`, `abliterate optimize --help` — CLI reference.
