# Analysis: Features, Abliteration Pipeline, and Codebase

This document summarizes analysis of the ollama-tools codebase: all user-facing features with improvement suggestions, a focused audit of the abliteration pipeline from direction computation through export to Ollama, and codebase-wide findings.

---

## 1. Feature overview and improvement suggestions

### 1.1 Quickstart / onboarding

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `check` | Verifies ollama, Hugging Face, optional deps, llama.cpp | Add optional `--fix` to suggest or run setup (e.g. `ollama serve`). |
| `doctor` | Environment diagnostics | Consider exporting a machine-readable report (JSON) for CI or support. |
| `plan quickstart` / `plan auto` | Suggests next steps | Persist “last plan” so `plan` can show “continue from here” with one command. |
| `quickstart` / `start` | Guided first-run flow | Add `--non-interactive` for scripts; document exit codes. |

### 1.2 Fetch, convert, create-from-base

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `fetch` | HF GGUF download → create | Optional `--verify-checksum` when HF exposes hashes. |
| `convert` | GGUF → Modelfile → create | Support `--adapter` to stack an adapter on the GGUF base (align with retrain UX). |
| `create-from-base` | Base + optional adapter + system/params | Allow `--template-from` to copy TEMPLATE from another Ollama model (like abliterate run). |
| `refresh-template` | Recreate model with base’s template | Document that this replaces the existing model; add `--dry-run` to show Modelfile only. |

### 1.3 Recipes

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `build` (from recipe) | YAML/JSON → fetch or convert or create-from-base | Support variables (e.g. `base: "{{ base_model }}"`) and env substitution. |
| `validate-recipe` | Schema/field checks | Emit JSON with per-field status; support validating against a remote base (e.g. HF repo exists). |

### 1.4 Adapters

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `fetch-adapter` | Download adapter from HF | Verify adapter format (e.g. PEFT dir or single .bin/.gguf) and fail early with a clear message. |
| `adapters search` / `recommend` | Search/recommend adapters | Cache results for TTL; add `--apply` flow to run `retrain` in one step (already noted in explore). |
| `retrain` | create-from-base + adapter | Document that adapter can be LoRA .bin/.gguf or PEFT dir; add `--template-from` for tool support. |

### 1.5 Training data and training pipeline

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `validate-training-data` | Validates paths and format | Add `--schema` to output expected JSON schema for custom validators. |
| `prepare-training-data` | Converts to llama.cpp etc. format | Support multiple output formats in one run; document which formats each trainer expects. |
| `train-data init` | Scaffold training data layout | Optional templates (e.g. alpaca, chat, completion-only). |
| `train` | Prints bash script | Consider a `--execute` mode that runs the steps (with confirmation) to reduce copy-paste. |
| `train-run` | validate → prepare → finetune → retrain | Add `--skip-retrain` to stop after LoRA/adapter is produced; document base vs adapter compatibility. |
| `finetune` | Script for llama.cpp (or other trainer) | Unify default trainer (e.g. llama.cpp) across train/train-run/finetune and document in wiki. |

### 1.6 Abliteration

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `abliterate compute-dir` | Harmful/harmless → refusal .pt | Add `--layer-frac` single value for faster single-layer run; optional JSON summary (layer chosen, gap size). |
| `abliterate run` | compute → apply → GGUF → create | Optional `--only-compute` / `--only-apply` / `--only-export` for resumable runs; document disk space for full checkpoint. |
| `abliterate download-lists` | Download bundled list sources | Add `--curated-only` to fetch only the small curated lists. |
| `abliterate chat` / `serve` / `proxy` | Chat/serve with HF tokenizer or proxy to Ollama | Document that proxy is the path for tool/function-calling with abliterated Ollama models. |
| `abliterate evaluate` | Refusal rate on harmful set | Support custom marker file path; output machine-readable (e.g. JSON) for CI. |
| `abliterate optimize` | Strength/layer search | Add `--max-evals` and timeout; optional parallel eval (with resource limits). |
| `abliterate fix-ollama-template` | Fix template of existing Ollama model | Document that this recreates the model (destructive); add `--dry-run`. |

### 1.7 Security eval

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `security-eval run` | Prompt set → model → CSV/JSON | Support `--baseline` to compare two models in one run; document prompt set schema (wiki has some). |
| `security-eval ui` | Streamlit UI | Add “compare runs” view using saved history. |
| `security-eval compare` | Compare runs from history | Export comparison to CSV/HTML for reports. |

### 1.8 Utilities

| Feature | Current behavior | Suggestions |
|--------|------------------|-------------|
| `setup-llama-cpp` | Clone/build llama.cpp | Option to use a system or conda llama.cpp; document minimal CMake options for convert/quantize. |
| `hf-cache ls` / `rm` | Inspect/clear HF cache | Add `--size` to show disk usage per blob. |
| `downsize pipeline` | Guidance for GKD/downsize + convert | Provide a minimal example recipe or script that runs student + convert. |

---

## 2. Abliteration pipeline audit (beginning → export to Ollama)

### 2.1 Pipeline stages

1. **Direction computation** (`compute_refusal_dir` in `abliterate.py`)
   - **Inputs:** Model (HF id or path, or GGUF path), harmful/harmless instruction files (or built-in merged lists).
   - **Process:** Tokenize instructions, forward with `output_hidden_states=True`, take hidden state at chosen layer and position (default last token). Either try several `layer_fracs` and pick the layer with largest harmful–harmless gap (single or top-k SVD directions), or compute per-layer directions.
   - **Output:** `.pt` file: either raw tensor `(hidden_size, k)` or dict `{per_layer: True, directions: (L, H)}`.

2. **Apply** (`apply_refusal_dir_and_save`)
   - **Inputs:** Same model (or compatible), refusal `.pt`, output dir, strength/kernel options.
   - **Process:** For each transformer layer (with optional skip first/last), form `I - s*D@D.T` and apply to attention (Q/K/V/O) and MLP (gate/up right, down left); optional norm-preserving scaling.
   - **Output:** Full HF checkpoint in `output_dir` (e.g. `save_pretrained` or manual `pytorch_model.bin` + config + tokenizer).

3. **Export to GGUF**
   - **Process:** Subprocess to llama.cpp `convert_hf_to_gguf.py` on the **applied** checkpoint directory. Optional requantization (e.g. Q4_K_M) via llama.cpp `quantize`.

4. **Export to Ollama**
   - **Process:** `build_modelfile(gguf_path)` → `FROM <gguf_path>`. Then:
     - If `--template-from <ollama_model>`: merge reference TEMPLATE (and optionally FROM) so tool/chat API format matches.
     - Else if no TEMPLATE in Modelfile and checkpoint is local HF: `template_from_hf_checkpoint(checkpoint_dir)` and `get_stop_tokens_from_checkpoint(checkpoint_dir)` from the **applied** checkpoint dir; append PARAMETER num_predict.
   - **Final step:** `run_ollama_create(name, content)` (writes Modelfile to temp or out_path, runs `ollama create -f <path>`).

### 2.2 Correctness checks

- **Layer indexing:** `get_layers` supports `model.model.layers`, `model.model.language_model.layers`, `model.transformer.h`. Hidden state index uses `h_layer = layer_idx + 1` (0 = embeddings), and `pos=-1` (last token). This matches standard HF hidden_states layout. **No bug found.**
- **Direction application:** For each linear, right-multiply `(I - s*D@D.T)` where the weight’s input dim is `hidden_size` (Q/K/V/O, gate_proj, up_proj), and left-multiply for down_proj (output dim = hidden_size). Norm-preserving rescaling is applied per layer. **Implementation matches “remove projection onto refusal direction” in activation space.**
- **Refusal .pt format:** Single/multi direction `(hidden_size, k)`; per-layer `(L, H)`. Reject `(k, hidden_size)` via `d.size(0) < d.size(1)` check. **Correct.**
- **Template and stop tokens for Ollama:** After apply, template and stop tokens are taken from the **same** checkpoint directory that is converted to GGUF (`checkpoint_dir`). So Ollama gets a template consistent with the abliterated model’s tokenizer. **Correct.**
- **Modelfile:** `build_modelfile(gguf_path)` produces `FROM <absolute_gguf_path>`. No adapter path for abliteration (abliteration is baked into the full GGUF). **Consistent with design.**

### 2.3 Potential issues and edge cases

- **Layer count mismatch (per-layer directions):** If the model used in apply has more layers than the number of saved per-layer directions, the last saved direction is reused for extra layers. If fewer, only the first N are used. No warning is emitted. **Recommendation:** Add a warning when `n_saved_layers != n_layers` in per-layer mode (and optionally validate in apply).
- **Stale offload_folder cleanup:** In `compute_refusal_dir`, when using a temp offload folder, cleanup is done in a couple of places; the per-layer branch’s `if offload_folder: shutil.rmtree(...)` may be redundant depending on load path. **Low impact;** no functional bug.
- **Missing `data/refusal_markers.txt`:** `evaluate_abliteration` uses `Path(__file__).parent / "data" / "refusal_markers.txt"` and falls back to a hardcoded list if missing. **No crash;** consider documenting the optional file in data/README.md.
- **Doc vs code:** `docs/ABLITERATE.md` still describes the old flow (compute-dir → use Sumandora for inference; convert abliterated GGUF manually). The actual CLI provides `abliterate run` (compute → apply → GGUF → create) and proxy/serve. **Recommendation:** Update `docs/ABLITERATE.md` to describe the full built-in pipeline and point to wiki/Abliterate.md for user-facing docs.

### 2.4 Summary: wrong implementations?

- **No wrong implementations identified** in the abliteration math, layer indexing, or export path.
- **Improvements suggested:** Warn on layer count mismatch for per-layer directions; update `docs/ABLITERATE.md`; optionally document optional `refusal_markers.txt` and add dry-run / resumable flags for run.

---

## 3. Codebase-wide analysis

### 3.1 Structure and consistency

- **CLI:** Single entrypoint in `cli.py` with subparsers and `set_defaults(handler=...)`; dispatch via `args.handler(parser, args)`. Consistent and easy to extend.
- **Optional dependencies:** Abliteration behind `abliterate` extra; security-eval and training have their own deps. Good separation.
- **Shared helpers:** `run_helpers.run_ollama_create`, `modelfile.build_modelfile`, `modelfile.template_from_hf_checkpoint` used by convert, create-from-base, retrain, and abliterate run. Reduces duplication.

### 3.2 Error handling and UX

- **print_actionable_error:** Used in many CLI paths with cause and next_steps. Improves debuggability.
- **Timeouts:** GGUF conversion and `ollama create` use timeouts (e.g. 3600s); subprocess failures are caught and reported. Good.
- **Gaps:** Some code paths could validate inputs earlier (e.g. adapter path exists and is valid format before starting retrain). Document exit codes for scripting.

### 3.3 Testing and docs

- **Tests:** `tests/` cover cli, abliterate_proxy, security_eval, training_data, recipe, modelfile, chat_util. Abliteration core (compute/apply) is integration-heavy (model loading); consider small unit tests for `_strength_kernel_scale`, `_get_D_for_layer`-style logic with mocks.
- **Docs:** `docs/` for design and analysis; `wiki/` for user-facing reference. Some docs (e.g. ABLITERATE.md) are outdated relative to code; wiki is more aligned.

### 3.4 Security and robustness

- **Torch load:** `torch.load(..., weights_only=True)` in apply limits deserialization risk. Good.
- **Paths:** Use `Path` and `resolve()` for GGUF in Modelfile to avoid relative-path issues when using temp Modelfiles.
- **Env:** `.env` loaded from home and cwd without overwriting existing env; reasonable for secrets.

### 3.5 Suggested codebase improvements

- Add a **CHANGELOG** (or keep it in docs) so feature and flag changes are easy to track.
- **Type hints:** Many functions are annotated; complete remaining public APIs for stricter static checking.
- **Logging:** Consider a single logger (e.g. `logging.getLogger("ollama_forge")`) and optional `--verbose` to reduce ad-hoc `print(..., file=sys.stderr)`.
- **Config:** Some commands have many flags; consider a config file (e.g. YAML) for abliterate run and train-run to support repeatable runs.

---

## 4. Summary table

| Area | Status | Priority improvements |
|------|--------|------------------------|
| Abliteration pipeline | Correct end-to-end; no wrong math or export path | Warn on layer count mismatch; update docs/ABLITERATE.md; optional resumable run |
| Features (all) | Broad and coherent | See §1 for per-feature suggestions (dry-run, machine-readable output, template-from, doc updates) |
| Codebase | Clear structure; good error messages | Unify logging; add CHANGELOG; more unit tests for pure logic; config file support for long commands |

This analysis is based on exploration of the repository and reading of `abliterate.py`, `modelfile.py`, `run_helpers.py`, `cli.py` (abliterate and export sections), and related docs and wiki.
