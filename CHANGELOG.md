# Changelog

All notable changes to this project are documented here. The project version is not updated until explicitly requested.

---

## [1.0.1]

### Plan

- **plan continue** — New subcommand: show or run the last saved plan. Save a plan with e.g. `plan quickstart --json`; then `plan continue` shows it and `plan continue --execute` runs the stored command(s).
- Plan output (when using `--json`) is persisted to `.ollama-forge-last-plan.json` (or `OLLAMA_FORGE_PLAN_FILE`) so you can "continue from here" later.

### Fetch, Import & Convert

- **import** — New command: download HF safetensors, convert to GGUF via llama.cpp, optionally quantize, derive chat template, and create an Ollama model in one step. Works with remote HF repos (`import meta-llama/Llama-3.2-1B-Instruct --name llama3.2-1b`) or local checkpoint directories. Key flags: `--quant`, `--no-requantize`, `--outtype`, `--template-from`, `--system`, `--temperature`, `--num-ctx`.
- **auto** — Local directories with `config.json` now route to `import` (previously unsupported).
- **fetch** — `--verify-checksum`: after download, verify file SHA256 against Hub ETag when available (LFS files).
- **convert** — `--adapter <path>`: stack an adapter on the GGUF base; adds `ADAPTER` to the Modelfile (directory or .bin/.gguf).

### Recipes

- **build** — Recipe variables and env substitution: use `{{ varname }}` or `${varname}` in recipe values; context is `os.environ` plus optional `variables` section in the recipe.
- **validate-recipe** — `--json`: output per-field validation result. `--validate-remote`: when source is `hf_repo`, check that the Hugging Face repo exists.

### Adapters

- **fetch-adapter** — After download, verify adapter format (PEFT or single .bin/.gguf); fail early with a clear message if invalid.
- **adapters recommend** — `--cache-ttl SECONDS` (default 3600): cache recommendations; `0` disables. Cache path: `~/.cache/ollama-forge/adapters-recommend/` or `OLLAMA_FORGE_CACHE`.

### Training data & train

- **validate-training-data** — `--schema`: print expected JSON schema (Alpaca + messages) and exit; no paths required.
- **prepare-training-data** — Multiple formats in one run: `--format llama.cpp,alpaca_plain` writes `<output_stem>_<format>.txt` for each. `--list-formats`: list supported formats and which trainer expects them. New format: `alpaca_plain`.
- **train-data init** — `--template alpaca|chat` (default alpaca): sample.jsonl as Alpaca-style or messages-style.
- **train** — `--execute`: run validate → prepare → (finetune if `--base-gguf` and `--run-trainer`), then print the retrain command.

### Abliteration

- **abliterate compute-dir** — `--layer-frac F`: single layer fraction (overrides `--layer-fracs`). `--json`: print summary (layer_frac, layer_index, gap_norm). `compute_refusal_dir` now returns this summary when not using per-layer directions.
- **abliterate optimize** — `--max-evals N`: overrides `--n-trials` when set. `--timeout SECONDS`: stop optimization after this many seconds. `--max-parallel N`: run up to N Optuna trials in parallel (default 1; use only if enough CPU/memory).

### Security eval

- **security-eval run** — `--baseline MODEL`: run the same prompt set against baseline and primary model; print comparison. `--schema`: print prompt set schema (TXT/JSONL); no prompt_set path required.
- **security-eval compare** — `--export PATH`: export comparison to CSV or HTML (by file suffix).
- **security-eval UI** — Dedicated "Compare runs" view using saved history; export comparison as CSV/HTML.

### Utilities

- **setup-llama-cpp** — `--use-system`: do not clone/build; verify finetune/quantize on PATH. `--use-conda`: print instructions for conda-installed llama.cpp. Post-build message mentions minimal CMake options.
- **hf-cache ls** — `--size`: print total cache disk usage (human-readable).

### Examples & docs

- **Downsize** — Minimal example: `examples/downsize/README.md` and `examples/recipes/downsize-student.yaml` for post-distillation student GGUF.
- **Docs** — Disk space note for abliterate run; ABLITERATE.md and wiki aligned with full pipeline.

### Codebase

- **Config** — `--config FILE` for abliterate run, train-run, finetune; YAML/JSON with CLI overrides.
- **Logging** — Single logger (`ollama_forge.log`), `--verbose` / `-v`; many `print(..., sys.stderr)` replaced with logger.
- **Type hints** — Filled in for abliterate, config_loader, and related modules.
- **Unit tests** — `tests/test_abliterate.py`: `_strength_kernel_scale`, `get_D_for_layer`. Recipe variable substitution tests.

### Bug fixes (Feb 2026)

- **o_proj left-multiply bug** — `o_proj` was silently skipped during ablation due to shape mismatch; now correctly left-multiplied. This was the primary cause of gibberish output.
- **GGUF bf16 overflow** — `convert_hf_to_gguf.py` now called with `--outtype bf16`; previously defaulted to f16, causing overflow of renormalized weights.
- **skip_begin/end_layers default 0** — Argparse defaults were `0`; corrected to `1` so embedding-adjacent layers are skipped by default.
- **`--no-norm-preserving` flag** — Added CLI flag to disable Frobenius-norm rescaling; critical for small models where 18× compounding causes activation explosion.
- **Per-layer direction: O(N×layers) → O(N) forward passes** — Single-pass hidden-state cache eliminates redundant forward passes (was 6400 passes for 32 layers × 100 instructions; now 200).
- **`compute_refusal_dir` returned wrong `layer_index`** — Returned loop variable (always last frac) instead of `best_layer_idx`; JSON summary now shows the actual best layer.
- **Tool-call templates stripped** — HF Jinja2 rendering dropped tool sections; priority-1 family `template_override` (Llama3, Mistral, Qwen2) now preserves full tool-calling support.
- **Model family coverage** — Added phi4, qwen2_5, qwen3, gemma3_text; `_auto_family_from_config` fallback for unknown architectures.
- **Live-server test race condition** — Added `_wait_for_server()` poll in test fixtures; eliminates socket `TimeoutError` on slow systems.
- **386 tests passing** — Expanded test suite covering modelfile template derivation, model family detection, layer-skip edge cases, and pipeline return values.

---

## [1.0.0]

Initial release.
