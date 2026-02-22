# Changelog

All notable changes to this project are documented here. The project version is not updated until explicitly requested.

---

## [1.0.1] (unreleased)

### Plan

- **plan continue** — New subcommand: show or run the last saved plan. Save a plan with e.g. `plan quickstart --json`; then `plan continue` shows it and `plan continue --execute` runs the stored command(s).
- Plan output (when using `--json`) is persisted to `.ollama-forge-last-plan.json` (or `OLLAMA_FORGE_PLAN_FILE`) so you can "continue from here" later.

### Fetch & Convert

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

---

## [1.0.0]

Initial release.
