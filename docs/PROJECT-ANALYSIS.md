# Project Analysis: Optimizations, Improvements, and New Features

Analysis of the **ollama-forge** codebase for performance optimizations, code quality improvements, and new feature opportunities. Tied to existing tests and docs where relevant.

---

## 1. Optimizations

### 1.1 Code deduplication

| Issue | Location | Recommendation |
|-------|----------|----------------|
| **Duplicate `_is_gemma_checkpoint`** | `modelfile.py` (line 144), `abliterate.py` (line 15) | Centralize in `model_family.py`: add `is_gemma_checkpoint(checkpoint_dir)` that uses `detect_model_family()` and checks `family.name == "gemma"`. Remove duplicates and import from one place. |
| **URL normalization** | `abliterate_proxy._get_ollama_base()`, `security_eval/client._normalize_base_url()` | Add shared `ollama_forge.http_util.normalize_base_url(url: str) -> str` (strip slash, add `http://` if missing). Reuse in proxy and security_eval. |
| **Tool conversion (Ollama → HF)** | `abliterate_proxy._convert_tools_to_hf`, `abliterate_serve._ollama_forge_to_hf` | Extract to `ollama_forge.chat_util.ollama_tools_to_hf(tools)` and use in both proxy and serve. |

**Tests to add:** `tests/test_model_family.py` — add `test_is_gemma_checkpoint`; optional `tests/test_chat_util.py` for shared chat/tool helpers if new module is created.

---

### 1.2 Performance

| Area | Current | Improvement |
|------|---------|-------------|
| **Tokenizer cache** | `abliterate_proxy._load_tokenizer` uses `@lru_cache(maxsize=4)` | Good. Consider making `maxsize` configurable via env (e.g. `OLLAMA_PROXY_TOKENIZER_CACHE_SIZE`) for multi-model setups. |
| **Download lists** | `cli.py` uses sequential `urlopen()` for harmful/harmless URLs | Use `concurrent.futures.ThreadPoolExecutor` to fetch URLs in parallel (same as current timeouts). Reduces total wait for `download-lists`. |
| **Proxy streaming** | Reads Ollama response in 4KB chunks | Consider larger buffer (e.g. 16KB) or line-by-line for NDJSON to reduce syscalls; measure before changing. |
| **subprocess** | `run_ollama_create`, `run_ollama_show_modelfile`, GGUF convert use `subprocess.run` without timeout | Add `timeout=300` (or configurable) to avoid hung processes. |

**Tests:** Existing tests remain valid. Add integration-style test for parallel download (e.g. mock URLs and assert concurrent calls).

---

### 1.3 Dependencies

| Current | Suggestion |
|---------|------------|
| **urllib only** | No `requests` dependency; `urllib.request` used in proxy, serve, security_eval, cli. | Keep stdlib for zero deps; optional: add `requests` as extra for nicer errors and retries (e.g. `ollama-forge[net]`). |
| **Optional abliterate** | `torch`, `transformers`, `accelerate`, `gguf`, `protobuf`, `optuna` | Document minimal subset for proxy-only (e.g. `transformers` only) so users can avoid installing `torch` if they only run proxy. |

---

## 2. Improvements

### 2.1 Documentation and discoverability

| Gap | Fix |
|-----|-----|
| **Command Reference** | Add `abliterate proxy` and `abliterate fix-ollama-template` to [wiki/Command-Reference.md](wiki/Command-Reference.md) and to the “Commands” / “Quick reference” tables. |
| **README** | Add one line under “Refusal removal” or “Abliterate”: “For agents: `abliterate proxy` (lightweight tokenizer proxy).” |
| **Doc parity** | [docs/suggestions.md](docs/suggestions.md) #14: Ensure README and wiki match CLI (e.g. `top_p`, `repeat_penalty`). Audit and fix. |
| **Recipe validate** | [docs/suggestions.md](docs/suggestions.md) #4: Add `validate-recipe` (or `build --validate-only`) to check recipe without running build. |

---

### 2.2 Error handling and UX

| Improvement | Where | Tied to |
|-------------|--------|--------|
| **Next steps on failure** | [docs/suggestions.md](docs/suggestions.md) #10: Every major failure prints: (1) problem summary, (2) likely cause, (3) exact next command(s). | `run_helpers.print_actionable_error` already exists; ensure all CLI handlers use it and pass `next_steps`. Grep for `print(` and `sys.stderr` to find ad-hoc errors. |
| **Proxy: unknown model** | When `POST /api/chat` uses a model name not registered with the proxy, return 400 with message: “Model X not registered. Start proxy with --name X or --checkpoint <dir>.” | Already implemented; add test in `tests/test_abliterate_proxy.py` that mocks handler and asserts 400 body. |
| **Ollama connection** | Proxy and security_eval: if Ollama is down, fail fast with clear message and suggest `ollama serve` or check `OLLAMA_HOST`. | Optional: add `ollama_forge.run_helpers.ping_ollama(base_url) -> bool` and call from proxy startup and security_eval. |

---

### 2.3 Consistency

| Item | Change |
|------|--------|
| **Abliterate subcommands** | Document all in Command Reference: `compute-dir`, `run`, `chat`, `serve`, `proxy`, `evaluate`, `optimize`, `fix-ollama-template`, `download-lists`. |
| **Ports** | Defaults: serve 11435, proxy 11436. Document in wiki (Abliterate / Heretic-Integration) and in `--help`. |
| **Recipe format** | [docs/suggestions.md](docs/suggestions.md) #15: Add recipe keys for `quant` and `quantize` to mirror CLI. |

---

### 2.4 Testing

| Gap | Recommendation |
|-----|----------------|
| **modelfile template derivation** | Add tests in `tests/test_modelfile.py` for `template_from_hf_checkpoint_with_reason` with a minimal fixture (e.g. temp dir with `config.json` + `tokenizer_config.json` and a tiny Jinja template). |
| **run_helpers** | Add `tests/test_run_helpers.py`: `print_actionable_error` output shape, `require_ollama` when `ollama` missing vs present (mock `shutil.which`), `get_jsonl_paths_or_exit` with empty list. |
| **recipe YAML** | In `tests/test_recipe.py` add at least one test that loads a YAML recipe (temp file) and asserts parsed keys. |
| **Golden path** | [docs/suggestions.md](docs/suggestions.md) #17: One integration-style test per workflow: e.g. `plan quickstart`, `plan doctor-fix`, `auto --plan` with nonexistent file (already partially covered). |

---

## 3. New Features

### 3.1 High value (from suggestions and roadmap)

| Feature | Description | Effort |
|---------|-------------|--------|
| **Recipe from HF** | Recipe key `hf_repo` (+ optional `gguf_file`): `build recipe.yaml` runs fetch-from-HF then convert. [docs/SIMPLICITY_ANALYSIS.md](docs/SIMPLICITY_ANALYSIS.md) §2. | M |
| **Auto-pick GGUF when multiple** | When repo has many GGUF files, auto-pick by policy (e.g. Q4_K_M) and print “We auto-picked `<file>`; use `--gguf-file` to override.” [docs/SIMPLICITY_ANALYSIS.md](docs/SIMPLICITY_ANALYSIS.md) §1. | S (partially done via `pick_one_gguf`; ensure fetch path uses it and message is clear). |
| **validate-recipe** | `ollama-forge validate-recipe recipe.yaml` (or `build --validate-only`): validate schema and paths without running build. [docs/suggestions.md](docs/suggestions.md) #4. | S |
| **Interactive mode for missing args** | When required options are missing, prompt in terminal with defaults (e.g. `fetch` asks for repo, name). [docs/suggestions.md](docs/suggestions.md) #5. | M |
| **Adapter compatibility preflight** | Before create: verify adapter files and base-model compatibility; clear errors. [docs/suggestions.md](docs/suggestions.md) #9. | S/M |

---

### 3.2 Abliterate and proxy

| Feature | Description | Effort |
|---------|-------------|--------|
| **Proxy: multiple models** | Register multiple model names → checkpoint dirs (e.g. config file or repeated `--model NAME --checkpoint DIR`). Single proxy serves several abliterated models. | M |
| **Proxy: health endpoint** | `GET /` or `GET /api/tags` returns 200 and list of registered models so agents can check availability. | S |
| **Ollama ping on startup** | Proxy and serve: on startup, optionally ping Ollama (or target URL); warn or exit if unreachable. | S |
| **Abliterate: resume** | If `abliterate run` fails after GGUF step, allow resuming from checkpoint (e.g. `--from-checkpoint ./abliterate-name/checkpoint`) to skip compute/apply. | M |

---

### 3.3 Security eval and tooling

| Feature | Description | Effort |
|---------|-------------|--------|
| **Security eval: retry** | Configurable retries for transient API errors in `run_eval`. | S |
| **Security eval: export** | Export run results to JSON/CSV for CI or external dashboards. | S |
| **Unified output mode** | [docs/suggestions.md](docs/suggestions.md) #13: `--output json` for machine-readable output (e.g. plan, check) for scripting. | M |

---

### 3.4 Developer and CI

| Feature | Description | Effort |
|---------|-------------|--------|
| **Ruff rule set** | Expand `tool.ruff.lint` (e.g. add B, C4, SIM) for consistency; fix or exclude where necessary. | S |
| **Pre-commit or githooks** | Document or add `ruff check` and `pytest` in `.githooks` so contributors run them before push. | S (githooks README exists). |
| **Coverage** | Add `pytest-cov` and `coverage` report in CI; gate on minimum coverage for new code. | S |

---

## 4. Summary Table

| Category | Count | Examples |
|----------|-------|----------|
| **Optimizations** | 6 | Dedupe `_is_gemma_checkpoint`, shared URL/tool helpers, parallel download-lists, subprocess timeouts |
| **Improvements** | 10+ | Command Reference update, recipe validate, error next-steps, test coverage for modelfile/run_helpers/recipe |
| **New features** | 12+ | Recipe from HF, validate-recipe, interactive mode, proxy multi-model, health endpoint, abliterate resume, security-eval export, `--output json` |

---

## 5. Suggested implementation order

1. **Quick wins (1–2 days)**  
   - Update Command Reference (proxy, fix-ollama-template).  
   - Add `abliterate proxy` to README.  
   - Centralize `_is_gemma_checkpoint` in `model_family` and use it from modelfile + abliterate.  
   - Add subprocess timeout in `run_ollama_create` and GGUF convert.

2. **Short term (≈1 week)**  
   - Recipe: add `quant` / `quantize` keys; implement `validate-recipe` or `build --validate-only`.  
   - Shared `normalize_base_url` and optional `ollama_tools_to_hf` in a small util module.  
   - Tests: `test_run_helpers.py`, `test_modelfile.py` template derivation, recipe YAML.

3. **Medium term (2–4 weeks)**  
   - Recipe `hf_repo` in build.  
   - Auto-pick GGUF in fetch path with clear message.  
   - Proxy: health endpoint and optional Ollama ping on startup.  
   - Interactive mode for missing args (e.g. fetch).

4. **Longer term**  
   - Proxy multi-model registration.  
   - Abliterate resume from checkpoint.  
   - Unified `--output json` and security-eval export/retry.  
   - Ruff expansion and coverage in CI.

This document can be updated as items are completed or reprioritized.

---

## 6. Re-analysis (post-implementation)

After completing §§1–5, a full pass over the repo (cli, wiki, suggestions, SIMPLICITY_ANALYSIS, ROADMAP) surfaces the following.

### 6.1 Completed from original analysis

- **Optimizations:** All done (is_gemma_checkpoint, normalize_base_url, ollama_tools_to_hf, parallel download-lists, subprocess timeouts, tokenizer cache env).
- **Improvements:** Command Reference, README proxy line, doc parity (top_p/repeat_penalty), build --validate-only, recipe quant/quantize in RECIPE.md, print_actionable_error + next_steps across CLI, proxy 400 test, ping_ollama + optional proxy/security-eval check, ports 11435/11436 in wiki and --help, test_run_helpers, modelfile template tests, recipe YAML tests.
- **New features:** Auto-pick GGUF message, recipe hf_repo in build, proxy health GET / and GET /api/tags, proxy multi-model (--add-model name:path), security-eval retry (--retries) and export (CSV/JSON), pytest-cov in dev deps (Ruff B/C4/SIM deferred).

### 6.2 Remaining improvements (done)

| Item | Source | Description | Effort | Status |
|------|--------|-------------|--------|--------|
| **Unified --output json** | suggestions #13 | `check`, `doctor` support `--json` for machine-readable output. | M | Done |
| **README: HF auth one-liner** | SIMPLICITY §3 | One line under Fetch for gated/private repos (HF_TOKEN / huggingface-cli login). | S | Done |
| **README: Run from anywhere** | SIMPLICITY §4 | Clarify repo vs global (uv run vs pip/uv tool install). | S | Done |
| **Check Ollama before download** | SIMPLICITY §8 | Fetch and fetch-adapter already call `require_ollama()` before download. | — | N/A |
| **Golden-path integration tests** | suggestions #17 | plan quickstart, plan doctor-fix, auto --plan; check/json, doctor/json. | M | Done |
| **Error UX snapshot tests** | suggestions #18 | stderr contains "Next:" and "ollama-forge" / "Run: ollama-forge". | S | Done |
| **Ruff rule expansion** | §3.4 | B, C4, SIM enabled; fixes applied; E501 ignored. | M | Done |
| **Docs tracking** | suggestions #16 | docs/ no longer in .gitignore so product docs are versioned. | S | Done |

### 6.3 Remaining new features (done)

| Item | Source | Description | Effort | Status |
|------|--------|-------------|--------|--------|
| **Interactive mode for missing args** | suggestions #5, §3.1 | fetch / fetch-adapter prompt in TTY when repo/name/base missing. | M | Done |
| **Abliterate: resume from checkpoint** | §3.2 | `abliterate run --from-checkpoint DIR` skips compute/apply, resumes at GGUF step. | M | Done |
| **Adapter compatibility preflight** | suggestions #9, §3.1 | create-from-base verifies adapter dir and base path before create. | S/M | Done |
| **Proxy: config file for multi-model** | §3.2 (extend) | `abliterate proxy --config FILE` (YAML: models with name/checkpoint). | M | Done |
| **End-to-end train run** | suggestions #7 | `train-run` command: validate → prepare → finetune (if --base-gguf) → retrain. | L | Done |
| **Optional requests extra** | §1.3 | `ollama-forge[net]` with `requests`; documented in README. | S | Done |
| **Document proxy-only deps** | §1.3 | wiki/Abliterate.md: proxy-only needs transformers only (no torch). | S | Done |

### 6.4 Quick wins (done)

1. **README:** HF auth line and run-from-repo vs install-from-anywhere — done.
2. **check / doctor:** `--json` added; structured status for CI/scripting — done.
3. **Tests:** Error-message tests for "Next:" and "Run: ollama-forge" (build missing/invalid recipe) — done.

### 6.5 Summary table (re-analysis)

| Category | Status |
|----------|--------|
| **Improvements (§6.2)** | All done: --json for check/doctor, README HF/install, golden-path and error UX tests, Ruff B/C4/SIM, docs tracking |
| **New features (§6.3)** | All done: interactive mode, abliterate --from-checkpoint, adapter preflight, proxy --config, train-run e2e, [net] extra, proxy-only doc |

This re-analysis reflected the backlog as of the last update; §§6.2–6.4 items above have been implemented.

### 6.6 Optional next (backlog)

| Item | Source | Description | Effort | Status |
|------|--------|-------------|--------|--------|
| **Command Reference** | §2.1 | Keep wiki/Command-Reference.md in sync (train-run, convert-training-data-format, validate-recipe, security-eval). | S | Done |
| **Standalone validate-recipe** | suggestions #4 | Optional `validate-recipe recipe.yaml` in addition to `build --validate-only` for discoverability. | S | Done |
| **Proxy 400 test** | §2.2 | Test in test_abliterate_proxy: unknown model returns 400 with expected body. | S | Done |
| **Recipe quant/quantize keys** | §2.3, suggestions #15 | Document recipe keys for `quant` and `quantize` in RECIPE.md; add validate-recipe pointer. | S | Done |
| **Coverage gate in CI** | §3.4 | Run pytest-cov in CI and optionally fail on coverage drop (pytest-cov already in dev deps). | S | Done (.github/workflows/test.yml; add --cov-fail-under when baseline set) |
| **E501 line length** | §3.4 | Ruff currently ignores E501; optionally fix long lines incrementally. | M | Incremental fixes in cli, abliterate, abliterate_proxy; E501 still ignored in Ruff |
| **security-eval in Command Reference** | — | Add security-eval run/ui to wiki Command Reference if missing. | S | Done |
