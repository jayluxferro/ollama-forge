# Agent memory (ollama-forge)

Persistent context for AI assistants working on this repo. Update as we progress.

---

## Project

- **Name:** ollama-forge  
- **Purpose:** Create, retrain, ablate, and convert models for local Ollama. CLI + wiki docs.  
- **Key docs:** [PROJECT-ANALYSIS.md](PROJECT-ANALYSIS.md) (optimizations, improvements, features), [suggestions.md](suggestions.md), [ROADMAP.md](ROADMAP.md).

---

## Conventions

- **Skills:** Follow **Make No Mistakes** when editing: verify references, no guessing, double-check code and links.  
- **Tests:** New behavior gets tests; run `uv run pytest tests/` before committing.  
- **CLI errors:** Use `run_helpers.print_actionable_error` with `next_steps` where possible.  
- **Abliterate:** Proxy (port 11436) = tokenizer-only; serve (11435) = full model. Command Reference and README document proxy.

---

## Progress (todo-driven)

- **Done:** Centralized `_is_gemma_checkpoint` in `model_family.is_gemma_checkpoint`; modelfile and abliterate import it. Tests: `TestIsGemmaCheckpoint` in test_model_family.py.  
- **Done:** Added `http_util.normalize_base_url`; proxy and security_eval/client use it. Tests: `tests/test_http_util.py`.  
- **Done:** Subprocess timeouts: run_ollama_create (300s), run_ollama_show_modelfile (60s), GGUF convert and quantize (3600s); TimeoutExpired handled with actionable messages.  
- **Done:** chat_util.ollama_tools_to_hf shared; proxy + serve use it. Parallel fetch for download-lists (ThreadPoolExecutor). Proxy tokenizer cache size via OLLAMA_PROXY_TOKENIZER_CACHE_SIZE. Test for proxy 400 when model not registered (TestProxyUnknownModel).  
- **Done:** Doc parity: README + Fetch-and-Convert mention top_p/repeat_penalty; build --validate-only added (Recipes.md, tests in test_cli.py). docs/RECIPE.md: quant/quantize documented with CLI mirroring (fetch --quant, convert --quantize); command name fixed to ollama-forge; --validate-only and wiki link added.  
- **Done:** CLI failures now use print_actionable_error + next_steps: hf-cache (ls/rm), security-eval (run + ui), download-lists, abliterate (chat, proxy, serve, evaluate, optimize, compute-dir, run, fix-ollama-template), adapters search, validate-training-data, prepare-training-data, GGUF conversion/quantization in abliterate run.  
- **Done:** run_helpers.ping_ollama(base_url); optional startup check: proxy and security-eval run check Ollama reachable (--no-check-ollama to skip). Ports 11434/11435/11436 documented in wiki (Abliterate, Heretic-Integration, Command-Reference) and in --help for proxy/serve. Tests: test_run_helpers.py, test_load_recipe_yaml_*, template_from_hf_checkpoint_with_reason in test_modelfile.py.  
- **Done (features):** Fetch auto-pick message: "We auto-picked …; use --gguf-file to override." Recipe hf_repo in build (already supported). Proxy: GET / and GET /api/tags health (200 + models); multi-model via --add-model name:path. Security-eval: --retries (default 2); CSV/JSON export. pytest-cov in dev deps; Ruff E,F,I (B/C4/SIM deferred).  
- **Todo list:** See PROJECT-ANALYSIS.md §5 and Cursor todo list (optimizations → improvements → new features).

---

## User preferences

- Prefer hybrid approach for abliterated models: better GGUF conversion + lightweight proxy; tool support required.  
- Wants todo list covering all optimizations, improvements, new features; then work through it.  
- Save memory and update skills as we go based on interaction and instructions.

---

*Last updated: all PROJECT-ANALYSIS todos completed (opt, imp, feat); proxy multi-model, health, security-eval retry, pytest-cov.*
