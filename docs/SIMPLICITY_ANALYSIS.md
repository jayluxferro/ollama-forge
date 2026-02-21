# Simplicity analysis — what’s left to make the tool really simple

This doc summarizes friction points and concrete improvements so ollama-tools stays “really simple to use” for non-experts.

---

## Already in good shape

- **One-command fetch:** `fetch <repo_id> --name <name>` for HF GGUF → Ollama.
- **One-command fetch-adapter:** `fetch-adapter <repo_id> --base <base> --name <name>`.
- **Clear README:** Simplest workflows first, no internal doc links, table of tools.
- **GETTING_STARTED:** Goal-based sections (I want X → do Y).
- **Example recipes:** `examples/recipes/` with README; build works with JSON without extras.
- **Help:** Subcommands and `--help` are clear.

---

## 1. Multiple GGUF files → still two steps for many repos

**Issue:** Repos like TheBloke/Llama-2-7B-GGUF have many `.gguf` files. We error and ask for `--gguf-file <filename>`. So the “one command” often becomes two (run once, see list, run with `--gguf-file`).

**Improvement:** When there are multiple GGUF files, **auto-pick one** by a simple policy, e.g.:

- Prefer a file containing `Q4_K_M` or `q4_k_m` (common default recommendation).
- Else prefer smallest file size (by name heuristics or by listing).
- Document: “We auto-picked `<file>`; use `--gguf-file <other>` to choose another.”

**Impact:** High — “one command” works for most popular GGUF repos without the user knowing filenames.

---

## 2. Recipe can’t say “from Hugging Face”

**Issue:** Recipes only support `base` (Ollama name/path) or `gguf` (local path). To use a model from HF you must run `fetch` then `build`, or remember `fetch` flags.

**Improvement:** Extend recipe format with optional **`hf_repo`** (and optional **`gguf_file`**):

- If `hf_repo` is set, `build` runs the same logic as `fetch` (download GGUF, then convert) then uses `name` / `system` / `temperature` / `num_ctx` from the recipe.
- If both `gguf` and `hf_repo` exist, reject or prefer one (e.g. prefer `hf_repo` if we want “HF first”).

**Impact:** Medium — “one file, one command” then covers “I want this HF model” without learning the `fetch` CLI.

---

## 3. Hugging Face auth (gated/private models)

**Issue:** Gated or private HF repos need login. `huggingface_hub` uses `HF_TOKEN` or `huggingface-cli login`, but we never mention it.

**Improvement:** One line in README and GETTING_STARTED (e.g. in the fetch section): “For gated or private models, set `HF_TOKEN` or run `huggingface-cli login`.”

**Impact:** Medium — avoids confusing “401” or “not found” when the repo is gated.

---

## 4. Running the CLI from anywhere

**Issue:** Docs only show `uv run ollama-tools` from the repo. Users who don’t clone the repo (e.g. after `pip install ollama-tools`) or want a global command don’t see how.

**Improvement:**

- In README “Setup”: add “From this repo: `uv run ollama-tools`. To use from anywhere: `uv tool install /path/to/ollama-tools` or `pip install -e .` then run `ollama-tools`.”
- Or, once on PyPI: “`pip install ollama-tools` or `uv tool install ollama-tools`.”

**Impact:** Medium — lowers friction for “I just want the command on my PATH.”

---

## 5. YAML recipe requires an extra

**Issue:** `build recipe.yaml` needs YAML support; we say “uv sync --extra recipe” but new users may run `build` first and see “PyYAML required”.

**Options:**

- **A:** Add `pyyaml` to **default** dependencies so `uv sync` is enough for YAML recipes.
- **B:** Keep it optional but make the first recipe example in GETTING_STARTED use JSON (so no extra needed), and say “For YAML, run `uv sync --extra recipe`.”

**Impact:** Low–medium — either one command less to remember (A) or fewer “why doesn’t build work?” moments (B).

---

## 6. Main help: “start here”

**Issue:** `ollama-tools --help` lists all commands but doesn’t point newcomers to the simplest path.

**Improvement:** Add an epilog (or a short line in the description) to the main parser, e.g. “Quick start: ollama-tools fetch <HF_REPO> --name my-model”.

**Impact:** Low — helps the unsure user pick the right first command.

---

## 7. Example repo in README

**Issue:** README uses `TheBloke/Llama-2-7B-GGUF`; that repo has many GGUF files, so the copy-paste example may hit “Multiple .gguf files found” until we auto-pick (see §1).

**Improvement:** Either (a) switch the example to a repo that has a single GGUF, or (b) implement auto-pick and keep the example; then the example “just works.”

**Impact:** Low if we fix §1; otherwise medium (avoid first-run failure).

---

## 8. Check Ollama before downloading (fetch / fetch-adapter)

**Issue:** We download the GGUF (or adapter) and then run `ollama create`. If `ollama` isn’t installed, we fail after a potentially large download.

**Improvement:** At the start of `fetch` and `fetch-adapter`, check that `ollama` is on PATH (e.g. `shutil.which("ollama")`). If not, exit with “Install Ollama and ensure it is on PATH” before downloading.

**Impact:** Low–medium — better UX and fewer wasted downloads.

---

## 9. Adapter format expectations (fetch-adapter)

**Issue:** Ollama’s Modelfile `ADAPTER` may expect a specific format (e.g. llama.cpp adapter). HF adapter repos are often PEFT/safetensors. If they’re not compatible, users get a confusing failure at `ollama create`.

**Improvement:** In GETTING_STARTED (and maybe ADAPTER.md): state that “fetch-adapter works with adapter repos in the format Ollama expects (e.g. …). If your HF repo is PEFT-only, you may need to export to that format first.” (Exact wording after checking Ollama docs.)

**Impact:** Medium if we get support questions; low if most HF adapters are already in the right format.

---

## 10. One-line “quick start” at top of README

**Issue:** README has “Setup” then “Simplest workflows”. Someone in a hurry might miss the single best path.

**Improvement:** Add a one-line **Quick start** at the very top (e.g. under the tagline): “Quick start: `uv sync && uv run ollama-tools fetch TheBloke/… --name my-model && ollama run my-model`” (with a real repo that works with current behavior or with §1).

**Impact:** Low — makes the “absolute minimum” obvious.

---

## Suggested order of work

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Auto-pick one GGUF when multiple (§1) | Small | High |
| 2 | HF auth one-liner in docs (§3) | Tiny | Medium |
| 3 | Check Ollama before download (§8) | Tiny | Medium |
| 4 | Recipe `hf_repo` support (§2) | Medium | Medium |
| 5 | “Start here” in main --help (§6) | Tiny | Low |
| 6 | Run from anywhere / install note (§4) | Tiny | Medium |
| 7 | YAML default or JSON-first example (§5) | Small | Low–Medium |
| 8 | Example repo or note (§7) | Tiny | Low (if §1 done) |
| 9 | Quick start line in README (§10) | Tiny | Low |
| 10 | Adapter format note (§9) | Small | Medium (if needed) |

---

## Summary

The tool is already simple for the main flows (fetch, convert, create-from-base, retrain, build). The changes that would make it **really** simple with minimal effort are:

1. **Auto-pick one GGUF** when a repo has multiple files, so the documented “one command” works for most HF GGUF repos.
2. **Document HF auth** and **check for Ollama** before downloading, so users don’t hit auth or “ollama not found” after a long download.
3. **Recipe `hf_repo`** so “one file, one command” covers “from Hugging Face” without learning the fetch CLI.
4. **Install/run-from-anywhere** and a **“start here”** hint in `--help` so the first run is obvious.

Everything else above is incremental polish.
