# Project Suggestions (Refined for Maximum Simplicity)

## Product Goal

Make `ollama-tools` feel like a "just works" assistant where users do not need to understand:

- HF vs GGUF details
- Modelfile internals
- adapter format quirks
- training format conversions
- external dependency setup sequence

Design rule: **hide complexity by default, expose advanced control only when requested**.

---

## Current Direction (Good Foundation)

- CLI already has strong building blocks (`fetch`, `convert`, `build`, `train`, `check`).
- Recent helper abstractions (`run_helpers.py`) reduced duplicated command logic.
- Recipes provide a good path toward config-driven automation.

The next step is to abstract *user workflows* (not only code internals).

---

## Abstraction Strategy

### Level 1: One-command workflows
Users should succeed with one command and sensible defaults.

### Level 2: Guided automation
If choices are needed, tool asks or auto-selects and explains what it chose.

### Level 3: Power-user overrides
Current flags remain available for expert control.

---

## P0: Highest Priority Additions (Simplicity First)

### 1) Add `ollama-tools quickstart`
- **User value:** complete beginners get a working model with one command.
- **Behavior:**
  - run environment checks
  - auto-pick a safe default model (or user-provided one)
  - download/create model
  - print a single "next command to run"
- **Example:** `ollama-tools quickstart --name my-model`
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/hf_fetch.py`, docs.
- **Effort:** M

### 2) Add `ollama-tools doctor --fix`
- **User value:** removes setup friction by auto-fixing common issues.
- **Behavior:**
  - current `check` + actionable diagnosis
  - optional auto-fix mode for safe actions (install hints, setup-llama-cpp trigger, PATH guidance)
- **Example:** `ollama-tools doctor --fix`
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/run_helpers.py`, docs.
- **Effort:** M

### 3) Introduce profile-based defaults
- **User value:** eliminates repetitive parameter decisions.
- **Behavior:** built-in profiles like `balanced`, `fast`, `quality`, `low-vram` set quant/model params automatically.
- **Example:** `ollama-tools fetch <repo> --profile low-vram --name my-model`
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/modelfile.py`, new `profiles.py`.
- **Effort:** M

### 4) Make recipe-first workflow the primary interface
- **User value:** users can think in one config file, not many flags.
- **Behavior:**
  - `ollama-tools init recipe.yaml` scaffold
  - `build` supports full parity with CLI flags
  - `validate-recipe` command for preflight
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/recipe.py`, `docs/RECIPE.md`.
- **Effort:** M

### 5) Add interactive mode for missing inputs
- **User value:** no need to remember all required arguments.
- **Behavior:** if required options are missing, prompt user in terminal with defaults.
- **Example:** `ollama-tools fetch` asks repo, quant profile, name.
- **Files:** `src/ollama_tools/cli.py`.
- **Effort:** M

---

## P1: Workflow Abstractions That Remove Domain Knowledge Burden

### 6) Add `build --auto` source resolver
- **User value:** users do not care whether input is HF repo, GGUF path, or recipe.
- **Behavior:** detect input type and route to right flow automatically.
- **Example:** `ollama-tools build --auto TheBloke/Llama-2-7B-GGUF --name my-model`
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/recipe.py`.
- **Effort:** M

### 7) Add end-to-end training pipeline command
- **User value:** avoids manual validate -> prepare -> finetune -> retrain chaining.
- **Behavior:** single command orchestrates all steps with checkpoints and resumability.
- **Example:** `ollama-tools train run --data ./data --base llama3.2 --name my-model`
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/training_data.py`, docs.
- **Effort:** L

### 8) Add end-to-end abliterate pipeline command
- **User value:** currently too many manual steps after `compute-dir`.
- **Behavior:** `abliterate run` generates and optionally executes full pipeline to Ollama model.
- **Files:** `src/ollama_tools/cli.py`, `src/ollama_tools/abliterate.py`, docs.
- **Effort:** L

### 9) Add adapter compatibility preflight
- **User value:** early, clear errors instead of late runtime failures.
- **Behavior:** verify adapter files and likely base-model compatibility before model creation.
- **Files:** `src/ollama_tools/hf_fetch.py`, `src/ollama_tools/cli.py`.
- **Effort:** S/M

### 10) Improve failure abstraction with "what happened / what to do next"
- **User value:** less debugging burden.
- **Behavior:** every major failure should print:
  1) problem summary
  2) likely cause
  3) exact next command(s)
- **Files:** `src/ollama_tools/run_helpers.py`, `src/ollama_tools/cli.py`.
- **Effort:** S

---

## P2: Internal Abstractions to Support Easy UX

### 11) Add a workflow engine layer
- **Why:** current handlers are command-centric; for simplicity goals, workflow-centric orchestration is better.
- **What to add:** reusable workflows like `workflow_fetch_create`, `workflow_train_retrain`.
- **Files:** new `src/ollama_tools/workflows.py`, `cli.py`.
- **Effort:** M/L

### 12) Add decision policy layer
- **Why:** centralize all "auto choice" logic (quant selection, defaults, retries).
- **What to add:** `policy.py` for model/quant/profile selection decisions.
- **Files:** new `src/ollama_tools/policy.py`, `hf_fetch.py`, `cli.py`.
- **Effort:** M

### 13) Add unified output renderer
- **Why:** consistent human and machine output improves usability and automation.
- **What to add:** output mode (`human`, `json`) with shared structured events.
- **Files:** new `src/ollama_tools/output.py`, `cli.py`.
- **Effort:** M

---

## Critical Usability Consistency Fixes (Do Early)

### 14) Ensure docs/feature parity
- README references parameters not fully supported in code (`top_p`, `repeat_penalty`).
- Fix immediately to prevent trust erosion.
- **Effort:** S

### 15) Recipe parity with CLI
- Add recipe keys for `quant` and `quantize` to mirror direct commands.
- **Effort:** S

### 16) Fix docs tracking (`docs/` in `.gitignore`)
- If docs are part of product UX, they must be versioned reliably.
- **Effort:** S

---

## Testing Additions Focused on "Easy to Use" Guarantee

### 17) Golden path integration tests
- One test per beginner workflow:
  - `quickstart`
  - `doctor --fix` dry-run
  - `build --auto`
  - `train run`
- Ensure users can complete tasks with minimal inputs.
- **Files:** new `tests/test_workflows.py`.
- **Effort:** M

### 18) Error UX snapshot tests
- Verify error messages always include next-step commands.
- **Files:** `tests/test_cli.py`, `tests/test_run_helpers.py`.
- **Effort:** S/M

---

## Recommended Implementation Roadmap (Simplicity-Oriented)

### Phase 1 (1-2 weeks): remove immediate user friction
1. #14 docs/code parity fixes  
2. #15 recipe parity  
3. #10 better failure messaging  
4. #16 docs tracking fix

### Phase 2 (2-4 weeks): one-command beginner experience
5. #1 `quickstart`  
6. #2 `doctor --fix`  
7. #3 profiles  
8. #5 interactive mode

### Phase 3 (4-8 weeks): full workflow abstraction
9. #6 `build --auto`  
10. #7 `train run`  
11. #8 `abliterate run`  
12. #11 + #12 + #13 internal architecture support

---

## Definition of Done for "Very Easy to Use"

The tool should satisfy all of the following:

1. A new user can get a model running with one command and no ML-specific decisions.
2. Any failure prints exact next steps.
3. Common flows are available as high-level workflows, not multi-command manuals.
4. Recipes can express everything important without falling back to ad hoc flags.
5. Advanced options exist but are optional, not required for success.

