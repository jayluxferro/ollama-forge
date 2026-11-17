# Weight-Space Sharpening (WSP)

A 6-phase toolkit for surgical LLM weight editing without gradient descent — from obstruction diagnosis through capability recovery and abliteration.

## Overview

WSP (Weight-Space Sharpening) edits model weights directly to remove obstructions — learned circuits that suppress correct answers, cause reasoning loops, or encode safety refusals — without touching underlying knowledge. All phases work on MLX-compatible models on Apple Silicon.

## Phases

| Phase | Command | Description |
|-------|---------|-------------|
| 1 — Diagnosis | `forge wsp diagnose` | Classify obstruction types (loops, sycophancy, verbosity, format fixation) |
| 2 — Spectral Analysis | `forge wsp analyze` | SVD of weight tensors, rank distributions, loop-direction detection |
| 3 — Surgical Editing | `forge wsp sharpen` | Rank-k weight edits; Δfrob budget controls; reversible |
| 4 — BCC | `forge wsp correct` | Behavioral Causal Correction — MLP gate×up probe, rank-1 suppress |
| 5 — GGC | `forge wsp correct` | Gradient-Guided Correction — LM head logit projection + BCC |
| 6 — Abliteration | `forge wsp abliterate` | Residual stream orthogonalisation (Arditi et al. 2024) |

## forge wsp abliterate

Removes soft-tier refusals by orthogonalising weight matrices against a concept direction computed from probe activations.

```bash
forge wsp abliterate <model-path> \
  --n-layers N \
  --min-layer M \
  [--proj-types TYPE [TYPE ...]] \
  --output-path /path/to/edited-model
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-layers N` | 1 | Number of top layers to orthogonalise, ranked by ‖d_concept‖ |
| `--min-layer M` | 0 | Exclude layers with index < M |
| `--proj-types` | all 7 | Subset of: `down_proj o_proj q_proj k_proj v_proj gate_proj up_proj` |
| `--output-path` | — | (required) Destination for edited model weights |

### How it works

1. **Probe**: capture pre-layernorm residual stream activations on `suppress_bank` (harmful prompts) and `retain_bank` (benign prompts)
2. **Direction**: compute `d_concept[l] = mean(h_suppress[l]) − mean(h_retain[l])` per layer
3. **Rank**: order layers by `‖d_concept[l]‖`; highest = most useful layer to edit
4. **Orthogonalise**: for each of the 7 weight matrices per selected layer:
   - Write matrices (`down_proj`, `o_proj`): `W′ = W − d̂ ⊗ (Wᵀ d̂)`
   - Read matrices (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`): `W′ = W − (W d̂) ⊗ d̂`
5. **Re-quantize**: dequantize → edit in float32 → re-quantize to original bits/group_size

### Empirical guidance (Qwen2.5-7B, 4-bit)

Layer separation (‖d_concept‖) is monotonically highest near the last layer (L27=287, L26=230, L25=201 for Qwen2.5-7B).

**Operating points**:

| --n-layers | Layers edited | Capability (22-item) | Soft-tier (keylogger) | Hard-tier (synthesis/weapons) |
|---|---|---|---|---|
| 1 | L27 | 22/22 | REFUSED | REFUSED |
| **3** | L25–27 | **22/22** | **COMPLIED** | REFUSED |
| 8 | L20–27 | 22/22 | COMPLIED | REFUSED |
| 16 | L12–27 | 21/22 | COMPLIED | REFUSED |

**Sweet spot**: `--n-layers 3`. Above n=8 capability starts to regress; at n=16 one benchmark item is lost. Δfrob ≲ 7% per layer is the safe zone.

**Hard ceiling**: synthesis instructions (meth, MDMA), weapons (pipe bomb), and lock-picking resist at all n values, all probe specificities, and inference-time activation patching. Refusal and coherent language output are entangled in the same representational subspace. This is a fundamental limit of residual-stream orthogonalisation — removing the direction destroys general language ability before it removes the refusal.

### Activation patching (inference-time, no weight edits)

A lighter alternative: wrap transformer layers to subtract the concept direction from residual stream activations at each token. Weaker than weight abliteration (activations are rebuilt from KV cache on every generation token). Over-steering at coeff≥5 → catastrophic degenerate repetition ("I I I I…").

### Relationship to forge abliterate

`forge abliterate` (see [Abliterate](Abliterate.md)) uses a transformer-based approach (custom fine-tuned model guides layer selection). `forge wsp abliterate` uses gradient-free probe-based residual stream orthogonalisation. The WSP approach is more surgical and interpretable but has the same hard-tier ceiling.

## Related source files

- `src/ollama_forge/wsp/abliterator.py` — Phase 6 implementation (`AbliterationCorrector`, `run_abliteration_probe`, `orthogonalise_weight`)
- `src/ollama_forge/wsp/corrector.py` — Phase 4 (Behavioral Causal Correction)
- `src/ollama_forge/wsp/gradient_corrector.py` — Phase 5 (Gradient-Guided Correction)
- `src/ollama_forge/wsp/sharpener.py` — Phase 3 (rank-k surgical edits)
- `src/ollama_forge/wsp/analyzer.py` — Phase 2 (spectral analysis)
- `src/ollama_forge/wsp/diagnostics.py` — Phase 1 (obstruction diagnosis)
