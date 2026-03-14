# OBLITERATUS Migration: Decisions and Deviations

## Overview

This document records key decisions made when porting features from
[OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) into ollama-forge.

## What Was Ported (and How)

### Core Algorithm Improvements
- **Whitened SVD**: Ported as `--svd-method whitened`. Uses Cholesky decomposition of
  harmless covariance for whitening before SVD. Same math as OBLITERATUS.
- **Bias projection**: Ported as `--project-bias`. Projects refusal from bias vectors
  alongside weight matrices. Default enabled.
- **Iterative refinement**: Ported as `--refine-passes N`. Re-probes model for residual
  refusal direction after each pass. Simpler than OBLITERATUS's Ouroboros compensation
  (no automatic threshold adaptation), but effective.
- **Sparse surgery**: Ported as `--sparse-surgery --surgery-top-k`. Row-level masking
  based on projection magnitude. Same concept as OBLITERATUS's SparseDirectionSurgeon.

### Analysis Modules
All 15 OBLITERATUS analysis modules were ported or have equivalents:
- activation_probe, cross_layer_similarity, logit_lens, tuned_lens, residual_stream
- causal_tracing, conditional_similarity, activation_patching, causal_patching
- steering_vectors, concept_geometry, architecture_profile, defense_robustness
- cross_model_transfer, sparsity_analysis

### Study Framework
- Strategy registry with 4 strategies (layer_removal, head_pruning, ffn_ablation,
  embedding_ablation) matches OBLITERATUS.
- YAML config system, preset registry, interactive wizard.
- Model presets organized by compute tier.

### Informed Pipeline
- Analysis-driven recommendations with automatic feature selection.
- Recommends whitened SVD, refinement, sparse surgery based on analysis results.

## Key Deviations from OBLITERATUS

### 1. No Telemetry / Community Hub Sync
OBLITERATUS has anonymous telemetry with HuggingFace Hub sync for a community
leaderboard. We have local-only contribution storage with no remote sync.
**Reason**: Privacy-first approach. Users can share results manually.

### 2. No Gradio Web UI (Streamlit instead)
OBLITERATUS uses Gradio with 7 tabs. We use a simpler Streamlit UI.
**Reason**: Streamlit is lighter-weight and already a dependency pattern in the project.

### 3. Simplified Informed Pipeline
OBLITERATUS has a 7-stage closed-loop `InformedAbliterationPipeline` class with
Ouroboros compensation. We have a recommendation engine + CLI pipeline.
**Reason**: Our pipeline is CLI-first. The recommendation engine provides the same
intelligence without requiring a monolithic class.

### 4. No Bayesian Optimization (Bell-Curve Kernel)
OBLITERATUS uses Optuna TPE with a parametric bell-curve kernel. We have grid-based
strength optimization in `study_optimize.py`.
**Reason**: Optuna is a heavy optional dependency. Grid search is simpler and sufficient
for the 1-2 parameter sweeps typical in abliteration.

### 5. LoRA Adapter Computation from Checkpoint (Not In-Place)
OBLITERATUS computes LoRA adapters during the ablation pass. We compute them
post-ablation from the checkpoint.
**Reason**: Keeps the core ablation code simpler. The mathematical result is identical.

### 6. GGUF/Ollama Integration (Our Unique Feature)
OBLITERATUS saves HuggingFace checkpoints only. We have the full pipeline:
HF checkpoint -> GGUF conversion -> quantization -> Ollama Modelfile -> `ollama create`.
This is our primary value proposition and has no OBLITERATUS equivalent.

## Research-Only Modules (Not Ported)

These OBLITERATUS modules were not ported due to limited practical value or
heavy dependencies:

- **Wasserstein optimal transport**: Requires scipy/POT. Marginal benefit over standard SVD.
- **Riemannian manifold geometry**: Research curiosity, not actionable for abliteration.
- **Spectral certification**: DCT-based frequency decomposition. Novel but unproven.
- **SAE (Sparse Autoencoder) features**: Requires pre-trained SAE models (separate training).

These can be added later if there's user demand.
