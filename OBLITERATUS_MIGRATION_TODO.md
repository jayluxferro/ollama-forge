# OBLITERATUS Migration TODO

Scope: port `OBLITERATUS` capabilities into `ollama-forge` without telemetry, remote sync, or auto-contribution upload.

## Foundation

- [x] Add structured abliterate run reports, local contributions, aggregation, and benchmark wrapper
- [x] Add study preset registry for generic ablation experiments
- [x] Add YAML study config loader with preset expansion
- [x] Add reusable model preset registry for hardware-tier recommendations
- [x] Add shared reporting schema for study runs and analysis outputs

## Product Surfaces

- [x] Add guided interactive CLI for study setup and model selection
- [x] Add local UI for ablation/abliteration workflows
- [x] Add side-by-side model comparison workflows (abliterate compare)
- [x] Add richer export/report regeneration commands (markdown, HTML, JSON, CSV, LaTeX)
- [x] Add curated examples for study configs and benchmark recipes

## Generic Ablation Suite

- [x] Add model loader/handle abstraction for transformer studies
- [x] Add strategy registry
- [x] Add layer removal strategy
- [x] Add attention head pruning strategy
- [x] Add FFN ablation strategy
- [x] Add embedding ablation strategy
- [x] Add study runner from YAML config
- [x] Add plot outputs for study runs

## Evaluation

- [x] Add baseline evaluator abstraction (StudyEvaluator: perplexity, entropy, effective_rank, accuracy, f1)
- [x] Add advanced metrics module
- [x] Add benchmark catalog and baseline comparison helpers
- [x] Add plot generation for study and benchmark outputs (plot_impact + plot_heatmap)
- [x] Add lm-eval integration (study lm-eval command, build/run/save plan)
- [x] Add Heretic-style eval integration (evaluate_abliteration with refusal marker detection)

## Abliteration Pipeline Expansion

- [x] Expand method catalog toward `OBLITERATUS` parity (surgical, optimized, nuclear profiles)
- [x] Add informed pipeline scaffolding (recommend_abliterate_settings with new feature recommendations)
- [x] Add reusable stage/result objects (InformedPipelineResult, PipelineStage)
- [x] Add richer intermediate artifact capture (build_informed_run_artifact, pipeline exports)
- [x] Add community/local result envelopes without telemetry (save/load/aggregate contributions)

## Mechanistic Analysis Modules

- [x] Add analysis module registry
- [x] Add activation probing
- [x] Add causal tracing
- [x] Add logit lens
- [x] Add tuned lens hooks (analyze_tuned_lens with LayerNorm + KL convergence)
- [x] Add concept geometry analysis
- [x] Add cross-layer alignment analysis
- [x] Add cross-model transfer analysis
- [x] Add steering vector analysis
- [x] Add residual stream analysis
- [x] Add activation patching / causal patching
- [x] Add conditional abliteration analysis
- [x] Add defense robustness analysis
- [x] Add sparse surgery / SAE-oriented analysis (analyze_sparsity with Gini, concentration, recommended top_k)
- [x] Document Wasserstein / spectral / manifold as research-only (see OBLITERATUS_MIGRATION_NOTES.md)

## Core Algorithm Improvements

- [x] Add bias projection (project_bias=True, --project-bias / --no-project-bias)
- [x] Add whitened SVD direction extraction (--svd-method whitened)
- [x] Add iterative refinement (--refine-passes N, --refine-threshold T)
- [x] Add sparse surgery mode (--sparse-surgery --surgery-top-k)

## Reversible / Advanced Interventions

- [x] Add LoRA-based reversible ablation
- [x] Add Bayesian/grid optimizer for ablation parameters
- [x] Add architecture profile system
- [x] Add MoE-aware interventions (--moe-expert-scale, surgical profile with 0.4 scale)
- [x] Add reproducibility helpers and run manifests (enhanced with git hash, versions)

## Documentation

- [x] Document study presets and config format (examples/studies/README.md)
- [x] Document migration decisions and deviations (OBLITERATUS_MIGRATION_NOTES.md)
- [x] Document unsupported research-only modules (OBLITERATUS_MIGRATION_NOTES.md)
- [x] Add wiki pages for new study and analysis commands (Study-Framework.md, Abliterate-Profiles.md)

## Current Slice

- [x] Write migration backlog
- [x] Port study presets
- [x] Port study config loader
- [x] Add tests for presets/configs
- [x] Add strategy registry and basic study runner
- [x] Add transformer-backed study runtime and `study run`
- [x] Add study model presets, report inspection, and baseline analysis modules
- [x] Add guided study config generation, local study UI, and deeper baseline analysis modules
- [x] Add advanced metrics, benchmark catalog, strength optimization, and informed abliterate planning
