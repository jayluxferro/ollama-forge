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
- [ ] Add side-by-side model comparison workflows
- [ ] Add richer export/report regeneration commands
- [ ] Add curated examples for study configs and benchmark recipes

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

- [ ] Add baseline evaluator abstraction
- [x] Add advanced metrics module
- [x] Add benchmark catalog and baseline comparison helpers
- [ ] Add plot generation for study and benchmark outputs
- [ ] Add lm-eval integration where feasible
- [ ] Add Heretic-style eval integration where feasible

## Abliteration Pipeline Expansion

- [ ] Expand method catalog toward `OBLITERATUS` parity
- [ ] Add informed pipeline scaffolding
- [ ] Add reusable stage/result objects
- [ ] Add richer intermediate artifact capture
- [ ] Add community/local result envelopes without telemetry

## Mechanistic Analysis Modules

- [x] Add analysis module registry
- [x] Add activation probing
- [ ] Add causal tracing
- [x] Add causal tracing
- [x] Add logit lens
- [ ] Add tuned lens hooks where feasible
- [ ] Add concept geometry analysis
- [x] Add cross-layer alignment analysis
- [ ] Add cross-model transfer analysis
- [ ] Add steering vector analysis
- [x] Add residual stream analysis
- [ ] Add activation patching / causal patching
- [ ] Add conditional abliteration analysis
- [x] Add conditional abliteration analysis
- [ ] Add defense robustness analysis
- [ ] Add sparse surgery / SAE-oriented analysis where feasible
- [ ] Add Wasserstein / spectral / manifold modules where justified

## Reversible / Advanced Interventions

- [ ] Add LoRA-based reversible ablation
- [x] Add Bayesian/grid optimizer for ablation parameters
- [ ] Add architecture profile system
- [ ] Add MoE-aware interventions where architectures allow it
- [ ] Add reproducibility helpers and run manifests

## Documentation

- [ ] Document study presets and config format
- [ ] Document migration decisions and deviations from `OBLITERATUS`
- [ ] Document unsupported research-only modules and rationale
- [ ] Add wiki pages for new study and analysis commands

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
