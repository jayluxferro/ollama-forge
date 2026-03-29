# Study Framework (Generic Ablation Suite)

Systematic ablation studies for understanding how LLM capabilities are distributed across layers, attention heads, FFN blocks, and embedding dimensions.

---

## Quick Start

```bash
# Install study dependencies
uv sync

# Interactive setup (detects hardware, suggests models)
uv run ollama-forge study interactive

# Or use a preset config
uv run ollama-forge study run examples/studies/quick-wikitext.yaml
```

---

## Study Presets

| Preset | Strategies | Samples | Use Case |
|--------|-----------|---------|----------|
| `quick` | layer_removal, ffn_ablation | 25 | Fast sanity check |
| `full` | all 4 strategies | 200 | Thorough analysis |
| `attention` | head_pruning | 100 | Attention allocation |
| `layers` | layer_removal, ffn_ablation | 100 | Depth profile |
| `knowledge` | ffn_ablation, embedding_ablation | 150 | Knowledge localization |
| `pruning` | head_pruning, ffn_ablation | 100 | Compression candidates |
| `jailbreak` | head_pruning, ffn_ablation, embedding_ablation | 400 | Refusal localization |
| `guardrail` | all 4 strategies | 300 | Safety encoding |
| `robustness` | layer_removal, head_pruning, ffn_ablation | 500 | Stress test |

```bash
uv run ollama-forge study presets     # List all presets
uv run ollama-forge study models      # List curated model presets by tier
```

---

## YAML Config Format

```yaml
preset: quick                        # Study preset name
model:
  name: Qwen/Qwen2.5-0.5B-Instruct  # HuggingFace model ID
  task: causal_lm                    # causal_lm or classification
  dtype: float16                     # float16, bfloat16, float32
  device: auto                       # auto, cpu, cuda, mps
dataset:
  name: wikitext                     # HF dataset or local path
  subset: wikitext-2-raw-v1          # Dataset config/subset
  split: test                        # Dataset split
  text_column: text                  # Text column name
max_samples: 100                     # Max evaluation samples
batch_size: 4                        # Batch size
output_dir: study-results/my-study   # Output directory
```

---

## Ablation Strategies

| Strategy | What It Does |
|----------|-------------|
| `layer_removal` | Zeros entire transformer layers (soft removal) |
| `head_pruning` | Zeros Q/K/V/O weights for specific attention heads |
| `ffn_ablation` | Zeros FFN/MLP blocks per layer |
| `embedding_ablation` | Zeros contiguous ranges of embedding dimensions |

---

## Analysis Modules

Run mechanistic analysis on a model:

```bash
# Single module
uv run ollama-forge study analyze config.yaml --module activation_probe

# Bundle all modules
uv run ollama-forge study analyze-bundle config.yaml -o analysis.json
```

| Module | Purpose |
|--------|---------|
| `activation_probe` | Per-layer activation statistics and top layers by norm |
| `cross_layer_similarity` | Cosine similarity matrix between layer directions |
| `logit_lens` | Project hidden states through unembedding (raw) |
| `tuned_lens` | Project through LayerNorm + unembedding (normalized) |
| `residual_stream` | Track residual stream norm changes across layers |
| `causal_tracing` | Layer knockout KL divergence |
| `conditional_similarity` | Group-conditional activation similarity |
| `activation_patching` | Cross-group activation transplant effects |
| `causal_patching` | Prompt-level causal intervention |
| `steering_vectors` | Contrastive group direction strength per layer |
| `concept_geometry` | Polyhedral cone detection (is refusal multi-directional?) |
| `architecture_profile` | Dense/MoE/Reasoning classification |
| `defense_robustness` | Self-repair risk and safety-capability entanglement |
| `cross_model_transfer` | Direction universality between two models |
| `sparsity_analysis` | Gini coefficient and concentration of refusal signal |

---

## Benchmarks

```bash
# Run a benchmark preset
uv run ollama-forge study benchmark run --preset quick --model my-model

# Plan an lm-eval command
uv run ollama-forge study lm-eval --model hf --tasks hellaswag,mmlu --plan
```

---

## Strength Optimization

```bash
uv run ollama-forge study optimize config.yaml --strengths 0.25,0.5,0.75,1.0
```

Grid search over ablation strengths to find the sweet spot between capability preservation and component importance.

---

## Reports

```bash
# View a study report
uv run ollama-forge study report study-results/study-results.json

# Export to different formats
uv run ollama-forge study report results.json --export report.html
uv run ollama-forge study report results.json --export report.md

# Compare two study runs
uv run ollama-forge study compare run_a.json run_b.json
```

---

## Model Presets

116 curated models organized by compute tier:

| Tier | VRAM | Examples |
|------|------|---------|
| `tiny` | <1GB | distilgpt2, TinyLlama, Qwen2.5-0.5B |
| `small` | ~4GB | Qwen2.5-3B, Phi-3.5, Yi-1.5-6B |
| `medium` | ~8-16GB | Qwen2.5-7B, LLaMA-3.1-8B, Mistral-7B |
| `large` | 24GB+ | Qwen2.5-14B, Qwen3-32B, LLaMA-3.1-70B |
| `frontier` | 100GB+ | Qwen3-235B, DeepSeek-V3, GLM-5 |

```bash
uv run ollama-forge study models --tier small
```
