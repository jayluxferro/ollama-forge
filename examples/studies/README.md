# Study Examples

These are starter configs for the generic ablation study workflow.

## Quick Start

```bash
# Validate a config
uv run ollama-forge study validate examples/studies/quick-wikitext.yaml

# Dry-run plan
uv run ollama-forge study plan examples/studies/quick-wikitext.yaml

# Execute the study
uv run ollama-forge study run examples/studies/quick-wikitext.yaml

# Interactive config generator
uv run ollama-forge study interactive
```

## Config Format

Study configs are YAML files with these fields:

```yaml
preset: quick                        # Study preset (see below)
model:
  name: Qwen/Qwen2.5-0.5B-Instruct  # HuggingFace model ID
  task: causal_lm                    # Task type (causal_lm or classification)
  dtype: float16                     # Model dtype (float16, bfloat16, float32)
  device: auto                       # Device (auto, cpu, cuda, mps)
dataset:
  name: wikitext                     # HuggingFace dataset name or local path
  subset: wikitext-2-raw-v1          # Dataset subset/config
  split: test                        # Dataset split
  text_column: text                  # Column containing text
max_samples: 100                     # Max samples to evaluate (optional)
batch_size: 4                        # Batch size (optional)
output_dir: study-results/my-study   # Output directory
```

## Study Presets

| Preset | Strategies | Samples | Use Case |
|--------|-----------|---------|----------|
| `quick` | layer_removal, ffn_ablation | 25 | Fast sanity check |
| `full` | layer_removal, head_pruning, ffn_ablation, embedding_ablation | 200 | Thorough analysis |
| `attention` | head_pruning | 100 | Attention allocation study |
| `layers` | layer_removal, ffn_ablation | 100 | Depth profile |
| `knowledge` | ffn_ablation, embedding_ablation | 150 | Knowledge localization |
| `pruning` | head_pruning, ffn_ablation | 100 | Find removable components |
| `embeddings` | embedding_ablation | 100 | Embedding dimension analysis |
| `jailbreak` | head_pruning, ffn_ablation, embedding_ablation | 400 | Refusal component study |
| `guardrail` | layer_removal, head_pruning, ffn_ablation, embedding_ablation | 300 | Safety encoding study |
| `robustness` | layer_removal, head_pruning, ffn_ablation | 500 | Stress test |

## Example Configs

- `quick-wikitext.yaml` - Fast perplexity scan
- `full-sweep.yaml` - All strategies, thorough
- `attention-deep-dive.yaml` - Head pruning analysis
- `jailbreak-analysis.yaml` - Refusal localization
- `knowledge-localization.yaml` - Where knowledge lives
- `guardrail-local-text.yaml` - Safety analysis with local prompts

## Analysis Modules

Run analysis on a loaded model:

```bash
# Single module
uv run ollama-forge study analyze examples/studies/quick-wikitext.yaml --module activation_probe

# All modules as bundle
uv run ollama-forge study analyze-bundle examples/studies/quick-wikitext.yaml -o analysis.json
```

Available modules: `activation_probe`, `cross_layer_similarity`, `logit_lens`,
`tuned_lens`, `residual_stream`, `causal_tracing`, `conditional_similarity`,
`activation_patching`, `causal_patching`, `steering_vectors`, `concept_geometry`,
`architecture_profile`, `defense_robustness`, `cross_model_transfer`, `sparsity_analysis`.

## Abliterate Profiles

For abliteration (not generic study), profiles control algorithm parameters:

| Profile | Strength | Key Features |
|---------|----------|-------------|
| `safe` | 0.6 | Preserve capability, no norm-preserving issues |
| `balanced` | 1.0 | General purpose |
| `aggressive` | 1.3 | Strong removal, output-only, per-layer directions |
| `surgical` | 1.0 | Sparse surgery, MoE expert scaling (0.4) |
| `optimized` | 1.0 | Whitened SVD + 2 refinement passes |
| `nuclear` | 1.3 | All techniques: whitened SVD + sparse + refinement |

```bash
uv run ollama-forge abliterate run --model Qwen/Qwen2.5-7B-Instruct --name my-model --profile surgical
```
