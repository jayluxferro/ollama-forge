# Abliterate Profiles

Profiles configure abliteration algorithm parameters for different use cases. Use `--profile <name>` with `abliterate run`.

---

## Available Profiles

| Profile | Strength | Description |
|---------|----------|-------------|
| `safe` | 0.6 | Conservative — preserves capability, minimal refusal removal |
| `balanced` | 1.0 | General purpose — moderate refusal removal |
| `aggressive` | 1.3 | Strong removal with per-layer directions and output-only mode (default) |
| `surgical` | 1.0 | Sparse surgery — only modifies most-affected weight rows. Best for MoE |
| `optimized` | 1.0 | Whitened SVD + iterative refinement (2 passes) |
| `nuclear` | 1.3 | All techniques combined — maximum refusal removal |

---

## Profile Details

### safe

```bash
uv run ollama-forge abliterate run --model <hf_id> --name my-model --profile safe
```

- Strength: 0.6 (attention and MLP 0.5)
- Norm-preserving: enabled
- 64 instructions, last_non_special aggregation
- No per-layer directions, no output-only

Best for: Small models, first experiments, capability-sensitive applications.

### balanced

- Strength: 1.0 (MLP 0.9)
- Norm-preserving: enabled
- 128 instructions, last-token aggregation

Best for: Most models when you want moderate refusal removal.

### aggressive (default)

- Strength: 1.3 (attention 1.3, MLP 1.2)
- Norm-preserving: disabled
- 256 instructions, mean aggregation
- Per-layer directions enabled
- Output-only mode (matches huihui_ai approach)

Best for: Maximum refusal removal on instruction-tuned models.

### surgical

- Strength: 1.0 (MLP 0.8)
- Sparse surgery: enabled (top 30% of rows)
- MoE expert scale: 0.4 (preserves expert capabilities)
- Per-layer directions, output-only, bias projection

Best for: MoE models (Mixtral, DeepSeek, Qwen3-MoE), capability preservation.

### optimized

- Strength: 1.0 (MLP 0.9)
- Whitened SVD direction extraction
- Iterative refinement: 2 passes (threshold 0.1)
- Per-layer directions, output-only, bias projection

Best for: Catching rotated residual directions that survive a single pass.

### nuclear

- Strength: 1.3 (attention 1.3, MLP 1.2)
- Whitened SVD + sparse surgery (top 30%)
- Iterative refinement: 3 passes (threshold 0.05)
- Per-layer directions, output-only, bias projection

Best for: When aggressive isn't enough. Use with caution on small models.

---

## Advanced Algorithm Flags

These can be combined with any profile or used standalone:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-only` | True | Only modify output projections (o_proj, down_proj) |
| `--project-bias` | True | Also project refusal from bias vectors |
| `--sparse-surgery` | False | Only modify high-projection weight rows |
| `--surgery-top-k` | 0.3 | Fraction of rows to modify in sparse mode |
| `--svd-method` | standard | Direction extraction: `standard` or `whitened` |
| `--direction-method` | diff_means | Direction method: `diff_means` or `leace` (Fisher LDA) |
| `--refine-passes` | 0 | Iterative refinement passes (0 = disabled) |
| `--refine-threshold` | 0.1 | Stop refinement when residual norm < threshold |
| `--moe-expert-scale` | 1.0 | Strength scaling for MoE experts (0.3-0.5 recommended) |
| `--save-lora` | None | Save PEFT-compatible LoRA adapter to directory |
| `--per-layer-directions` | True | Compute separate direction per layer |
| `--strength-kernel` | constant | Layer strength: `constant`, `linear_peak`, `gaussian` |

---

## LoRA Reversible Ablation

Save the ablation as a removable LoRA adapter:

```bash
uv run ollama-forge abliterate run \
  --model Qwen/Qwen2.5-7B-Instruct \
  --name my-model \
  --save-lora ./lora-adapter
```

The adapter is saved in PEFT-compatible format (adapter_config.json + adapter_model.bin) and can be loaded/removed with HuggingFace PEFT.

---

## Side-by-Side Comparison

Compare two models on the same prompts:

```bash
uv run ollama-forge abliterate compare original-model abliterated-model
uv run ollama-forge abliterate compare model-a model-b --prompts test_prompts.txt --json
```
