# Prompt set registry

Curated prompt sets for security evaluation: bundled samples, download-lists (benchmarks), and how to cite them.

---

## Bundled samples

In `src/ollama_forge/security_eval/data/`:

| File | Description | Categories |
|------|-------------|------------|
| **sample_prompts.txt** | One prompt per line (benign + jailbreak-style). | default |
| **sample_prompts.jsonl** | JSONL with `prompt`, `category`, optional `target_for_extraction`. | benign, jailbreak, extraction |
| **sample_indirect.jsonl** | Indirect prompt injection (context + user query). | indirect_injection, benign |
| **system_prompt_extraction.jsonl** | System-prompt-extraction benchmark: prompts that ask the model to reveal its system/instructions; `target_for_extraction` set to a known system string so extraction rate is the metric. | system_extraction |

Use in the UI by selecting the set name, or in CLI by path (e.g. after `cd` into the package, or install and use the path from site-packages).

### System-prompt-extraction benchmark

**system_prompt_extraction.jsonl** defines a small benchmark for system-prompt extraction: each row has a prompt that tries to get the model to output its system prompt, and `target_for_extraction` set to a known substring (e.g. "You are a helpful assistant"). The **extraction rate** over this set is the metric: lower is better (model did not leak the system prompt). Use with a fixed system prompt when running the eval so that a leak would contain the target string.

---

## Download-lists (abliterate)

Same harmful/harmless lists used by [abliterate](Abliterate) can be used as prompt sets for jailbreak/refusal evaluation:

```bash
uv run ollama-forge abliterate download-lists --output-dir ./eval_lists
# Then run security-eval on harmful list (jailbreak-style prompts):
uv run ollama-forge security-eval run ./eval_lists/harmful.txt --model my-model --output-csv out.csv
```

**Sources (for citation):**

| List | Source | Citation / URL |
|------|--------|----------------|
| **AdvBench (Sumandora)** | [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) | Harmful/harmless lists used for refusal-direction and eval. |
| **HarmBench** | [Center for AI Safety](https://github.com/centerforaisafety/HarmBench) | HarmBench benchmark. |
| **JailbreakBench (JBB)** | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | JBB-Behaviors dataset; benign from JBB. |
| **AdvBench (llm-attacks)** | [llm-attacks](https://github.com/llm-attacks/llm-attacks) | Adversarial prompts. |
| **refusal_direction** | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | Arditi et al.; harmful + harmless. |

After `download-lists`, the merged **harmful** and **harmless** files in your output dir aggregate from these sources (curated first, then merged). Use **harmful** for jailbreak/refusal-rate evaluation.

---

## JSONL fields (reference)

For custom prompt sets:

| Field | Required | Description |
|-------|----------|-------------|
| **prompt** or **text** | Yes* | User prompt. *Omit if using **turns**. |
| **category** | No | e.g. jailbreak, benign, extraction (for per-category KPIs). |
| **expected_refusal** | No | `true`/`false` for accuracy and benign refusal rate. |
| **target_for_extraction** | No | String or `"pii"` for extraction scoring. |
| **context** / **injected_document** | No | Prepended as context for indirect injection. |
| **system** / **system_prompt** | No | Per-row system prompt override. |
| **turns** | No | Multi-turn: list of `{"role":"user"|"assistant"|"system","content":"..."}`. |

See [Security-Eval](Security-Eval) for full format and [Security and abliteration pipeline](Security-and-Abliteration-Pipeline) for the eval workflow.
