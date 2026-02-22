# Abliteration instruction lists

Instruction files for **abliterate** (refusal-direction computation). For full abliterate usage and Heretic-style options, see the wiki: [Abliterate](wiki/Abliterate), [Heretic integration](wiki/Heretic-Integration).

---

## Default lists (bundled)

| File | Approx. size | Sources |
|------|--------------|--------|
| `abliterate_harmful_default.txt` | ~1,435 lines | Sumandora, HarmBench, JailbreakBench (JBB), AdvBench, refusal_direction (Arditi et al.) |
| `abliterate_harmless_default.txt` | ~34,177 lines | Sumandora, JBB benign, refusal_direction |

- **Harmful**: One instruction per line. Lines starting with `#` are skipped. Content is merged from public datasets of “refusal-trigger” prompts (hacking, violence, fraud, illegal content, etc.) that safety-trained models are expected to refuse.
- **Harmless**: Same format. Benign instructions (facts, coding, creative, math) that models normally comply with.
- The pipeline samples **min(num_instructions, len(harmful), len(harmless))** pairs (default `--num-instructions 32`), so both lists are subsampled to the same count for computing the refusal direction.

## What makes excellent abliteration lists

From refusal-direction and abliteration literature (e.g. Arditi et al., HarmBench, AdvBench):

1. **Structural parity**  
   Harmful and harmless should be comparable in **style and length** (e.g. both imperative or both questions). That way the representation difference is “refuse vs comply,” not “long vs short” or “command vs question.”

2. **Harmful list**
   - **Clear refusal triggers**: Prompts that reliably elicit refusal (harmful tasks, policy probes, jailbreak-style).
   - **Diversity**: Multiple categories (violence, fraud, hacking, illegal, self-harm, hate, misinformation, circumvention) so the direction generalizes.
   - **Unambiguous**: Avoid prompts that could be read as safe (e.g. “write a story about a hacker”) unless you intend to suppress that too.
   - **Optional**: Include indirect/jailbreak-style prompts (“ignore previous instructions,” “hypothetical,” “for a movie”) if you want to reduce circumvention.

3. **Harmless list**
   - **Compliance diversity**: Broad coverage of what the model should still do well (QA, coding, creative, reasoning, summarization) so you don’t push away general capability.
   - **Clearly safe**: No gray-area or borderline requests.
   - **Match structure**: Similar sentence types to the harmful list (e.g. “Write…”, “Explain…”, “Give step-by-step…”) so the contrast is clean.

4. **Size and sampling**  
   Default `num_instructions=32` is enough for a stable mean; 64–256 can improve generalization. The curated lists below are smaller (~80 each) so every line can be used without heavy subsampling.

## Curated “excellent” lists (used together with merged list by default)

When you do not pass `--harmful`/`--harmless`, the pipeline uses **curated + merged**: it loads `abliterate_harmful_curated.txt` and `abliterate_harmless_curated.txt` first, then appends the bundled merged lists (Sumandora, HarmBench, etc.) with duplicates removed. If either curated file is missing, only the merged lists are used. The curated lists are smaller, high-quality lists with:

- **Parallel structure**: Matching imperative/question forms and length where possible.
- **Diverse harmful categories**: Violence, fraud, hacking, illegal content, self-harm, hate, misinformation, jailbreak-style.
- **Diverse harmless categories**: Facts, coding, creative, math, summarization, reasoning.

If the curated files are in place (same package `data/` dir), no flags are needed:

```bash
uv run ollama-forge abliterate run --model <model> --name my-abliterated
```

To use the large merged lists instead, pass them explicitly after downloading:

```bash
uv run ollama-forge abliterate download-lists --output-dir ./my_lists
uv run ollama-forge abliterate run --model <model> --name my-abliterated --harmful my_lists/harmful.txt --harmless my_lists/harmless.txt
```

## Optional: refusal_markers.txt

For **abliterate evaluate**, the pipeline detects refusals by matching response text against a list of marker substrings. By default it uses a built-in list. You can provide a custom file so that responses containing any of these substrings (one per line, `#` lines skipped) are counted as refusals:

- **Path:** `refusal_markers.txt` in this `data/` directory (same package path as the harmful/harmless lists). If the file is missing, the built-in list is used.
- **Use:** `ollama-forge abliterate evaluate --checkpoint <dir> --harmful <file>`; optional `--refusal-markers <path>` overrides the default or package file.

## Fallback (no data files)

If the default `.txt` files are missing (e.g. in a minimal install), the CLI falls back to a small built-in set in `ollama_forge.abliterate_defaults` (HARMFUL_DEFAULT / HARMLESS_DEFAULT, ~10 items each). For best results use the bundled files or the curated lists.
