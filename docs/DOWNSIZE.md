# Downsizing models (e.g. 30B → 3B)

Produce a **smaller model** from a larger one so you can run it locally in Ollama. Main approach: **knowledge distillation** (teacher → student). Optional: **pruning** for moderate size reduction.

---

## Pipeline overview

1. **Choose teacher and student** — Teacher = large model (e.g. 30B, HF or local). Student = smaller same-family or compatible model (e.g. 3B).
2. **Distill** — Train the student to mimic the teacher (data distillation: teacher generates (input, output) pairs; student trains on them; or logit distillation / GKD).
3. **Export student for Ollama** — Save student → convert to GGUF (llama.cpp) → `ollama-tools convert --gguf <path> --name <name>`.

Training (step 2) is done **outside** ollama-tools; we document the flow and the export step reuses existing tooling.

---

## Knowledge distillation

### Data distillation

- **Teacher** generates responses for a dataset (e.g. instructions, prompts).
- **Student** is fine-tuned (or trained) on (input, teacher_output) pairs.
- Tools: custom script with Hugging Face `transformers` + your training loop, or frameworks like **Axolotl**, **Unsloth**, or **TRL** (see below).

### Logit / response distillation

- Student is trained to match teacher **logits** or **responses** on the same inputs.
- **TRL (Hugging Face)** — [Generalized Knowledge Distillation (GKD) Trainer](https://huggingface.co/docs/trl/main/en/gkd_trainer): trains student on self-generated outputs with teacher feedback; addresses train–inference mismatch; configurable via `GKDConfig`.

### Pruning + distillation (optional)

- **Prune** the large model (e.g. remove layers, reduce width) then **distill** to recover quality. Good for moderate ratios (e.g. 2–4×) with less compute than full distillation from scratch.
- For aggressive ratios (e.g. 30B → 3B), standard teacher–student distillation is the main path.

---

## Student export to Ollama

Once you have a **trained student** (saved Hugging Face model):

1. **Convert to GGUF** — Use llama.cpp: `convert-hf-to-gguf.py` (or the script that matches the student architecture). Optionally quantize (e.g. Q4_K_M).
2. **Create Ollama model** — `ollama-tools convert --gguf /path/to/student.gguf --name my-downsized`
3. **Run** — `ollama run my-downsized`

Same flow as in HF_TO_OLLAMA.md; the student is just another HF model to convert.

---

## Pruning (optional, moderate downsizing)

For **moderate** size reduction (e.g. 7B → 3B, or 15B → 8B):

- **Structured pruning** — Remove layers (depth), heads, or MLP width; then often **fine-tune or distill** to recover performance.
- **Layer dropping** — Keep a subset of layers; may need alignment (e.g. copy every Nth layer into a smaller stack).
- Tools: research code or custom scripts; some Hugging Face / community pruning tools exist. Not bundled in ollama-tools; we document as an optional path.
- For **large** ratios (e.g. 10×), prefer **distillation** (teacher 30B → student 3B) over pruning alone.

---

## Trade-offs and hardware

- **Distillation** — Needs teacher inference (GPU/memory for 30B) and student training (GPU for 3B training). Data generation can be batched.
- **Pruning** — Less training than full distillation but may need careful tuning; good for 2–4× reduction.
- **Quality** — Smaller students usually lose some capability; distillation and data quality matter.

---

## Using ollama-tools

- **After you have a downsize student** (saved HF or GGUF):
  - If HF: convert to GGUF with llama.cpp, then:
  - `ollama-tools convert --gguf /path/to/student.gguf --name my-downsized`
- **Pipeline summary:** Run `ollama-tools downsize` or `ollama-tools downsize pipeline` to print the steps; then run distillation externally and use `convert` for the student.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Choose teacher (large) and student (small) model IDs or paths. |
| 2 | Run distillation (TRL GKD, Axolotl, or custom): teacher generates data or logits → student trains. |
| 3 | Export student: HF → GGUF (llama.cpp) → `ollama-tools convert --gguf <path> --name <name>`. |
| Optional | For moderate ratio: consider pruning + distillation; document in this guide. |
