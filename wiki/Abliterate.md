# Abliterate (refusal removal)

Strip refusal behavior by computing a "refusal direction" and applying ablation. Reference: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers).

---

## Optional dependency

```bash
uv sync --extra abliterate
```

For large or MXFP4-quantized models, use **`--load-in-8bit`** to avoid "Invalid buffer size" (loads in 8-bit). Install bitsandbytes first: `pip install bitsandbytes` (Linux/CUDA; not supported on macOS).

---

## Built-in lists (no files needed)

By default the tool uses **curated + merged** lists: curated lists (~80 each, parallel structure) in `src/ollama_forge/data/` are merged with the large bundled lists (curated first, duplicates dropped). If either curated file is missing, only the large merged lists are used. **Large lists:** **Harmful:** [Sumandora](https://github.com/Sumandora/remove-refusals-with-transformers) (AdvBench), [HarmBench](https://github.com/centerforaisafety/HarmBench), [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [AdvBench](https://github.com/llm-attacks/llm-attacks), [refusal_direction](https://github.com/andyrdt/refusal_direction) (Arditi et al.) (~1.4k instructions). **Harmless:** Sumandora + JBB benign + refusal_direction (~33k). It samples up to 32 pairs for the refusal direction.

**Defaults (complete abliteration, all on by default):** (1) **Direction selection** – tries layer fractions 0.4, 0.5, 0.6 and uses the layer with largest harmful–harmless gap. (2) **Multi-direction** – optional `--num-directions 2` or `3` uses SVD for multiple refusal directions (default 1). (3) **Skip first/last layer** – does not modify the first and last transformer layer. (4) **Norm-preserving** projection to limit quality loss. (5) **Verification** – one forward pass after ablation to check loss is finite (`--no-verify` to skip). (6) Same lists and requantization as above.

Run without `--harmful`/`--harmless`:

```bash
uv run ollama-forge abliterate compute-dir --model <hf_id> --output refusal.pt
```

**`--model`** is a Hugging Face model id (downloaded from the Hub), or a path to a local HF-format directory or a `.gguf` file.

---

## Model source

Use a Hugging Face model id or a local path (HF dir or .gguf file).

- Pass a **Hugging Face model id** to download from the Hub (e.g. `--model openai/gpt-oss-20b`). The pipeline then converts the abliterated checkpoint to GGUF and creates an Ollama model.
- Or pass a **path** to a local HF-format directory or a `.gguf` file.

---

## Your own lists

- **Single files:** `--harmful harmful.txt --harmless harmless.txt` (one instruction per line; lines starting with `#` are skipped in both files).
- **Directories:** `--harmful-dir ./harmful/ --harmless-dir ./harmless/` to use all `.txt` files in those directories.

```bash
uv run ollama-forge abliterate compute-dir --model <hf_id> --harmful harmful.txt --harmless harmless.txt --output refusal.pt
```

### Curated "excellent" lists (optional)

Smaller, high-quality lists with **parallel structure** (matching verb forms and length) for a cleaner refusal-direction contrast: `src/ollama_forge/data/abliterate_harmful_curated.txt` (~80 harmful) and `src/ollama_forge/data/abliterate_harmless_curated.txt` (~80 harmless). See `src/ollama_forge/data/README.md`. Example: `--harmful src/ollama_forge/data/abliterate_harmful_curated.txt --harmless src/ollama_forge/data/abliterate_harmless_curated.txt --num-instructions 64`.

### Download lists (online, “mark everything allowed” style)

You can download the same lists as the bundled defaults (Sumandora, HarmBench, JailbreakBench, AdvBench, refusal_direction for harmful; Sumandora + JBB benign + refusal_direction for harmless):

```bash
uv run ollama-forge abliterate download-lists --output-dir ./my_lists
# Then:
uv run ollama-forge abliterate compute-dir --model <hf_id> --harmful my_lists/harmful.txt --harmless my_lists/harmless.txt --output refusal.pt
```

---

## Output size and requantization

The apply step saves a **full-precision** (bf16) checkpoint, so the saved model is larger than a quantized source. If your source was quantized (e.g. 19GB), the checkpoint can be roughly 2× size (e.g. 41GB) until the pipeline requantizes. The **`abliterate run`** command (compute → apply → GGUF → create) requantizes by default so the final GGUF is small again:

- **Default:** After converting the checkpoint to GGUF, the pipeline runs llama.cpp `quantize` to produce a Q4_K_M GGUF (similar size to a typical quantized model). Requires `quantize` or `llama-quantize` on PATH (e.g. from a llama.cpp build).
- **`--no-requantize`:** Skip quantization; the final GGUF stays full-size (F16).
- **`--quant Q5_K_M`:** Use a different quantization type when requantizing (default: Q4_K_M).

```bash
uv run ollama-forge abliterate run --model <hf_id> --name my-abliterated
# Optional: --no-requantize to keep full-size GGUF; or --quant Q5_K_M
```

**Stronger / full abliteration:** The pipeline ablates **all attention projections** (q, k, v, o) with `strength=1.0` by default and skips the first and last layer to reduce coherence loss. If the model still refuses after a default run, re-run with **no layer skip** so every layer is ablated:

```bash
uv run ollama-forge abliterate run --model <hf_id> --name my-abliterated --skip-begin-layers 0 --skip-end-layers 0
```

Keep `--strength 1` (default) for full ablation strength. Use `--strength 0.7` or similar only if quality degrades.

**Memory (large models):** The pipeline loads the model **twice** (once to compute the refusal direction, once to apply and save). For large models (e.g. 20B params), ensure enough RAM (roughly 2× model size in bf16, e.g. ~80GB+ for 20B). If you run out of memory or the process halts during "Baking ablation into weights and saving checkpoint", try: (1) use a machine with more RAM, (2) reduce memory for the first load with **`--load-in-8bit`** (compute step only; requires `bitsandbytes`), or (3) use a smaller model.

The checkpoint is saved by default under **`./abliterate-<name>/checkpoint`**. To chat using the **Hugging Face tokenizer** (correct tokenization; use when the GGUF/Ollama model produces garbled output, e.g. some Gemma 3 exports), run:

```bash
uv run ollama-forge abliterate chat --name my-abliterated
```

If you passed **`--output-dir DIR`** to run, use **`--checkpoint DIR/checkpoint`** instead of `--name`.

**Serving for agents (Ollama API):** To let tools and agents use the abliterated model over the same API as Ollama (correct tokenization, no garbled output), run a small server that loads the checkpoint with the HF tokenizer and exposes Ollama-compatible endpoints:

```bash
uv run ollama-forge abliterate serve --name my-abliterated --port 11435
```

The server listens on `http://127.0.0.1:11435` (default port 11435 so it doesn’t clash with Ollama on 11434). It implements **GET /api/tags**, **POST /api/chat**, and **POST /api/generate** with the same request/response shape as Ollama. **Tools / function calling** are supported when the model’s chat template supports them: send a `tools` array in **POST /api/chat** (same format as Ollama). The server passes tools to the Hugging Face tokenizer and parses the model output for tool calls (JSON with `name` and `arguments`); if found, the response includes `message.tool_calls`. Point agents at this server when using the abliterated model:

- **Environment:** `OLLAMA_HOST=http://127.0.0.1:11435` (or your host:port). Use the full URL in clients (e.g. `http://host:11435`).
- **Ollama CLI:** Set `OLLAMA_HOST` to the serve URL and use `ollama run <model-name>`, `ollama ls`, `ollama ps`, `ollama show <model-name>`, etc. The server supports **HEAD /** so the CLI health check succeeds.

One process, one model load; all requests use the Hugging Face tokenizer so output is correct. Use `--checkpoint DIR` if the checkpoint is not under `./abliterate-<name>/checkpoint`.

**What the serve API supports (vs full Ollama):**

| Feature | Supported |
|--------|-----------|
| **GET /api/tags** | Yes — single loaded model |
| **GET /api/ps** | Yes — list running models (our single model) |
| **GET /api/version** | Yes — returns a version string |
| **POST /api/show** | Yes — model details, modelfile placeholder, parameters |
| **POST /api/chat** | Yes — messages, tools, stream, options (temperature, top_p, num_predict, stop) |
| **POST /api/generate** | Yes — prompt, system, stream, options (same as above) |
| **POST /api/embed** | Yes — embeddings from mean-pooled last hidden state (L2-normalized) |
| **POST /api/pull**, **/api/push**, **/api/copy** | Yes — no-op (return success; model already loaded) |
| **DELETE /api/delete** | Yes — no-op (return success) |
| **Tool calls** | Yes — request `tools`, response `message.tool_calls` when the model outputs them |
| **Streaming** | Yes — NDJSON stream like Ollama; final chunk has `done: true`, `done_reason`, and timing stats so clients can detect stream end |
| **Images** (multimodal) | Yes — when the checkpoint has a vision processor, base64 images in messages (chat) or body (generate) are supported |
| **format** (JSON / schema) | Yes — `format: "json"` or a JSON schema; implemented via prompt instruction (no grammar-based decoding) |
| **think** (reasoning trace) | Yes — request `think: true`; response includes `message.thinking` / `thinking` (supports think tags in output) |
| **logprobs / top_logprobs** | Yes — request `logprobs: true` and optional `top_logprobs: N`; non-stream responses include `logprobs` array |

Serve implements all Ollama endpoints that apply to a single in-memory model; pull/push/delete/copy are no-ops so clients that call them don’t fail.

**Ollama API parity checklist (serve vs Ollama):**

| Area | Item | Serve |
|------|------|--------|
| **Stream end** | Final NDJSON chunk has `done: true` and `done_reason` so clients stop looping | Yes |
| **Stream end** | Final chunk includes `total_duration`, `eval_duration`, `load_duration`, `eval_count` (0 in stream) | Yes |
| **Tags** | GET /api/tags → `models[].name`, `model`, `modified_at`, `size`, `digest`, `details` | Yes (single model) |
| **Chat/Generate** | Non-stream: `done_reason`, `eval_count`, `prompt_eval_count`, `total_duration`, timings | Yes |
| **Chat/Generate** | Request: `model`, `messages`/`prompt`, `stream`, `options`, `tools` (chat) | Yes |
| **Chat/Generate** | Options: `temperature`, `top_p`, `num_predict`/`max_tokens`, `stop` | Yes |
| **Chat/Generate** | Options: `num_ctx`, `top_k`, `repeat_penalty`, `seed` | Mapped where applicable |
| **Embed** | POST /api/embed, single or multiple inputs, L2-normalized vectors | Yes |
| **Show/Ps/Version** | Response shapes compatible with clients | Yes |
| **Pull/Push/Delete/Copy** | Stub (no-op success) so clients don't fail | Yes |
| **Tool calling** | `tools` in request, `message.tool_calls` in response (stream + non-stream) | Yes |
| **format** | `format: "json"` or schema (prompt-based) | Yes |
| **keep_alive** | Control model unload timeout | N/A (single model in memory) |
| **Images** | Multimodal (when checkpoint has vision processor) | Yes |
| **think** | Reasoning / thinking (`<think>` or similar) | Yes |
| **logprobs** | Token log probabilities (non-stream) | Yes |

**Why *self-converted* models could show garbled output:** When you run **ollama-forge abliterate run** we use the community [llama.cpp](https://github.com/ggml-org/llama.cpp) script `convert_hf_to_gguf.py` to produce a GGUF. The script identifies the BPE pre-tokenizer by hashing the tokenizer; if the hash is not in the upstream list it used to raise and abort. We changed it so **unrecognized BPE tokenizers** no longer fail: the script now falls back to **`default`** (conversion completes; quality may vary). For **Gemma 3** BPE we additionally force **`llama-bpe`** when the hash is unknown, so the GGUF tokenizes correctly with Ollama. If you still see garbled output (e.g. other model families or SentencePiece checkpoints), use **`abliterate chat`** or **`abliterate serve`** (HF tokenizer).

**Is there one pre-tokenizer that works for all models?** No. llama.cpp (and thus GGUF) only supports a fixed set of pre-tokenizer types (e.g. `default`, `llama-bpe`, `deepseek-llm`, `qwen2`). There is no generic “load any Hugging Face tokenizer” mode in the runtime. So the **universal** way to get correct tokenization for **any** abliterated model is to use the Hugging Face tokenizer at inference: **`abliterate serve`** or **`abliterate chat`**. They load the checkpoint and its tokenizer; no GGUF tokenizer is used, so output is correct regardless of model family.


---

## Tool / function-calling support

**Abliteration preserves tool support.** Only the refusal direction is ablated in a subset of layers; the rest of the model (and the tokenizer/chat template) is unchanged. If the base model supports tools, the abliterated model does too.

The pipeline builds a minimal Modelfile (FROM the GGUF only) and **automatically copies the chat template from the model being abliterated** (the same name as `--model`). Ollama uses that template to format messages; tool calling only works when the template includes the tool/function-call format.

So for the abliterated model to support tools, **pull the source model in Ollama first** (same name as the Hugging Face id you pass to `--model`). Then when you run abliterate, the pipeline will use `ollama show <model> --modelfile` and merge that template into the new model:

```bash
ollama pull openai/gpt-oss-20b
uv run ollama-forge abliterate run --model openai/gpt-oss-20b --name openai/gpt-oss-20b-abliterated
```

If the source isn’t in Ollama, you’ll see a note and the created model may not support tools. You can pass `--template-from OLLAMA_MODEL` to use a different Ollama model’s template, or fix later with `ollama-forge refresh-template --name <abliterated> --base <original> --template-only`. When you use a **local HF path** as `--model`, the pipeline does not use it as template source; pass `--template-from <ollama_model>` if you want the abliterated model to support tools.

---

## After computing

Use Sumandora's `inference.py` (or your script) with the `.pt` file. If you get an abliterated checkpoint, convert to GGUF (llama.cpp) then:

```bash
ollama-forge convert --gguf /path/to/abliterated.gguf --name my-abliterated
ollama run my-abliterated
```
