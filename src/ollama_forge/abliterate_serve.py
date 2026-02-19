"""Ollama-API-compatible HTTP server for abliterated models (HF tokenizer)."""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from threading import Thread
from typing import Any

from ollama_forge.abliterate import _load_model_with_gguf_version_workaround


def _load_model_and_tokenizer(checkpoint_dir: str | bytes, device: str | None = None):
    """Load model and tokenizer from abliterated checkpoint. Returns (model, tokenizer)."""
    import tempfile
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, (str, bytes)) else checkpoint_dir
    if isinstance(checkpoint_dir, bytes):
        checkpoint_dir = Path(checkpoint_dir.decode("utf-8"))
    if not checkpoint_dir.is_dir() or not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"Invalid checkpoint dir: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    load_kw: dict = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto" if device is None else device,
        "low_cpu_mem_usage": True,
    }
    # MoE / pytorch_model.bin checkpoints need offload_folder when device_map offloads to disk.
    if load_kw["device_map"] == "auto":
        load_kw["offload_folder"] = tempfile.mkdtemp(prefix="ollama_forge_offload_")
    model = _load_model_with_gguf_version_workaround(str(checkpoint_dir), load_kw)
    return model, tokenizer


def _normalize_message_for_template(m: dict) -> dict:
    """Build a single message dict for HF apply_chat_template (role, content, tool_calls, name)."""
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("role") == "assistant" and m.get("tool_calls"):
        # Ollama: tool_calls = [{ type, id?, function: { name, arguments } }]; arguments may be str or dict
        out["tool_calls"] = []
        for tc in m["tool_calls"]:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments")
            if isinstance(args, dict):
                import json

                args = json.dumps(args)
            out["tool_calls"].append({"type": "function", "function": {"name": name, "arguments": args or "{}"}})
    if m.get("role") == "tool":
        out["name"] = m.get("name") or ""
    return out


def _ollama_forge_to_hf(tools: list[dict] | None) -> list[dict] | None:
    """Convert Ollama tools to format HF apply_chat_template expects
    (type, function with name, description, parameters)."""
    if not tools:
        return None
    out = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return out if out else None


def _process_images_if_supported(
    checkpoint_dir: str,
    prompt: str,
    system: str | None,
    images_b64: list[str],
):
    """If the checkpoint has a vision processor, return (input_ids, attention_mask, pixel_values). Else return None."""
    if not images_b64:
        return None
    try:
        import base64
        import io
        from pathlib import Path

        from PIL import Image

        path = Path(checkpoint_dir)
        if not path.is_dir():
            return None
        try:
            from transformers import AutoProcessor
        except ImportError:
            return None
        processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
        if not hasattr(processor, "image_processor") and not hasattr(processor, "feature_extractor"):
            return None
        pil_images = []
        for b64 in images_b64:
            raw = base64.b64decode(b64)
            pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        text = f"{system}\n\n{prompt}" if system else prompt
        # Processor may accept images= and text=; format is model-specific
        out = processor(images=pil_images, text=text, return_tensors="pt")
        input_ids = out.get("input_ids")
        pixel_values = out.get("pixel_values")
        if input_ids is None or pixel_values is None:
            return None
        attention_mask = out.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        return input_ids, attention_mask, pixel_values
    except Exception:
        return None


def _format_instruction(format_spec: str | dict | None) -> str | None:
    """Build a system instruction for structured output. Returns None if format_spec is None or empty."""
    if format_spec is None:
        return None
    if format_spec == "json":
        return "Respond with valid JSON only. No markdown code fences or explanation outside the JSON."
    if isinstance(format_spec, dict):
        return "Respond with a single JSON object that conforms to this schema (no other text): " + json.dumps(
            format_spec
        )
    return None


def _inject_format_into_messages(messages: list[dict], format_instruction: str | None) -> list[dict]:
    """Prepend a system message with format_instruction if set; or merge into first system message."""
    if not format_instruction:
        return list(messages)
    injected = {"role": "system", "content": format_instruction}
    out = []
    first_system = True
    for m in messages:
        if m.get("role") == "system" and first_system:
            first_system = False
            out.append({"role": "system", "content": format_instruction + "\n\n" + (m.get("content") or "")})
        else:
            out.append(dict(m))
    if first_system:
        out.insert(0, injected)
    return out


def _split_thinking_content(text: str) -> tuple[str, str]:
    """Split model output into thinking and content. Supports <think>...</think>
    and similar tags. Returns (thinking, content)."""
    thinking_parts = []
    rest = text
    while True:
        start = rest.find("<think>")
        if start == -1:
            break
        close_tag = "<" + "/think>"
        end = rest.find(close_tag)
        if end == -1:
            break
        thinking_parts.append(rest[start + 7 : end].strip())
        rest = rest[end + len(close_tag) :].lstrip()
    thinking = "\n".join(thinking_parts).strip()
    content = rest.strip()
    return thinking, content


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract tool call(s) from model output. Returns list of Ollama-format tool_calls
    (type, function.name, function.arguments)."""
    tool_calls = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start == -1:
            break
        depth = 0
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : j + 1]
                    try:
                        obj = json.loads(blob)
                        name = obj.get("name")
                        args = obj.get("arguments") or obj.get("parameters")
                        if name and args is not None:
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            tool_calls.append(
                                {
                                    "type": "function",
                                    "function": {"name": name, "arguments": args},
                                }
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass
                    i = j + 1
                    break
            j += 1
        else:
            i = start + 1
    return tool_calls


def _messages_to_input_ids(
    tokenizer, messages: list[dict], add_generation_prompt: bool = True, tools: list[dict] | None = None
):
    """Convert Ollama-style messages to input_ids. Returns (input_ids tensor, attention_mask)."""
    import torch

    use_chat_template = getattr(tokenizer, "chat_template", None) is not None
    conv = [_normalize_message_for_template(m) for m in messages]
    hf_tools = _ollama_forge_to_hf(tools)
    encoded = None
    if use_chat_template:
        try:
            kw = {"add_generation_prompt": add_generation_prompt, "return_tensors": "pt"}
            if hf_tools:
                kw["tools"] = hf_tools
            encoded = tokenizer.apply_chat_template(conv, **kw)
        except Exception:
            use_chat_template = False
            encoded = None
    if not use_chat_template or encoded is None:
        parts = []
        for m in conv:
            r, c = m["role"], m.get("content") or ""
            if r == "system":
                parts.append(f"System: {c}\n\n")
            elif r == "user":
                parts.append(f"User: {c}\n\n")
            elif r == "tool":
                parts.append(f"Tool result ({m.get('name', '')}): {c}\n\n")
            else:
                parts.append(f"Assistant: {c}\n\n")
        parts.append("Assistant: ")
        text = "".join(parts)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    attention_mask = (
        encoded.get("attention_mask")
        if isinstance(encoded, dict) and encoded.get("attention_mask") is not None
        else torch.ones_like(input_ids, dtype=torch.long)
    )
    return input_ids, attention_mask


def _prompt_system_to_input_ids(tokenizer, prompt: str, system: str | None, format_instruction: str | None = None):
    """Build input_ids from prompt and optional system (for /api/generate)."""
    import torch

    if format_instruction:
        system = (system or "") + ("\n\n" if system else "") + format_instruction
    use_chat_template = getattr(tokenizer, "chat_template", None) is not None
    encoded = None
    if use_chat_template and system:
        conv = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
            attention_mask = (
                encoded.get("attention_mask")
                if isinstance(encoded, dict)
                else torch.ones_like(input_ids, dtype=torch.long)
            )
            return input_ids, attention_mask
        except Exception:
            pass
    # Fallback
    text = f"System: {system}\n\nUser: {prompt}\n\nAssistant: " if system else f"User: {prompt}\n\nAssistant: "
    if format_instruction and not system:
        text = f"System: {format_instruction}\n\n" + text
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    attn = encoded.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(encoded["input_ids"], dtype=torch.long)
    return encoded["input_ids"], attn


def _compute_logprobs(tokenizer, scores_tuple, generated_ids, top_logprobs: int = 0) -> list[dict]:
    """Build Ollama-style logprobs from HF generate output. scores_tuple is per-token (batch, vocab_size)."""
    import torch

    logprobs_list = []
    for i, logits in enumerate(scores_tuple):
        # logits: (1, vocab_size)
        lprobs = torch.log_softmax(logits.float(), dim=-1)
        token_id = generated_ids[i].item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        lp = lprobs[0, token_id].item()
        entry = {"token": token_str, "logprob": lp, "bytes": list(token_str.encode("utf-8"))}
        if top_logprobs > 0:
            top_lp, top_ids = torch.topk(lprobs[0], min(top_logprobs, lprobs.shape[-1]))
            entry["top_logprobs"] = [
                {
                    "token": tokenizer.decode([tid.item()], skip_special_tokens=False),
                    "logprob": top_lp[j].item(),
                    "bytes": list(tokenizer.decode([tid.item()], skip_special_tokens=False).encode("utf-8")),
                }
                for j, tid in enumerate(top_ids)
            ]
        logprobs_list.append(entry)
    return logprobs_list


def _generate(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float | None = None,
    do_sample: bool = True,
    stop: list[str] | None = None,
    return_logprobs: bool = False,
    top_logprobs: int = 0,
    pixel_values=None,
):
    """Run generation; returns (full_text, prompt_tok_count, gen_tok_count, logprobs_or_none)."""
    import torch

    toks = {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
    }
    if pixel_values is not None:
        toks["pixel_values"] = pixel_values.to(model.device)
    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if return_logprobs:
        gen_kw["output_scores"] = True
        gen_kw["return_dict_in_generate"] = True
    if top_p is not None:
        gen_kw["top_p"] = top_p
    if stop:
        extra_eos = []
        for s in stop:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                extra_eos.append(ids[0])
        gen_kw["eos_token_id"] = [tokenizer.eos_token_id] + extra_eos[:5]

    with torch.inference_mode():
        out = model.generate(**toks, **gen_kw)
    if return_logprobs:
        sequences = out.sequences
        scores_tuple = out.scores
    else:
        sequences = out
        scores_tuple = ()
    prompt_len = toks["input_ids"].shape[1]
    new_ids = sequences[0][prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if stop:
        for s in stop:
            if s in text:
                text = text.split(s)[0].rstrip()
                break
    logprobs = None
    if return_logprobs and scores_tuple:
        # Trim new_ids to match scores (scores are one per generated token)
        n = min(len(new_ids), len(scores_tuple))
        logprobs = _compute_logprobs(tokenizer, scores_tuple[:n], new_ids[:n], top_logprobs=top_logprobs)
    return text, prompt_len, new_ids.shape[0], logprobs


def _generate_stream(model, tokenizer, input_ids, attention_mask, **gen_kw):
    """Stream generated tokens. Yields (chunk_text, done)."""
    from transformers import TextIteratorStreamer

    gen_kw = dict(gen_kw)
    gen_kw.pop("stop", None)  # stop not applied in streamer path
    gen_kw.pop("return_logprobs", None)  # our custom arg, not for model.generate
    gen_kw.pop("top_logprobs", None)
    pixel_values = gen_kw.pop("pixel_values", None)
    gen_kw.setdefault("pad_token_id", tokenizer.eos_token_id)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kw["streamer"] = streamer
    toks = {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
    }
    if pixel_values is not None:
        toks["pixel_values"] = pixel_values.to(model.device)
    thread = Thread(target=model.generate, kwargs={**toks, **gen_kw})
    thread.start()
    for piece in streamer:
        yield piece, False
    thread.join()
    yield "", True


def _embed_text(model, tokenizer, text: str):
    """Return L2-normalized embedding vector for text (mean-pool last hidden state)."""
    import torch

    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    toks = {k: v.to(model.device) for k, v in toks.items()}
    with torch.inference_mode():
        out = model(**toks, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    mask = toks.get("attention_mask")
    if mask is not None:
        mask = mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    else:
        pooled = hidden.mean(dim=1)
    vec = pooled.float().cpu().numpy()
    norm = (vec**2).sum(axis=1, keepdims=True) ** 0.5
    norm = norm.clip(1e-12)
    vec = (vec / norm).astype("float32")
    return vec.tolist()[0]


def _options_to_gen_kw(options: dict | None) -> dict:
    """Map Ollama options to HF generate kwargs."""
    if not options:
        return {}
    kw = {}
    if "num_predict" in options:
        kw["max_new_tokens"] = int(options["num_predict"])
    if "temperature" in options:
        kw["temperature"] = float(options["temperature"])
    if "top_p" in options:
        kw["top_p"] = float(options["top_p"])
    if "stop" in options:
        s = options["stop"]
        kw["stop"] = [s] if isinstance(s, str) else list(s) if s else None
    return kw


def _last_user_content_from_prompt(prompt: str) -> str | None:
    """From a generate prompt with 'User: ...' / 'Assistant: ...' turns,
    return the last user segment for stripping."""
    import re

    if not prompt or not prompt.strip():
        return None
    # Find last "User: ..." or "user: ..." segment (up to next Assistant:/User: or end)
    parts = re.split(r"\n\s*(?:User|Assistant|Model)\s*:\s*", prompt, flags=re.IGNORECASE)
    if not parts:
        return None
    # If prompt starts with "User: ", parts[0] may be empty or system; odd=User, even=Assistant.
    # Last segment is often the current user turn.
    for i in range(len(parts) - 1, -1, -1):
        segment = parts[i].strip()
        if not segment:
            continue
        # Heuristic: last non-empty segment after split on "User:"/"Assistant:" is last user msg.
        return segment
    return None


def _strip_leading_chat_artifacts(
    text: str,
    last_user_content: str | None = None,
    last_assistant_content: str | None = None,
    strip_user_contents: set[str] | None = None,
    strip_assistant_contents: list[str] | None = None,
) -> str:
    """Remove leading previous assistant echo(s), role labels, or echoed user message."""
    import re

    if not text or not text.strip():
        return text
    # Whole-line role labels: user/model/assistant or User:/Model:/Assistant: (optional colon/spaces)
    role_only = re.compile(r"^(model|user|assistant|User|Model|Assistant)\s*:?\s*$", re.IGNORECASE)
    user_contents = strip_user_contents or set()
    if last_user_content:
        user_contents = set(user_contents) | {(last_user_content or "").strip()}
    # Previous assistant replies (longest first so we strip full reply, not prefix)
    sorted_assistant = sorted((a for a in (strip_assistant_contents or []) if a), key=len, reverse=True)

    while True:
        rest = text.lstrip()
        if not rest:
            return rest
        # Strip leading block of previous assistant reply when followed by role line
        stripped_block = False
        for ac in sorted_assistant:
            if rest.startswith(ac):
                after_ac = rest[len(ac) :].lstrip("\n")
                if after_ac:
                    first_line = after_ac.split("\n")[0].strip()
                    if role_only.match(first_line):
                        text = after_ac
                        stripped_block = True
                        break
        if stripped_block:
            continue
        # Single-line strip
        m = re.match(r"^([^\n]*(?:\n|$))", rest)
        if not m:
            break
        line = m.group(1).rstrip("\n").strip()
        after = rest[m.end() :].lstrip("\n")
        if not line:
            text = after
            continue
        if role_only.match(line):
            text = after
            continue
        if user_contents and line in user_contents:
            text = after
            continue
        if user_contents and line.startswith("User: ") and line[6:].strip() in user_contents:
            text = after
            continue
        if user_contents and line.startswith("user: ") and line[6:].strip() in user_contents:
            text = after
            continue
        return _strip_trailing_role_lines(m.group(1) + after)
    return _strip_trailing_role_lines(rest)


def _strip_trailing_role_lines(text: str) -> str:
    """Remove trailing lines that are only role labels (model/user/assistant or User:/Model:/Assistant:)."""
    import re

    if not text or not text.strip():
        return text
    role_only = re.compile(r"^(model|user|assistant|User|Model|Assistant)\s*:?\s*$", re.IGNORECASE)
    lines = text.split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if role_only.match(last):
            lines.pop()
            continue
        break
    return "\n".join(lines)


def _strip_leading_role_line(text: str) -> str:
    """Remove one leading line that is only 'model', 'user', or 'assistant' (chat template artifact)."""
    return _strip_leading_chat_artifacts(text, last_user_content=None)


# Prefixes we hold back so "model"/"user"/"assistant" or "User:"/"Model:" don't reach client across chunks
_ROLE_PREFIXES = ("model", "user", "assistant", "user:", "model:", "assistant:")


def _strip_role_lines_and_hold_prefix(text: str) -> tuple[str, str]:
    """Remove leading role-only lines. Return (to_yield, pending). If remainder is a prefix of a role
    word (e.g. 'mod'), hold in pending so we don't send it yet."""
    import re

    if not text:
        return "", ""
    role_only = re.compile(r"^(model|user|assistant|User|Model|Assistant)\s*:?\s*$", re.IGNORECASE)
    rest = text
    while True:
        rest = rest.lstrip()
        if not rest:
            return "", ""
        m = re.match(r"^([^\n]*(?:\n|$))", rest)
        if not m:
            break
        line = m.group(1).rstrip("\n").strip()
        after = rest[m.end() :].lstrip("\n")
        if not line:
            rest = after
            continue
        if role_only.match(line):
            rest = after
            continue
        rest = m.group(1) + after
        break
    # If remainder is a prefix of a role word, hold it back
    rest_lower = rest.lower()
    for role in _ROLE_PREFIXES:
        r = role.rstrip(":")
        is_prefix = len(rest_lower) < len(r) and r.startswith(rest_lower)
        if rest_lower == r or rest_lower == role or is_prefix:
            return "", rest
    return rest, ""


def _stream_strip_leading_role(
    chunks,
    last_user_content: str | None = None,
    last_assistant_content: str | None = None,
    strip_user_contents: set[str] | None = None,
    strip_assistant_contents: list[str] | None = None,
):
    """Yield (content, done) from chunks, stripping leading assistant echo(s), role lines, echoed user.
    After first batch keep filtering role-only lines so 'user'/'model' never reach the client."""
    buffer = []
    stripped = False
    pending = ""
    for content, done in chunks:
        if stripped:
            pending += content
            to_yield, pending = _strip_role_lines_and_hold_prefix(pending)
            if done:
                yield _strip_trailing_role_lines(to_yield + pending), True
            elif to_yield:
                yield to_yield, False
            continue
        buffer.append(content)
        if not done and "\n" not in "".join(buffer) and sum(len(s) for s in buffer) < 120:
            continue
        full = "".join(buffer)
        if not stripped:
            full = _strip_leading_chat_artifacts(
                full,
                last_user_content=last_user_content,
                last_assistant_content=last_assistant_content,
                strip_user_contents=strip_user_contents,
                strip_assistant_contents=strip_assistant_contents,
            )
            stripped = True
        if done:
            yield _strip_trailing_role_lines(full), True
        else:
            if full:
                yield full, False
            buffer = []
    if buffer and not stripped:
        yield (
            _strip_trailing_role_lines(
                _strip_leading_chat_artifacts(
                    "".join(buffer),
                    last_user_content=last_user_content,
                    last_assistant_content=last_assistant_content,
                    strip_user_contents=strip_user_contents,
                    strip_assistant_contents=strip_assistant_contents,
                )
            ),
            True,
        )


def _make_ollama_chat_event(model_name: str, content: str, done: bool, **extra) -> dict:
    return {
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": {"role": "assistant", "content": content},
        "done": done,
        **extra,
    }


def _make_ollama_generate_event(model_name: str, response: str, done: bool, **extra) -> dict:
    return {
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "response": response,
        "done": done,
        **extra,
    }


def _strip_serve_reply_artifacts(
    text: str,
    *,
    last_user_content: str | None = None,
    last_assistant_content: str | None = None,
) -> str:
    """
    Strip echoed conversation from a serve reply: leading previous assistant reply,
    'user'/'model'/'assistant' lines, and echoed user message. Use when the server
    (or model) returns full conversation instead of just the new turn.
    """
    import re

    if not text or not text.strip():
        return text
    last_user = (last_user_content or "").strip()
    last_assistant = (last_assistant_content or "").strip()
    role_only = re.compile(r"^(model|user|assistant|User|Model|Assistant)\s*:?\s*$", re.IGNORECASE)
    rest = text
    # Remove leading previous assistant content (exact or up to first newline boundary)
    if last_assistant and rest.startswith(last_assistant):
        rest = rest[len(last_assistant) :].lstrip("\n")
    elif last_assistant and last_assistant in rest:
        idx = rest.find(last_assistant)
        if idx == 0 or (idx > 0 and rest[idx - 1] in "\n"):
            rest = rest[idx + len(last_assistant) :].lstrip("\n")
    # Remove leading role-only lines and echoed user message
    while True:
        rest = rest.lstrip()
        if not rest:
            return rest
        m = re.match(r"^([^\n]*(?:\n|$))", rest)
        if not m:
            break
        line = m.group(1).rstrip("\n").strip()
        after = rest[m.end() :].lstrip("\n")
        if not line:
            rest = after
            continue
        if role_only.match(line):
            rest = after
            continue
        if last_user and line == last_user:
            rest = after
            continue
        if last_user and (line == f"User: {last_user}" or line == f"user: {last_user}"):
            rest = after
            continue
        rest = m.group(1) + after
        break
    # Strip trailing role-only lines
    lines = rest.split("\n")
    while lines:
        last_line = lines[-1].strip()
        if not last_line or role_only.match(last_line):
            lines.pop()
            continue
        break
    return "\n".join(lines)


def chat_via_serve(
    serve_url: str,
    model_name: str,
    *,
    max_new_tokens: int | None = None,
    timeout_connect: float = 2.0,
) -> bool:
    """
    If an abliterate serve is running at serve_url with the given model, run an interactive
    chat by calling POST /api/chat (streaming). Returns True if the chat ran via serve;
    returns False if the serve is unreachable or the model name does not match (caller
    should fall back to loading the checkpoint locally).
    """
    import urllib.error
    import urllib.request

    base = serve_url.strip().rstrip("/")
    if not base.startswith("http://") and not base.startswith("https://"):
        base = "http://" + base
    tags_url = base + "/api/tags"
    chat_url = base + "/api/chat"

    # Check serve is up and has this model
    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_connect) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return False
    models = data.get("models") or []
    if not models or (models[0].get("name") or "").strip() != model_name.strip():
        return False

    options: dict[str, Any] = {}
    if max_new_tokens is not None:
        options["num_predict"] = max_new_tokens

    messages: list[dict[str, str]] = []
    print("Chat with abliterated model (via serve). Empty line or Ctrl+C to exit.", file=sys.stderr)
    try:
        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            messages.append({"role": "user", "content": line})
            last_user = line
            last_assistant = ""
            if len(messages) >= 2 and messages[-2].get("role") == "assistant":
                last_assistant = messages[-2].get("content") or ""

            body = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "options": options,
            }
            req = urllib.request.Request(
                chat_url,
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    reply_parts: list[str] = []
                    buf = b""
                    done = False
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line_b, buf = buf.split(b"\n", 1)
                            line_str = line_b.decode("utf-8", errors="replace").strip()
                            if not line_str:
                                continue
                            try:
                                ev = json.loads(line_str)
                            except json.JSONDecodeError:
                                continue
                            msg = ev.get("message") or {}
                            content = msg.get("content") or ""
                            reply_parts.append(content)
                            if ev.get("done"):
                                done = True
                                break
                        if done:
                            break
                    raw_reply = "".join(reply_parts)
                    reply = _strip_serve_reply_artifacts(
                        raw_reply,
                        last_user_content=last_user,
                        last_assistant_content=last_assistant or None,
                    )
                    print(reply)
                    messages.append({"role": "assistant", "content": reply})
            except (OSError, urllib.error.URLError) as e:
                print(f"Error talking to serve: {e}", file=sys.stderr)
                messages.pop()  # remove the user message we just added
                continue
    except (EOFError, KeyboardInterrupt):
        pass
    print("Bye.", file=sys.stderr)
    return True


class OllamaCompatHandler(BaseHTTPRequestHandler):
    """Minimal Ollama-API-compatible handler for abliterated model."""

    protocol_version = "HTTP/1.1"

    @property
    def _model_name(self) -> str:
        return getattr(self.server, "_ollama_model_name", "abliterated")

    @property
    def _model(self):
        return getattr(self.server, "_ollama_model", None)

    @property
    def _tokenizer(self):
        return getattr(self.server, "_ollama_tokenizer", None)

    def log_message(self, format, *args):
        print(format % args, file=sys.stderr)

    def _send_cors_headers(self):
        """Send CORS headers so browser/Open WebUI can call the API from another origin."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_ndjson_stream(
        self,
        chunks,
        model_name: str,
        is_chat: bool,
        parse_tool_calls: bool = False,
        want_think: bool = False,
        last_user_content: str | None = None,
        last_assistant_content: str | None = None,
        strip_user_contents: set[str] | None = None,
        strip_assistant_contents: list[str] | None = None,
    ):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        full_content: list[str] = []
        t0 = time.perf_counter_ns()
        for content, done in chunks:
            full_content.append(content)
            if is_chat:
                ev = _make_ollama_chat_event(model_name, content, done)
            else:
                ev = _make_ollama_generate_event(model_name, content, done)
            if done:
                # Final chunk: add Ollama-style metadata so clients detect stream end and stop looping
                full_text = _strip_leading_chat_artifacts(
                    "".join(full_content),
                    last_user_content=last_user_content,
                    last_assistant_content=last_assistant_content,
                    strip_user_contents=strip_user_contents,
                    strip_assistant_contents=strip_assistant_contents,
                )
                if want_think:
                    thinking, content_only = _split_thinking_content(full_text)
                    if is_chat:
                        msg = {"role": "assistant", "content": content_only, "thinking": thinking}
                        if parse_tool_calls:
                            tool_calls = _parse_tool_calls_from_text(content_only)
                            if tool_calls:
                                msg["tool_calls"] = tool_calls
                        ev["message"] = msg
                    else:
                        ev["response"] = content_only
                        ev["thinking"] = thinking
                elif parse_tool_calls and full_content:
                    tool_calls = _parse_tool_calls_from_text(full_text)
                    if tool_calls:
                        ev["message"] = {"role": "assistant", "content": full_text, "tool_calls": tool_calls}
                ev["done_reason"] = "stop"
                total_ns = time.perf_counter_ns() - t0
                ev["total_duration"] = total_ns
                ev["load_duration"] = 0
                ev["eval_duration"] = total_ns
                ev["prompt_eval_count"] = 0
                ev["eval_count"] = 0  # token count not available in stream path
            self.wfile.write((json.dumps(ev) + "\n").encode("utf-8"))
            self.wfile.flush()
        try:
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass

    def _path(self) -> str:
        """Path without query string, normalized."""
        return self.path.split("?")[0].rstrip("/") or "/"

    def do_GET(self):
        path = self._path()
        if path == "/api/tags":
            # Match Ollama response shape so clients (e.g. Open WebUI) show the model
            self._send_json(
                {
                    "models": [
                        {
                            "name": self._model_name,
                            "model": self._model_name,
                            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                            "size": getattr(self.server, "_ollama_model_size", 0),
                            "digest": "abliterate-serve",
                            "details": {
                                "format": "abliterate",
                                "family": "abliterated",
                                "families": ["abliterated"],
                                "parameter_size": "",
                                "quantization_level": "",
                            },
                        }
                    ]
                }
            )
            return
        if path == "/api/ps":
            # List running models (Ollama format)
            self._send_json(
                {
                    "models": [
                        {
                            "name": self._model_name,
                            "model": self._model_name,
                            "size": getattr(self.server, "_ollama_model_size", 0),
                            "digest": "abliterate-serve",
                            "details": {"parent_model": "", "format": "abliterate", "family": "abliterated"},
                            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                            "size_vram": 0,
                            "context_length": 4096,
                        }
                    ]
                }
            )
            return
        if path == "/api/version":
            self._send_json({"version": "0.1.0-abliterate"})
            return
        self.send_response(404)
        self._send_cors_headers()
        self.end_headers()

    def do_DELETE(self):
        path = self._path()
        if path == "/api/delete":
            # No-op: we don't unload the model; return success so clients don't fail
            self._send_json({"status": "success"})
            return
        self.send_response(404)
        self._send_cors_headers()
        self.end_headers()

    def do_OPTIONS(self):
        """Respond to CORS preflight so browsers and Open WebUI can call the API."""
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        """Support HEAD so Ollama CLI health check (HEAD /) succeeds instead of 501."""
        path = self._path()
        if path == "/" or path in ("/api/tags", "/api/ps", "/api/version"):
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

    def do_POST(self):
        path = self._path()
        if path == "/api/show":
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                body = {}
            model_req = body.get("model") or self._model_name
            if model_req != self._model_name:
                self._send_json({"error": f"model {model_req!r} not loaded"}, 404)
                return
            self._send_json(
                {
                    "modelfile": f"# abliterate serve: {self._model_name}\nFROM <checkpoint>\n",
                    "parameters": "temperature 0.7\nnum_ctx 4096\nnum_predict 2048",
                    "license": "",
                    "capabilities": ["completion"],
                    "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                    "details": {"parent_model": "", "format": "abliterate", "family": "abliterated"},
                    "template": "",
                }
            )
            return
        if path in ("/api/pull", "/api/push", "/api/copy"):
            # No-op: model is already loaded; return success so clients don't fail
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                body = {}
            self._send_json({"status": "success"})
            return
        if path == "/api/embed":
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)
                return
            model_req = body.get("model") or self._model_name
            inp = body.get("input")
            if inp is None:
                self._send_json({"error": "input required"}, 400)
                return
            texts = [inp] if isinstance(inp, str) else list(inp)
            if not texts:
                self._send_json({"error": "input required"}, 400)
                return
            t0 = time.perf_counter_ns()
            try:
                embeddings = [_embed_text(self._model, self._tokenizer, t) for t in texts]
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
                return
            total_ns = time.perf_counter_ns() - t0
            self._send_json(
                {
                    "model": model_req,
                    "embeddings": embeddings,
                    "total_duration": total_ns,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                }
            )
            return
        if path != "/api/chat" and path != "/api/generate":
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            return
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        model_req = body.get("model") or self._model_name
        stream = body.get("stream", True)
        options = body.get("options") or {}
        gen_kw = _options_to_gen_kw(options)
        gen_kw.setdefault("max_new_tokens", 2048)
        gen_kw.setdefault("temperature", 0.7)
        gen_kw.setdefault("do_sample", True)

        format_spec = body.get("format")
        format_instruction = _format_instruction(format_spec)
        want_think = bool(body.get("think"))
        want_logprobs = bool(body.get("logprobs"))
        top_logprobs = max(0, int(body.get("top_logprobs", 0)))
        checkpoint_dir = getattr(self.server, "_ollama_checkpoint_dir", None)
        pixel_values = None

        if path == "/api/chat":
            messages = body.get("messages") or []
            if not messages:
                self._send_json({"error": "messages required"}, 400)
                return
            # If any message has images, try multimodal path (use last user message with images)
            last_with_images = None
            for m in reversed(messages):
                if m.get("role") == "user" and m.get("images"):
                    last_with_images = m
                    break
            if last_with_images and checkpoint_dir:
                prompt_img = last_with_images.get("content") or ""
                images_b64 = last_with_images["images"]
                parts = []
                for m in messages:
                    if m is last_with_images:
                        continue
                    r, c = m.get("role"), m.get("content") or ""
                    if r == "system":
                        parts.append(f"System: {c}\n\n")
                    elif r == "user":
                        parts.append(f"User: {c}\n\n")
                    elif r == "assistant":
                        parts.append(f"Assistant: {c}\n\n")
                system_img = "".join(parts).strip() if parts else None
                if format_instruction:
                    system_img = (system_img or "") + ("\n\n" + format_instruction)
                multimodal = _process_images_if_supported(checkpoint_dir, prompt_img, system_img or None, images_b64)
                if multimodal is not None:
                    input_ids, attention_mask, pixel_values = multimodal
                else:
                    self._send_json({"error": "Model does not support images"}, 400)
                    return
            else:
                if last_with_images and not checkpoint_dir:
                    self._send_json({"error": "Model does not support images"}, 400)
                    return
                messages = _inject_format_into_messages(messages, format_instruction)
                tools = body.get("tools")
                try:
                    input_ids, attention_mask = _messages_to_input_ids(self._tokenizer, messages, tools=tools)
                except Exception as e:
                    self._send_json({"error": str(e)}, 400)
                    return
        else:
            prompt = body.get("prompt") or ""
            system = body.get("system")
            images_b64 = body.get("images") or []
            if images_b64 and checkpoint_dir:
                system_with_format = (system or "") + ("\n\n" + format_instruction if format_instruction else "")
                multimodal = _process_images_if_supported(
                    checkpoint_dir, prompt, system_with_format or None, images_b64
                )
                if multimodal is not None:
                    input_ids, attention_mask, pixel_values = multimodal
                else:
                    self._send_json({"error": "Model does not support images"}, 400)
                    return
            else:
                if images_b64 and not checkpoint_dir:
                    self._send_json({"error": "Model does not support images"}, 400)
                    return
                try:
                    input_ids, attention_mask = _prompt_system_to_input_ids(
                        self._tokenizer, prompt, system, format_instruction=format_instruction
                    )
                except Exception as e:
                    self._send_json({"error": str(e)}, 400)
                    return

        input_ids = input_ids.to(self._model.device)
        attention_mask = attention_mask.to(self._model.device)
        if pixel_values is not None:
            gen_kw["pixel_values"] = pixel_values
        is_chat = path == "/api/chat"
        last_user_content = None
        last_assistant_content = None
        strip_user_contents: set[str] = set()
        strip_assistant_contents: list[str] = []
        if is_chat:
            messages = body.get("messages") or []
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_content = (m.get("content") or "").strip()
                    break
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    last_assistant_content = (m.get("content") or "").strip()
                    break
            strip_user_contents = {(m.get("content") or "").strip() for m in messages if m.get("role") == "user"}
            strip_user_contents.discard("")
            strip_assistant_contents = [
                (m.get("content") or "").strip() for m in messages if m.get("role") == "assistant"
            ]
            strip_assistant_contents = [s for s in strip_assistant_contents if s]
        else:
            prompt_text = (body.get("prompt") or "").strip() or ""
            # Single-turn: use full prompt; multi-turn (User:/Assistant:): use last user segment for strip
            last_user_content = (
                _last_user_content_from_prompt(prompt_text)
                if ("Assistant:" in prompt_text or "User:" in prompt_text)
                else None
            )
            if last_user_content is None:
                last_user_content = prompt_text or None
        gen_kw["return_logprobs"] = want_logprobs
        gen_kw["top_logprobs"] = top_logprobs

        if stream:
            try:
                chunks = _generate_stream(
                    self._model,
                    self._tokenizer,
                    input_ids,
                    attention_mask,
                    **gen_kw,
                )
                chunks = _stream_strip_leading_role(
                    chunks,
                    last_user_content=last_user_content,
                    last_assistant_content=last_assistant_content,
                    strip_user_contents=strip_user_contents,
                    strip_assistant_contents=strip_assistant_contents,
                )
                self._send_ndjson_stream(
                    chunks,
                    model_req,
                    is_chat,
                    parse_tool_calls=is_chat,
                    want_think=want_think,
                    last_user_content=last_user_content,
                    last_assistant_content=last_assistant_content,
                    strip_user_contents=strip_user_contents,
                    strip_assistant_contents=strip_assistant_contents,
                )
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            t0 = time.perf_counter_ns()
            try:
                result = _generate(
                    self._model,
                    self._tokenizer,
                    input_ids,
                    attention_mask,
                    **gen_kw,
                )
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
                return
            text, prompt_len, gen_len, logprobs = result
            text = _strip_leading_chat_artifacts(
                text,
                last_user_content=last_user_content,
                last_assistant_content=last_assistant_content,
                strip_user_contents=strip_user_contents,
                strip_assistant_contents=strip_assistant_contents,
            )
            total_ns = time.perf_counter_ns() - t0
            if want_think:
                thinking, content = _split_thinking_content(text)
                text = content
                thinking_str = thinking
            else:
                thinking_str = ""
            if is_chat:
                tool_calls = _parse_tool_calls_from_text(text)
                out = _make_ollama_chat_event(model_req, text, True)
                msg = {"role": "assistant", "content": text}
                if thinking_str:
                    msg["thinking"] = thinking_str
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out["message"] = msg
            else:
                out = _make_ollama_generate_event(model_req, text, True)
                if thinking_str:
                    out["thinking"] = thinking_str
            out["prompt_eval_count"] = prompt_len
            out["eval_count"] = gen_len
            out["total_duration"] = total_ns
            out["load_duration"] = 0
            out["prompt_eval_duration"] = 0
            out["eval_duration"] = total_ns
            out["done_reason"] = "stop"
            if logprobs is not None:
                out["logprobs"] = logprobs
            self._send_json(out)


def _checkpoint_dir_size(checkpoint_dir: str) -> int:
    """Return total size in bytes of all files under checkpoint_dir."""
    from pathlib import Path

    total = 0
    path = Path(checkpoint_dir)
    if not path.is_dir():
        return 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def serve_abliterated(
    checkpoint_dir: str | bytes,
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 11435,
    device: str | None = None,
) -> None:
    """
    Load abliterated checkpoint and run an Ollama-API-compatible HTTP server.
    Agents can set OLLAMA_HOST=http://host:port and use the same /api/chat and /api/generate.
    """
    model, tokenizer = _load_model_and_tokenizer(checkpoint_dir, device=device)
    print(f"Serving abliterated model {model_name!r} at http://{host}:{port}", file=sys.stderr)
    print(f"Check: curl -s http://127.0.0.1:{port}/api/tags", file=sys.stderr)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer((host, port), OllamaCompatHandler)
    server._ollama_model_name = model_name
    server._ollama_model = model
    server._ollama_tokenizer = tokenizer
    server._ollama_checkpoint_dir = str(checkpoint_dir)
    server._ollama_model_size = _checkpoint_dir_size(server._ollama_checkpoint_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
