"""Modelfile generation and creation from base."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _extract_template_block(content: str) -> str | None:
    """Extract the TEMPLATE \"\"\"...\"\"\" block from Modelfile text. Returns full block or None."""
    # Match TEMPLATE followed by optional whitespace and """, then content until closing """
    m = re.search(r"TEMPLATE\s+(\"\"\".*?\"\"\")", content, re.DOTALL | re.IGNORECASE)
    if m:
        return "TEMPLATE " + m.group(1)
    return None


def template_body_from_modelfile(content: str) -> str | None:
    """Extract the template body (content inside TEMPLATE \"\"\"...\"\"\") from Modelfile text."""
    m = re.search(r"TEMPLATE\s+\"\"\"(.*?)\"\"\"", content, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _replace_from_line(content: str, base: str) -> str:
    """Replace the first FROM line with FROM base."""
    return re.sub(r"^FROM\s+.+$", f"FROM {base}", content, count=1, flags=re.MULTILINE | re.IGNORECASE)


def merge_modelfile_with_reference_template(
    current_content: str,
    reference_content: str,
    base: str,
    *,
    template_only: bool = False,
) -> str:
    """
    Merge current Modelfile with reference's TEMPLATE.
    - If template_only=True: only replace TEMPLATE; keep current model's FROM (same weights).
    - If template_only=False: replace TEMPLATE and set FROM to base (recreate from base).
    """
    ref_template = _extract_template_block(reference_content)
    if not ref_template:
        if template_only:
            return current_content
        return _replace_from_line(current_content, base)

    # Replace TEMPLATE block in current with reference's
    current_merged = re.sub(
        r"TEMPLATE\s+\"\"\".*?\"\"\"",
        ref_template,
        current_content,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if current_merged == current_content:
        # No TEMPLATE in current; insert after FROM
        current_merged = re.sub(
            r"(^FROM\s+.+$)",
            r"\1\n" + ref_template,
            current_content,
            count=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    if template_only:
        return current_merged
    return _replace_from_line(current_merged, base)


# Placeholders we inject so we can find and replace with Ollama template vars.
_HF_SYS_PLACEHOLDER = "\x00ollama_sys\x00"
_HF_USER_PLACEHOLDER = "\x00ollama_user\x00"
_HF_RESP_PLACEHOLDER = "\x00ollama_resp\x00"


def _looks_like_jinja(s: str) -> bool:
    return "{%" in s or "{{" in s


def _extract_template_from_config(data: dict) -> str | None:
    """Get chat template string from tokenizer_config.json-style dict. Returns None if not found."""
    # Direct key (string or dict with "template"/"content")
    for key in ("chat_template", "chat_template_jinja"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
        if isinstance(raw, dict):
            for sub in ("template", "content"):
                if isinstance(raw.get(sub), str) and raw[sub].strip():
                    return raw[sub]
    # List of template configs (e.g. [{"name": "gemma", "template": "..."}])
    raw = data.get("chat_template")
    if isinstance(raw, list) and raw:
        first = raw[0] if isinstance(raw[0], dict) else None
        if first:
            for sub in ("template", "content"):
                if isinstance(first.get(sub), str) and first[sub].strip():
                    return first[sub]
    # Fallback: any string value in the config that looks like Jinja (e.g. Gemma or custom keys)
    def find_jinja(obj: object) -> str | None:
        if isinstance(obj, str) and obj.strip() and _looks_like_jinja(obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                found = find_jinja(v)
                if found:
                    return found
        if isinstance(obj, list):
            for v in obj:
                found = find_jinja(v)
                if found:
                    return found
        return None

    return find_jinja(data)


def _ensure_chat_template(tokenizer: object, checkpoint_dir: Path) -> bool:
    """If tokenizer has no chat_template, try loading from tokenizer_config.json or chat_template.jinja."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return True
    cfg_path = checkpoint_dir / "tokenizer_config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        template = _extract_template_from_config(data)
        if template:
            tokenizer.chat_template = template
            return True
    jinja_path = checkpoint_dir / "chat_template.jinja"
    if jinja_path.is_file():
        try:
            tokenizer.chat_template = jinja_path.read_text(encoding="utf-8")
            return True
        except Exception:
            pass
    return False


def _is_gemma_checkpoint(checkpoint_dir: Path) -> bool:
    """True if config or tokenizer_config indicates a Gemma model (Gemma 2/3)."""
    for name in ("config.json", "tokenizer_config.json"):
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        # tokenizer_config.json from Gemma has "tokenizer_class": "GemmaTokenizer"
        if data.get("tokenizer_class", "").lower().startswith("gemma"):
            return True
        # config.json has "model_type": "gemma2" or similar
        if "gemma" in (data.get("model_type") or "").lower():
            return True
    return False


# Built-in Ollama template for Gemma 2/3 when the checkpoint has no chat_template saved.
# BOS at start (Gemma expects it once); then <<start_of_turn>>user\n{prompt}<<end_of_turn>>\n<<start_of_turn>>model\n{response}
_GEMMA_OLLAMA_TEMPLATE = """<bos>{{ if .System }}{{ .System }}

{{ end }}<<start_of_turn>>user
{{ .Prompt }}<<end_of_turn>>
<<start_of_turn>>model
{{ .Response }}"""


def _apply_and_build(
    tokenizer: object,
    messages: list[dict],
    add_gen: bool,
) -> str | None:
    """Run apply_chat_template (tokenize=False then True) and return string or None."""
    for tokenize_val in (False, True):
        try:
            if tokenize_val:
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_gen,
                )
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                out = tokenizer.decode(ids, skip_special_tokens=False)
            else:
                out = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_gen,
                )
            if isinstance(out, str):
                return out
        except Exception:
            continue
    return None


def template_from_hf_checkpoint_with_reason(checkpoint_dir: str | Path) -> tuple[str | None, str | None]:
    """
    Derive an Ollama TEMPLATE from the checkpoint tokenizer. Returns (template_content, None) on
    success or (None, error_reason) on failure. Use this when you need to show why derivation failed.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "config.json").is_file():
        return (None, "checkpoint has no config.json")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return (None, "transformers not installed")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    except Exception as e:
        return (None, f"could not load tokenizer: {e}")
    if not _ensure_chat_template(tokenizer, checkpoint_dir):
        # Checkpoint has no chat_template (e.g. GemmaTokenizer doesn't persist it). Use built-in for Gemma.
        if _is_gemma_checkpoint(checkpoint_dir):
            return (_GEMMA_OLLAMA_TEMPLATE, None)
        return (None, "no chat_template on tokenizer and none in tokenizer_config.json or chat_template.jinja")
    messages_sys_user = [
        {"role": "system", "content": _HF_SYS_PLACEHOLDER},
        {"role": "user", "content": _HF_USER_PLACEHOLDER},
    ]
    messages_user_only = [{"role": "user", "content": _HF_USER_PLACEHOLDER}]
    out = _apply_and_build(tokenizer, messages_sys_user, add_gen=True)
    if out is None:
        out = _apply_and_build(tokenizer, messages_user_only, add_gen=True)
    if out is None:
        for role in ("assistant", "model"):
            messages_full = messages_sys_user + [{"role": role, "content": _HF_RESP_PLACEHOLDER}]
            out = _apply_and_build(tokenizer, messages_full, add_gen=False)
            if out is not None:
                break
    if out is None:
        return (None, "apply_chat_template failed for system+user, user-only, and full conversation (assistant/model)")
    # Build Ollama template: map placeholders and append {{ .Response }} or truncate after it
    if _HF_RESP_PLACEHOLDER in out:
        out = out.replace(_HF_SYS_PLACEHOLDER, "{{ .System }}")
        out = out.replace(_HF_USER_PLACEHOLDER, "{{ .Prompt }}")
        out = out.replace(_HF_RESP_PLACEHOLDER, "{{ .Response }}")
        out = out.strip()
        idx = out.index("{{ .Response }}") + len("{{ .Response }}")
        out = out[:idx]
    else:
        out = out.replace(_HF_SYS_PLACEHOLDER, "{{ .System }}")
        out = out.replace(_HF_USER_PLACEHOLDER, "{{ .Prompt }}")
        out = out.strip()
        out = out + "{{ .Response }}"
    return (out, None)


def template_from_hf_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """
    Derive an Ollama TEMPLATE string from the checkpoint's Hugging Face tokenizer chat template.
    Returns the template content or None if not possible. For a failure reason use template_from_hf_checkpoint_with_reason.
    """
    template, _ = template_from_hf_checkpoint_with_reason(checkpoint_dir)
    return template


def modelfile_append_template(modelfile_content: str, template_body: str) -> str:
    """Append or replace TEMPLATE block in modelfile content with the given template body."""
    # Do not escape quotes: inside """...""" content is literal; escaping would insert backslashes.
    if '"""' in template_body:
        template_body = template_body.replace('"""', '\\"\\"\\"')
    block = 'TEMPLATE """' + template_body + '"""'
    if re.search(r"TEMPLATE\s+\"\"\"", modelfile_content, re.IGNORECASE):
        return re.sub(
            r"TEMPLATE\s+\"\"\".*?\"\"\"",
            block,
            modelfile_content,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return modelfile_content.rstrip() + "\n" + block + "\n"


# When using built-in Gemma template, always add these stops so generation ends even if tokenizer has empty eos.
_GEMMA_STOP_SEQUENCES = (
    "<<end_of_turn>>",
    "<<start_of_turn>>",  # stop if model starts next turn (avoids runaway)
    "<eos>",
    "<|endoftext|>",
)


def get_stop_tokens_from_checkpoint(checkpoint_dir: str | Path) -> list[str]:
    """
    Get stop sequences from the checkpoint tokenizer (eos_token, pad_token, common end-of-turn).
    Uses model family detection to include family-specific stop tokens.
    """
    from ollama_forge.model_family import get_family_stop_tokens

    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "config.json").is_file():
        return []
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str | None) -> None:
        if not s or not s.strip() or s in seen:
            return
        seen.add(s)
        out.append(s)

    # Add family-specific stop tokens first (Gemma, Llama3, Mistral, etc.)
    for s in get_family_stop_tokens(checkpoint_dir):
        add(s)

    # Gemma: also read eos/pad from tokenizer_config so we add exact tokens from config
    if _is_gemma_checkpoint(checkpoint_dir):
        cfg_path = checkpoint_dir / "tokenizer_config.json"
        if cfg_path.is_file():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                for key in ("eos_token", "pad_token"):
                    if isinstance(data.get(key), str) and data[key].strip():
                        add(data[key])
            except Exception:
                pass

    try:
        from transformers import AutoTokenizer
    except ImportError:
        return out
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    except Exception:
        return out

    # Tokenizer EOS / PAD
    if getattr(tokenizer, "eos_token", None):
        add(tokenizer.eos_token)
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        try:
            add(tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False))
        except Exception:
            pass
    if getattr(tokenizer, "pad_token", None) and tokenizer.pad_token != getattr(tokenizer, "eos_token", None):
        add(tokenizer.pad_token)
    elif getattr(tokenizer, "pad_token_id", None) is not None and tokenizer.pad_token_id != getattr(tokenizer, "eos_token_id", None):
        try:
            add(tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False))
        except Exception:
            pass

    # Common end-of-turn / EOS in chat templates (avoid endless repetition when eos is empty)
    for candidate in (
        "<<end_of_turn>>",
        "<|end_of_turn|>",
        "<|end|>",
        "<|eot_id|>",
        "<|return|>",
    ):
        add(candidate)

    return out


def modelfile_append_stop_parameters(modelfile_content: str, stop_tokens: list[str]) -> str:
    """Append PARAMETER stop \"...\" for each stop token. Skips empty and dedupes by insertion order."""
    if not stop_tokens:
        return modelfile_content
    seen: set[str] = set()
    lines = modelfile_content.rstrip().split("\n")
    for s in stop_tokens:
        if not s or s in seen:
            continue
        seen.add(s)
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'PARAMETER stop "{escaped}"')
    return "\n".join(lines) + "\n"


def modelfile_append_num_predict(modelfile_content: str, num_predict: int = 2048) -> str:
    """Append PARAMETER num_predict N so generation stops after N tokens even if stop sequences don't fire."""
    if num_predict <= 0:
        return modelfile_content
    # Avoid duplicate
    if re.search(r"PARAMETER\s+num_predict\s+\d+", modelfile_content, re.IGNORECASE):
        return modelfile_content
    return modelfile_content.rstrip() + f"\nPARAMETER num_predict {num_predict}\n"


def build_modelfile(
    base: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    adapter: str | None = None,
) -> str:
    """Build Modelfile content from base and optional overrides."""
    lines = [f"FROM {base}"]
    if system is not None:
        escaped = system.replace('"""', '\\"\\"\\"')
        if "\n" in escaped:
            lines.append(f'SYSTEM """\n{escaped}\n"""')
        else:
            lines.append(f'SYSTEM """{escaped}"""')
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature}")
    if num_ctx is not None:
        lines.append(f"PARAMETER num_ctx {num_ctx}")
    if top_p is not None:
        lines.append(f"PARAMETER top_p {top_p}")
    if repeat_penalty is not None:
        lines.append(f"PARAMETER repeat_penalty {repeat_penalty}")
    if adapter is not None:
        lines.append(f"ADAPTER {adapter}")
    return "\n".join(lines) + "\n"
