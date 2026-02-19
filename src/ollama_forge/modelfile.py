"""Modelfile generation and creation from base."""

from __future__ import annotations

import re
from pathlib import Path


def _extract_template_block(content: str) -> str | None:
    """Extract the TEMPLATE \"\"\"...\"\"\" block from Modelfile text. Returns full block or None."""
    # Match TEMPLATE followed by optional whitespace and """, then content until closing """
    m = re.search(r"TEMPLATE\s+(\"\"\".*?\"\"\")", content, re.DOTALL | re.IGNORECASE)
    if m:
        return "TEMPLATE " + m.group(1)
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


def template_from_hf_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """
    Derive an Ollama TEMPLATE string from the checkpoint's Hugging Face tokenizer chat template.
    Uses add_generation_prompt=True (system + user only) then appends {{ .Response }} so the
    prompt ends with the assistant start token and Ollama generates into .Response. This works
    for templates like gpt-oss that expect generation to follow "<|start|>assistant".
    Returns the template content (inside the triple-quoted block) or None if not possible.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "config.json").is_file():
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    except Exception:
        return None
    if getattr(tokenizer, "chat_template", None) is None:
        return None
    # Prefer: system + user with add_generation_prompt=True, then append {{ .Response }}.
    # That matches templates (e.g. gpt-oss) where generation follows "<|start|>assistant".
    messages_gen = [
        {"role": "system", "content": _HF_SYS_PLACEHOLDER},
        {"role": "user", "content": _HF_USER_PLACEHOLDER},
    ]
    out = None
    for tokenize_val, add_gen in [(False, True), (True, True)]:
        try:
            if tokenize_val:
                ids = tokenizer.apply_chat_template(
                    messages_gen,
                    tokenize=True,
                    add_generation_prompt=add_gen,
                )
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                out = tokenizer.decode(ids, skip_special_tokens=False)
            else:
                out = tokenizer.apply_chat_template(
                    messages_gen,
                    tokenize=False,
                    add_generation_prompt=add_gen,
                )
            if isinstance(out, str):
                break
        except Exception:
            out = None
    if not isinstance(out, str):
        # Fallback: full conversation with assistant placeholder, then truncate after .Response
        messages_full = messages_gen + [{"role": "assistant", "content": _HF_RESP_PLACEHOLDER}]
        try:
            out = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            try:
                ids = tokenizer.apply_chat_template(
                    messages_full,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                out = tokenizer.decode(ids, skip_special_tokens=False)
            except Exception:
                return None
        if not isinstance(out, str):
            return None
        out = out.replace(_HF_SYS_PLACEHOLDER, "{{ .System }}")
        out = out.replace(_HF_USER_PLACEHOLDER, "{{ .Prompt }}")
        out = out.replace(_HF_RESP_PLACEHOLDER, "{{ .Response }}")
        out = out.strip()
        resp_marker = "{{ .Response }}"
        if resp_marker in out:
            idx = out.index(resp_marker) + len(resp_marker)
            out = out[:idx]
        return out
    out = out.replace(_HF_SYS_PLACEHOLDER, "{{ .System }}")
    out = out.replace(_HF_USER_PLACEHOLDER, "{{ .Prompt }}")
    out = out.strip()
    out = out + "{{ .Response }}"
    return out


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
