"""Modelfile generation and creation from base."""

from __future__ import annotations

import re


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
