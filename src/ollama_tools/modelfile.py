"""Modelfile generation and creation from base."""

from __future__ import annotations


def build_modelfile(
    base: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
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
    if adapter is not None:
        lines.append(f"ADAPTER {adapter}")
    return "\n".join(lines) + "\n"
