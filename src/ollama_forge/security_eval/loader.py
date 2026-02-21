"""Load prompt sets from TXT (one prompt per line) or JSONL (prompt, category, ...)."""

from __future__ import annotations

import base64
import json
from pathlib import Path


def _resolve_image_to_base64(image_spec: str, base_dir: Path | None = None) -> str | None:
    """Resolve image path or data URL to base64 string. Returns None if invalid."""
    s = (image_spec or "").strip()
    if not s:
        return None
    if s.startswith("data:") and "base64," in s:
        try:
            return s.split("base64,", 1)[1].strip()
        except IndexError:
            return None
    if s.startswith(("http://", "https://")):
        try:
            import urllib.request

            with urllib.request.urlopen(s, timeout=10) as resp:
                raw = resp.read()
            return base64.b64encode(raw).decode("ascii")
        except Exception:
            return None
    # file path (relative to base_dir or cwd)
    p = Path(s)
    if base_dir and not p.is_absolute():
        p = base_dir / p
    if p.is_file():
        try:
            return base64.b64encode(p.read_bytes()).decode("ascii")
        except Exception:
            return None
    return None


def load_prompt_set(path: str | Path) -> list[dict]:
    """
    Load a prompt set from a file.
    - .txt: one prompt per line; lines starting with # skipped. Each row gets category="default".
    - .jsonl: one JSON per line with "prompt"; may have category, expected_refusal, target_for_extraction,
      tools, dangerous_tool_names, image (path/URL/base64 data URL), turns.
    Returns list of dicts: prompt, category, expected_refusal, target_for_extraction, context, system,
      turns, tools, dangerous_tool_names, image (resolved to base64 if path/URL), image_reveal_keywords.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt set not found: {path}")

    base_dir = path.parent
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("text") or ""
                if not prompt and not obj.get("turns"):
                    continue
                if not prompt and obj.get("turns"):
                    prompt = "(multi-turn)"
                context = obj.get("context") or obj.get("injected_document") or ""
                if context:
                    context = context.strip()
                tools = obj.get("tools")
                if tools is not None and not isinstance(tools, list):
                    tools = None
                dangerous_tool_names = obj.get("dangerous_tool_names")
                if dangerous_tool_names is not None and not isinstance(dangerous_tool_names, list):
                    dangerous_tool_names = [str(dangerous_tool_names)] if dangerous_tool_names else []
                image_spec = obj.get("image") or obj.get("image_url")
                image_b64 = _resolve_image_to_base64(image_spec, base_dir) if image_spec else None
                row = {
                    "prompt": prompt.strip() if isinstance(prompt, str) else "(multi-turn)",
                    "category": obj.get("category") or obj.get("attack_type") or "default",
                    "expected_refusal": obj.get("expected_refusal"),
                    "target_for_extraction": obj.get("target_for_extraction"),
                    "context": context,
                    "system": obj.get("system") or obj.get("system_prompt"),
                    "turns": obj.get("turns") if isinstance(obj.get("turns"), list) else None,
                    "tools": tools,
                    "dangerous_tool_names": dangerous_tool_names or [],
                    "image": image_b64,
                    "image_reveal_keywords": (
                        list(kw) if isinstance((kw := obj.get("image_reveal_keywords")), list) else [kw] if kw else []
                    ),
                }
                rows.append(row)
    else:
        # .txt or any other: one prompt per line
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(
                    {
                        "prompt": line,
                        "category": "default",
                        "expected_refusal": None,
                        "target_for_extraction": None,
                        "context": "",
                    }
                )
    return rows
