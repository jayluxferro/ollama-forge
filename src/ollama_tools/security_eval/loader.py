"""Load prompt sets from TXT (one prompt per line) or JSONL (prompt, category, ...)."""

from __future__ import annotations

import json
from pathlib import Path


def load_prompt_set(path: str | Path) -> list[dict]:
    """
    Load a prompt set from a file.
    - .txt: one prompt per line; lines starting with # are skipped. Each row gets category="default".
    - .jsonl: one JSON object per line with at least "prompt"; may have "category", "expected_refusal", "target_for_extraction".
    Returns a list of dicts with keys: prompt, category, expected_refusal (bool|None), target_for_extraction (str|None).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt set not found: {path}")

    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("text") or ""
                if not prompt:
                    continue
                # Optional context for indirect prompt injection (RAG-style: model sees context then user query)
                context = obj.get("context") or obj.get("injected_document") or ""
                if context:
                    context = context.strip()
                rows.append({
                    "prompt": prompt.strip(),
                    "category": obj.get("category") or obj.get("attack_type") or "default",
                    "expected_refusal": obj.get("expected_refusal"),
                    "target_for_extraction": obj.get("target_for_extraction"),
                    "context": context,
                })
    else:
        # .txt or any other: one prompt per line
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append({
                    "prompt": line,
                    "category": "default",
                    "expected_refusal": None,
                    "target_for_extraction": None,
                    "context": "",
                })
    return rows
