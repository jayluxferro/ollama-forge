"""Context budget manager — tracks token usage across a conversation and warns near limits.

For local llama.cpp / mlx backends the KV cache is a prefix cache: re-sending
the same prefix is fast (cache hit) but each new token counts against the
context window. This manager tracks the running total and warns before
the model starts silently truncating.

Token estimation: 4 characters per token is a rough average that holds for most
English prose and code. Install `tiktoken` for precise counts (auto-detected).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _messages_tokens(messages: list[dict]) -> int:
    # 4 overhead tokens per message (role + separators) is a tiktoken-calibrated heuristic.
    total = 0
    for m in messages:
        total += _estimate_tokens(m.get("content") or "")
        total += 4
    return total + 2  # reply priming


@dataclass
class BudgetSnapshot:
    used: int
    max_tokens: int
    messages: int

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    @property
    def fraction_used(self) -> float:
        return self.used / self.max_tokens if self.max_tokens else 0.0

    def __str__(self) -> str:
        pct = self.fraction_used * 100
        return f"tokens {self.used}/{self.max_tokens} ({pct:.0f}%) — {self.remaining} remaining"


class ContextBudget:
    """
    Running token-usage tracker for a conversation.

    Usage:
        budget = ContextBudget(max_tokens=8192)
        budget.add(messages)
        if budget.fraction_used > 0.8:
            # trim oldest non-system messages
        snapshot = budget.snapshot()
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        warn_threshold: float = 0.8,
        warn_fn: object = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.warn_threshold = warn_threshold
        self._warn_fn = warn_fn or print
        self._used: int = 0
        self._message_count: int = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._used)

    @property
    def fraction_used(self) -> float:
        return self._used / self.max_tokens if self.max_tokens else 0.0

    def add(self, messages: list[dict]) -> BudgetSnapshot:
        """Add token counts for *messages* and optionally warn if near the limit."""
        n = _messages_tokens(messages)
        self._used += n
        self._message_count += len(messages)
        snap = self.snapshot()
        if self.fraction_used >= self.warn_threshold:
            self._warn_fn(
                f"[ContextBudget] ⚠ {snap.fraction_used*100:.0f}% full — "
                f"{snap.remaining} tokens remaining (max {self.max_tokens})"
            )
        return snap

    def reset(self) -> None:
        self._used = 0
        self._message_count = 0

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            used=self._used,
            max_tokens=self.max_tokens,
            messages=self._message_count,
        )

    @contextmanager
    def track(self) -> Generator["ContextBudget", None, None]:
        """Context manager that resets the budget on exit."""
        self.reset()
        try:
            yield self
        finally:
            self.reset()

    def trim_to_fit(self, messages: list[dict], reserve: int = 512) -> list[dict]:
        """
        Remove oldest non-system messages until the remaining batch fits.

        Preserves the first system message if present. Useful before each API
        call in an agentic loop to avoid silently hitting the context limit.
        """
        target = self.max_tokens - reserve
        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        while rest and _messages_tokens(system + rest) > target:
            rest.pop(0)

        return system + rest


_SHARED_BUDGETS: dict[str, ContextBudget] = {}


def get_shared(name: str = "default", max_tokens: int = 4096) -> ContextBudget:
    """Return a named shared ContextBudget (created on first access)."""
    if name not in _SHARED_BUDGETS:
        _SHARED_BUDGETS[name] = ContextBudget(max_tokens=max_tokens)
    return _SHARED_BUDGETS[name]
