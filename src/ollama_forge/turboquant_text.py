"""Helpers for formatting model text emitted by TurboQuant runtimes."""

from __future__ import annotations

from typing import Any

_REASONING_OPEN_TAGS = ("<think>", "<thinking>", "<reasoning>", "<analysis>")
_REASONING_CLOSE_TAGS = ("</think>", "</thinking>", "</reasoning>", "</analysis>")
_ALL_REASONING_TAGS = _REASONING_OPEN_TAGS + _REASONING_CLOSE_TAGS
_MAX_TAG_LEN = max(len(tag) for tag in _ALL_REASONING_TAGS)
_INITIAL_HOLD_CHARS = 16384


def _strip_special_tokens(text: str, tokenizer: Any) -> str:
    """Remove tokenizer special-token strings from decoded text."""
    if tokenizer is None:
        return text
    special_tokens = getattr(tokenizer, "all_special_tokens", None) or []
    for token in special_tokens:
        if token and token not in _ALL_REASONING_TAGS:
            text = text.replace(token, "")
    return text


def _tail_keep(tokenizer: Any) -> int:
    """Number of trailing characters to keep for split tag/token detection."""
    special_tokens = getattr(tokenizer, "all_special_tokens", None) or []
    max_special_len = max((len(token) for token in special_tokens if token), default=0)
    if max_special_len:
        return _MAX_TAG_LEN + max_special_len
    return _MAX_TAG_LEN


class ReasoningScrubber:
    """Incrementally normalizes hidden reasoning into visible <think> blocks."""

    def __init__(self):
        self._buffer = ""
        self._in_reasoning = False
        self._emitted_visible_text = False
        self._prefix_confirmed_clean = False
        self._emitted_reasoning_open = False

    def feed(self, text: str, tokenizer: Any = None) -> str:
        self._buffer += text
        out: list[str] = []
        tail_keep = _tail_keep(tokenizer)

        while True:
            if self._in_reasoning:
                close_idx, close_tag = _find_earliest_tag(self._buffer, _REASONING_CLOSE_TAGS)
                if close_tag is not None:
                    reasoning = _strip_special_tokens(self._buffer[:close_idx], tokenizer)
                    if reasoning:
                        out.append(reasoning)
                    out.append("</think>")
                    self._buffer = self._buffer[close_idx + len(close_tag):]
                    self._in_reasoning = False
                    self._emitted_reasoning_open = False
                    continue

                if not self._emitted_reasoning_open:
                    out.append("<think>")
                    self._emitted_reasoning_open = True

                if len(self._buffer) <= tail_keep:
                    break
                reasoning = _strip_special_tokens(self._buffer[:-(tail_keep - 1)], tokenizer)
                if reasoning:
                    out.append(reasoning)
                self._buffer = self._buffer[-(tail_keep - 1):]
                break

            open_idx, open_tag = _find_earliest_tag(self._buffer, _REASONING_OPEN_TAGS)
            close_idx, close_tag = _find_earliest_tag(self._buffer, _REASONING_CLOSE_TAGS)

            if (
                close_tag is not None
                and not self._emitted_visible_text
                and (open_tag is None or close_idx < open_idx)
            ):
                reasoning = _strip_special_tokens(self._buffer[:close_idx], tokenizer)
                out.append("<think>")
                out.append(reasoning)
                out.append("</think>")
                self._buffer = self._buffer[close_idx + len(close_tag):]
                self._prefix_confirmed_clean = True
                continue

            if open_tag is not None:
                visible = self._buffer[:open_idx]
                if visible:
                    out.append(_strip_special_tokens(visible, tokenizer))
                    if visible.strip():
                        self._emitted_visible_text = True
                        self._prefix_confirmed_clean = True
                self._buffer = self._buffer[open_idx + len(open_tag):]
                self._in_reasoning = True
                out.append("<think>")
                self._emitted_reasoning_open = True
                continue

            if not self._prefix_confirmed_clean:
                if len(self._buffer) <= _INITIAL_HOLD_CHARS:
                    break
                safe = self._buffer[:-(tail_keep - 1)]
                if safe:
                    out.append(_strip_special_tokens(safe, tokenizer))
                    if safe.strip():
                        self._emitted_visible_text = True
                        self._prefix_confirmed_clean = True
                self._buffer = self._buffer[-(tail_keep - 1):]
                break

            if len(self._buffer) <= tail_keep:
                break

            safe = self._buffer[:-(tail_keep - 1)]
            if safe:
                out.append(_strip_special_tokens(safe, tokenizer))
                if safe.strip():
                    self._emitted_visible_text = True
                    self._prefix_confirmed_clean = True
            self._buffer = self._buffer[-(tail_keep - 1):]
            break

        return "".join(out)

    def finalize(self, tokenizer: Any = None) -> str:
        if self._in_reasoning:
            tail = _strip_special_tokens(self._buffer, tokenizer)
            self._buffer = ""
            self._in_reasoning = False
            self._emitted_reasoning_open = False
            return f"{tail}</think>"
        tail = _strip_special_tokens(self._buffer, tokenizer)
        self._buffer = ""
        if tail.strip():
            self._emitted_visible_text = True
            self._prefix_confirmed_clean = True
        return tail


def clean_generated_text(text: str, tokenizer: Any = None) -> str:
    """Normalize hidden reasoning and strip stray special tokens."""
    scrubber = ReasoningScrubber()
    visible = scrubber.feed(text, tokenizer)
    visible += scrubber.finalize(tokenizer)
    return visible


def _find_earliest_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str | None]:
    """Return the earliest matching tag and its index."""
    best_idx = -1
    best_tag = None
    for tag in tags:
        idx = text.find(tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
            best_tag = tag
    return best_idx, best_tag
