"""Helpers for formatting model text emitted by TurboQuant runtimes."""

from __future__ import annotations

from typing import Any

_REASONING_OPEN_TAGS = ("<think>", "<thinking>", "<reasoning>", "<analysis>")
_REASONING_CLOSE_TAGS = ("</think>", "</thinking>", "</reasoning>", "</analysis>")
_ALL_REASONING_TAGS = _REASONING_OPEN_TAGS + _REASONING_CLOSE_TAGS
_MAX_TAG_LEN = max(len(tag) for tag in _ALL_REASONING_TAGS)
_INITIAL_HOLD_CHARS = 16384
_HARD_BOUNDARY_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|tool|>",
    "<|im_start|>",
    "<|eot_id|>",
    "<|end_of_turn|>",
    "<end_of_turn>",
    "<<start_of_turn>>",
    "<<end_of_turn>>",
    "<|start_header_id|>",
    "<|endoftext|>",
    "<|end_of_text|>",
)
_BOUNDARY_HINTS = (
    "end_of_turn",
    "start_of_turn",
    "eot_id",
    "im_start",
    "endoftext",
    "end_of_text",
    "start_header_id",
)
_SOFT_STRIP_MARKERS = (
    "<|im_end|>",
    "<|end_header_id|>",
    "<|end|>",
)


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _looks_like_boundary_marker(token: str) -> bool:
    lowered = token.lower()
    return any(hint in lowered for hint in _BOUNDARY_HINTS)


def _boundary_markers(tokenizer: Any = None) -> tuple[str, ...]:
    tok = _unwrap_tokenizer(tokenizer)
    markers = list(_HARD_BOUNDARY_MARKERS)
    seen = set(markers)

    eos_token = getattr(tok, "eos_token", None)
    if isinstance(eos_token, str) and eos_token and _looks_like_boundary_marker(eos_token) and eos_token not in seen:
        markers.append(eos_token)
        seen.add(eos_token)

    special_tokens = getattr(tok, "all_special_tokens", None) or []
    for token in special_tokens:
        if not token or token in _ALL_REASONING_TAGS or not _looks_like_boundary_marker(token) or token in seen:
            continue
        markers.append(token)
        seen.add(token)

    return tuple(markers)


def _soft_strip_markers(tokenizer: Any = None) -> tuple[str, ...]:
    tok = _unwrap_tokenizer(tokenizer)
    markers = list(_SOFT_STRIP_MARKERS)
    seen = set(markers)

    eos_token = getattr(tok, "eos_token", None)
    if isinstance(eos_token, str) and eos_token and eos_token not in seen:
        markers.append(eos_token)
        seen.add(eos_token)

    return tuple(markers)


def _strip_special_tokens(text: str, tokenizer: Any) -> str:
    """Remove tokenizer special-token strings from decoded text."""
    tok = _unwrap_tokenizer(tokenizer)
    special_tokens = list(getattr(tok, "all_special_tokens", None) or [])
    special_tokens.extend(_soft_strip_markers(tokenizer))
    for token in special_tokens:
        if token and token not in _ALL_REASONING_TAGS:
            text = text.replace(token, "")
    return text


def _tail_keep(tokenizer: Any) -> int:
    """Number of trailing characters to keep for split tag/token detection."""
    tok = _unwrap_tokenizer(tokenizer)
    special_tokens = list(getattr(tok, "all_special_tokens", None) or [])
    special_tokens.extend(_soft_strip_markers(tokenizer))
    max_special_len = max((len(token) for token in special_tokens if token), default=0)
    max_boundary_len = max((len(marker) for marker in _boundary_markers(tokenizer)), default=0)
    max_len = max(_MAX_TAG_LEN, max_boundary_len)
    if max_special_len:
        return max_len + max_special_len
    return max_len


def _truncate_at_boundary(text: str, tokenizer: Any = None) -> tuple[str, bool]:
    """Trim leaked chat-template/control markers from model output."""
    best_idx = -1
    for marker in _boundary_markers(tokenizer):
        idx = text.find(marker)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
    if best_idx == -1:
        return text, False
    return text[:best_idx].rstrip(), True


class ReasoningScrubber:
    """Incrementally normalizes hidden reasoning into visible <think> blocks."""

    def __init__(self):
        self._buffer = ""
        self._in_reasoning = False
        self._emitted_visible_text = False
        self._prefix_confirmed_clean = False
        self._emitted_reasoning_open = False
        self._stopped = False

    def feed(self, text: str, tokenizer: Any = None) -> str:
        if self._stopped:
            return ""
        self._buffer += text
        out: list[str] = []
        tail_keep = _tail_keep(tokenizer)

        while True:
            if self._in_reasoning:
                boundary_idx, _ = _find_earliest_tag(self._buffer, _boundary_markers(tokenizer))
                close_idx, close_tag = _find_earliest_tag(self._buffer, _REASONING_CLOSE_TAGS)
                if boundary_idx != -1 and (close_idx == -1 or boundary_idx < close_idx):
                    reasoning = _strip_special_tokens(self._buffer[:boundary_idx], tokenizer)
                    if reasoning:
                        out.append(reasoning.rstrip())
                    out.append("</think>")
                    self._buffer = ""
                    self._in_reasoning = False
                    self._emitted_reasoning_open = False
                    self._stopped = True
                    break
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
            boundary_idx, boundary_tag = _find_earliest_tag(self._buffer, _boundary_markers(tokenizer))

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

            if boundary_tag is not None and (open_tag is None or boundary_idx < open_idx):
                visible = _strip_special_tokens(self._buffer[:boundary_idx], tokenizer).rstrip()
                if visible:
                    out.append(visible)
                    if visible.strip():
                        self._emitted_visible_text = True
                        self._prefix_confirmed_clean = True
                self._buffer = ""
                self._stopped = True
                break

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
        if self._stopped:
            self._buffer = ""
            return ""
        if self._in_reasoning:
            tail = _strip_special_tokens(self._buffer, tokenizer)
            self._buffer = ""
            self._in_reasoning = False
            self._emitted_reasoning_open = False
            tail, _ = _truncate_at_boundary(tail, tokenizer)
            return f"{tail}</think>"
        tail = _strip_special_tokens(self._buffer, tokenizer)
        self._buffer = ""
        tail, _ = _truncate_at_boundary(tail, tokenizer)
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
