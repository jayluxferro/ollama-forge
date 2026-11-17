"""Lightweight benchmark for before/after comparisons.

Intentionally small (5–10 questions per task) so it can run in under a minute.
For serious evaluation use MATH-500 / GSM8K / IFEval directly.
"""
from __future__ import annotations

import json
import re
import urllib.request

# ── question banks ────────────────────────────────────────────────────────────

_MATH_QA: list[tuple[str, str]] = [
    ("What is 15% of 240?", "36"),
    ("Solve: 3x + 7 = 22. What is x?", "5"),
    ("A rectangle has length 11 cm and perimeter 36 cm. What is its width?", "7"),
    ("What is the square root of 196?", "14"),
    ("If a train travels at 60 mph for 2 hours, how far does it travel?", "120"),
    ("What is 2^8?", "256"),
]

_FACTUAL_QA: list[tuple[str, str]] = [
    ("What is the capital of Japan?", "tokyo"),
    ("How many planets are in our solar system?", "8"),
    ("What is the chemical symbol for gold?", "au"),
    ("In which year did World War II end?", "1945"),
    ("What is the speed of light in km/s (approximate)?", "300000"),
    ("What is the largest ocean on Earth?", "pacific"),
]

_CODING_QA: list[tuple[str, str]] = [
    (
        "Write a Python one-liner to reverse a list called lst. Reply with ONLY the code.",
        "[::-1]",
    ),
    (
        "What Python method removes and returns the last element of a list? Reply with just the method name.",
        "pop",
    ),
    (
        "What is the time complexity of binary search? Reply with ONLY Big-O notation.",
        "o(log",
    ),
    (
        "In Python, what does len([1,2,3]) return? Reply with ONLY the number.",
        "3",
    ),
    (
        "What keyword is used to define a function in Python? Reply with ONE word.",
        "def",
    ),
]

_REASONING_QA: list[tuple[str, str]] = [
    ("If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Reply yes or no.", "yes"),
    (
        "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball."
        " How much does the ball cost in cents?",
        "5",
    ),
    ("What comes next in the sequence: 2, 4, 8, 16, __?", "32"),
    ("If you have 3 apples and take away 2, how many apples do YOU have?", "2"),
    ("A farmer has 17 sheep. All but 9 die. How many are left?", "9"),
]

_TASK_BANKS = {
    "math": _MATH_QA,
    "factual": _FACTUAL_QA,
    "coding": _CODING_QA,
    "reasoning": _REASONING_QA,
}


def _chat_once(prompt: str, model: str, base_url: str, timeout: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return (data["choices"][0]["message"].get("content") or "").strip().lower()


def _answer_matches(response: str, expected: str) -> bool:
    r = response.lower()
    e = expected.lower()
    return e in r or bool(re.search(rf"\b{re.escape(e)}\b", r))


def run_bench(
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    tasks: list[str] | None = None,
    timeout: float = 60.0,
) -> dict[str, tuple[int, int]]:
    """Run benchmark, return {task: (correct, total)} dict."""
    if tasks is None:
        tasks = list(_TASK_BANKS.keys())

    results: dict[str, tuple[int, int]] = {}
    for task in tasks:
        bank = _TASK_BANKS.get(task, [])
        correct = 0
        for prompt, expected in bank:
            try:
                resp = _chat_once(prompt, model, base_url, timeout)
                if _answer_matches(resp, expected):
                    correct += 1
            except Exception:
                pass
        results[task] = (correct, len(bank))
    return results
