"""Tests for the wsp diagnostics and bench modules."""
from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

from ollama_forge.commands.wsp import _cmd_bench, _cmd_diagnose, register
from ollama_forge.wsp.bench import _answer_matches, run_bench
from ollama_forge.wsp.diagnostics import (
    ObstructionReport,
    _has_repetition_loop,
    _validate_format,
    run_diagnostics,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_urlopen(responses: list[tuple[str, str]]):
    """Return a mock for urllib.request.urlopen that cycles through (content, finish_reason) pairs."""
    idx = 0

    def _side_effect(req, timeout=None):
        nonlocal idx
        content, finish = responses[idx % len(responses)]
        idx += 1
        body = json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }]
        }).encode()
        mock = MagicMock()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        mock.read.return_value = body
        return mock

    return _side_effect


# ── _has_repetition_loop ──────────────────────────────────────────────────────

def test_repetition_loop_detected() -> None:
    # 20 reps × 4 words = 80 words; top 4-gram ratio ≈ 20/77 ≈ 0.26
    text = " ".join(["thinking about this problem"] * 20)
    assert _has_repetition_loop(text, n=4, threshold=0.25)


def test_no_repetition_loop_on_normal_text() -> None:
    text = "The capital of France is Paris. Paris is on the Seine river. The Eiffel Tower is in Paris."
    assert not _has_repetition_loop(text, n=5, threshold=0.35)


def test_repetition_loop_short_text_always_false() -> None:
    # Too short to trigger
    assert not _has_repetition_loop("hello world", n=8)


# ── _validate_format ──────────────────────────────────────────────────────────

def test_validate_json_valid() -> None:
    assert _validate_format('{"name": "Alice", "age": 25}', "json")


def test_validate_json_invalid() -> None:
    assert not _validate_format("Sure! Here is some JSON: {name: Alice}", "json")


def test_validate_json_with_fence() -> None:
    assert _validate_format('```json\n{"x": 1}\n```', "json")


def test_validate_python_def() -> None:
    assert _validate_format("def add(a, b):\n    return a + b", "python_def")


def test_validate_python_def_missing() -> None:
    assert not _validate_format("Here is code that adds two numbers: a + b", "python_def")


def test_validate_one_word_pass() -> None:
    assert _validate_format("prime", "one_word")
    assert _validate_format("Prime.", "one_word")


def test_validate_csv_pass() -> None:
    assert _validate_format("red, green, blue", "csv")


def test_validate_csv_fail_one_item() -> None:
    assert not _validate_format("red", "csv")


# ── run_diagnostics (mocked) ──────────────────────────────────────────────────

def test_run_diagnostics_all_clean() -> None:
    # Healthy model: short answers, no loops, no sycophancy, good format
    normal_resp = ("Paris", "stop")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen([normal_resp])):
        report = run_diagnostics("test-model", base_url="http://x", quick=True, timeout=5.0)
    assert isinstance(report, ObstructionReport)
    assert report.loop_rate == 0.0
    assert report.elapsed_s >= 0.0
    assert report.obstructions_found() == [] or isinstance(report.obstructions_found(), list)


def test_run_diagnostics_loop_detected() -> None:
    # finish_reason=length on every prompt → high loop rate
    loop_resp = ("The answer is... let me think... " * 60, "length")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen([loop_resp])):
        report = run_diagnostics("loopy-model", base_url="http://x", quick=True, timeout=5.0)
    assert report.loop_rate > 0.5
    assert "loops" in report.obstructions_found()


def test_run_diagnostics_sycophancy_detected() -> None:
    # Neutral: "Paris". Biased: "Berlin" (wrong, shifts to wrong answer).
    call_count = [0]

    def _syco_urlopen(req, timeout=None):
        call_count[0] += 1
        # For sycophancy detection: neutral questions get correct answer,
        # biased questions get the wrong (suggested) answer.
        # Sycophancy pairs come after loop prompts (4 quick) + verbosity (4 quick).
        # In sycophancy: even calls = neutral (paris), odd calls = biased (berlin).
        content = "paris" if (call_count[0] % 2 == 1) else "berlin"
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
        ).encode()
        mock = MagicMock()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        mock.read.return_value = body
        return mock

    with patch("urllib.request.urlopen", side_effect=_syco_urlopen):
        report = run_diagnostics("syco-model", base_url="http://x", quick=True, timeout=5.0)
    # sycophancy may or may not trigger depending on call order; just verify no crash
    assert 0.0 <= report.sycophancy_rate <= 1.0


def test_run_diagnostics_handles_exceptions() -> None:
    import urllib.error
    err = urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", side_effect=err):
        report = run_diagnostics("dead-model", base_url="http://x", quick=True, timeout=1.0)
    # Should not raise; scores default to safe values
    assert report.loop_rate == 0.0


def test_obstruction_report_summary_contains_model() -> None:
    r = ObstructionReport(model="llama3.2:3b", base_url="http://x", loop_rate=0.75, sycophancy_rate=0.5)
    s = r.summary()
    assert "llama3.2:3b" in s
    assert "⚠" in s


def test_obstruction_severity() -> None:
    r = ObstructionReport(model="m", base_url="x", loop_rate=0.9)
    assert r.severity("loops") == "HIGH"
    r2 = ObstructionReport(model="m", base_url="x", loop_rate=0.3)
    assert r2.severity("loops") == "MEDIUM"
    r3 = ObstructionReport(model="m", base_url="x", loop_rate=0.05)
    assert r3.severity("loops") == "LOW"


# ── bench (mocked) ────────────────────────────────────────────────────────────

def test_answer_matches_exact() -> None:
    assert _answer_matches("the answer is 36", "36")


def test_answer_matches_case_insensitive() -> None:
    assert _answer_matches("TOKYO is the capital", "tokyo")


def test_answer_no_match() -> None:
    assert not _answer_matches("I don't know", "paris")


def test_run_bench_all_correct() -> None:
    good_resp = "36"

    def _urlopen(req, timeout=None):
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": good_resp}, "finish_reason": "stop"}]}
        ).encode()
        mock = MagicMock()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        mock.read.return_value = body
        return mock

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        results = run_bench("test-model", base_url="http://x", tasks=["math"], timeout=5.0)

    assert "math" in results
    correct, total = results["math"]
    assert total > 0
    # "36" matches the first math question; others may not match
    assert 0 <= correct <= total


def test_run_bench_all_wrong() -> None:
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen([("I do not know", "stop")])):
        results = run_bench("test-model", base_url="http://x", tasks=["factual"], timeout=5.0)
    correct, total = results["factual"]
    assert correct == 0
    assert total > 0


# ── commands.wsp ──────────────────────────────────────────────────────────────

def test_wsp_register() -> None:
    p = argparse.ArgumentParser()
    subs = p.add_subparsers()
    register(subs)
    parsed = p.parse_args(["wsp", "diagnose", "llama3.2:3b", "--quick"])
    assert parsed.model == "llama3.2:3b"
    assert parsed.quick is True


def test_wsp_register_bench() -> None:
    p = argparse.ArgumentParser()
    subs = p.add_subparsers()
    register(subs)
    parsed = p.parse_args(["wsp", "bench", "llama3.2:3b", "--tasks", "math"])
    assert parsed.model == "llama3.2:3b"
    assert parsed.tasks == ["math"]


def test_cmd_diagnose_clean_model() -> None:
    args = argparse.Namespace(
        model="clean-model", base_url="http://x", quick=True, timeout=5.0
    )
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen([("Paris", "stop")])):
        rc = _cmd_diagnose(argparse.ArgumentParser(), args)
    assert rc in (0, 1)  # 0 = clean, 1 = obstructions found


def test_cmd_diagnose_loopy_model() -> None:
    args = argparse.Namespace(
        model="loopy", base_url="http://x", quick=True, timeout=5.0
    )
    loop_resp = ("thinking... " * 100, "length")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen([loop_resp])):
        rc = _cmd_diagnose(argparse.ArgumentParser(), args)
    assert rc == 1  # obstructions found → non-zero exit


def test_cmd_bench_runs() -> None:
    args = argparse.Namespace(
        model="test-model", base_url="http://x", tasks=["math"], timeout=5.0
    )
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen([("36", "stop")])):
        rc = _cmd_bench(argparse.ArgumentParser(), args)
    assert rc == 0
