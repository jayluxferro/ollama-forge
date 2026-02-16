"""Tests for security_eval: loader, scorers, run (with mocked client)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ollama_forge.security_eval.loader import load_prompt_set
from ollama_forge.security_eval.scorers import score_extraction, score_refusal


def test_load_prompt_set_txt() -> None:
    """TXT: one prompt per line; comments and blank lines skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("# comment\n")
        f.write("first prompt\n")
        f.write("\n")
        f.write("second prompt\n")
        path = f.name
    try:
        rows = load_prompt_set(path)
        assert len(rows) == 2
        assert rows[0]["prompt"] == "first prompt"
        assert rows[0]["category"] == "default"
        assert rows[1]["prompt"] == "second prompt"
        assert rows[1].get("context") == ""
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_prompt_set_jsonl() -> None:
    """JSONL: prompt, category, target_for_extraction, context."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "Hello", "category": "benign"}\n')
        f.write('{"prompt": "Extract", "category": "extraction", "target_for_extraction": "secret"}\n')
        f.write('{"prompt": "Query", "context": "Injected doc here.", "category": "indirect"}\n')
        path = f.name
    try:
        rows = load_prompt_set(path)
        assert len(rows) == 3
        assert rows[0]["prompt"] == "Hello" and rows[0]["category"] == "benign"
        assert rows[1]["target_for_extraction"] == "secret"
        assert rows[2]["context"] == "Injected doc here."
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_prompt_set_missing_file() -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_prompt_set("/nonexistent/prompts.txt")


def test_score_refusal_empty() -> None:
    """Empty or very short response is refusal."""
    assert score_refusal("") is True
    assert score_refusal("   ") is True
    assert score_refusal("No") is True


def test_score_refusal_keywords() -> None:
    """Refusal keywords trigger True."""
    assert score_refusal("I cannot help with that.") is True
    assert score_refusal("I'm not able to assist.") is True
    assert score_refusal("Sorry, I refuse.") is True


def test_score_refusal_compliance() -> None:
    """Normal compliance response is False."""
    assert score_refusal("The answer is 4.") is False
    assert score_refusal("Here is a haiku about the ocean.") is False


def test_score_extraction_target() -> None:
    """Target string in response returns True."""
    assert score_extraction("The secret code is ABC123.", target="ABC123") is True
    assert score_extraction("Nothing here.", target="ABC123") is False


def test_score_extraction_pattern() -> None:
    """Regex pattern works."""
    assert score_extraction("Email: foo@bar.com", pattern=r"[\w.+-]+@[\w.-]+\.\w+") is True
    assert score_extraction("No email here.", pattern=r"[\w.+-]+@[\w.-]+\.\w+") is False


def test_score_extraction_pii() -> None:
    """PII-like target triggers regex check."""
    assert score_extraction("Contact: test@example.com", target="pii") is True
    assert score_extraction("My number is 555-123-4567.", target="pii") is True


def test_run_eval_mocked() -> None:
    """run_eval with mocked client returns KPIs and results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("What is 2+2?\n")
        f.write("I cannot do that.\n")
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            mock_query.side_effect = [
                ("The answer is 4.", 0.1),  # long enough not to be refusal-by-length
                ("I cannot assist with that.", 0.2),
            ]
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(path, base_url="http://localhost:9999", model="test", verbose=False)
        kpis = run_meta["kpis"]
        assert kpis["total"] == 2
        assert kpis["refusals"] == 1
        assert kpis["compliances"] == 1
        assert kpis["asr_pct"] == 50.0
        assert kpis["refusal_rate_pct"] == 50.0
        assert len(run_meta["results"]) == 2
        assert run_meta["results"][0]["compliance"] is True
        assert run_meta["results"][1]["refusal"] is True
    finally:
        Path(path).unlink(missing_ok=True)
