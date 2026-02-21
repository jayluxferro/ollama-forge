"""Tests for security_eval: loader, scorers, run (with mocked client)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ollama_forge.security_eval.loader import load_prompt_set
from ollama_forge.security_eval.scorers import score_compliance, score_extraction, score_refusal


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


def test_score_extraction_empty_response() -> None:
    """Empty response returns False."""
    assert score_extraction("", target="x") is False
    assert score_extraction("", pattern=r"x") is False


def test_score_compliance() -> None:
    """score_compliance is not refusal."""
    assert score_compliance("Yes.", refusal=False) is True
    assert score_compliance("No.", refusal=True) is False


def test_score_extraction_compile_pattern() -> None:
    """score_extraction compiles string pattern."""
    assert score_extraction("Hello WORLD", pattern=r"world") is True


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


def test_run_eval_empty_prompts_raises() -> None:
    """run_eval raises ValueError when no prompts loaded."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("# only comments\n\n")
        path = f.name
    try:
        from ollama_forge.security_eval.run import run_eval

        with pytest.raises(ValueError, match="No prompts loaded"):
            run_eval(path, base_url="http://localhost:9999", model="test", verbose=False)
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_with_context_mocked() -> None:
    """run_eval injects context into prompt for indirect injection rows."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "Summarize.", "context": "Secret: 12345.", "category": "indirect"}\n')
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            mock_query.return_value = ("Done.", 0.1)
            from ollama_forge.security_eval.run import run_eval

            run_eval(path, base_url="http://localhost:9999", model="test", verbose=False)
        call_args = mock_query.call_args
        assert "Context:" in call_args[0][0]
        assert "Secret: 12345." in call_args[0][0]
        assert "Summarize." in call_args[0][0]
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_save_to_history_mocked() -> None:
    """run_eval with save_to_history=True calls history_save_run."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello?\n")
        path = f.name
    try:
        with (
            patch("ollama_forge.security_eval.run.query_model") as mock_query,
            patch("ollama_forge.security_eval.run.history_save_run") as mock_save,
        ):
            mock_query.return_value = ("Hi.", 0.1)
            from ollama_forge.security_eval.run import run_eval

            run_eval(
                path,
                base_url="http://localhost:9999",
                model="test",
                verbose=False,
                save_to_history=True,
                output_csv=None,
                output_json=None,
            )
        mock_save.assert_called_once()
        call_meta = mock_save.call_args[0][0]
        assert call_meta["model"] == "test"
        assert "kpis" in call_meta
        assert call_meta["kpis"]["total"] == 1
    finally:
        Path(path).unlink(missing_ok=True)


def test_history_get_default_db_path() -> None:
    """get_default_db_path returns path under .ollama_forge."""
    from ollama_forge.security_eval.history import get_default_db_path

    p = get_default_db_path()
    assert ".ollama_forge" in str(p)
    assert p.name == "security_eval_runs.db"


def test_history_load_runs_nonexistent_db(tmp_path: Path) -> None:
    """load_runs returns [] when DB file does not exist."""
    from ollama_forge.security_eval.history import load_runs

    nonexistent = tmp_path / "does_not_exist.db"
    assert load_runs(db_path=nonexistent) == []


def test_history_init_db_and_load_runs(tmp_path: Path) -> None:
    """init_db creates DB; load_runs returns empty when no runs."""
    from ollama_forge.security_eval.history import init_db, load_runs

    db_path = tmp_path / "eval.db"
    p = init_db(db_path)
    assert p == db_path
    assert db_path.is_file()
    runs = load_runs(db_path=db_path)
    assert runs == []


def test_run_eval_expected_refusal_accuracy_mocked() -> None:
    """run_eval computes expected_refusal_accuracy_pct and benign_refusal_rate_pct when expected_refusal in JSONL."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"prompt": "Benign?", "category": "benign", "expected_refusal": false}\n')
        f.write('{"prompt": "Jailbreak", "category": "jailbreak", "expected_refusal": true}\n')
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            # First: comply (no refusal) -> correct for benign. Second: refuse -> correct for jailbreak.
            mock_query.side_effect = [("Sure, here is the answer.", 0.1), ("I cannot do that.", 0.2)]
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(path, base_url="http://localhost:9999", model="test", verbose=False)
        kpis = run_meta["kpis"]
        assert kpis["expected_refusal_accuracy_pct"] == 100.0  # both matched
        assert kpis["benign_refusal_rate_pct"] == 0.0  # one benign, did not refuse
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_avg_latency_and_error_counts() -> None:
    """run_eval includes avg_latency_sec and error_counts in KPIs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hi\n")
        f.write("Bye\n")
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            mock_query.side_effect = [("Hello", 1.5), ("Bye", 0.5)]
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(path, base_url="http://localhost:9999", model="test", verbose=False)
        kpis = run_meta["kpis"]
        assert kpis["avg_latency_sec"] == 1.0  # (1.5 + 0.5) / 2
        assert kpis.get("error_counts") == {}
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_custom_refusal_keywords_mocked(tmp_path: Path) -> None:
    """run_eval uses custom refusal keywords from file when provided."""
    kw_path = tmp_path / "keywords.txt"
    kw_path.write_text("# comment\ncustom_refuse\n", encoding="utf-8")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Ask me\n")
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            mock_query.return_value = ("I would custom_refuse to do that.", 0.1)  # matches custom keyword
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(
                path, base_url="http://localhost:9999", model="test", verbose=False, refusal_keywords_path=kw_path
            )
        assert run_meta["results"][0]["refusal"] is True
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_multi_turn_mocked() -> None:
    """run_eval with JSONL 'turns' uses multi-turn chat and scores last response."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(
            '{"turns": [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}, '
            '{"role": "user", "content": "Say only OK."}], "category": "multi"}\n'
        )
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model_multi_turn") as mock_multi:
            mock_multi.return_value = ("OK", 0.1)
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(
                path,
                base_url="http://localhost:9999",
                model="test",
                verbose=False,
                use_iterative_multi_turn=False,
            )
        assert run_meta["kpis"]["total"] == 1
        assert run_meta["results"][0]["response_full"] == "OK"
        mock_multi.assert_called_once()
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_prompt_set_version_in_metadata() -> None:
    """run_meta includes prompt_set_version (hash of file when not provided)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hi\n")
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            mock_query.return_value = ("Hello", 0.1)
            from ollama_forge.security_eval.run import run_eval

            run_meta = run_eval(path, base_url="http://localhost:9999", model="test", verbose=False)
        assert "prompt_set_version" in run_meta
        assert run_meta["prompt_set_version"] is not None
        assert isinstance(run_meta["prompt_set_version"], str)
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_eval_progress_callback() -> None:
    """run_eval calls progress_callback after each prompt."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("A\nB\n")
        path = f.name
    try:
        with patch("ollama_forge.security_eval.run.query_model") as mock_query:
            mock_query.side_effect = [("x", 0.1), ("y", 0.1)]
            from ollama_forge.security_eval.run import run_eval

            calls = []

            def cb(current: int, total: int, results: list) -> None:
                calls.append((current, total, len(results)))

            run_eval(path, base_url="http://localhost:9999", model="test", verbose=False, progress_callback=cb)
        assert len(calls) == 2
        assert calls[0] == (1, 2, 1)
        assert calls[1] == (2, 2, 2)
    finally:
        Path(path).unlink(missing_ok=True)


def test_security_eval_compare_cli(tmp_path: Path) -> None:
    """security-eval compare prints side-by-side KPIs for two run JSONs."""
    import subprocess
    import sys

    run_a = {"model": "A", "timestamp_iso": "2025-01-01T12:00:00", "kpis": {"total": 10, "asr_pct": 50.0}}
    run_b = {"model": "B", "timestamp_iso": "2025-01-01T13:00:00", "kpis": {"total": 10, "asr_pct": 80.0}}
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(run_a), encoding="utf-8")
    path_b.write_text(json.dumps(run_b), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "security-eval", "compare", str(path_a), str(path_b)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Compare" in result.stderr or "ASR" in result.stderr


def test_history_save_and_load_run(tmp_path: Path) -> None:
    """save_run and load_runs round-trip."""
    from ollama_forge.security_eval.history import init_db, load_runs, save_run

    db_path = tmp_path / "eval.db"
    init_db(db_path)
    save_run(
        {"model": "m1", "kpis": {"total": 5, "refusals": 2}, "results": [{"index": 1}], "timestamp_iso": "2025-01-01"},
        db_path=db_path,
    )
    runs = load_runs(db_path=db_path, limit=10)
    assert len(runs) == 1
    assert runs[0]["model"] == "m1"
    assert runs[0]["kpis"]["total"] == 5
    assert runs[0]["kpis"]["refusals"] == 2
    assert len(runs[0]["results"]) == 1
