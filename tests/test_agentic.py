"""Tests for agentic quality modules: eval, router, budget."""
from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from ollama_forge.agentic.budget import BudgetSnapshot, ContextBudget, get_shared
from ollama_forge.agentic.eval import (
    BUILTIN_SUITE,
    EvalResult,
    ToolCallCase,
    format_results,
    run_eval,
)
from ollama_forge.agentic.router import ModelRouter, auto_config
from ollama_forge.commands.eval import _cmd_eval, register

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_openai_response(fn_name: str, arguments: dict) -> bytes:
    return json.dumps({
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(arguments)},
                }],
            }
        }]
    }).encode()


def _fake_urlopen(fn_name: str, arguments: dict) -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = _make_openai_response(fn_name, arguments)
    return mock


def _str_param(name: str) -> dict:
    return {"type": "string", "description": name}


def _simple_tool(fn_name: str, *param_names: str) -> dict:
    props = {p: _str_param(p) for p in param_names}
    return {
        "type": "function",
        "function": {
            "name": fn_name,
            "description": f"{fn_name} tool",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(param_names),
            },
        },
    }


# ── agentic.eval ─────────────────────────────────────────────────────────────


def test_builtin_suite_has_entries() -> None:
    assert len(BUILTIN_SUITE) >= 6
    for case in BUILTIN_SUITE:
        assert case.name
        assert case.prompt
        assert case.expected_fn
        assert case.tools


def test_run_eval_pass() -> None:
    case = ToolCallCase(
        name="test_weather",
        prompt="Weather in London?",
        tools=[_simple_tool("get_weather", "location")],
        expected_fn="get_weather",
        expected_arg_keys=["location"],
    )
    with patch("urllib.request.urlopen", return_value=_fake_urlopen("get_weather", {"location": "London"})):
        results = run_eval("test-model", cases=[case])
    assert len(results) == 1
    assert results[0].passed
    assert results[0].fn_matched
    assert results[0].args_matched


def test_run_eval_wrong_fn() -> None:
    case = ToolCallCase(
        name="test_calc",
        prompt="Calculate 2+2",
        tools=[_simple_tool("calculate", "expression")],
        expected_fn="calculate",
        expected_arg_keys=["expression"],
    )
    with patch("urllib.request.urlopen", return_value=_fake_urlopen("wrong_fn", {"expression": "2+2"})):
        results = run_eval("test-model", cases=[case])
    assert not results[0].passed
    assert not results[0].fn_matched


def test_run_eval_missing_arg() -> None:
    case = ToolCallCase(
        name="test_send",
        prompt="Send email to bob",
        tools=[_simple_tool("send_email", "to", "subject")],
        expected_fn="send_email",
        expected_arg_keys=["to", "subject"],
    )
    with patch("urllib.request.urlopen", return_value=_fake_urlopen("send_email", {"to": "bob@x.com"})):
        results = run_eval("test-model", cases=[case])
    assert not results[0].passed
    assert results[0].fn_matched
    assert not results[0].args_matched


def test_run_eval_http_error() -> None:
    case = BUILTIN_SUITE[0]
    err = urllib.error.HTTPError(url="", code=500, msg="Internal Server Error", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=err):
        results = run_eval("test-model", cases=[case])
    assert not results[0].passed
    assert results[0].error is not None


def test_format_results_pass() -> None:
    case = BUILTIN_SUITE[0]
    r = EvalResult(case=case, fn_matched=True, args_matched=True, latency_ms=42.0, raw_tool_calls=[])
    text = format_results([r], "test-model")
    assert "✓" in text
    assert "Score: 1/1" in text


def test_format_results_failure() -> None:
    case = BUILTIN_SUITE[0]
    r = EvalResult(case=case, fn_matched=False, args_matched=False, latency_ms=99.0, raw_tool_calls=[])
    text = format_results([r], "test-model")
    assert "✗" in text
    assert "Score: 0/1" in text


# ── agentic.router ────────────────────────────────────────────────────────────


def test_auto_config_code_route() -> None:
    models = ["codellama:7b", "llava:7b", "llama3.2:3b"]
    cfg = auto_config(models)
    assert cfg["code"] == "codellama:7b"


def test_auto_config_vision_route() -> None:
    models = ["llava:7b", "llama3.2:3b", "qwen2.5-coder:7b"]
    cfg = auto_config(models)
    assert cfg["vision"] == "llava:7b"


def test_auto_config_empty() -> None:
    assert auto_config([]) == {}


def test_router_explicit_config() -> None:
    r = ModelRouter(config={"code": "codellama", "default": "llama3"})
    assert r.route("code") == "codellama"
    assert r.route("vision") == "llama3"  # falls back to default
    assert r.route("unknown") == "llama3"


def test_router_no_default_returns_none() -> None:
    r = ModelRouter(config={"code": "codellama"})
    assert r.route("vision") is None


def test_router_save_load(tmp_path: Path) -> None:
    r = ModelRouter(config={"code": "codellama", "default": "llama3"})
    p = tmp_path / "routes.yaml"
    r.save(p)
    r2 = ModelRouter(config_path=p)
    assert r2.route("code") == "codellama"
    assert r2.route("default") == "llama3"


def test_router_from_available() -> None:
    r = ModelRouter.from_available(["codellama:7b", "llava:7b", "llama3.2:3b"])
    assert r.route("code") == "codellama:7b"
    assert r.route("vision") == "llava:7b"


def test_router_set_updates_route() -> None:
    r = ModelRouter(config={"default": "llama3"})
    r.set("code", "deepseek-coder")
    assert r.route("code") == "deepseek-coder"


# ── agentic.budget ────────────────────────────────────────────────────────────


def test_budget_add_accumulates() -> None:
    b = ContextBudget(max_tokens=1000)
    msgs = [{"role": "user", "content": "Hello, how are you?"}]
    snap = b.add(msgs)
    assert snap.used > 0
    assert snap.remaining < 1000
    b.add(msgs)
    assert b.used > snap.used


def test_budget_remaining_capped_at_zero() -> None:
    b = ContextBudget(max_tokens=1)
    b.add([{"role": "user", "content": "x" * 1000}])
    assert b.remaining == 0


def test_budget_fraction_used() -> None:
    b = ContextBudget(max_tokens=1000)
    b._used = 500
    assert b.fraction_used == 0.5


def test_budget_warn_called_near_limit() -> None:
    warnings: list[str] = []
    b = ContextBudget(max_tokens=100, warn_threshold=0.5, warn_fn=warnings.append)
    b._used = 60  # already 60%
    b.add([{"role": "user", "content": "hi"}])
    assert len(warnings) == 1
    assert "⚠" in warnings[0]


def test_budget_reset() -> None:
    b = ContextBudget(max_tokens=1000)
    b.add([{"role": "user", "content": "hello"}])
    b.reset()
    assert b.used == 0


def test_budget_track_context_manager() -> None:
    b = ContextBudget(max_tokens=1000)
    with b.track() as b2:
        b2.add([{"role": "user", "content": "hello"}])
        assert b2.used > 0
    assert b.used == 0


def test_budget_snapshot_str() -> None:
    snap = BudgetSnapshot(used=512, max_tokens=1024, messages=4)
    s = str(snap)
    assert "512" in s
    assert "1024" in s


def test_budget_trim_to_fit() -> None:
    b = ContextBudget(max_tokens=50)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    trimmed = b.trim_to_fit(messages, reserve=0)
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) <= len(messages)


def test_get_shared_returns_same_instance() -> None:
    a = get_shared("test-budget-123")
    b_inst = get_shared("test-budget-123")
    assert a is b_inst


# ── commands.eval ─────────────────────────────────────────────────────────────


def test_eval_register() -> None:
    p = argparse.ArgumentParser()
    subs = p.add_subparsers()
    register(subs)
    parsed = p.parse_args(["eval", "--model", "llama3"])
    assert parsed.model == "llama3"
    assert parsed.base_url == "http://127.0.0.1:11435"
    assert parsed.suite == "builtins"


def test_cmd_eval_all_pass() -> None:
    args = argparse.Namespace(
        model="test-model", base_url="http://127.0.0.1:11435",
        suite="builtins", timeout=60.0,
    )
    mock_results = [
        EvalResult(case=c, fn_matched=True, args_matched=True, latency_ms=10.0, raw_tool_calls=[])
        for c in BUILTIN_SUITE[:2]
    ]
    with patch("ollama_forge.agentic.eval.run_eval", return_value=mock_results):
        rc = _cmd_eval(argparse.ArgumentParser(), args)
    assert rc == 0


def test_cmd_eval_some_fail() -> None:
    args = argparse.Namespace(
        model="test-model", base_url="http://127.0.0.1:11435",
        suite="builtins", timeout=60.0,
    )
    mock_results = [
        EvalResult(case=BUILTIN_SUITE[0], fn_matched=True, args_matched=True, latency_ms=10.0, raw_tool_calls=[]),
        EvalResult(case=BUILTIN_SUITE[1], fn_matched=False, args_matched=False, latency_ms=10.0, raw_tool_calls=[]),
    ]
    with patch("ollama_forge.agentic.eval.run_eval", return_value=mock_results):
        rc = _cmd_eval(argparse.ArgumentParser(), args)
    assert rc == 1


def test_cmd_eval_custom_suite(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.dump({
        "cases": [{
            "name": "custom",
            "prompt": "What time is it?",
            "tools": [_simple_tool("get_time")],
            "expected_fn": "get_time",
            "expected_arg_keys": [],
        }]
    }))
    args = argparse.Namespace(
        model="test-model", base_url="http://127.0.0.1:11435",
        suite=str(suite_path), timeout=60.0,
    )
    mock_results = [
        EvalResult(
            case=ToolCallCase(name="custom", prompt="What time is it?", tools=[], expected_fn="get_time"),
            fn_matched=True, args_matched=True, latency_ms=5.0, raw_tool_calls=[],
        )
    ]
    with patch("ollama_forge.agentic.eval.run_eval", return_value=mock_results):
        rc = _cmd_eval(argparse.ArgumentParser(), args)
    assert rc == 0
