"""Tool-call correctness mini-eval.

Tests whether a model reliably calls the right function with the right args.
Uses the OpenAI-compatible /v1/chat/completions endpoint (forge api or Ollama
with OpenAI compat enabled).

Why a separate eval from security_eval: security_eval tests *misuse* (prompt
injection, jailbreak). This tests *correctness* — does the model call the right
function with the right arguments given an unambiguous instruction?
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallCase:
    name: str
    prompt: str
    tools: list[dict[str, Any]]
    expected_fn: str
    expected_arg_keys: list[str] = field(default_factory=list)
    system: str | None = None


@dataclass
class EvalResult:
    case: ToolCallCase
    fn_matched: bool
    args_matched: bool
    latency_ms: float | None
    raw_tool_calls: list[dict]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.fn_matched and self.args_matched and self.error is None


def _make_tool(name: str, description: str, props: dict[str, dict], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


BUILTIN_SUITE: list[ToolCallCase] = [
    ToolCallCase(
        name="get_weather",
        prompt="What is the weather in Paris right now?",
        tools=[_make_tool(
            "get_current_weather",
            "Get the current weather for a location.",
            {"location": {"type": "string", "description": "City name"}},
            ["location"],
        )],
        expected_fn="get_current_weather",
        expected_arg_keys=["location"],
    ),
    ToolCallCase(
        name="search_web",
        prompt="Search the web for the latest news about large language models.",
        tools=[_make_tool(
            "search_web",
            "Search the web for information.",
            {"query": {"type": "string", "description": "Search query"}},
            ["query"],
        )],
        expected_fn="search_web",
        expected_arg_keys=["query"],
    ),
    ToolCallCase(
        name="calculate",
        prompt="Calculate 347 multiplied by 89.",
        tools=[_make_tool(
            "calculate",
            "Evaluate a math expression.",
            {"expression": {"type": "string", "description": "Math expression to evaluate"}},
            ["expression"],
        )],
        expected_fn="calculate",
        expected_arg_keys=["expression"],
    ),
    ToolCallCase(
        name="get_stock_price",
        prompt="What is the current stock price for Apple (AAPL)?",
        tools=[_make_tool(
            "get_stock_price",
            "Get the current stock price for a ticker symbol.",
            {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
            ["ticker"],
        )],
        expected_fn="get_stock_price",
        expected_arg_keys=["ticker"],
    ),
    ToolCallCase(
        name="send_email",
        prompt="Send an email to alice@example.com with subject 'Meeting tomorrow' and body 'See you at 10am'.",
        tools=[_make_tool(
            "send_email",
            "Send an email to a recipient.",
            {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            ["to", "subject", "body"],
        )],
        expected_fn="send_email",
        expected_arg_keys=["to", "subject"],
    ),
    ToolCallCase(
        name="translate",
        prompt="Translate 'Good morning' to French.",
        tools=[_make_tool(
            "translate",
            "Translate text to a target language.",
            {
                "text": {"type": "string"},
                "target_language": {"type": "string", "description": "Target language name or code"},
            },
            ["text", "target_language"],
        )],
        expected_fn="translate",
        expected_arg_keys=["text", "target_language"],
    ),
    ToolCallCase(
        name="get_time",
        prompt="What time is it in Tokyo?",
        tools=[_make_tool(
            "get_current_time",
            "Get the current time in a timezone.",
            {"timezone": {"type": "string", "description": "IANA timezone name, e.g. Asia/Tokyo"}},
            ["timezone"],
        )],
        expected_fn="get_current_time",
        expected_arg_keys=["timezone"],
    ),
    ToolCallCase(
        name="read_file",
        prompt="Read the contents of the file at /home/user/notes.txt.",
        tools=[_make_tool(
            "read_file",
            "Read the contents of a file.",
            {"path": {"type": "string", "description": "Absolute path to the file"}},
            ["path"],
        )],
        expected_fn="read_file",
        expected_arg_keys=["path"],
    ),
]


def run_eval(
    model: str,
    base_url: str = "http://127.0.0.1:11435",
    cases: list[ToolCallCase] | None = None,
    timeout: float = 60.0,
) -> list[EvalResult]:
    """Run the tool-call eval suite against an OpenAI-compatible endpoint."""
    if cases is None:
        cases = BUILTIN_SUITE

    base = base_url.rstrip("/")
    results: list[EvalResult] = []

    for case in cases:
        messages: list[dict] = []
        if case.system:
            messages.append({"role": "system", "content": case.system})
        messages.append({"role": "user", "content": case.prompt})

        body = json.dumps({
            "model": model,
            "messages": messages,
            "tools": case.tools,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raw = e.read() if e.fp else b""
            try:
                msg = json.loads(raw).get("error", raw.decode())
            except Exception:
                msg = str(e)
            results.append(EvalResult(
                case=case, fn_matched=False, args_matched=False,
                latency_ms=None, raw_tool_calls=[], error=msg,
            ))
            continue
        except Exception as e:
            results.append(EvalResult(
                case=case, fn_matched=False, args_matched=False,
                latency_ms=None, raw_tool_calls=[], error=str(e),
            ))
            continue

        latency_ms = (time.monotonic() - t0) * 1000
        message = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []

        fn_matched = False
        args_matched = False
        for tc in tool_calls:
            fn_name = (tc.get("function") or {}).get("name") or ""
            if fn_name == case.expected_fn:
                fn_matched = True
                raw_args_str = (tc.get("function") or {}).get("arguments") or "{}"
                try:
                    args = json.loads(raw_args_str)
                except Exception:
                    args = {}
                args_matched = all(k in args for k in case.expected_arg_keys)
                break

        results.append(EvalResult(
            case=case,
            fn_matched=fn_matched,
            args_matched=args_matched,
            latency_ms=latency_ms,
            raw_tool_calls=tool_calls,
        ))

    return results


def format_results(results: list[EvalResult], model: str) -> str:
    lines = [f"Tool-call eval — model: {model}", ""]
    passed = sum(1 for r in results if r.passed)
    for r in results:
        icon = "✓" if r.passed else "✗"
        lat = f"{r.latency_ms:.0f}ms" if r.latency_ms is not None else "err"
        detail = ""
        if not r.passed:
            if r.error:
                detail = f"  error: {r.error}"
            elif not r.fn_matched:
                got = [tc.get("function", {}).get("name") for tc in r.raw_tool_calls]
                detail = f"  expected fn={r.case.expected_fn!r}, got {got}"
            elif not r.args_matched:
                detail = f"  missing args from {r.case.expected_arg_keys}"
        lines.append(f"  {icon} {r.case.name:<20} {lat}{detail}")

    lines.append("")
    lines.append(f"Score: {passed}/{len(results)}")
    return "\n".join(lines)
