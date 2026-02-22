"""Tests for abliterate_serve module (pure-Python utilities and HTTP handler)."""

from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import pytest

from ollama_forge.abliterate_serve import (
    OllamaCompatHandler,
    _checkpoint_dir_size,
    _format_instruction,
    _inject_format_into_messages,
    _last_user_content_from_prompt,
    _make_ollama_chat_event,
    _make_ollama_generate_event,
    _options_to_gen_kw,
    _parse_tool_calls_from_text,
    _split_thinking_content,
    _stream_strip_leading_role,
    _strip_leading_chat_artifacts,
    _strip_role_lines_and_hold_prefix,
    _strip_serve_reply_artifacts,
    _strip_trailing_role_lines,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ThreadedTestServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_server(model_name: str = "test-model") -> _ThreadedTestServer:
    """Create a server with dummy attributes (no real model or tokenizer)."""
    server = _ThreadedTestServer(("127.0.0.1", 0), OllamaCompatHandler)
    server._ollama_model_name = model_name
    server._ollama_model = None
    server._ollama_tokenizer = None
    server._ollama_checkpoint_dir = None
    server._ollama_model_size = 42
    return server


# ---------------------------------------------------------------------------
# _format_instruction
# ---------------------------------------------------------------------------

class TestFormatInstruction:
    def test_none_returns_none(self) -> None:
        assert _format_instruction(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _format_instruction("") is None

    def test_json_string(self) -> None:
        result = _format_instruction("json")
        assert result is not None
        assert "JSON" in result

    def test_schema_dict(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _format_instruction(schema)
        assert result is not None
        assert "name" in result
        assert "schema" in result.lower() or "JSON" in result

    def test_unknown_string_returns_none(self) -> None:
        assert _format_instruction("xml") is None


# ---------------------------------------------------------------------------
# _inject_format_into_messages
# ---------------------------------------------------------------------------

class TestInjectFormatIntoMessages:
    def test_none_instruction_returns_copy(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = _inject_format_into_messages(msgs, None)
        assert result == msgs
        assert result is not msgs  # returns a copy

    def test_prepends_system_when_no_system(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = _inject_format_into_messages(msgs, "Return JSON")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Return JSON"
        assert result[1]["role"] == "user"

    def test_merges_with_existing_system(self) -> None:
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
        result = _inject_format_into_messages(msgs, "Return JSON")
        assert result[0]["role"] == "system"
        assert "Return JSON" in result[0]["content"]
        assert "Be helpful" in result[0]["content"]

    def test_only_merges_first_system(self) -> None:
        msgs = [
            {"role": "system", "content": "Sys1"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Sys2"},
        ]
        result = _inject_format_into_messages(msgs, "JSON")
        assert "JSON" in result[0]["content"]
        assert result[2]["content"] == "Sys2"


# ---------------------------------------------------------------------------
# _split_thinking_content
# ---------------------------------------------------------------------------

class TestSplitThinkingContent:
    def test_no_think_tags(self) -> None:
        thinking, content = _split_thinking_content("Hello world")
        assert thinking == ""
        assert content == "Hello world"

    def test_with_think_tags(self) -> None:
        text = "<think>I need to reason</think>The answer is 42"
        thinking, content = _split_thinking_content(text)
        assert "reason" in thinking
        assert content == "The answer is 42"

    def test_empty_think_tags(self) -> None:
        thinking, content = _split_thinking_content("<think></think>Answer")
        assert thinking == ""
        assert content == "Answer"

    def test_unclosed_think_tag_ignored(self) -> None:
        thinking, content = _split_thinking_content("<think>unfinished")
        assert thinking == ""
        assert "unfinished" in content


# ---------------------------------------------------------------------------
# _parse_tool_calls_from_text
# ---------------------------------------------------------------------------

class TestParseToolCallsFromText:
    def test_no_tool_calls(self) -> None:
        result = _parse_tool_calls_from_text("Just a normal response")
        assert result == []

    def test_simple_tool_call(self) -> None:
        text = '{"name": "get_weather", "arguments": {"city": "London"}}'
        result = _parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["arguments"] == '{"city": "London"}'

    def test_multiple_tool_calls(self) -> None:
        text = (
            '{"name": "func1", "arguments": {"a": 1}} '
            '{"name": "func2", "arguments": {"b": 2}}'
        )
        result = _parse_tool_calls_from_text(text)
        names = [r["function"]["name"] for r in result]
        assert "func1" in names
        assert "func2" in names

    def test_invalid_json_ignored(self) -> None:
        result = _parse_tool_calls_from_text("{not json}")
        assert result == []

    def test_dict_without_name_ignored(self) -> None:
        result = _parse_tool_calls_from_text('{"arguments": {"a": 1}}')
        assert result == []

    def test_returns_type_function(self) -> None:
        text = '{"name": "myFunc", "arguments": {"key": "val"}}'
        result = _parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["type"] == "function"


# ---------------------------------------------------------------------------
# _options_to_gen_kw
# ---------------------------------------------------------------------------

class TestOptionsToGenKw:
    def test_none_returns_empty(self) -> None:
        assert _options_to_gen_kw(None) == {}

    def test_empty_returns_empty(self) -> None:
        assert _options_to_gen_kw({}) == {}

    def test_num_predict(self) -> None:
        result = _options_to_gen_kw({"num_predict": 256})
        assert result["max_new_tokens"] == 256

    def test_temperature(self) -> None:
        result = _options_to_gen_kw({"temperature": 0.5})
        assert result["temperature"] == pytest.approx(0.5)

    def test_top_p(self) -> None:
        result = _options_to_gen_kw({"top_p": 0.9})
        assert result["top_p"] == pytest.approx(0.9)

    def test_stop_string(self) -> None:
        result = _options_to_gen_kw({"stop": "</s>"})
        assert result["stop"] == ["</s>"]

    def test_stop_list(self) -> None:
        result = _options_to_gen_kw({"stop": ["<eos>", "</s>"]})
        assert result["stop"] == ["<eos>", "</s>"]

    def test_unknown_key_ignored(self) -> None:
        result = _options_to_gen_kw({"unknown_key": 42})
        assert "unknown_key" not in result


# ---------------------------------------------------------------------------
# _last_user_content_from_prompt
# ---------------------------------------------------------------------------

class TestLastUserContentFromPrompt:
    def test_empty_returns_none(self) -> None:
        assert _last_user_content_from_prompt("") is None

    def test_no_role_returns_segment(self) -> None:
        result = _last_user_content_from_prompt("Hello there")
        assert result is not None

    def test_multi_turn_returns_last_user(self) -> None:
        prompt = "User: First\n\nAssistant: Reply\n\nUser: Second"
        result = _last_user_content_from_prompt(prompt)
        assert result is not None
        assert "Second" in result


# ---------------------------------------------------------------------------
# _strip_trailing_role_lines
# ---------------------------------------------------------------------------

class TestStripTrailingRoleLines:
    def test_no_trailing_role(self) -> None:
        text = "The answer is 42"
        assert _strip_trailing_role_lines(text) == text

    def test_strips_trailing_model(self) -> None:
        text = "The answer is 42\nmodel"
        result = _strip_trailing_role_lines(text)
        assert "model" not in result.strip().split("\n")[-1]

    def test_strips_trailing_user(self) -> None:
        text = "Response\nUser:"
        result = _strip_trailing_role_lines(text)
        assert result.strip() == "Response"

    def test_empty_string(self) -> None:
        assert _strip_trailing_role_lines("") == ""

    def test_only_role_lines_stripped(self) -> None:
        text = "assistant\nmodel\nuser"
        result = _strip_trailing_role_lines(text)
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# _strip_leading_chat_artifacts
# ---------------------------------------------------------------------------

class TestStripLeadingChatArtifacts:
    def test_no_artifacts(self) -> None:
        text = "The actual response"
        assert _strip_leading_chat_artifacts(text) == text

    def test_strips_leading_role_line(self) -> None:
        text = "assistant\nThe actual response"
        result = _strip_leading_chat_artifacts(text)
        assert not result.strip().startswith("assistant")
        assert "actual response" in result

    def test_strips_echoed_user_content(self) -> None:
        text = "What is 2+2?\nThe answer is 4"
        result = _strip_leading_chat_artifacts(text, last_user_content="What is 2+2?")
        assert "What is 2+2?" not in result
        assert "answer is 4" in result

    def test_empty_string(self) -> None:
        assert _strip_leading_chat_artifacts("").strip() == ""


# ---------------------------------------------------------------------------
# _strip_role_lines_and_hold_prefix
# ---------------------------------------------------------------------------

class TestStripRoleLinesAndHoldPrefix:
    def test_empty_string(self) -> None:
        to_yield, pending = _strip_role_lines_and_hold_prefix("")
        assert to_yield == ""
        assert pending == ""

    def test_regular_content_passed_through(self) -> None:
        to_yield, pending = _strip_role_lines_and_hold_prefix("Hello world")
        assert to_yield == "Hello world"
        assert pending == ""

    def test_role_prefix_held_back(self) -> None:
        # "mod" is a prefix of "model" — should be held pending
        to_yield, pending = _strip_role_lines_and_hold_prefix("mod")
        assert to_yield == ""
        assert pending == "mod"

    def test_full_role_word_stripped(self) -> None:
        to_yield, pending = _strip_role_lines_and_hold_prefix("model\nactual content")
        # "model" line should be stripped; content passes through
        assert "actual content" in to_yield or "actual content" in pending


# ---------------------------------------------------------------------------
# _stream_strip_leading_role
# ---------------------------------------------------------------------------

class TestStreamStripLeadingRole:
    def _collect(self, chunks, **kw) -> list[tuple[str, bool]]:
        return list(_stream_strip_leading_role(chunks, **kw))

    def test_passthrough_regular_content(self) -> None:
        chunks = [("Hello world", False), ("done text", True)]
        result = self._collect(chunks)
        full = "".join(c for c, _ in result)
        assert "Hello" in full

    def test_strips_leading_role_word(self) -> None:
        chunks = [("model\n", False), ("actual content", True)]
        result = self._collect(chunks)
        full = "".join(c for c, _ in result)
        assert "model\n" not in full or "actual content" in full

    def test_last_chunk_is_done(self) -> None:
        chunks = [("chunk1", False), ("chunk2", True)]
        result = self._collect(chunks)
        assert result[-1][1] is True

    def test_empty_stream(self) -> None:
        result = self._collect([("", True)])
        # Should not raise
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _make_ollama_chat_event / _make_ollama_generate_event
# ---------------------------------------------------------------------------

class TestMakeOllamaEvents:
    def test_chat_event_shape(self) -> None:
        ev = _make_ollama_chat_event("my-model", "Hello", False)
        assert ev["model"] == "my-model"
        assert ev["message"]["role"] == "assistant"
        assert ev["message"]["content"] == "Hello"
        assert ev["done"] is False
        assert "created_at" in ev

    def test_chat_event_done(self) -> None:
        ev = _make_ollama_chat_event("m", "text", True, done_reason="stop")
        assert ev["done"] is True
        assert ev["done_reason"] == "stop"

    def test_generate_event_shape(self) -> None:
        ev = _make_ollama_generate_event("my-model", "token", False)
        assert ev["model"] == "my-model"
        assert ev["response"] == "token"
        assert ev["done"] is False

    def test_generate_event_extra_fields(self) -> None:
        ev = _make_ollama_generate_event("m", "t", True, eval_count=5)
        assert ev["eval_count"] == 5


# ---------------------------------------------------------------------------
# _strip_serve_reply_artifacts
# ---------------------------------------------------------------------------

class TestStripServeReplyArtifacts:
    def test_no_artifacts(self) -> None:
        text = "Clean response"
        assert _strip_serve_reply_artifacts(text) == text

    def test_strips_leading_previous_assistant(self) -> None:
        prev = "Previous reply"
        text = prev + "\nmodel\nNew reply"
        result = _strip_serve_reply_artifacts(text, last_assistant_content=prev)
        assert prev not in result
        assert "New reply" in result

    def test_strips_leading_role_lines(self) -> None:
        text = "model\nuser\nActual response"
        result = _strip_serve_reply_artifacts(text)
        assert "Actual response" in result
        assert not result.strip().startswith("model")

    def test_strips_leading_user_echo(self) -> None:
        text = "What is 2+2?\nThe answer is 4"
        result = _strip_serve_reply_artifacts(text, last_user_content="What is 2+2?")
        assert "What is 2+2?" not in result
        assert "answer is 4" in result

    def test_empty_string(self) -> None:
        assert _strip_serve_reply_artifacts("").strip() == ""


# ---------------------------------------------------------------------------
# _checkpoint_dir_size
# ---------------------------------------------------------------------------

class TestCheckpointDirSize:
    def test_nonexistent_dir_returns_zero(self) -> None:
        assert _checkpoint_dir_size("/nonexistent/path/xyz") == 0

    def test_empty_dir_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            assert _checkpoint_dir_size(d) == 0

    def test_counts_file_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "file.txt"
            p.write_bytes(b"hello")
            size = _checkpoint_dir_size(d)
            assert size == 5

    def test_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "sub"
            sub.mkdir()
            (sub / "a.bin").write_bytes(b"x" * 100)
            (Path(d) / "b.bin").write_bytes(b"y" * 50)
            assert _checkpoint_dir_size(d) == 150


# ---------------------------------------------------------------------------
# OllamaCompatHandler — GET routes (live server, no real model)
# ---------------------------------------------------------------------------

class TestOllamaCompatHandlerGet:
    """Test GET routes that don't require a real model."""

    @pytest.fixture()
    def server_and_port(self):
        server = _make_server("abliterated-test")
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        port = server.server_address[1]
        yield server, port
        server.shutdown()

    def test_get_api_tags_200(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body["models"][0]["name"] == "abliterated-test"

    def test_get_api_ps_200(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert "models" in body

    def test_get_api_version_200(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert "version" in body

    def test_get_unknown_path_404(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/unknown", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2)
        assert exc.value.code == 404

    def test_head_api_tags_200(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/tags", method="HEAD")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200

    def test_options_cors_200(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            assert "Access-Control-Allow-Origin" in resp.headers


# ---------------------------------------------------------------------------
# OllamaCompatHandler — POST routes (live server, no real model)
# ---------------------------------------------------------------------------

class TestOllamaCompatHandlerPost:
    """Test POST routes — focus on error paths that don't need a real model."""

    @pytest.fixture()
    def server_and_port(self):
        server = _make_server("test-model")
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        port = server.server_address[1]
        yield server, port
        server.shutdown()

    def _post(self, port: int, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, json.loads(raw) if raw else {}

    def test_api_show_returns_modelfile(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/show", {"model": "test-model"})
        assert status == 200
        assert "modelfile" in body

    def test_api_show_wrong_model_404(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/show", {"model": "other-model"})
        assert status == 404
        assert "error" in body

    def test_api_pull_returns_success(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/pull", {"model": "anything"})
        assert status == 200
        assert body.get("status") == "success"

    def test_api_push_returns_success(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/push", {})
        assert status == 200

    def test_api_copy_returns_success(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/copy", {})
        assert status == 200

    def test_api_delete_returns_success(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/delete",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200

    def test_api_chat_invalid_json_400(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/chat",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2)
        assert exc.value.code == 400

    def test_api_chat_no_messages_400(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/chat", {"model": "test-model", "messages": []})
        assert status == 400
        assert "error" in body

    def test_api_generate_invalid_json_400(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/generate",
            data=b"bad",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2)
        assert exc.value.code == 400

    def test_api_embed_invalid_json_400(self, server_and_port) -> None:
        _, port = server_and_port
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/embed",
            data=b"bad",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2)
        assert exc.value.code == 400

    def test_api_embed_missing_input_400(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/embed", {"model": "test-model"})
        assert status == 400
        assert "error" in body

    def test_unknown_post_path_404(self, server_and_port) -> None:
        _, port = server_and_port
        status, body = self._post(port, "/api/unknown", {})
        assert status == 404
