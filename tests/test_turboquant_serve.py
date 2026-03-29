"""Tests for TurboQuant API server utilities."""

from __future__ import annotations

from ollama_forge.turboquant_serve import (
    _build_gen_config,
    _GenConfig,
    _infer_context_window,
    _resolve_default_max_tokens,
    TurboQuantServer,
)
from ollama_forge.turboquant_text import ReasoningScrubber, clean_generated_text


class TestBuildGenConfig:
    def test_defaults(self):
        cfg = _build_gen_config({}, None)
        assert isinstance(cfg, _GenConfig)
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.7

    def test_custom_params(self):
        cfg = _build_gen_config({
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 10,
        }, None)
        assert cfg.max_new_tokens == 100
        assert cfg.temperature == 0.5
        assert cfg.top_p == 0.8
        assert cfg.top_k == 10

    def test_with_tokenizer_eos(self):
        class FakeTok:
            eos_token_id = 99

        cfg = _build_gen_config({}, FakeTok())
        assert 99 in cfg.stop_tokens

    def test_no_eos_without_tokenizer(self):
        cfg = _build_gen_config({}, None)
        assert cfg.stop_tokens is None

    def test_uses_provided_default_max_tokens(self):
        cfg = _build_gen_config({}, None, default_max_tokens=4096)
        assert cfg.max_new_tokens == 4096


class TestContextWindowHelpers:
    def test_infer_context_window_from_hf_model(self):
        class FakeConfig:
            max_position_embeddings = 32768

        class FakeModel:
            config = FakeConfig()

        assert _infer_context_window(FakeModel()) == 32768

    def test_resolve_default_max_tokens_uses_remaining_context(self):
        class FakeConfig:
            max_position_embeddings = 128

        class FakeModel:
            config = FakeConfig()

        assert _resolve_default_max_tokens(FakeModel(), None, prompt_len=20, requested=None) == 108

    def test_resolve_default_max_tokens_prefers_user_request(self):
        class FakeConfig:
            max_position_embeddings = 128

        class FakeModel:
            config = FakeConfig()

        assert _resolve_default_max_tokens(FakeModel(), None, prompt_len=20, requested=64) == 64


class TestEncodeMessages:
    """Test that _encode_messages handles various apply_chat_template returns."""

    def _make_server(self, tokenizer):
        return TurboQuantServer(
            model=None, tokenizer=tokenizer, model_name="test",
            generate_fn=lambda *a: iter([]), gen_config_cls=_GenConfig,
        )

    def test_dict_like_return(self):
        """apply_chat_template returning a dict-like BatchEncoding."""
        class FakeTok:
            chat_template = "fake"
            def apply_chat_template(self, msgs, **kw):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        srv = self._make_server(FakeTok())
        assert srv._encode_messages([{"role": "user", "content": "hi"}]) == [1, 2, 3]

    def test_list_return(self):
        """apply_chat_template returning a plain list of ints."""
        class FakeTok:
            chat_template = "fake"
            def apply_chat_template(self, msgs, **kw):
                return [10, 20, 30]
        srv = self._make_server(FakeTok())
        assert srv._encode_messages([{"role": "user", "content": "hi"}]) == [10, 20, 30]

    def test_nested_list_return(self):
        """apply_chat_template returning nested list [[ids]]."""
        class FakeTok:
            chat_template = "fake"
            def apply_chat_template(self, msgs, **kw):
                return [[5, 6, 7]]
        srv = self._make_server(FakeTok())
        assert srv._encode_messages([{"role": "user", "content": "hi"}]) == [5, 6, 7]


class TestReasoningScrubber:
    def test_clean_generated_text_keeps_think_block(self):
        text = "<think>private reasoning</think>\nFinal answer."
        assert clean_generated_text(text) == "<think>private reasoning</think>\nFinal answer."

    def test_clean_generated_text_synthesizes_missing_open_tag(self):
        text = "hidden reasoning that leaked</think>\nVisible answer."
        assert clean_generated_text(text) == "<think>hidden reasoning that leaked</think>\nVisible answer."

    def test_streaming_scrubber_handles_split_tags(self):
        scrubber = ReasoningScrubber()
        parts = [
            "<thi",
            "nk>private",
            " reasoning</thi",
            "nk>Hello",
            " world",
        ]
        out = "".join(scrubber.feed(part) for part in parts)
        out += scrubber.finalize()
        assert out == "<think>private reasoning</think>Hello world"

    def test_streaming_scrubber_holds_prefix_until_closing_tag(self):
        scrubber = ReasoningScrubber()
        parts = [
            "private reasoning that should not be shown",
            " and more hidden thoughts",
            "</think>\nVisible answer.",
        ]
        out = "".join(scrubber.feed(part) for part in parts)
        out += scrubber.finalize()
        assert out == "<think>private reasoning that should not be shown and more hidden thoughts</think>\nVisible answer."

    def test_streaming_scrubber_removes_special_tokens(self):
        class FakeTok:
            all_special_tokens = ["<|im_end|>"]

        scrubber = ReasoningScrubber()
        out = scrubber.feed("<|im_end|>Hello", FakeTok())
        out += scrubber.finalize(FakeTok())
        assert out == "Hello"

    def test_reasoning_tags_are_not_removed_when_marked_special(self):
        class FakeTok:
            all_special_tokens = ["<think>", "</think>", "<|im_end|>"]

        text = "<think>reasoning</think><|im_end|>Answer"
        assert clean_generated_text(text, FakeTok()) == "<think>reasoning</think>Answer"
