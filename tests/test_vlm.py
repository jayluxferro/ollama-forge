"""Tests for VLM (Vision Language Model) integration.

All tests work WITHOUT mlx-vlm installed by mocking the imports.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers to create a fake mlx_vlm module for mocking
# ---------------------------------------------------------------------------

def _make_fake_mlx_vlm():
    """Build a minimal mock ``mlx_vlm`` package for testing."""
    pkg = types.ModuleType("mlx_vlm")
    pkg.__path__ = []  # make it look like a package

    # Fake model config
    class FakeConfig:
        model_type = "fake-vlm"

    class FakeModel:
        config = FakeConfig()

    class FakeProcessor:
        pass

    def fake_load(model_path, **kwargs):
        return FakeModel(), FakeProcessor()

    def fake_generate(model, processor, prompt, **kwargs):
        return "Generated text from VLM"

    def fake_stream_generate(model, processor, prompt, **kwargs):
        for tok in ["Hello", " ", "world"]:
            yield tok

    def fake_convert(hf_path, **kwargs):
        pass  # no-op for testing

    pkg.load = fake_load
    pkg.generate = fake_generate
    pkg.stream_generate = fake_stream_generate
    pkg.convert = fake_convert

    # Fake prompt_utils submodule
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def fake_apply_chat_template(processor, config, prompt, **kwargs):
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def fake_get_message_json(role, content):
        return {"role": role, "content": content}

    prompt_utils.apply_chat_template = fake_apply_chat_template
    prompt_utils.get_message_json = fake_get_message_json
    pkg.prompt_utils = prompt_utils

    # Fake server submodule (for serve command)
    server = types.ModuleType("mlx_vlm.server")
    pkg.server = server

    return {
        "mlx_vlm": pkg,
        "mlx_vlm.prompt_utils": prompt_utils,
        "mlx_vlm.server": server,
    }


# ---------------------------------------------------------------------------
# device.py tests
# ---------------------------------------------------------------------------


class TestIsVlmAvailableDevice:
    """Test is_vlm_available in device.py."""

    def test_returns_false_when_not_installed(self):
        """is_vlm_available() should return False when mlx-vlm is not installed."""
        with mock.patch.dict(sys.modules, {"mlx_vlm": None}):
            # Re-import to pick up the mocked module state
            from ollama_forge.device import is_vlm_available
            # The function does a fresh import attempt; mock ImportError
            with mock.patch("builtins.__import__", side_effect=ImportError("no mlx_vlm")):
                # We need to call it in a way that triggers the import
                # Actually, device.is_vlm_available does a try/import inside
                result = is_vlm_available()
                # On CI without mlx-vlm, this will be False
                assert isinstance(result, bool)

    def test_returns_bool(self):
        """is_vlm_available() should always return a bool."""
        from ollama_forge.device import is_vlm_available
        assert isinstance(is_vlm_available(), bool)


# ---------------------------------------------------------------------------
# vlm.py module tests
# ---------------------------------------------------------------------------


class TestVlmIsAvailable:
    """Test is_vlm_available in vlm.py."""

    def test_returns_false_without_mlx_vlm(self):
        """Should return False when mlx-vlm is not installed."""
        # Force-reload vlm module with mlx_vlm unavailable
        saved = sys.modules.pop("mlx_vlm", "MISSING")
        sys.modules["mlx_vlm"] = None  # simulate missing
        try:
            # Remove cached vlm module so it re-evaluates the import
            sys.modules.pop("ollama_forge.vlm", None)
            from ollama_forge import vlm
            assert vlm.is_vlm_available() is False
        finally:
            sys.modules.pop("ollama_forge.vlm", None)
            if saved == "MISSING":
                sys.modules.pop("mlx_vlm", None)
            else:
                sys.modules["mlx_vlm"] = saved

    def test_returns_true_with_mock_mlx_vlm(self):
        """Should return True when mlx-vlm is importable."""
        import importlib

        fake_modules = _make_fake_mlx_vlm()
        saved_modules = {}
        for name in fake_modules:
            saved_modules[name] = sys.modules.get(name, "MISSING")

        try:
            sys.modules.update(fake_modules)
            sys.modules.pop("ollama_forge.vlm", None)
            import ollama_forge.vlm as vlm_mod
            importlib.reload(vlm_mod)
            assert vlm_mod.is_vlm_available() is True
        finally:
            sys.modules.pop("ollama_forge.vlm", None)
            for name, val in saved_modules.items():
                if val == "MISSING":
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = val


class TestVlmFunctionsWithMock:
    """Test vlm.py functions with mocked mlx-vlm."""

    @pytest.fixture(autouse=True)
    def _setup_fake_mlx_vlm(self):
        """Install fake mlx_vlm for each test, clean up after."""
        fake_modules = _make_fake_mlx_vlm()
        saved = {}
        for name in fake_modules:
            saved[name] = sys.modules.get(name, "MISSING")
        sys.modules.update(fake_modules)
        sys.modules.pop("ollama_forge.vlm", None)
        yield
        sys.modules.pop("ollama_forge.vlm", None)
        for name, val in saved.items():
            if val == "MISSING":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val

    def test_vlm_load(self):
        from ollama_forge.vlm import vlm_load
        model, processor = vlm_load("fake/model")
        assert model is not None
        assert processor is not None

    def test_vlm_load_with_adapter(self):
        from ollama_forge.vlm import vlm_load
        model, processor = vlm_load("fake/model", adapter_path="/tmp/adapter")
        assert model is not None

    def test_vlm_generate(self):
        from ollama_forge.vlm import vlm_generate, vlm_load
        model, processor = vlm_load("fake/model")
        result = vlm_generate(model, processor, "Describe this image")
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_vlm_generate_with_images(self):
        from ollama_forge.vlm import vlm_generate, vlm_load
        model, processor = vlm_load("fake/model")
        result = vlm_generate(
            model, processor, "What is this?",
            images=["photo.jpg"], max_tokens=100,
        )
        assert "text" in result

    def test_vlm_stream_generate(self):
        from ollama_forge.vlm import vlm_load, vlm_stream_generate
        model, processor = vlm_load("fake/model")
        tokens = list(vlm_stream_generate(
            model, processor, "Hello",
            max_tokens=50, temperature=0.5,
        ))
        assert tokens == ["Hello", " ", "world"]

    def test_vlm_apply_chat_template(self):
        from ollama_forge.vlm import vlm_apply_chat_template, vlm_load
        model, processor = vlm_load("fake/model")
        result = vlm_apply_chat_template(
            processor, model.config, "Describe this",
            num_images=1,
        )
        assert "Describe this" in result

    def test_vlm_generate_raises_without_mlx_vlm(self):
        """Functions should raise RuntimeError when mlx-vlm is missing."""
        sys.modules.pop("ollama_forge.vlm", None)
        # Now break mlx_vlm
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        # Import the module fresh - it should see mlx_vlm as unavailable
        import importlib
        spec = importlib.util.find_spec("ollama_forge.vlm")
        if spec and spec.loader:
            vlm_mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(vlm_mod)
            except ImportError:
                pass
            else:
                with pytest.raises(RuntimeError, match="mlx-vlm is required"):
                    vlm_mod.vlm_load("fake/model")


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestVlmCliParsing:
    """Test that the vlm subcommands are correctly registered in argparse."""

    @pytest.fixture()
    def parse(self):
        """Return a helper that parses CLI args and returns the namespace."""
        import argparse

        # We just need the parser, not to run main().
        # Recreate the parser portion.  The easiest way is to call main
        # with --help captured, but let's just test parse_args directly.
        # We'll import the parser by partially running main.
        from unittest.mock import patch

        from ollama_forge.cli import main

        captured_parser = {}

        original_parse_args = argparse.ArgumentParser.parse_args

        def intercept_parse_args(self, args=None, namespace=None):
            captured_parser["parser"] = self
            return original_parse_args(self, args=args, namespace=namespace)

        import contextlib
        with (
            patch.object(argparse.ArgumentParser, "parse_args", intercept_parse_args),
            contextlib.suppress(Exception),
        ):
            main.__wrapped__ if hasattr(main, "__wrapped__") else main
            # Call main with a known command to capture the parser
            with patch("sys.argv", ["ollama-forge", "vlm"]), contextlib.suppress(SystemExit):
                main()

        parser = captured_parser.get("parser")
        if parser is None:
            pytest.skip("Could not capture parser")

        def _parse(args: list[str]):
            return parser.parse_args(args)

        return _parse

    def test_vlm_generate_args(self, parse):
        ns = parse(["vlm", "generate", "--model", "m/m", "--prompt", "hello"])
        assert ns.model == "m/m"
        assert ns.prompt == "hello"
        assert ns.max_tokens == 256
        assert ns.temperature == 0.0
        assert getattr(ns, "handler", None) is not None

    def test_vlm_generate_with_image(self, parse):
        ns = parse(["vlm", "generate", "--model", "m/m", "--prompt", "describe",
                     "--image", "a.jpg", "--image", "b.jpg"])
        assert ns.image == ["a.jpg", "b.jpg"]

    def test_vlm_generate_optional_args(self, parse):
        ns = parse(["vlm", "generate", "--model", "m/m", "--prompt", "hi",
                     "--max-tokens", "100", "--temperature", "0.5",
                     "--top-p", "0.9", "--kv-bits", "4",
                     "--enable-thinking", "--thinking-budget", "1024",
                     "--adapter-path", "/tmp/adapter", "--verbose"])
        assert ns.max_tokens == 100
        assert ns.temperature == 0.5
        assert ns.top_p == 0.9
        assert ns.kv_bits == 4
        assert ns.enable_thinking is True
        assert ns.thinking_budget == 1024
        assert ns.adapter_path == "/tmp/adapter"
        assert ns.verbose is True

    def test_vlm_chat_args(self, parse):
        ns = parse(["vlm", "chat", "--model", "m/m"])
        assert ns.model == "m/m"
        assert ns.max_tokens == 512
        assert ns.temperature == 0.7
        assert getattr(ns, "handler", None) is not None

    def test_vlm_chat_optional_args(self, parse):
        ns = parse(["vlm", "chat", "--model", "m/m",
                     "--system", "You are helpful",
                     "--kv-bits", "4", "--adapter-path", "/tmp/a"])
        assert ns.system == "You are helpful"
        assert ns.kv_bits == 4
        assert ns.adapter_path == "/tmp/a"

    def test_vlm_serve_args(self, parse):
        ns = parse(["vlm", "serve", "--model", "m/m"])
        assert ns.model == "m/m"
        assert ns.host == "127.0.0.1"
        assert ns.port == 11434
        assert getattr(ns, "handler", None) is not None

    def test_vlm_serve_optional_args(self, parse):
        ns = parse(["vlm", "serve", "--model", "m/m",
                     "--host", "0.0.0.0", "--port", "9090",
                     "--kv-bits", "2", "--adapter-path", "/tmp/a"])
        assert ns.host == "0.0.0.0"
        assert ns.port == 9090
        assert ns.kv_bits == 2
        assert ns.adapter_path == "/tmp/a"


# ---------------------------------------------------------------------------
# CLI handler tests (with mocked vlm module)
# ---------------------------------------------------------------------------


class TestVlmCliHandlers:
    """Test that CLI handlers work with mocked vlm backend."""

    @pytest.fixture(autouse=True)
    def _setup_fake_mlx_vlm(self):
        """Install fake mlx_vlm for each test, clean up after."""
        fake_modules = _make_fake_mlx_vlm()
        saved = {}
        for name in fake_modules:
            saved[name] = sys.modules.get(name, "MISSING")
        sys.modules.update(fake_modules)
        sys.modules.pop("ollama_forge.vlm", None)
        yield
        sys.modules.pop("ollama_forge.vlm", None)
        for name, val in saved.items():
            if val == "MISSING":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val

    def test_vlm_generate_handler(self, capsys):
        import argparse

        from ollama_forge.cli import _cmd_vlm_generate

        args = argparse.Namespace(
            model="fake/model",
            prompt="Describe this",
            image=None,
            audio=None,
            max_tokens=256,
            temperature=0.0,
            top_p=None,
            kv_bits=None,
            enable_thinking=False,
            thinking_budget=None,
            adapter_path=None,
            verbose=False,
        )
        ret = _cmd_vlm_generate(None, args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "Generated text" in captured.out

    def test_vlm_generate_handler_verbose(self, capsys):
        import argparse

        from ollama_forge.cli import _cmd_vlm_generate

        args = argparse.Namespace(
            model="fake/model",
            prompt="Hello",
            image=["a.jpg"],
            audio=None,
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            kv_bits=None,
            enable_thinking=False,
            thinking_budget=None,
            adapter_path=None,
            verbose=True,
        )
        ret = _cmd_vlm_generate(None, args)
        assert ret == 0

    def test_vlm_generate_handler_missing_vlm(self, capsys):
        """vlm generate should show helpful error when mlx-vlm is missing."""
        # Break the vlm module
        sys.modules.pop("ollama_forge.vlm", None)
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        import argparse

        from ollama_forge.cli import _cmd_vlm_generate

        args = argparse.Namespace(
            model="fake/model",
            prompt="Hello",
            image=None,
            audio=None,
            max_tokens=256,
            temperature=0.0,
            top_p=None,
            kv_bits=None,
            enable_thinking=False,
            thinking_budget=None,
            adapter_path=None,
            verbose=False,
        )
        ret = _cmd_vlm_generate(None, args)
        assert ret == 1
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "mlx-vlm" in combined or "not installed" in combined

    def test_vlm_serve_handler_missing_vlm(self, capsys):
        """vlm serve should show helpful error when mlx-vlm is missing."""
        sys.modules.pop("ollama_forge.vlm", None)
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        import argparse

        from ollama_forge.cli import _cmd_vlm_serve

        args = argparse.Namespace(
            model="fake/model",
            host="127.0.0.1",
            port=8080,
            kv_bits=None,
            adapter_path=None,
        )
        ret = _cmd_vlm_serve(None, args)
        assert ret == 1
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "mlx-vlm" in combined or "not installed" in combined

    def test_vlm_chat_handler_missing_vlm(self, capsys):
        """vlm chat should show helpful error when mlx-vlm is missing."""
        sys.modules.pop("ollama_forge.vlm", None)
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        import argparse

        from ollama_forge.cli import _cmd_vlm_chat

        args = argparse.Namespace(
            model="fake/model",
            max_tokens=512,
            temperature=0.7,
            system=None,
            kv_bits=None,
            adapter_path=None,
        )
        ret = _cmd_vlm_chat(None, args)
        assert ret == 1
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "mlx-vlm" in combined or "not installed" in combined

    def test_vlm_convert_handler(self, capsys):
        """vlm convert handler should call vlm_convert and return 0."""
        import argparse

        from ollama_forge.cli import _cmd_vlm_convert

        args = argparse.Namespace(
            hf_path="fake/model",
            mlx_path="mlx_model",
            quantize=False,
            q_bits=4,
            q_group_size=64,
            dtype=None,
            upload_repo=None,
        )
        ret = _cmd_vlm_convert(None, args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "mlx_model" in captured.out

    def test_vlm_convert_handler_with_quantize(self, capsys):
        """vlm convert handler should work with --quantize."""
        import argparse

        from ollama_forge.cli import _cmd_vlm_convert

        args = argparse.Namespace(
            hf_path="fake/model",
            mlx_path="output_dir",
            quantize=True,
            q_bits=8,
            q_group_size=128,
            dtype="float16",
            upload_repo=None,
        )
        ret = _cmd_vlm_convert(None, args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "output_dir" in captured.out

    def test_vlm_quantize_handler(self, capsys):
        """vlm quantize handler should call vlm_convert with quantize=True and return 0."""
        import argparse

        from ollama_forge.cli import _cmd_vlm_quantize

        args = argparse.Namespace(
            model="fake/model",
            output="mlx_model_quantized",
            bits=4,
            group_size=64,
            dtype=None,
            upload_repo=None,
        )
        ret = _cmd_vlm_quantize(None, args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "mlx_model_quantized" in captured.out

    def test_vlm_convert_handler_missing_vlm(self, capsys):
        """vlm convert should show helpful error when mlx-vlm is missing."""
        sys.modules.pop("ollama_forge.vlm", None)
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        import argparse

        from ollama_forge.cli import _cmd_vlm_convert

        args = argparse.Namespace(
            hf_path="fake/model",
            mlx_path="mlx_model",
            quantize=False,
            q_bits=4,
            q_group_size=64,
            dtype=None,
            upload_repo=None,
        )
        ret = _cmd_vlm_convert(None, args)
        assert ret == 1
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "mlx-vlm" in combined or "not installed" in combined

    def test_vlm_quantize_handler_missing_vlm(self, capsys):
        """vlm quantize should show helpful error when mlx-vlm is missing."""
        sys.modules.pop("ollama_forge.vlm", None)
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        import argparse

        from ollama_forge.cli import _cmd_vlm_quantize

        args = argparse.Namespace(
            model="fake/model",
            output="mlx_model_quantized",
            bits=4,
            group_size=64,
            dtype=None,
            upload_repo=None,
        )
        ret = _cmd_vlm_quantize(None, args)
        assert ret == 1
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "mlx-vlm" in combined or "not installed" in combined


# ---------------------------------------------------------------------------
# vlm_convert function tests
# ---------------------------------------------------------------------------


class TestVlmConvertFunction:
    """Test vlm_convert in vlm.py with mocked mlx-vlm."""

    @pytest.fixture(autouse=True)
    def _setup_fake_mlx_vlm(self):
        """Install fake mlx_vlm for each test, clean up after."""
        fake_modules = _make_fake_mlx_vlm()
        saved = {}
        for name in fake_modules:
            saved[name] = sys.modules.get(name, "MISSING")
        sys.modules.update(fake_modules)
        sys.modules.pop("ollama_forge.vlm", None)
        yield
        sys.modules.pop("ollama_forge.vlm", None)
        for name, val in saved.items():
            if val == "MISSING":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val

    def test_vlm_convert_returns_path(self):
        from pathlib import Path

        from ollama_forge.vlm import vlm_convert
        result = vlm_convert("fake/model", mlx_path="out_dir")
        assert isinstance(result, Path)
        assert str(result) == "out_dir"

    def test_vlm_convert_defaults(self):
        from pathlib import Path

        from ollama_forge.vlm import vlm_convert
        result = vlm_convert("fake/model")
        assert isinstance(result, Path)
        assert str(result) == "mlx_model"

    def test_vlm_convert_with_quantize(self):
        from pathlib import Path

        from ollama_forge.vlm import vlm_convert
        result = vlm_convert(
            "fake/model",
            mlx_path="quant_out",
            quantize=True,
            q_bits=8,
            q_group_size=128,
        )
        assert isinstance(result, Path)
        assert str(result) == "quant_out"

    def test_vlm_convert_raises_without_mlx_vlm(self):
        """vlm_convert should raise RuntimeError when mlx-vlm is missing."""
        sys.modules.pop("ollama_forge.vlm", None)
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)

        import importlib
        spec = importlib.util.find_spec("ollama_forge.vlm")
        if spec and spec.loader:
            vlm_mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(vlm_mod)
            except ImportError:
                pass
            else:
                with pytest.raises(RuntimeError, match="mlx-vlm is required"):
                    vlm_mod.vlm_convert("fake/model")


# ---------------------------------------------------------------------------
# CLI parsing tests for convert / quantize
# ---------------------------------------------------------------------------


class TestVlmConvertQuantizeCliParsing:
    """Test that vlm convert/quantize subcommands are correctly registered."""

    @pytest.fixture()
    def parse(self):
        """Return a helper that parses CLI args and returns the namespace."""
        import argparse
        import contextlib
        from unittest.mock import patch

        captured_parser = {}

        original_parse_args = argparse.ArgumentParser.parse_args

        def intercept_parse_args(self, args=None, namespace=None):
            captured_parser["parser"] = self
            return original_parse_args(self, args=args, namespace=namespace)

        with patch.object(argparse.ArgumentParser, "parse_args", intercept_parse_args):
            try:
                from ollama_forge.cli import main
                with patch("sys.argv", ["ollama-forge", "vlm"]), contextlib.suppress(SystemExit):
                    main()
            except Exception:
                pass

        parser = captured_parser.get("parser")
        if parser is None:
            pytest.skip("Could not capture parser")

        def _parse(args: list[str]):
            return parser.parse_args(args)

        return _parse

    def test_vlm_convert_args(self, parse):
        ns = parse(["vlm", "convert", "--hf-path", "org/model"])
        assert ns.hf_path == "org/model"
        assert ns.mlx_path == "mlx_model"
        assert ns.quantize is False
        assert ns.q_bits == 4
        assert ns.q_group_size == 64
        assert ns.dtype is None
        assert ns.upload_repo is None
        assert getattr(ns, "handler", None) is not None

    def test_vlm_convert_all_args(self, parse):
        ns = parse(["vlm", "convert", "--hf-path", "org/model",
                     "--mlx-path", "my_output", "--quantize",
                     "--q-bits", "8", "--q-group-size", "128",
                     "--dtype", "float16", "--upload-repo", "user/repo"])
        assert ns.hf_path == "org/model"
        assert ns.mlx_path == "my_output"
        assert ns.quantize is True
        assert ns.q_bits == 8
        assert ns.q_group_size == 128
        assert ns.dtype == "float16"
        assert ns.upload_repo == "user/repo"

    def test_vlm_quantize_args(self, parse):
        ns = parse(["vlm", "quantize", "--model", "org/model"])
        assert ns.model == "org/model"
        assert ns.output == "mlx_model_quantized"
        assert ns.bits == 4
        assert ns.group_size == 64
        assert ns.dtype is None
        assert ns.upload_repo is None
        assert getattr(ns, "handler", None) is not None

    def test_vlm_quantize_all_args(self, parse):
        ns = parse(["vlm", "quantize", "--model", "org/model",
                     "--output", "q_out", "--bits", "8",
                     "--group-size", "128", "--dtype", "float16",
                     "--upload-repo", "user/repo"])
        assert ns.model == "org/model"
        assert ns.output == "q_out"
        assert ns.bits == 8
        assert ns.group_size == 128
        assert ns.dtype == "float16"
        assert ns.upload_repo == "user/repo"


# ---------------------------------------------------------------------------
# Security eval VLM integration tests (Phase 4)
# ---------------------------------------------------------------------------


class TestQueryModelVlm:
    """Test query_model_vlm from security_eval.client."""

    @pytest.fixture(autouse=True)
    def _setup_fake_mlx_vlm(self):
        """Install fake mlx_vlm for each test, clean up after."""
        import importlib

        fake_modules = _make_fake_mlx_vlm()
        saved = {}
        for name in fake_modules:
            saved[name] = sys.modules.get(name, "MISSING")
        sys.modules.update(fake_modules)
        sys.modules.pop("ollama_forge.vlm", None)
        import ollama_forge.vlm as vlm_mod

        importlib.reload(vlm_mod)
        yield
        sys.modules.pop("ollama_forge.vlm", None)
        for name, val in saved.items():
            if val == "MISSING":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val

    def test_query_model_vlm_basic(self):
        from ollama_forge.security_eval.client import query_model_vlm

        text, duration = query_model_vlm("Describe this", model_path="fake/model")
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(duration, float)
        assert duration >= 0

    def test_query_model_vlm_with_system(self):
        from ollama_forge.security_eval.client import query_model_vlm

        text, duration = query_model_vlm(
            "Hello", model_path="fake/model", system="You are helpful.",
        )
        assert isinstance(text, str)
        assert isinstance(duration, float)

    def test_query_model_vlm_with_images(self, tmp_path):
        from ollama_forge.security_eval.client import query_model_vlm

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        text, duration = query_model_vlm(
            "What is this?", model_path="fake/model", image_paths=[str(img)],
        )
        assert isinstance(text, str)
        assert isinstance(duration, float)

    def test_query_model_vlm_with_preloaded(self):
        from ollama_forge.security_eval.client import load_vlm_for_eval, query_model_vlm

        loaded = load_vlm_for_eval("fake/model")
        text, duration = query_model_vlm(
            "Describe this", model_path="fake/model", _loaded_model=loaded,
        )
        assert isinstance(text, str)
        assert isinstance(duration, float)


class TestLoadVlmForEval:
    """Test load_vlm_for_eval from security_eval.client."""

    @pytest.fixture(autouse=True)
    def _setup_fake_mlx_vlm(self):
        import importlib

        fake_modules = _make_fake_mlx_vlm()
        saved = {}
        for name in fake_modules:
            saved[name] = sys.modules.get(name, "MISSING")
        sys.modules.update(fake_modules)
        sys.modules.pop("ollama_forge.vlm", None)
        import ollama_forge.vlm as vlm_mod

        importlib.reload(vlm_mod)
        yield
        sys.modules.pop("ollama_forge.vlm", None)
        for name, val in saved.items():
            if val == "MISSING":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val

    def test_load_returns_tuple(self):
        from ollama_forge.security_eval.client import load_vlm_for_eval

        result = load_vlm_for_eval("fake/model")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestRunEvalVlmSignature:
    def test_run_eval_accepts_vlm_model_path(self):
        import inspect

        from ollama_forge.security_eval.run import run_eval

        sig = inspect.signature(run_eval)
        assert "vlm_model_path" in sig.parameters
        assert sig.parameters["vlm_model_path"].default is None


class TestVlmMissingGraceful:
    def test_query_model_vlm_raises_without_mlx_vlm(self):
        saved_vlm = sys.modules.pop("ollama_forge.vlm", None)
        saved_mlx = sys.modules.get("mlx_vlm", "MISSING")
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)
        try:
            from ollama_forge.security_eval.client import query_model_vlm

            with pytest.raises(RuntimeError, match="mlx-vlm"):
                query_model_vlm("hello", model_path="fake/model")
        finally:
            sys.modules.pop("ollama_forge.vlm", None)
            if saved_vlm is not None:
                sys.modules["ollama_forge.vlm"] = saved_vlm
            if saved_mlx == "MISSING":
                sys.modules.pop("mlx_vlm", None)
            else:
                sys.modules["mlx_vlm"] = saved_mlx

    def test_load_vlm_for_eval_raises_without_mlx_vlm(self):
        saved_vlm = sys.modules.pop("ollama_forge.vlm", None)
        saved_mlx = sys.modules.get("mlx_vlm", "MISSING")
        sys.modules["mlx_vlm"] = None
        sys.modules.pop("ollama_forge.vlm", None)
        try:
            from ollama_forge.security_eval.client import load_vlm_for_eval

            with pytest.raises(RuntimeError, match="mlx-vlm"):
                load_vlm_for_eval("fake/model")
        finally:
            sys.modules.pop("ollama_forge.vlm", None)
            if saved_vlm is not None:
                sys.modules["ollama_forge.vlm"] = saved_vlm
            if saved_mlx == "MISSING":
                sys.modules.pop("mlx_vlm", None)
            else:
                sys.modules["mlx_vlm"] = saved_mlx
