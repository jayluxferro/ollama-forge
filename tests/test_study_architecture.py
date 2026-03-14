"""Tests for study architecture profiles."""

from types import SimpleNamespace

from ollama_forge.study_architecture import detect_architecture_profile


class _Handle:
    def __init__(self, name: str):
        self.model = SimpleNamespace(config=SimpleNamespace(_name_or_path=name, model_type="llama"))
        self.num_layers = 32
        self.hidden_size = 4096
        self.num_heads = 32
        self.architecture = "TestModel"


def test_detect_architecture_profile_dense_standard() -> None:
    profile = detect_architecture_profile(_Handle("Qwen/Qwen2.5-7B-Instruct"))
    assert profile.arch_class == "dense"
    assert profile.reasoning_class == "standard"


def test_detect_architecture_profile_moe_reasoning() -> None:
    profile = detect_architecture_profile(_Handle("deepseek-r1-a22b-moe"))
    assert profile.arch_class == "moe"
    assert profile.reasoning_class == "reasoning"
    assert profile.recommended_profile == "safe"
