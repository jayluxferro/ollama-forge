"""Modelfile generation tests."""

from ollama_tools.modelfile import build_modelfile


def test_build_modelfile_minimal() -> None:
    """Only FROM when no options."""
    out = build_modelfile("llama3.2")
    assert out.strip() == "FROM llama3.2"


def test_build_modelfile_with_system_and_params() -> None:
    """System and parameters are emitted."""
    out = build_modelfile(
        "llama3.2",
        system="You are helpful.",
        temperature=0.7,
        num_ctx=4096,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    assert "FROM llama3.2" in out
    assert "You are helpful." in out
    assert "PARAMETER temperature 0.7" in out
    assert "PARAMETER num_ctx 4096" in out
    assert "PARAMETER top_p 0.9" in out
    assert "PARAMETER repeat_penalty 1.1" in out


def test_build_modelfile_with_adapter() -> None:
    """ADAPTER line when adapter path given."""
    out = build_modelfile("llama3.2", adapter="/path/to/adapter")
    assert "FROM llama3.2" in out
    assert "ADAPTER /path/to/adapter" in out