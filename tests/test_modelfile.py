"""Modelfile generation tests."""

from pathlib import Path

from ollama_forge.modelfile import (
    build_modelfile,
    merge_modelfile_with_reference_template,
    modelfile_append_num_predict,
    modelfile_append_stop_parameters,
    modelfile_append_template,
    template_body_from_modelfile,
    template_from_hf_checkpoint_with_reason,
)


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


def test_template_from_hf_checkpoint_with_reason_no_config(tmp_path: Path) -> None:
    """template_from_hf_checkpoint_with_reason returns (None, reason) when checkpoint has no config.json."""
    # Empty dir or dir without config.json
    template, reason = template_from_hf_checkpoint_with_reason(tmp_path)
    assert template is None
    assert reason is not None
    assert "config.json" in reason


def test_template_from_hf_checkpoint_with_reason_nonexistent_dir() -> None:
    """template_from_hf_checkpoint_with_reason returns (None, reason) for nonexistent path."""
    template, reason = template_from_hf_checkpoint_with_reason("/nonexistent/checkpoint/dir")
    assert template is None
    assert reason is not None


def test_template_body_from_modelfile() -> None:
    """template_body_from_modelfile extracts content inside TEMPLATE triple-quotes."""
    content = 'FROM x\nTEMPLATE """hello {{ .Prompt }}"""\nPARAMETER x 1'
    assert template_body_from_modelfile(content) == "hello {{ .Prompt }}"
    assert template_body_from_modelfile("FROM x") is None
    assert template_body_from_modelfile('FROM x\nTEMPLATE """\n\n"""') == ""


def test_merge_modelfile_with_reference_template_no_ref_template() -> None:
    """When reference has no TEMPLATE block, FROM is replaced (template_only=False)."""
    current = "FROM old\nPARAMETER x 1"
    ref = "FROM ref\nPARAMETER y 2"
    out = merge_modelfile_with_reference_template(current, ref, "new_base")
    assert "FROM new_base" in out
    assert "PARAMETER x 1" in out


def test_merge_modelfile_with_reference_template_template_only() -> None:
    """When template_only=True and ref has no TEMPLATE, current is unchanged."""
    current = "FROM old\nPARAMETER x 1"
    ref = "FROM ref"
    out = merge_modelfile_with_reference_template(current, ref, "new_base", template_only=True)
    assert out == current


def test_merge_modelfile_with_reference_template_replaces_template_and_from() -> None:
    """Reference TEMPLATE is merged and FROM updated."""
    current = 'FROM old\nTEMPLATE """old template"""'
    ref = 'FROM ref\nTEMPLATE """new {{ .Prompt }}"""'
    out = merge_modelfile_with_reference_template(current, ref, "new_base")
    assert "FROM new_base" in out
    assert "new {{ .Prompt }}" in out


def test_modelfile_append_template() -> None:
    """modelfile_append_template inserts TEMPLATE block after FROM."""
    content = "FROM llama3.2\n"
    body = "{{ .Prompt }}"
    out = modelfile_append_template(content, body)
    assert "FROM llama3.2" in out
    assert "TEMPLATE" in out
    assert "{{ .Prompt }}" in out


def test_modelfile_append_stop_parameters() -> None:
    """modelfile_append_stop_parameters adds PARAMETER stop lines (quoted)."""
    content = "FROM x\n"
    out = modelfile_append_stop_parameters(content, ["</s>", "<|eot|>"])
    assert 'PARAMETER stop "</s>"' in out
    assert 'PARAMETER stop "<|eot|>"' in out


def test_modelfile_append_num_predict() -> None:
    """modelfile_append_num_predict adds PARAMETER num_predict."""
    content = "FROM x\n"
    out = modelfile_append_num_predict(content, 4096)
    assert "PARAMETER num_predict 4096" in out
