"""Tests for external study eval integration helpers."""

from ollama_forge.study_eval_integrations import build_lm_eval_command


def test_build_lm_eval_command_includes_tasks_and_output() -> None:
    command = build_lm_eval_command(
        model="hf",
        tasks=["hellaswag", "arc_easy"],
        model_args="pretrained=Qwen/Qwen2.5-0.5B-Instruct",
        output_path="out.json",
        device="cpu",
        batch_size="4",
        limit=10,
    )
    assert command.command[0] == "lm_eval"
    assert "--tasks" in command.command
    assert "hellaswag,arc_easy" in command.command
    assert "--output_path" in command.command
