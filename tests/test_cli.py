"""CLI tests."""

import subprocess
import sys


def test_cli_help() -> None:
    """ollama-tools --help exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ollama-tools" in result.stdout or "usage" in result.stdout.lower()


def test_cli_prog_name() -> None:
    """Help mentions the program name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Ollama" in result.stdout


def test_create_from_base_help() -> None:
    """create-from-base --help lists required args."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "create-from-base", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--base" in result.stdout and "--name" in result.stdout


def test_convert_help() -> None:
    """convert --help lists --gguf and --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gguf" in result.stdout and "--name" in result.stdout


def test_fetch_help() -> None:
    """fetch --help lists repo_id, --name, --gguf-file."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "fetch", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "repo_id" in result.stdout and "--name" in result.stdout
    assert "fetch" in result.stdout.lower() or "Hugging" in result.stdout


def test_fetch_adapter_help() -> None:
    """fetch-adapter --help lists repo_id, --base, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "fetch-adapter", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "repo_id" in result.stdout and "--base" in result.stdout and "--name" in result.stdout


def test_adapters_search_help() -> None:
    """adapters search --help lists query and limit."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "adapters", "search", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "query" in result.stdout and "search" in result.stdout.lower()


def test_validate_training_data_help() -> None:
    """validate-training-data --help lists data argument."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "validate-training-data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "data" in result.stdout or "jsonl" in result.stdout.lower()


def test_retrain_help() -> None:
    """retrain --help lists --base, --adapter, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "retrain", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--base" in result.stdout and "--adapter" in result.stdout and "--name" in result.stdout


def test_abliterate_help() -> None:
    """abliterate --help lists compute-dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "abliterate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "compute-dir" in result.stdout


def test_abliterate_compute_dir_help() -> None:
    """abliterate compute-dir --help lists --model, --output, and harmful/harmless options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "abliterate", "compute-dir", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout and "--output" in result.stdout
    assert "harmful" in result.stdout.lower()


def test_check_runs() -> None:
    """check command runs and prints ollama and huggingface status."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)  # 1 if something missing
    assert "ollama" in result.stdout.lower()
    assert "huggingface" in result.stdout.lower() or "HF" in result.stdout


def test_setup_llama_cpp_help() -> None:
    """setup-llama-cpp --help lists --dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "setup-llama-cpp", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dir" in result.stdout or "clone" in result.stdout.lower()


def test_downsize_prints_pipeline() -> None:
    """downsize (no subcommand) prints pipeline steps."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "downsize"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Downsize" in result.stdout and "convert" in result.stdout


def test_downsize_pipeline_subcommand() -> None:
    """downsize pipeline prints pipeline steps."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "downsize", "pipeline"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "teacher" in result.stdout.lower() or "student" in result.stdout.lower()


def test_build_help() -> None:
    """build --help lists recipe and --out-modelfile."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "build", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "recipe" in result.stdout and "build" in result.stdout.lower()


def test_build_nonexistent_recipe_fails() -> None:
    """build with nonexistent recipe file exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "build", "/nonexistent/recipe.json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "Error" in result.stderr


def test_prepare_training_data_help() -> None:
    """prepare-training-data --help lists data, output, format."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "prepare-training-data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "data" in result.stdout and "output" in result.stdout


def test_train_help() -> None:
    """train --help lists --data, --base, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data" in result.stdout and "--base" in result.stdout and "--name" in result.stdout
