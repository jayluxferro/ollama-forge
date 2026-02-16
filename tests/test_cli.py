"""CLI tests."""

import json
import subprocess
import sys


def test_cli_help() -> None:
    """ollama-forge --help exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ollama-forge" in result.stdout or "usage" in result.stdout.lower()


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
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_convert_help() -> None:
    """convert --help lists --gguf and --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gguf" in result.stdout and "--name" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_fetch_help() -> None:
    """fetch --help lists repo_id, --name, --gguf-file."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "fetch", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "repo_id" in result.stdout and "--name" in result.stdout
    assert "--quant" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout
    assert "fetch" in result.stdout.lower() or "Hugging" in result.stdout


def test_quickstart_help() -> None:
    """quickstart --help lists beginner defaults and common options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "quickstart", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--name" in result.stdout and "--repo-id" in result.stdout
    assert "--quant" in result.stdout
    assert "--profile" in result.stdout
    assert "--task" in result.stdout


def test_start_help() -> None:
    """start --help lists alias options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "start", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--name" in result.stdout and "--profile" in result.stdout
    assert "--task" in result.stdout


def test_plan_help() -> None:
    """plan --help lists plan subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "plan", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "quickstart" in result.stdout and "auto" in result.stdout
    assert "doctor-fix" in result.stdout


def test_plan_quickstart_runs() -> None:
    """plan quickstart prints quickstart plan and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "plan", "quickstart", "--name", "m"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Quickstart plan" in result.stderr


def test_plan_quickstart_json() -> None:
    """plan quickstart --json returns machine-readable output."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ollama_tools.cli",
            "plan",
            "quickstart",
            "--name",
            "m",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["route"] == "quickstart"
    assert payload["name"] == "m"


def test_plan_doctor_fix_runs() -> None:
    """plan doctor-fix prints fix plan and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "plan", "doctor-fix"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Fix plan" in result.stdout


def test_plan_doctor_fix_json() -> None:
    """plan doctor-fix --json returns machine-readable output."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ollama_tools.cli",
            "plan",
            "doctor-fix",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["route"] == "doctor-fix"
    assert isinstance(payload["actions"], list)


def test_auto_help() -> None:
    """auto --help lists source and routing options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "auto", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "source" in result.stdout.lower()
    assert "--quantize" in result.stdout and "--quant" in result.stdout
    assert "--no-prompt" in result.stdout
    assert "--plan" in result.stdout
    assert "--base" in result.stdout and "--output" in result.stdout


def test_auto_recipe_nonexistent_fails_fast() -> None:
    """auto with .yaml source routes to build and fails with file-not-found."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "auto", "/nonexistent/recipe.yaml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "Error" in result.stderr


def test_auto_gguf_nonexistent_fails_fast() -> None:
    """auto with .gguf source routes to convert and fails with file-not-found."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "auto", "/nonexistent/model.gguf"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "gguf file not found" in result.stderr.lower() or "Error" in result.stderr


def test_auto_plan_does_not_execute() -> None:
    """auto --plan prints route and exits 0 without running command."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ollama_tools.cli",
            "auto",
            "/nonexistent/model.gguf",
            "--plan",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Auto plan" in result.stdout
    assert "route: convert" in result.stdout


def test_fetch_adapter_help() -> None:
    """fetch-adapter --help lists repo_id, --base, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "fetch-adapter", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "repo_id" in result.stdout and "--base" in result.stdout and "--name" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_adapters_search_help() -> None:
    """adapters search --help lists query and limit."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "adapters", "search", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "query" in result.stdout and "search" in result.stdout.lower()


def test_adapters_recommend_help() -> None:
    """adapters recommend --help lists base/query/apply options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "adapters", "recommend", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--base" in result.stdout and "--query" in result.stdout
    assert "--apply" in result.stdout
    assert "--plan" in result.stdout


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
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


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


def test_doctor_help() -> None:
    """doctor --help lists --fix options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "doctor", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--fix" in result.stdout and "--fix-llama-cpp" in result.stdout
    assert "--plan" in result.stdout


def test_doctor_runs() -> None:
    """doctor command runs and prints report."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "doctor"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)
    assert "Doctor report" in result.stdout


def test_doctor_fix_plan_runs() -> None:
    """doctor --fix --plan prints plan and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_tools.cli", "doctor", "--fix", "--plan"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Fix plan" in result.stdout


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
