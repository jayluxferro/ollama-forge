"""CLI tests."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_cli_help() -> None:
    """ollama-forge --help exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ollama-forge" in result.stdout or "usage" in result.stdout.lower()


def test_cli_prog_name() -> None:
    """Help mentions the program name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Ollama" in result.stdout


def test_create_from_base_help() -> None:
    """create-from-base --help lists required args."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "create-from-base", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--base" in result.stdout and "--name" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_convert_help() -> None:
    """convert --help lists --gguf and --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gguf" in result.stdout and "--name" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_fetch_help() -> None:
    """fetch --help lists repo_id, --name, --gguf-file."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "fetch", "--help"],
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
        [sys.executable, "-m", "ollama_forge.cli", "quickstart", "--help"],
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
        [sys.executable, "-m", "ollama_forge.cli", "start", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--name" in result.stdout and "--profile" in result.stdout
    assert "--task" in result.stdout


def test_plan_help() -> None:
    """plan --help lists plan subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "plan", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "quickstart" in result.stdout and "auto" in result.stdout
    assert "doctor-fix" in result.stdout


def test_plan_quickstart_runs() -> None:
    """plan quickstart prints quickstart plan and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "plan", "quickstart", "--name", "m"],
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
            "ollama_forge.cli",
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
        [sys.executable, "-m", "ollama_forge.cli", "plan", "doctor-fix"],
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
            "ollama_forge.cli",
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
        [sys.executable, "-m", "ollama_forge.cli", "auto", "--help"],
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
        [sys.executable, "-m", "ollama_forge.cli", "auto", "/nonexistent/recipe.yaml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "Error" in result.stderr


def test_auto_gguf_nonexistent_fails_fast() -> None:
    """auto with .gguf source routes to convert and fails with file-not-found."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "auto", "/nonexistent/model.gguf"],
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
            "ollama_forge.cli",
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


def test_auto_plan_local_checkpoint_routes_to_import() -> None:
    """auto --plan with a local HF checkpoint dir routes to import."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text("{}")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ollama_forge.cli",
                "auto",
                d,
                "--plan",
            ],
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0
    assert "route: import" in result.stdout


def test_fetch_adapter_help() -> None:
    """fetch-adapter --help lists repo_id, --base, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "fetch-adapter", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "repo_id" in result.stdout and "--base" in result.stdout and "--name" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_adapters_search_help() -> None:
    """adapters search --help lists query and limit."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "adapters", "search", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "query" in result.stdout and "search" in result.stdout.lower()


def test_adapters_recommend_help() -> None:
    """adapters recommend --help lists base/query/apply options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "adapters", "recommend", "--help"],
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
        [sys.executable, "-m", "ollama_forge.cli", "validate-training-data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "data" in result.stdout or "jsonl" in result.stdout.lower()


def test_retrain_help() -> None:
    """retrain --help lists --base, --adapter, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "retrain", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--base" in result.stdout and "--adapter" in result.stdout and "--name" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_abliterate_help() -> None:
    """abliterate --help lists compute-dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "compute-dir" in result.stdout


def test_abliterate_compute_dir_help() -> None:
    """abliterate compute-dir --help lists --model, --output, and harmful/harmless options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "compute-dir", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout and "--output" in result.stdout
    assert "harmful" in result.stdout.lower()


def test_check_runs() -> None:
    """check command runs and prints ollama and huggingface status."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)  # 1 if something missing
    assert "ollama" in result.stdout.lower()
    assert "huggingface" in result.stdout.lower() or "HF" in result.stdout


def test_check_json() -> None:
    """check --json outputs machine-readable status (same shape as doctor --json)."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "check", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert "ollama" in payload and "huggingface_hub" in payload
    assert "pyyaml" in payload and "hf_token" in payload
    assert all(isinstance(v, bool) for v in payload.values())


def test_doctor_help() -> None:
    """doctor --help lists --fix options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "doctor", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--fix" in result.stdout and "--fix-llama-cpp" in result.stdout
    assert "--plan" in result.stdout


def test_doctor_runs() -> None:
    """doctor command runs and prints report."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "doctor"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)
    assert "Doctor report" in result.stdout


def test_doctor_json() -> None:
    """doctor --json outputs machine-readable status for CI/scripting."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "doctor", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert "ollama" in payload and "huggingface_hub" in payload
    assert all(isinstance(v, bool) for v in payload.values())


def test_doctor_fix_plan_runs() -> None:
    """doctor --fix --plan prints plan and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "doctor", "--fix", "--plan"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Fix plan" in result.stdout


def test_setup_llama_cpp_help() -> None:
    """setup-llama-cpp --help lists --dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "setup-llama-cpp", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dir" in result.stdout or "clone" in result.stdout.lower()


def test_downsize_prints_pipeline() -> None:
    """downsize (no subcommand) prints pipeline steps."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "downsize"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Downsize" in result.stdout and "convert" in result.stdout


def test_downsize_pipeline_subcommand() -> None:
    """downsize pipeline prints pipeline steps."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "downsize", "pipeline"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "teacher" in result.stdout.lower() or "student" in result.stdout.lower()


def test_build_help() -> None:
    """build --help lists recipe, --validate-only, and --out-modelfile."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "build", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "recipe" in result.stdout and "build" in result.stdout.lower()
    assert "validate-only" in result.stdout


def test_build_nonexistent_recipe_fails() -> None:
    """build with nonexistent recipe file exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "build", "/nonexistent/recipe.json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "Error" in result.stderr


def test_build_missing_recipe_error_ux() -> None:
    """build with missing recipe path prints Next: and ollama-forge in stderr."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "build", "/nonexistent/recipe.yaml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Next:" in result.stderr
    assert "ollama-forge" in result.stderr


def test_build_invalid_recipe_error_ux() -> None:
    """build with invalid recipe (no name) prints Next: and Run: ollama-forge in stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"base": "llama3.2"}')
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ollama_forge.cli", "build", path, "--validate-only"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Next:" in result.stderr
        assert "Run: ollama-forge" in result.stderr
    finally:
        Path(path).unlink(missing_ok=True)


def test_build_validate_only_success() -> None:
    """build --validate-only with valid recipe exits 0 and prints summary."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"name": "my-model", "base": "llama3.2"}')
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ollama_forge.cli", "build", path, "--validate-only"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Recipe valid" in result.stdout
        assert "my-model" in result.stdout
        assert "base" in result.stdout
    finally:
        Path(path).unlink(missing_ok=True)


def test_build_validate_only_invalid_recipe_fails() -> None:
    """build --validate-only with invalid recipe (no name) exits non-zero."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"base": "llama3.2"}')
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ollama_forge.cli", "build", path, "--validate-only"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "name" in result.stderr.lower()
    finally:
        Path(path).unlink(missing_ok=True)


def test_prepare_training_data_help() -> None:
    """prepare-training-data --help lists data, output, format."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "prepare-training-data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "data" in result.stdout and "output" in result.stdout


def test_convert_training_data_format_help() -> None:
    """convert-training-data-format --help lists input and output."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "convert-training-data-format", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "input" in result.stdout and "output" in result.stdout


def test_convert_training_data_format_runs() -> None:
    """convert-training-data-format converts messages JSONL to Alpaca-style."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"messages": [{"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello!"}]}\n')
        in_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        out_path = f.name
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ollama_forge.cli",
                "convert-training-data-format",
                in_path,
                "-o",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(Path(out_path).read_text().strip())
        assert data["instruction"] == "Hi?"
        assert data["output"] == "Hello!"
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def test_import_help() -> None:
    """import --help lists source, --name, --quant, and template options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "import", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "source" in result.stdout
    assert "--name" in result.stdout
    assert "--quant" in result.stdout
    assert "--template-from" in result.stdout
    assert "--outtype" in result.stdout
    assert "--no-requantize" in result.stdout
    assert "--top-p" in result.stdout and "--repeat-penalty" in result.stdout


def test_import_missing_name() -> None:
    """import without --name exits non-zero with argparse error."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "import", "some/repo"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--name" in result.stderr


def test_train_help() -> None:
    """train --help lists --data, --base, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data" in result.stdout and "--base" in result.stdout and "--name" in result.stdout
