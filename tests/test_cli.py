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


def test_abliterate_profiles_help() -> None:
    """abliterate profiles --help lists JSON option."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "profiles", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--json" in result.stdout


def test_abliterate_report_help() -> None:
    """abliterate report --help lists report path."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "report", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "path" in result.stdout.lower()
    assert "--export" in result.stdout


def test_abliterate_regenerate_report_help() -> None:
    """abliterate regenerate-report --help lists output dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "regenerate-report", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout


def test_abliterate_pipeline_report_help() -> None:
    """abliterate pipeline-report --help lists export option."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "pipeline-report", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--export" in result.stdout


def test_abliterate_pipeline_compare_help() -> None:
    """abliterate pipeline-compare --help lists two pipeline paths."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "pipeline-compare", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pipeline_a" in result.stdout and "pipeline_b" in result.stdout
    assert "--export" in result.stdout


def test_abliterate_aggregate_help() -> None:
    """abliterate aggregate --help lists dir and metric options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "aggregate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dir" in result.stdout and "--metric" in result.stdout


def test_abliterate_benchmark_help() -> None:
    """abliterate benchmark --help lists model and prompt set options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "benchmark", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout and "prompt_set" in result.stdout


def test_abliterate_ui_help() -> None:
    """abliterate ui --help is available."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "ui", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ui" in result.stdout.lower() or "streamlit" in result.stdout.lower()


def test_abliterate_informed_plan_help() -> None:
    """abliterate informed-plan --help lists analysis input."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-plan", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--analysis" in result.stdout


def test_abliterate_informed_run_help() -> None:
    """abliterate informed-run --help lists analysis, model, and name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--analysis" in result.stdout and "--model" in result.stdout and "--name" in result.stdout
    assert "--artifact-file" in result.stdout


def test_abliterate_informed_refine_help() -> None:
    """abliterate informed-refine --help lists artifact input."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-refine", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "artifact" in result.stdout.lower()


def test_abliterate_informed_attach_eval_help() -> None:
    """abliterate informed-attach-eval --help lists artifact and eval paths."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-attach-eval", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "artifact" in result.stdout.lower() and "eval_report" in result.stdout


def test_abliterate_informed_artifact_help() -> None:
    """abliterate informed-artifact --help lists export option."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-artifact", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--export" in result.stdout


def test_abliterate_informed_compare_help() -> None:
    """abliterate informed-compare --help lists two artifact paths."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-compare", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "artifact_a" in result.stdout and "artifact_b" in result.stdout


def test_abliterate_informed_pipeline_help() -> None:
    """abliterate informed-pipeline --help lists study-config and model/name args."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "informed-pipeline", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--study-config" in result.stdout and "--model" in result.stdout and "--name" in result.stdout
    assert "--benchmark-preset" in result.stdout and "--compare-eval-report" in result.stdout
    assert "--auto-refine-run" in result.stdout and "--refine-output-dir" in result.stdout


def test_study_help() -> None:
    """study --help lists subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "presets" in result.stdout and "validate" in result.stdout and "plan" in result.stdout
    assert "init" in result.stdout and "interactive" in result.stdout


def test_study_presets_help() -> None:
    """study presets --help lists JSON option."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "presets", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--json" in result.stdout


def test_study_models_help() -> None:
    """study models --help lists recommendation flags."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "models", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--recommend" in result.stdout and "--tier" in result.stdout


def test_study_benchmarks_help() -> None:
    """study benchmarks --help lists kind filter."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "benchmarks", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--kind" in result.stdout


def test_study_benchmark_run_help() -> None:
    """study benchmark-run --help lists preset and model args."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "benchmark-run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--preset" in result.stdout and "--model" in result.stdout
    assert "--base-url" in result.stdout
    assert "--compare-model" in result.stdout


def test_study_lm_eval_help() -> None:
    """study lm-eval --help lists tasks and plan options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "lm-eval", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tasks" in result.stdout and "--plan" in result.stdout
    assert "--model-args" in result.stdout


def test_study_eval_compare_help() -> None:
    """study eval-compare --help lists two report paths."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "eval-compare", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "report_a" in result.stdout and "report_b" in result.stdout


def test_study_analysis_modules_help() -> None:
    """study analysis-modules --help lists JSON option."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "analysis-modules", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--json" in result.stdout


def test_study_analyze_bundle_help() -> None:
    """study analyze-bundle --help lists modules and output options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "analyze-bundle", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--modules" in result.stdout and "--output-file" in result.stdout


def test_study_analyze_help() -> None:
    """study analyze --help lists module and output options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "analyze", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--module" in result.stdout and "--output-file" in result.stdout
    assert "logit_lens" in result.stdout and "residual_stream" in result.stdout
    assert "causal_tracing" in result.stdout and "conditional_similarity" in result.stdout
    assert "activation_patching" in result.stdout
    assert "steering_vectors" in result.stdout and "concept_geometry" in result.stdout
    assert "defense_robustness" in result.stdout


def test_study_report_help() -> None:
    """study report --help lists report path."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "report", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "study-results" in result.stdout.lower() or "path" in result.stdout.lower()
    assert "--export" in result.stdout


def test_study_regenerate_report_help() -> None:
    """study regenerate-report --help lists output-dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "regenerate-report", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout


def test_study_compare_help() -> None:
    """study compare --help lists two report paths."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "compare", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "report_a" in result.stdout and "report_b" in result.stdout
    assert "--export" in result.stdout


def test_study_contribute_help() -> None:
    """study contribute --help lists report path and dir."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "contribute", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "report" in result.stdout.lower() and "--dir" in result.stdout


def test_study_aggregate_help() -> None:
    """study aggregate --help lists dir and json flag."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "aggregate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dir" in result.stdout and "--json" in result.stdout


def test_study_strategies_help() -> None:
    """study strategies --help lists JSON option."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "strategies", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--json" in result.stdout


def test_study_validate_success() -> None:
    """study validate accepts a valid preset-backed config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(
            "preset: quick\n"
            "model:\n"
            "  name: Qwen/Qwen2.5-0.5B\n"
            "dataset:\n"
            "  name: wikitext\n"
            "  split: test\n"
        )
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ollama_forge.cli", "study", "validate", path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Qwen/Qwen2.5-0.5B" in result.stdout
    finally:
        Path(path).unlink(missing_ok=True)


def test_study_plan_json() -> None:
    """study plan --json returns a plan payload."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(
            "preset: quick\n"
            "model:\n"
            "  name: Qwen/Qwen2.5-0.5B\n"
            "dataset:\n"
            "  name: wikitext\n"
            "  split: test\n"
        )
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ollama_forge.cli", "study", "plan", path, "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["model_name"] == "Qwen/Qwen2.5-0.5B"
        assert payload["strategies"][0]["strategy"] == "layer_removal"
    finally:
        Path(path).unlink(missing_ok=True)


def test_study_run_help() -> None:
    """study run --help lists config and output override options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "config" in result.stdout.lower()
    assert "--output-dir" in result.stdout


def test_study_optimize_help() -> None:
    """study optimize --help lists strengths and metric options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "optimize", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--strengths" in result.stdout and "--metric" in result.stdout


def test_study_init_help() -> None:
    """study init --help lists output and preset options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "init", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--out" in result.stdout and "--preset" in result.stdout


def test_study_interactive_help() -> None:
    """study interactive --help lists non-interactive and run options."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "interactive", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--non-interactive" in result.stdout and "--run" in result.stdout


def test_study_ui_help() -> None:
    """study ui --help is available."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "study", "ui", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "streamlit" in result.stdout.lower() or "ui" in result.stdout.lower()


def test_study_init_writes_config() -> None:
    """study init writes a starter config file."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "study.yaml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ollama_forge.cli",
                "study",
                "init",
                "--out",
                str(out),
                "--preset",
                "quick",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "preset: quick" in text
        assert "Qwen/Qwen2.5-0.5B-Instruct" in text


def test_study_interactive_non_interactive_writes_config() -> None:
    """study interactive --non-interactive writes a config using detected/default values."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "study.yaml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ollama_forge.cli",
                "study",
                "interactive",
                "--non-interactive",
                "--out",
                str(out),
                "--preset",
                "quick",
                "--tier",
                "tiny",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out.is_file()
        assert "Plan: ollama-forge study plan" in result.stdout


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
    """setup-llama-cpp --help lists --dir and --update."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "setup-llama-cpp", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dir" in result.stdout
    assert "--update" in result.stdout


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


def test_import_gguf_converter_in_help() -> None:
    """import --help lists --gguf-converter with choices."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "import", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gguf-converter" in result.stdout
    assert "llama-cpp" in result.stdout
    assert "unsloth" in result.stdout
    assert "auto" in result.stdout


def test_abliterate_run_gguf_converter_in_help() -> None:
    """abliterate run --help lists --gguf-converter."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gguf-converter" in result.stdout
    assert "unsloth" in result.stdout


def test_abliterate_run_dry_run() -> None:
    """abliterate run --dry-run prints config without loading model."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ollama_forge.cli", "abliterate", "run",
            "--model", "fake/model", "--name", "test-dry",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dry run" in result.stdout.lower()
    assert "strength" in result.stdout
    assert "test-dry" in result.stdout


def test_abliterate_compare_help() -> None:
    """abliterate compare --help lists model_a, model_b."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "compare", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "model_a" in result.stdout
    assert "model_b" in result.stdout
    assert "--prompts" in result.stdout


def test_abliterate_profiles_json() -> None:
    """abliterate profiles --json returns valid JSON with all profiles."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "profiles", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    import json

    profiles = json.loads(result.stdout)
    assert "safe" in profiles
    assert "aggressive" in profiles
    assert "surgical" in profiles
    assert "nuclear" in profiles


def test_abliterate_run_help_shows_new_flags() -> None:
    """abliterate run --help shows all new algorithm flags."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "abliterate", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in (
        "--project-bias", "--sparse-surgery", "--surgery-top-k",
        "--svd-method", "--direction-method", "--refine-passes",
        "--moe-expert-scale", "--save-lora", "--dry-run",
    ):
        assert flag in result.stdout, f"Missing flag: {flag}"


def test_train_help() -> None:
    """train --help lists --data, --base, --name."""
    result = subprocess.run(
        [sys.executable, "-m", "ollama_forge.cli", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data" in result.stdout and "--base" in result.stdout and "--name" in result.stdout
