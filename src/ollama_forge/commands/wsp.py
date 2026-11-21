"""forge wsp — weight-space programming toolkit (Phases 1–6)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_diagnose(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.diagnostics import run_diagnostics

    model = args.model
    base_url = args.base_url
    quick = getattr(args, "quick", False)
    timeout = getattr(args, "timeout", 60.0)

    print(f"Diagnosing {model}  ({base_url}){'  [quick]' if quick else ''}", file=sys.stderr)
    print("", file=sys.stderr)

    steps: list[str] = []

    def progress(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    report = run_diagnostics(
        model=model,
        base_url=base_url,
        quick=quick,
        timeout=timeout,
        progress_fn=progress,
    )

    print()
    print(report.summary())

    found = report.obstructions_found()
    if found:
        print()
        print("Suggested next steps:")
        if "loops" in found:
            print(f"  forge wsp locate {model} --obstruction loops   (Phase 2: find loop-inducing weight directions)")
        if "sycophancy" in found:
            print(f"  forge wsp locate {model} --obstruction sycophancy")
        if "verbosity" in found:
            print(f"  forge wsp locate {model} --obstruction verbosity")
        if "format_fixation" in found:
            print(f"  forge wsp locate {model} --obstruction format_fixation")
        print()
        print("  Once located:")
        print(f"  forge wsp sharpen {model} --fix {'|'.join(found)} --out {model}-sharp")

    return 1 if found else 0


def _cmd_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Lightweight benchmark runner (counts correct answers on a small fixed set)."""
    from ollama_forge.wsp.bench import run_bench

    model = args.model
    base_url = args.base_url
    tasks = getattr(args, "tasks", None) or ["math", "factual", "coding"]
    timeout = getattr(args, "timeout", 60.0)

    print(f"Benchmarking {model}  ({base_url})  tasks={tasks}", file=sys.stderr)
    results = run_bench(model=model, base_url=base_url, tasks=tasks, timeout=timeout)

    print()
    print(f"Benchmark results — {model}")
    print()
    total_correct = 0
    total_n = 0
    for task, (correct, n) in results.items():
        pct = correct / n * 100 if n else 0.0
        mark = "✓" if pct >= 60 else "⚠"
        print(f"  {task:20s}  {correct}/{n}  ({pct:.0f}%)  {mark}")
        total_correct += correct
        total_n += n
    if total_n:
        print()
        print(f"  Overall: {total_correct}/{total_n}  ({total_correct/total_n*100:.0f}%)")
    return 0


def _cmd_locate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.analyzer import WeightSpaceAnalyzer, find_model_path

    model_path_str = getattr(args, "model_path", None)
    model_name = args.model
    obstruction = getattr(args, "obstruction", "loops")
    output_json = getattr(args, "output_json", None)
    proj_type_filter = getattr(args, "proj_type", None)
    n_top = getattr(args, "top", 10)

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = find_model_path(model_name)
        if model_path is None:
            print(
                f"Error: cannot find local weights for '{model_name}'.\n"
                "Use --model-path to specify the directory containing .safetensors files.",
                file=sys.stderr,
            )
            return 2

    print(f"Locating {obstruction} circuits in {model_path.name}", file=sys.stderr)
    print("", file=sys.stderr)

    analyzer = WeightSpaceAnalyzer(model_path)

    steps: list[str] = []

    def progress(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    try:
        weight_map = analyzer.scan(
            proj_types=([proj_type_filter] if proj_type_filter else None),
            progress_fn=progress,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error during scan: {exc}", file=sys.stderr)
        return 1

    print()
    print(weight_map.report(obstruction=obstruction, n=n_top))

    if output_json:
        out = Path(output_json)
        out.write_text(json.dumps(weight_map.to_json(), indent=2))
        print(f"\nWeight map saved → {out}")
        print(f"Run: forge wsp sharpen {model_name} --weight-map {out}")

    return 0


def _cmd_sharpen(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.analyzer import WeightMap, WeightSpaceAnalyzer, find_model_path
    from ollama_forge.wsp.sharpener import Sharpener

    model_name = args.model
    model_path_str = getattr(args, "model_path", None)
    weight_map_path = getattr(args, "weight_map", None)
    obstruction = getattr(args, "obstruction", "loops")
    n_layers = getattr(args, "n_layers", 3)
    n_directions = getattr(args, "n_directions", 1)
    proj_type = getattr(args, "proj_type", "down_proj")
    sparse_fraction = getattr(args, "sparse_fraction", 0.0)
    output_path = getattr(args, "output_path", None)

    # Resolve model path
    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = find_model_path(model_name)
        if model_path is None:
            print(
                f"Error: cannot find local weights for '{model_name}'.\n"
                "Use --model-path to specify the directory containing .safetensors files.",
                file=sys.stderr,
            )
            return 2

    # Load or build weight map
    if weight_map_path:
        wm_data = json.loads(Path(weight_map_path).read_text())
        weight_map = WeightMap.from_json(wm_data)
        print(f"Loaded weight map from {weight_map_path}", file=sys.stderr)
    else:
        print("No --weight-map given; running locate scan first...", file=sys.stderr)
        analyzer = WeightSpaceAnalyzer(model_path)
        steps: list[str] = []

        def _loc_progress(msg: str) -> None:
            steps.append(msg)
            print(f"  {msg}", file=sys.stderr)

        try:
            weight_map = analyzer.scan(progress_fn=_loc_progress)
        except (FileNotFoundError, Exception) as exc:
            print(f"Error during scan: {exc}", file=sys.stderr)
            return 1

    # Run sharpener
    print(f"\nSharpening {model_path.name}  [{obstruction}, {n_layers} layer(s), {n_directions} dir(s)]",
          file=sys.stderr)
    sharpener = Sharpener(model_path)

    sharp_steps: list[str] = []

    def _sharp_progress(msg: str) -> None:
        sharp_steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    try:
        result = sharpener.sharpen(
            weight_map=weight_map,
            obstruction=obstruction,
            n_layers=n_layers,
            n_directions=n_directions,
            proj_type=proj_type,
            sparse_fraction=sparse_fraction,
            output_path=output_path,
            progress_fn=_sharp_progress,
        )
    except (FileNotFoundError, ValueError, Exception) as exc:
        print(f"Error during sharpen: {exc}", file=sys.stderr)
        return 1

    print()
    print(result.report())
    return 0


def _cmd_correct(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.analyzer import find_model_path
    from ollama_forge.wsp.corrector import BehavioralCorrector

    model_name = args.model
    model_path_str = getattr(args, "model_path", None)
    n_layers = getattr(args, "n_layers", 2)
    proj_type = getattr(args, "proj_type", "down_proj")
    alpha = getattr(args, "alpha", 0.1)
    min_layer = getattr(args, "min_layer", 0)
    output_path = getattr(args, "output_path", None)

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = find_model_path(model_name)
        if model_path is None:
            print(
                f"Error: cannot find local weights for '{model_name}'.\n"
                "Use --model-path to specify the directory.",
                file=sys.stderr,
            )
            return 2

    print(
        f"Behavioral Causal Correction — {model_path.name}  "
        f"[{n_layers} layer(s), α={alpha}]",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    corrector = BehavioralCorrector(model_path)

    steps: list[str] = []

    def _prog(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    try:
        result = corrector.correct(
            n_layers=n_layers,
            proj_type=proj_type,
            alpha=alpha,
            min_layer=min_layer,
            output_path=output_path,
            progress_fn=_prog,
        )
    except (FileNotFoundError, ValueError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(result.report())
    return 0


def _cmd_gradient_correct(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.analyzer import find_model_path
    from ollama_forge.wsp.gradient_corrector import GradientCorrector

    model_name = args.model
    model_path_str = getattr(args, "model_path", None)
    n_layers = getattr(args, "n_layers", 2)
    proj_type = getattr(args, "proj_type", "down_proj")
    alpha = getattr(args, "alpha", 0.1)
    beta = getattr(args, "beta", 0.3)
    min_layer = getattr(args, "min_layer", 20)
    output_path = getattr(args, "output_path", None)

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = find_model_path(model_name)
        if model_path is None:
            print(
                f"Error: cannot find local weights for '{model_name}'.\n"
                "Use --model-path to specify the directory.",
                file=sys.stderr,
            )
            return 2

    print(
        f"Gradient-Guided Correction — {model_path.name}  "
        f"[{n_layers} layer(s), α={alpha}, β={beta}, min_layer={min_layer}]",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    corrector = GradientCorrector(model_path)

    steps: list[str] = []

    def _prog(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    try:
        result = corrector.correct(
            n_layers=n_layers,
            proj_type=proj_type,
            alpha=alpha,
            beta=beta,
            min_layer=min_layer,
            output_path=output_path,
            progress_fn=_prog,
        )
    except (FileNotFoundError, ValueError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(result.report())
    return 0


def _cmd_abliterate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ollama_forge.wsp.abliterator import AbliterationCorrector
    from ollama_forge.wsp.analyzer import find_model_path

    model_name = args.model
    model_path_str = getattr(args, "model_path", None)
    n_layers = getattr(args, "n_layers", 3)
    min_layer = getattr(args, "min_layer", 0)
    proj_types = getattr(args, "proj_types", None)
    output_path = getattr(args, "output_path", None)
    probe_mode = getattr(args, "probe_mode", "last_token")
    full_response_tokens = getattr(args, "full_response_tokens", 50)
    suppress_bank_path = getattr(args, "suppress_bank", None)
    retain_bank_path = getattr(args, "retain_bank", None)

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = find_model_path(model_name)
        if model_path is None:
            print(
                f"Error: cannot find local weights for '{model_name}'.\n"
                "Use --model-path to specify the directory.",
                file=sys.stderr,
            )
            return 2

    # Load custom probe banks from JSON files if provided
    suppress_bank = None
    retain_bank = None
    if suppress_bank_path is not None:
        import json as _json
        suppress_bank = _json.loads(Path(suppress_bank_path).read_text())
        if not isinstance(suppress_bank, list) or not all(isinstance(s, str) for s in suppress_bank):
            print("Error: --suppress-bank must be a JSON file containing a list of strings.", file=sys.stderr)
            return 2
    if retain_bank_path is not None:
        import json as _json
        retain_bank = _json.loads(Path(retain_bank_path).read_text())
        if not isinstance(retain_bank, list) or not all(isinstance(s, str) for s in retain_bank):
            print("Error: --retain-bank must be a JSON file containing a list of strings.", file=sys.stderr)
            return 2

    mode_label = f"probe={probe_mode}" + (f"/{full_response_tokens}tok" if probe_mode == "full_response" else "")
    print(
        f"Abliteration — {model_path.name}  "
        f"[n={n_layers}, min_layer={min_layer}, {mode_label}]",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    corrector = AbliterationCorrector(model_path)

    steps: list[str] = []

    def _prog(msg: str) -> None:
        steps.append(msg)
        print(f"  {msg}", file=sys.stderr)

    try:
        result = corrector.abliterate(
            suppress_bank=suppress_bank,
            retain_bank=retain_bank,
            n_layers=n_layers,
            min_layer=min_layer,
            proj_types=proj_types,
            output_path=output_path,
            progress_fn=_prog,
            probe_mode=probe_mode,
            full_response_tokens=full_response_tokens,
        )
    except (FileNotFoundError, ValueError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(result.report())
    return 0


def _cmd_to_mlx(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Convert a HuggingFace model (or local safetensors dir) to MLX format."""
    import json

    hf_path = args.hf_path
    output = getattr(args, "output", None) or hf_path.replace("/", "--") + "-mlx"
    quantize = getattr(args, "quantize", False)
    q_bits = getattr(args, "q_bits", 4)
    q_group_size = getattr(args, "q_group_size", 64)
    dtype = getattr(args, "dtype", None)
    trust_remote_code = getattr(args, "trust_remote_code", False)

    try:
        from mlx_lm import convert
    except ImportError:
        print("Error: mlx_lm not available. Run: uv add mlx-lm", file=sys.stderr)
        return 1

    print(f"Converting {hf_path} → {output}", file=sys.stderr)
    if quantize:
        print(f"  Quantise: {q_bits}-bit, group size {q_group_size}", file=sys.stderr)

    # Pre-flight: check for quantization formats MLX cannot dequantize (NVFP4, FP8 block).
    # These need an external dequantization pass before MLX conversion.
    _quant_blocker: str | None = None
    _arch_blocker: str | None = None
    _check_local: str | None = None
    if Path(hf_path).is_dir():
        _check_local = hf_path
    else:
        _hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        _slug = "models--" + hf_path.replace("/", "--")
        _snaps = _hf_cache / _slug / "snapshots"
        if _snaps.is_dir():
            for _snap in sorted(_snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if (_snap / "config.json").exists():
                    _check_local = str(_snap)
                    break
    if _check_local:
        _cfg_text = Path(_check_local) / "config.json"
        if _cfg_text.exists():
            _cfg = json.loads(_cfg_text.read_text())
            _qcfg = _cfg.get("quantization_config", {})
            _quant_algo = (_qcfg.get("quant_algo") or _qcfg.get("quant_method") or "").upper()
            if any(x in _quant_algo for x in ("NVFP4", "FP4", "FP8")):
                _quant_blocker = _quant_algo
            _mtype = _cfg.get("model_type", "") or (
                _cfg.get("text_config", {}) or {}
            ).get("model_type", "")
            _unsupported_archs = {
                "inkling2_loop": (
                    "inkling2_loop has 146 extra controller weights (loop-gating tensors) "
                    "that mlx_lm and mlx_vlm do not support. The architecture is not a simple "
                    "rename of inkling2 — the loop parameters must be loaded too."
                ),
            }
            if _mtype in _unsupported_archs:
                _arch_blocker = _unsupported_archs[_mtype]
    if _quant_blocker or _arch_blocker:
        print("Error: this model cannot be converted to MLX format yet:", file=sys.stderr)
        if _arch_blocker:
            print(f"  Architecture: {_arch_blocker}", file=sys.stderr)
        if _quant_blocker:
            print(
                f"  Weights: {_quant_blocker} quantization (FP4/FP8 block format) requires "
                "dequantization before MLX can load the safetensors. This is not yet "
                "implemented in the forge.",
                file=sys.stderr,
            )
        print(
            "\nTo run WSP on this model today, use it via its Ollama tag directly:\n"
            "  ollama run <model-tag>  # or forge wsp diagnose <model-tag>",
            file=sys.stderr,
        )
        return 1

    try:
        convert(
            hf_path=hf_path,
            mlx_path=output,
            quantize=quantize,
            q_bits=q_bits,
            q_group_size=q_group_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        msg = str(exc)
        if "not supported" not in msg:
            print(f"Error: {msg}", file=sys.stderr)
            return 1

        # Architecture not in mlx_lm — fall back to mlx_vlm (supports VLMs like inkling2)
        import shutil
        import subprocess

        print(f"  mlx_lm: {msg}", file=sys.stderr)
        print("  Retrying with mlx_vlm (subprocess)…", file=sys.stderr)

        # Find a Python that can import mlx_vlm (may differ from the forge venv's Python
        # if scipy has an ABI issue on this OS version)
        vlm_py: str | None = None
        for candidate in ("python3.11", "python3.12", "python3"):
            exe = shutil.which(candidate)
            if exe is None:
                continue
            if subprocess.run([exe, "-c", "import mlx_vlm"], capture_output=True).returncode == 0:
                vlm_py = exe
                break

        if vlm_py is None:
            print(
                "Error: architecture not in mlx_lm's registry and no Python with mlx_vlm found.\n"
                "  pip3.11 install mlx-vlm",
                file=sys.stderr,
            )
            return 1

        def _vlm_convert(src: str) -> int:
            cmd = [vlm_py, "-m", "mlx_vlm", "convert", "--hf-path", src, "--mlx-path", output]
            if quantize:
                cmd += ["--quantize", "--q-bits", str(q_bits), "--q-group-size", str(q_group_size)]
            if trust_remote_code:
                cmd += ["--trust-remote-code"]
            return subprocess.run(cmd, check=False).returncode

        rc = _vlm_convert(hf_path)
        if rc != 0:
            # mlx_vlm also rejected the model — try patching model_type to base variant.
            # e.g. "inkling2_loop" → "inkling2": same weights, loop runs in Ollama runtime.
            import os
            import tempfile

            # Resolve hf_path to a local snapshot if it's a HF repo id
            local_src: str | None = None
            if os.path.isdir(hf_path):
                local_src = hf_path
            else:
                # snapshot_download can throw IncompleteSnapshotError when minor files
                # (README, .gitattributes) are missing.  Fall back to scanning the HF
                # cache directory directly for a snapshot that has config.json.
                try:
                    from huggingface_hub import snapshot_download
                    local_src = snapshot_download(
                        hf_path,
                        local_files_only=True,
                        ignore_patterns=["*.md", ".gitattributes"],
                    )
                except Exception:
                    pass
                if local_src is None:
                    # Scan ~/.cache/huggingface/hub manually
                    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
                    slug = "models--" + hf_path.replace("/", "--")
                    snaps_dir = cache_dir / slug / "snapshots"
                    if snaps_dir.is_dir():
                        for snap in sorted(snaps_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                            if (snap / "config.json").exists():
                                local_src = str(snap)
                                break

            if local_src is None:
                print(
                    "Error: mlx_vlm could not convert this architecture and the model is "
                    "not in the local HF cache; re-download with network access and retry.",
                    file=sys.stderr,
                )
                return 1

            cfg_path = Path(local_src) / "config.json"
            if not cfg_path.exists():
                return rc  # no config to patch

            cfg = json.loads(cfg_path.read_text())
            orig_type = cfg.get("model_type", "")
            # Strip loop/variant suffixes to get the base model family name
            base_type = orig_type.replace("_loop", "").replace("_looped", "")
            if base_type == orig_type:
                return rc  # no variant suffix to strip

            print(
                f"  mlx_vlm: {orig_type!r} not supported — retrying as {base_type!r} "
                f"(stripping variant suffix; standard transformer weights should be loadable).",
                file=sys.stderr,
            )
            with tempfile.TemporaryDirectory(prefix="forge_vlm_patch_") as tmp:
                tmp_path = Path(tmp)
                patched_cfg = dict(cfg)
                patched_cfg["model_type"] = base_type
                (tmp_path / "config.json").write_text(json.dumps(patched_cfg, indent=2))
                for f in Path(local_src).iterdir():
                    if f.name != "config.json":
                        (tmp_path / f.name).symlink_to(f)
                rc = _vlm_convert(str(tmp_path))
        if rc != 0:
            return rc

    print(f"\nMLX model saved to {output}")
    print(f"Next:  forge wsp abliterate {output} --output {output}-abliterated")
    return 0


def _cmd_to_gguf(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Convert an MLX safetensors directory to GGUF (for Ollama / llama.cpp)."""
    import json
    import subprocess

    mlx_path = Path(args.mlx_path).resolve()
    output = getattr(args, "output", None) or str(mlx_path.parent / (mlx_path.name + ".gguf"))
    ollama_name = getattr(args, "ollama_name", None)

    config_path = mlx_path / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found — is {mlx_path} an MLX model directory?", file=sys.stderr)
        return 1

    # Prefer llama.cpp converter (supports all architectures, handles quantized weights)
    llama_script = Path(__file__).parent.parent.parent.parent / "llama.cpp" / "convert_hf_to_gguf.py"
    if llama_script.exists():
        print(f"Using llama.cpp converter: {llama_script.name}", file=sys.stderr)
        result = subprocess.run(
            [sys.executable, str(llama_script), str(mlx_path), "--outfile", output],
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    else:
        # Fallback: mlx_lm.gguf (unquantized LLaMA-style only)
        with open(config_path) as f:
            config = json.load(f)
        if config.get("quantization"):
            print(
                "Error: mlx_lm.gguf does not support quantized models. "
                "Put llama.cpp/convert_hf_to_gguf.py on PATH or use --llama-cpp-dir.",
                file=sys.stderr,
            )
            return 1
        try:
            import mlx.core as mx
            from mlx_lm.gguf import convert_to_gguf
        except ImportError:
            print("Error: mlx_lm not available. Run: uv add mlx-lm", file=sys.stderr)
            return 1
        weights: dict = {}
        for sf in sorted(mlx_path.glob("*.safetensors")):
            weights.update(mx.load(str(sf)))
        convert_to_gguf(mlx_path, weights, config, output)

    print(f"\nGGUF saved to {output}")
    if ollama_name:
        print(f"Loading into Ollama as '{ollama_name}'…", file=sys.stderr)
        mf = f"FROM {output}\n"
        mf_path = str(mlx_path.parent / (mlx_path.name + ".Modelfile"))
        Path(mf_path).write_text(mf)
        result = subprocess.run(["ollama", "create", ollama_name, "-f", mf_path], check=False)
        if result.returncode == 0:
            print(f"Ollama model '{ollama_name}' ready.")
        return result.returncode
    print(f"To load into Ollama:  forge convert --gguf {output} --name <model-name>")
    return 0


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "wsp",
        help="Weight-Space Programming toolkit — diagnose and sharpen any model",
    )
    sub = p.add_subparsers(dest="wsp_command")

    # ── diagnose ──────────────────────────────────────────────────────────────
    p_diag = sub.add_parser(
        "diagnose",
        help="Measure obstruction signals: loop rate, sycophancy, verbosity, format compliance",
    )
    p_diag.add_argument("model", help="Model name (e.g. llama3.2:3b)")
    p_diag.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        dest="base_url",
        help="OpenAI-compatible endpoint (default: Ollama at 11434)",
    )
    p_diag.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller prompt banks (faster, less accurate)",
    )
    p_diag.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    p_diag.set_defaults(handler=_cmd_diagnose)

    # ── bench ─────────────────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "bench",
        help="Lightweight benchmark: math, factual, coding tasks",
    )
    p_bench.add_argument("model", help="Model name (e.g. llama3.2:3b)")
    p_bench.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        dest="base_url",
        help="OpenAI-compatible endpoint (default: Ollama at 11434)",
    )
    p_bench.add_argument(
        "--tasks",
        nargs="+",
        choices=["math", "factual", "coding", "reasoning"],
        help="Task categories to benchmark (default: all)",
    )
    p_bench.add_argument(
        "--timeout",
        type=float,
        default=60.0,
    )
    p_bench.set_defaults(handler=_cmd_bench)

    # ── locate ────────────────────────────────────────────────────────────────
    p_locate = sub.add_parser(
        "locate",
        help="Scan model weights (Phase 2): rank MLP layers by obstruction likelihood",
    )
    p_locate.add_argument("model", help="Model name or HF repo id (e.g. mlx-community/Qwen2.5-1.5B-Instruct-4bit)")
    p_locate.add_argument(
        "--model-path",
        dest="model_path",
        metavar="PATH",
        help="Path to directory containing .safetensors files (overrides auto-discovery)",
    )
    p_locate.add_argument(
        "--obstruction",
        choices=["loops", "sycophancy", "verbosity", "format_fixation"],
        default="loops",
        help="Which obstruction type to rank for (default: loops)",
    )
    p_locate.add_argument(
        "--proj-type",
        dest="proj_type",
        choices=["gate_proj", "up_proj", "down_proj"],
        default=None,
        help="Only scan this MLP projection type (default: all three)",
    )
    p_locate.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many candidate layers to show (default: 10)",
    )
    p_locate.add_argument(
        "--output-json",
        dest="output_json",
        metavar="PATH",
        help="Save weight map to this JSON file for use with forge wsp sharpen",
    )
    p_locate.set_defaults(handler=_cmd_locate)

    # ── sharpen ───────────────────────────────────────────────────────────────
    p_sharp = sub.add_parser(
        "sharpen",
        help="Edit model weights (Phase 3): zero obstruction-circuit directions",
    )
    p_sharp.add_argument("model", help="Model name or HF repo id")
    p_sharp.add_argument(
        "--model-path",
        dest="model_path",
        metavar="PATH",
        help="Path to source .safetensors directory (overrides auto-discovery)",
    )
    p_sharp.add_argument(
        "--weight-map",
        dest="weight_map",
        metavar="JSON",
        help="Path to weight map JSON from forge wsp locate (runs locate if omitted)",
    )
    p_sharp.add_argument(
        "--obstruction",
        choices=["loops", "sycophancy", "verbosity", "format_fixation"],
        default="loops",
        help="Which obstruction to target (default: loops)",
    )
    p_sharp.add_argument(
        "--n-layers",
        dest="n_layers",
        type=int,
        default=3,
        help="How many candidate layers to edit (default: 3)",
    )
    p_sharp.add_argument(
        "--n-directions",
        dest="n_directions",
        type=int,
        default=1,
        help="Singular directions to zero per layer (default: 1; paper used 1)",
    )
    p_sharp.add_argument(
        "--proj-type",
        dest="proj_type",
        choices=["gate_proj", "up_proj", "down_proj"],
        default="down_proj",
        help="MLP projection to edit (default: down_proj)",
    )
    p_sharp.add_argument(
        "--sparse-fraction",
        dest="sparse_fraction",
        type=float,
        default=0.0,
        help="Only zero the top fraction of values aligned with the dominant direction "
        "(0.005 ≈ 65k values/layer matching the paper; 0 = full rank-k removal)",
    )
    p_sharp.add_argument(
        "--output-path",
        dest="output_path",
        metavar="DIR",
        help="Output directory for the sharpened model (default: <source>-sharp/)",
    )
    p_sharp.set_defaults(handler=_cmd_sharpen)

    # ── correct ───────────────────────────────────────────────────────────────
    p_corr = sub.add_parser(
        "correct",
        help="Behavioral Causal Correction (Phase 4): probe → attribute → rank-1 fix",
    )
    p_corr.add_argument("model", help="Model name or HF repo id")
    p_corr.add_argument(
        "--model-path",
        dest="model_path",
        metavar="PATH",
        help="Path to source .safetensors directory (overrides auto-discovery)",
    )
    p_corr.add_argument(
        "--n-layers",
        dest="n_layers",
        type=int,
        default=2,
        help="How many top-attributed layers to correct (default: 2)",
    )
    p_corr.add_argument(
        "--proj-type",
        dest="proj_type",
        choices=["gate_proj", "up_proj", "down_proj"],
        default="down_proj",
        help="MLP projection to correct (default: down_proj)",
    )
    p_corr.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Correction strength 0–1 (default: 0.1; larger = stronger damping)",
    )
    p_corr.add_argument(
        "--min-layer",
        dest="min_layer",
        type=int,
        default=0,
        help="Skip layers below this index during attribution (default: 0). "
        "Set to ~half the model depth to avoid early-layer edits that degrade "
        "factual retrieval (e.g. 14 for a 28-layer model).",
    )
    p_corr.add_argument(
        "--output-path",
        dest="output_path",
        metavar="DIR",
        help="Output directory for the corrected model (default: <source>-bcc/)",
    )
    p_corr.set_defaults(handler=_cmd_correct)

    # ── gradient-correct ──────────────────────────────────────────────────────
    p_ggc = sub.add_parser(
        "gradient-correct",
        help="Gradient-Guided Correction (Phase 5): probe → LM head direction → rank-1 fix",
    )
    p_ggc.add_argument("model", help="Model name or HF repo id")
    p_ggc.add_argument(
        "--model-path",
        dest="model_path",
        metavar="PATH",
        help="Path to source .safetensors directory (overrides auto-discovery)",
    )
    p_ggc.add_argument(
        "--n-layers",
        dest="n_layers",
        type=int,
        default=2,
        help="How many top-attributed layers to correct (default: 2)",
    )
    p_ggc.add_argument(
        "--proj-type",
        dest="proj_type",
        choices=["gate_proj", "up_proj", "down_proj"],
        default="down_proj",
        help="MLP projection to edit (default: down_proj)",
    )
    p_ggc.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="BCC suppression strength 0–1 (default: 0.1; 0 = pure GGC)",
    )
    p_ggc.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="LM head logit-projection strength (default: 0.3; 0 = pure BCC)",
    )
    p_ggc.add_argument(
        "--min-layer",
        dest="min_layer",
        type=int,
        default=20,
        help="Skip layers below this index (default: 20). "
        "Phase 5 targets late layers where the LM head signal is most direct.",
    )
    p_ggc.add_argument(
        "--output-path",
        dest="output_path",
        metavar="DIR",
        help="Output directory for the corrected model (default: <source>-ggc/)",
    )
    p_ggc.set_defaults(handler=_cmd_gradient_correct)

    # ── abliterate ────────────────────────────────────────────────────────────
    p_abl = sub.add_parser(
        "abliterate",
        help="Abliteration (Phase 6): suppress a concept direction via residual stream orthogonalisation",
    )
    p_abl.add_argument("model", help="Model name or HF repo id")
    p_abl.add_argument(
        "--model-path",
        dest="model_path",
        metavar="PATH",
        help="Path to source .safetensors directory (overrides auto-discovery)",
    )
    p_abl.add_argument(
        "--n-layers",
        dest="n_layers",
        type=int,
        default=3,
        help="How many top-separation layers to orthogonalise (default: 3). "
        "n=3 is the empirically validated sweet spot for 7B models.",
    )
    p_abl.add_argument(
        "--min-layer",
        dest="min_layer",
        type=int,
        default=0,
        help="Skip layers below this index during ranking (default: 0). "
        "Set to 10–15 to protect early semantic layers on very small models.",
    )
    p_abl.add_argument(
        "--proj-types",
        dest="proj_types",
        nargs="+",
        choices=["down_proj", "o_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        default=None,
        help="Which projections to orthogonalise per layer (default: all seven).",
    )
    p_abl.add_argument(
        "--output-path",
        dest="output_path",
        metavar="DIR",
        help="Output directory for the abliterated model (default: <source>-abliterated/)",
    )
    p_abl.add_argument(
        "--probe-mode",
        dest="probe_mode",
        choices=["last_token", "full_response"],
        default="last_token",
        help="How to collect residual stream activations (default: last_token). "
        "'last_token' captures only the final prompt-token position — fast and "
        "sufficient for most cases. 'full_response' generates --response-tokens "
        "tokens and averages activations across all positions; use this when "
        "last_token leaves the target behaviour intact.",
    )
    p_abl.add_argument(
        "--response-tokens",
        dest="full_response_tokens",
        type=int,
        default=50,
        metavar="N",
        help="Tokens to generate per probe prompt when --probe-mode=full_response (default: 50).",
    )
    p_abl.add_argument(
        "--suppress-bank",
        dest="suppress_bank",
        metavar="PATH",
        help="JSON file containing a list of strings — prompts that trigger the concept "
        "to remove (overrides the built-in refusal bank).",
    )
    p_abl.add_argument(
        "--retain-bank",
        dest="retain_bank",
        metavar="PATH",
        help="JSON file containing a list of benign strings to anchor the retain direction "
        "(overrides the built-in retain bank).",
    )
    p_abl.set_defaults(handler=_cmd_abliterate)

    # ── to-mlx ────────────────────────────────────────────────────────────────
    p_tml = sub.add_parser(
        "to-mlx",
        help="Convert a HuggingFace model to MLX safetensors (required before WSP editing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Download and convert a HuggingFace model to MLX safetensors format.\n"
            "The resulting directory can then be passed to forge wsp abliterate.\n\n"
            "mlx_lm is tried first; if the architecture is not in its registry (e.g.\n"
            "inkling2_loop / Inkling-2-Mini), mlx_vlm is used automatically as a fallback.\n"
            "Both packages are included in the forge install on Apple Silicon."
        ),
    )
    p_tml.add_argument("hf_path", metavar="HF_REPO_OR_PATH", help="HuggingFace repo id or local safetensors directory")
    p_tml.add_argument(
        "--output",
        metavar="DIR",
        help="Output directory for the MLX model (default: <hf_path>-mlx)",
    )
    p_tml.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Quantise the weights after conversion (uses --q-bits)",
    )
    p_tml.add_argument(
        "--q-bits",
        dest="q_bits",
        type=int,
        default=4,
        metavar="N",
        help="Bits per weight when --quantize is set (default: 4)",
    )
    p_tml.add_argument(
        "--q-group-size",
        dest="q_group_size",
        type=int,
        default=64,
        metavar="N",
        help="Group size for quantisation (default: 64)",
    )
    p_tml.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default=None,
        help="Cast non-quantised weights to this dtype (default: from config)",
    )
    p_tml.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow custom model code from the HuggingFace repo",
    )
    p_tml.set_defaults(handler=_cmd_to_mlx)

    # ── to-gguf ───────────────────────────────────────────────────────────────
    p_tgf = sub.add_parser(
        "to-gguf",
        help="Convert an MLX model directory to GGUF (for Ollama / llama.cpp)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Convert an MLX safetensors directory (e.g. from forge wsp abliterate) to\n"
            "a GGUF file ready for Ollama.\n\n"
            "Uses llama.cpp/convert_hf_to_gguf.py when available (supports all architectures\n"
            "and quantized weights), otherwise falls back to mlx_lm.gguf (unquantized\n"
            "LLaMA-compatible models only).\n\n"
            "Full WSP pipeline:\n"
            "  forge wsp to-mlx  <hf-repo>  --output ./model-mlx\n"
            "  forge wsp abliterate ./model-mlx --output ./model-mlx-abliterated\n"
            "  forge wsp to-gguf  ./model-mlx-abliterated --output model.gguf --ollama-name my-model\n"
            "  ollama run my-model"
        ),
    )
    p_tgf.add_argument("mlx_path", metavar="MLX_DIR", help="MLX model directory (contains config.json + *.safetensors)")
    p_tgf.add_argument(
        "--output",
        metavar="PATH",
        help="Output GGUF file path (default: <MLX_DIR>.gguf)",
    )
    p_tgf.add_argument(
        "--ollama-name",
        dest="ollama_name",
        metavar="NAME",
        help="If set, automatically load the GGUF into Ollama under this name after conversion",
    )
    p_tgf.set_defaults(handler=_cmd_to_gguf)

    p.set_defaults(handler=lambda parser, args: (p.print_help(), 0)[1])
    return p
