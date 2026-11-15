"""VLM command group: vision-language model inference via mlx-vlm."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Any

from ollama_forge.run_helpers import print_actionable_error


def _cmd_vlm_generate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Generate text from a vision-language model with optional image/audio."""
    try:
        from ollama_forge.vlm import (
            is_vlm_available,
            vlm_apply_chat_template,
            vlm_generate,
            vlm_load,
        )
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    model_path = args.model
    prompt = args.prompt
    images = getattr(args, "image", None) or []
    audio = getattr(args, "audio", None)
    max_tokens = getattr(args, "max_tokens", 256)
    temperature = getattr(args, "temperature", 0.0)
    verbose = getattr(args, "verbose", False)
    adapter_path = getattr(args, "adapter_path", None)
    trust_remote_code = getattr(args, "trust_remote_code", False)
    quantize_activations = getattr(args, "quantize_activations", False)

    extra_kwargs: dict[str, Any] = {}
    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    kv_bits = getattr(args, "kv_bits", None)
    if kv_bits is not None:
        extra_kwargs["kv_bits"] = kv_bits
    kv_quant_scheme = getattr(args, "kv_quant_scheme", None)
    if kv_quant_scheme is not None:
        extra_kwargs["kv_quant_scheme"] = kv_quant_scheme
    kv_group_size = getattr(args, "kv_group_size", None)
    if kv_group_size is not None:
        extra_kwargs["kv_group_size"] = kv_group_size
    quantized_kv_start = getattr(args, "quantized_kv_start", None)
    if quantized_kv_start is not None:
        extra_kwargs["quantized_kv_start"] = quantized_kv_start
    max_kv_size = getattr(args, "max_kv_size", None)
    if max_kv_size is not None:
        extra_kwargs["max_kv_size"] = max_kv_size
    prefill_step_size = getattr(args, "prefill_step_size", None)
    if prefill_step_size is not None:
        extra_kwargs["prefill_step_size"] = prefill_step_size
    if getattr(args, "enable_thinking", False):
        extra_kwargs["enable_thinking"] = True
    thinking_budget = getattr(args, "thinking_budget", None)
    if thinking_budget is not None:
        extra_kwargs["thinking_budget"] = thinking_budget
    thinking_start_token = getattr(args, "thinking_start_token", None)
    if thinking_start_token is not None:
        extra_kwargs["thinking_start_token"] = thinking_start_token
    thinking_end_token = getattr(args, "thinking_end_token", None)
    if thinking_end_token is not None:
        extra_kwargs["thinking_end_token"] = thinking_end_token
    resize_shape = getattr(args, "resize_shape", None)
    if resize_shape is not None:
        extra_kwargs["resize_shape"] = resize_shape

    print(f"Loading model: {model_path} ...", file=sys.stderr)
    try:
        model, processor = vlm_load(
            model_path, adapter_path=adapter_path,
            trust_remote_code=trust_remote_code,
            quantize_activations=quantize_activations,
        )
    except Exception as exc:
        print_actionable_error(
            f"Failed to load model: {exc}",
            next_steps=[
                "Check the model path or HuggingFace repo ID",
                "Ensure you have enough memory for the model",
            ],
        )
        return 1

    num_images = len(images) if images else 0
    num_audios = 1 if audio else 0
    try:
        formatted_prompt = vlm_apply_chat_template(
            processor, model.config, prompt,
            num_images=num_images, num_audios=num_audios,
        )

        result = vlm_generate(
            model, processor, formatted_prompt,
            images=images or None, audio=audio,
            max_tokens=max_tokens, temperature=temperature,
            **extra_kwargs,
        )
    except Exception as exc:
        print_actionable_error(
            f"Generation failed: {exc}",
            next_steps=[
                "Check that image/audio paths exist and are valid",
                "Try reducing --max-tokens if running out of memory",
            ],
        )
        return 1

    print(result["text"])

    if verbose:
        stats_parts = []
        if result.get("prompt_tps") is not None:
            stats_parts.append(f"Prompt: {result['prompt_tps']:.1f} tok/s")
        if result.get("generation_tps") is not None:
            stats_parts.append(f"Generation: {result['generation_tps']:.1f} tok/s")
        if result.get("peak_memory") is not None:
            stats_parts.append(f"Peak memory: {result['peak_memory']:.2f} GB")
        if stats_parts:
            print(f"\n[{' | '.join(stats_parts)}]", file=sys.stderr)
    return 0


def _cmd_vlm_chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Interactive chat with a vision-language model."""
    try:
        from ollama_forge.vlm import (
            is_vlm_available,
            vlm_apply_chat_template,
            vlm_create_vision_cache,
            vlm_load,
            vlm_stream_generate,
        )
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    model_path = args.model
    max_tokens = getattr(args, "max_tokens", 512)
    temperature = getattr(args, "temperature", 0.7)
    system_prompt = getattr(args, "system", None)
    adapter_path = getattr(args, "adapter_path", None)
    trust_remote_code = getattr(args, "trust_remote_code", False)
    vision_cache_size = getattr(args, "vision_cache_size", 0)

    extra_kwargs: dict[str, Any] = {}
    kv_bits = getattr(args, "kv_bits", None)
    if kv_bits is not None:
        extra_kwargs["kv_bits"] = kv_bits
    kv_quant_scheme = getattr(args, "kv_quant_scheme", None)
    if kv_quant_scheme is not None:
        extra_kwargs["kv_quant_scheme"] = kv_quant_scheme
    kv_group_size = getattr(args, "kv_group_size", None)
    if kv_group_size is not None:
        extra_kwargs["kv_group_size"] = kv_group_size
    quantized_kv_start = getattr(args, "quantized_kv_start", None)
    if quantized_kv_start is not None:
        extra_kwargs["quantized_kv_start"] = quantized_kv_start
    max_kv_size = getattr(args, "max_kv_size", None)
    if max_kv_size is not None:
        extra_kwargs["max_kv_size"] = max_kv_size
    prefill_step_size = getattr(args, "prefill_step_size", None)
    if prefill_step_size is not None:
        extra_kwargs["prefill_step_size"] = prefill_step_size
    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    if getattr(args, "enable_thinking", False):
        extra_kwargs["enable_thinking"] = True
    thinking_budget = getattr(args, "thinking_budget", None)
    if thinking_budget is not None:
        extra_kwargs["thinking_budget"] = thinking_budget

    # Vision feature cache for multi-turn image caching (11x+ speedup)
    vision_cache = None
    if vision_cache_size > 0:
        try:
            vision_cache = vlm_create_vision_cache(max_size=vision_cache_size)
            print(f"Vision feature cache enabled (max {vision_cache_size} images)", file=sys.stderr)
        except Exception:
            print("Warning: VisionFeatureCache not available in this mlx-vlm version", file=sys.stderr)

    print(f"Loading model: {model_path} ...", file=sys.stderr)
    try:
        model, processor = vlm_load(
            model_path, adapter_path=adapter_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print_actionable_error(
            f"Failed to load model: {exc}",
            next_steps=[
                "Check the model path or HuggingFace repo ID",
                "Ensure you have enough memory for the model",
            ],
        )
        return 1

    print("Ready. Commands: /image <path>, /audio <path>, /clear, /quit", file=sys.stderr)
    if system_prompt:
        print(f"System: {system_prompt}", file=sys.stderr)
    print()

    current_images: list[str] = []
    current_audio: str | None = None
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            if stripped.lower() in ("/quit", "/exit"):
                break

            if stripped.lower() == "/clear":
                history.clear()
                if system_prompt:
                    history.append({"role": "system", "content": system_prompt})
                current_images = []
                current_audio = None
                if vision_cache is not None:
                    vision_cache.clear()
                print("  [History, attachments, and vision cache cleared]", file=sys.stderr)
                continue

            if stripped.lower().startswith("/image "):
                img_path = stripped[7:].strip()
                if not os.path.isfile(img_path):
                    print(f"  [File not found: {img_path}]", file=sys.stderr)
                    continue
                current_images.append(img_path)
                print(f"  [Attached image: {img_path}]", file=sys.stderr)
                continue

            if stripped.lower().startswith("/audio "):
                audio_path = stripped[7:].strip()
                if not os.path.isfile(audio_path):
                    print(f"  [File not found: {audio_path}]", file=sys.stderr)
                    continue
                current_audio = audio_path
                print(f"  [Attached audio: {audio_path}]", file=sys.stderr)
                continue

            # Build prompt with conversation history
            history.append({"role": "user", "content": stripped})
            conversation_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in history
            )

            num_images = len(current_images)
            num_audios = 1 if current_audio else 0
            formatted_prompt = vlm_apply_chat_template(
                processor, model.config, conversation_text,
                num_images=num_images, num_audios=num_audios,
            )

            print("Assistant: ", end="", flush=True)
            response_tokens: list[str] = []
            try:
                for token in vlm_stream_generate(
                    model, processor, formatted_prompt,
                    images=current_images or None,
                    audio=current_audio,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra_kwargs,
                ):
                    print(token, end="", flush=True)
                    response_tokens.append(token)
            except Exception as exc:
                print(f"\n  [Generation error: {exc}]", file=sys.stderr)
            print()

            # Record assistant response in history
            history.append({"role": "assistant", "content": "".join(response_tokens)})

            # Clear attachments after each turn
            current_images = []
            current_audio = None

    except KeyboardInterrupt:
        print("\nBye.")
    return 0


def _cmd_vlm_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Serve a VLM via mlx-vlm's built-in OpenAI-compatible server."""
    try:
        from ollama_forge.vlm import is_vlm_available
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    model_path = args.model
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 11434)
    adapter_path = getattr(args, "adapter_path", None)

    cmd = [
        sys.executable, "-m", "mlx_vlm.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
    ]
    kv_bits = getattr(args, "kv_bits", None)
    if kv_bits is not None:
        cmd += ["--kv-bits", str(kv_bits)]
    kv_quant_scheme = getattr(args, "kv_quant_scheme", None)
    if kv_quant_scheme is not None:
        cmd += ["--kv-quant-scheme", kv_quant_scheme]
    kv_group_size = getattr(args, "kv_group_size", None)
    if kv_group_size is not None:
        cmd += ["--kv-group-size", str(kv_group_size)]
    quantized_kv_start = getattr(args, "quantized_kv_start", None)
    if quantized_kv_start is not None:
        cmd += ["--quantized-kv-start", str(quantized_kv_start)]
    max_kv_size = getattr(args, "max_kv_size", None)
    if max_kv_size is not None:
        cmd += ["--max-kv-size", str(max_kv_size)]
    prefill_step_size = getattr(args, "prefill_step_size", None)
    if prefill_step_size is not None:
        cmd += ["--prefill-step-size", str(prefill_step_size)]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]
    if getattr(args, "trust_remote_code", False):
        cmd.append("--trust-remote-code")
    if getattr(args, "reload", False):
        cmd.append("--reload")

    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: Binding to {host} — the VLM server has no authentication. "
            "Only bind to non-localhost in trusted networks.",
            file=sys.stderr,
        )
    base_url = f"http://{host}:{port}"
    print(f"Starting mlx-vlm server: {shlex.join(cmd)}", file=sys.stderr)
    print(f"Endpoints: {base_url}/v1/chat/completions, {base_url}/v1/responses", file=sys.stderr)

    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        print_actionable_error(
            "Could not execute mlx_vlm.server",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down ...", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return proc.returncode or 0


def _cmd_vlm_video_generate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Generate text from video input using a vision-language model."""
    try:
        from ollama_forge.vlm import is_vlm_available, vlm_video_generate
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    model_path = args.model
    video = args.video
    prompt = getattr(args, "prompt", "Describe this video.")
    system = getattr(args, "system", None)
    max_tokens = getattr(args, "max_tokens", 100)
    temperature = getattr(args, "temperature", 0.7)
    max_pixels = getattr(args, "max_pixels", None)
    max_frames = getattr(args, "max_frames", None)
    fps = getattr(args, "fps", 1.0)
    verbose = getattr(args, "verbose", True)

    max_pixels_tuple = tuple(max_pixels) if max_pixels else None

    print(f"Processing {len(video)} video(s) with {model_path} ...", file=sys.stderr)
    try:
        rc = vlm_video_generate(
            model_path=model_path,
            video=video,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            max_pixels=max_pixels_tuple,
            max_frames=max_frames,
            fps=fps,
            verbose=verbose,
        )
    except Exception as exc:
        print_actionable_error(
            f"Video generation failed: {exc}",
            next_steps=[
                "Check that video paths exist and are valid",
                "Try reducing --max-frames if running out of memory",
            ],
        )
        return 1

    return rc


def _cmd_vlm_chat_ui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Launch the mlx-vlm Gradio chat UI for visual interaction with a VLM."""
    try:
        from ollama_forge.vlm import is_vlm_available
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    model_path = args.model

    cmd = [sys.executable, "-m", "mlx_vlm.chat_ui", "--model", model_path]

    print(f"Launching Gradio chat UI for {model_path} ...", file=sys.stderr)
    print("The browser should open automatically. Press Ctrl+C to stop.", file=sys.stderr)

    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        print_actionable_error(
            "Could not execute mlx_vlm.chat_ui",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down ...", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return proc.returncode or 0


def _cmd_vlm_convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Convert a HuggingFace VLM to MLX format."""
    try:
        from ollama_forge.vlm import is_vlm_available, vlm_convert
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    hf_path = args.hf_path
    mlx_path = getattr(args, "mlx_path", "mlx_model")
    quantize = getattr(args, "quantize", False)
    q_bits = getattr(args, "q_bits", 4)
    q_group_size = getattr(args, "q_group_size", 64)
    q_mode = getattr(args, "q_mode", "affine")
    dtype = getattr(args, "dtype", None)
    upload_repo = getattr(args, "upload_repo", None)
    revision = getattr(args, "revision", None)
    dequantize = getattr(args, "dequantize", False)
    trust_remote_code = getattr(args, "trust_remote_code", False)
    quant_predicate = getattr(args, "quant_predicate", None)

    action = "Dequantizing" if dequantize else "Converting"
    print(f"{action} {hf_path} -> {mlx_path} ...", file=sys.stderr)
    try:
        out = vlm_convert(
            hf_path,
            mlx_path=mlx_path,
            quantize=quantize,
            q_bits=q_bits,
            q_group_size=q_group_size,
            q_mode=q_mode,
            dtype=dtype,
            upload_repo=upload_repo,
            revision=revision,
            dequantize=dequantize,
            trust_remote_code=trust_remote_code,
            quant_predicate=quant_predicate,
        )
    except Exception as exc:
        print_actionable_error(
            f"Conversion failed: {exc}",
            next_steps=[
                "Check the model path or HuggingFace repo ID",
                "Ensure you have enough disk space",
            ],
        )
        return 1

    print(f"Converted model saved to: {out}")
    return 0


def _cmd_vlm_quantize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Quantize a VLM model (convenience wrapper around convert with --quantize)."""
    try:
        from ollama_forge.vlm import is_vlm_available, vlm_convert
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    model = args.model
    output = getattr(args, "output", "mlx_model_quantized")
    bits = getattr(args, "bits", 4)
    group_size = getattr(args, "group_size", 64)
    q_mode = getattr(args, "q_mode", "affine")
    dtype = getattr(args, "dtype", None)
    upload_repo = getattr(args, "upload_repo", None)
    trust_remote_code = getattr(args, "trust_remote_code", False)

    print(
        f"Quantizing {model} -> {output} ({bits}-bit, mode={q_mode}, group_size={group_size}) ...",
        file=sys.stderr,
    )
    try:
        out = vlm_convert(
            model,
            mlx_path=output,
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
            q_mode=q_mode,
            dtype=dtype,
            upload_repo=upload_repo,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print_actionable_error(
            f"Quantization failed: {exc}",
            next_steps=[
                "Check the model path or HuggingFace repo ID",
                "Ensure you have enough disk space",
            ],
        )
        return 1

    print(f"Quantized model saved to: {out}")
    return 0


def _cmd_vlm_finetune(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Fine-tune a VLM using LoRA/QLoRA/full via mlx-vlm."""
    try:
        from ollama_forge.vlm import is_vlm_available, vlm_finetune
    except ImportError as exc:
        print_actionable_error(
            f"Missing dependency: {exc}",
            next_steps=["Install mlx-vlm: pip install 'mlx-vlm>=0.4.4'"],
        )
        return 1

    if not is_vlm_available():
        print_actionable_error(
            "mlx-vlm is not installed",
            next_steps=[
                "Install with: pip install 'mlx-vlm>=0.4.4'",
                "Requires Apple Silicon (M1/M2/M3/M4)",
            ],
        )
        return 1

    dataset = args.dataset
    if not os.path.exists(dataset):
        print_actionable_error(
            f"Dataset file not found: {dataset}",
            next_steps=[
                "Check the path and ensure the file exists",
                "Dataset should be a JSONL file with 'images' and 'messages' columns",
            ],
        )
        return 1

    train_mode = getattr(args, "train_mode", "sft")
    print(f"Fine-tuning model: {args.model} (mode: {train_mode})", file=sys.stderr)
    print(f"Dataset: {dataset}", file=sys.stderr)
    print(f"Output: {args.output_path}", file=sys.stderr)

    image_resize_shape = getattr(args, "image_resize_shape", None)
    image_resize_tuple = tuple(image_resize_shape) if image_resize_shape else None

    try:
        rc = vlm_finetune(
            model_path=args.model,
            dataset=dataset,
            output_path=args.output_path,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=getattr(args, "epochs", None),
            iters=getattr(args, "iters", 1000),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision=args.train_vision,
            full_finetune=args.full_finetune,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            grad_checkpoint=args.grad_checkpoint,
            grad_clip=getattr(args, "grad_clip", None),
            adapter_path=getattr(args, "adapter_path", None),
            split=getattr(args, "split", "train"),
            dataset_config=getattr(args, "dataset_config", None),
            image_resize_shape=image_resize_tuple,
            custom_prompt_format=getattr(args, "custom_prompt_format", None),
            steps_per_report=getattr(args, "steps_per_report", 10),
            steps_per_eval=getattr(args, "steps_per_eval", 200),
            steps_per_save=getattr(args, "steps_per_save", 100),
            val_batches=getattr(args, "val_batches", 25),
            max_seq_length=getattr(args, "max_seq_length", 2048),
            train_on_completions=getattr(args, "train_on_completions", False),
            train_mode=train_mode,
            beta=getattr(args, "beta", 0.1),
            eps=getattr(args, "eps", 1e-8),
            assistant_id=getattr(args, "assistant_id", 77091),
        )
    except Exception as exc:
        print_actionable_error(
            f"Fine-tuning failed: {exc}",
            next_steps=[
                "Check model path and dataset format",
                "Try reducing batch size if running out of memory",
            ],
        )
        return 1

    if rc == 0:
        print(f"Fine-tuning complete. Adapter saved to: {args.output_path}", file=sys.stderr)
    else:
        print_actionable_error(
            f"Fine-tuning failed (exit code {rc})",
            next_steps=[
                "Check the training output above for errors",
                "Ensure the dataset format is correct (JSONL with images + messages)",
                "Try reducing batch size if running out of memory",
            ],
        )
    return rc


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Register the ``vlm`` subcommand group and return its parser."""
    p_vlm = subparsers.add_parser(
        "vlm",
        help="Vision Language Model inference on Apple Silicon (mlx-vlm >=0.4.4)",
    )
    vlm_sub = p_vlm.add_subparsers(dest="vlm_command")

    # vlm generate
    p_vlm_gen = vlm_sub.add_parser(
        "generate",
        help="Generate text from a vision-language model",
    )
    p_vlm_gen.add_argument("--model", required=True,
                           help="HF repo id or local path (e.g. mlx-community/Qwen2-VL-2B-Instruct-4bit)")
    p_vlm_gen.add_argument("--prompt", required=True, help="Text prompt")
    p_vlm_gen.add_argument("--image", action="append", default=None,
                           help="Image path or URL (can be repeated)")
    p_vlm_gen.add_argument("--audio", action="append", default=None,
                           help="Audio file path (can be repeated)")
    p_vlm_gen.add_argument("--resize-shape", type=int, nargs="+", default=None,
                           help="Resize shape for input images (e.g. 224 224)")
    p_vlm_gen.add_argument("--system", default=None, help="System message")
    p_vlm_gen.add_argument("--max-tokens", type=int, default=256,
                           help="Max tokens to generate (default: 256)")
    p_vlm_gen.add_argument("--temperature", type=float, default=0.0,
                           help="Sampling temperature (default: 0.0)")
    p_vlm_gen.add_argument("--top-p", type=float, default=None,
                           help="Nucleus sampling threshold")
    # TurboQuant KV cache options
    p_vlm_gen.add_argument("--kv-bits", type=float, default=None,
                           help="KV cache quantization bits (e.g. 4, 3.5 for TurboQuant)")
    p_vlm_gen.add_argument("--kv-quant-scheme", default=None,
                           choices=("uniform", "turboquant"),
                           help="KV cache quantization backend (fractional --kv-bits auto-selects turboquant)")
    p_vlm_gen.add_argument("--kv-group-size", type=int, default=None,
                           help="Group size for uniform KV cache quantization")
    p_vlm_gen.add_argument("--quantized-kv-start", type=int, default=None,
                           help="Start token index for quantized KV cache")
    p_vlm_gen.add_argument("--max-kv-size", type=int, default=None,
                           help="Maximum KV cache size in tokens")
    p_vlm_gen.add_argument("--prefill-step-size", type=int, default=None,
                           help="Tokens per prefill step (lower = less peak memory)")
    # Thinking / reasoning
    p_vlm_gen.add_argument("--enable-thinking", action="store_true", default=False,
                           help="Enable chain-of-thought / thinking mode")
    p_vlm_gen.add_argument("--thinking-budget", type=int, default=None,
                           help="Max thinking tokens")
    p_vlm_gen.add_argument("--thinking-start-token", default=None,
                           help="Token marking start of thinking block (default: <think>)")
    p_vlm_gen.add_argument("--thinking-end-token", default=None,
                           help="Token marking end of thinking block (default: </think>)")
    # Model options
    p_vlm_gen.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    p_vlm_gen.add_argument("--trust-remote-code", action="store_true", default=False,
                           help="Trust remote code when loading from HF Hub")
    p_vlm_gen.add_argument("--quantize-activations", action="store_true", default=False,
                           help="Enable activation quantization for mxfp8 models")
    p_vlm_gen.add_argument("--verbose", action="store_true", default=False,
                           help="Show timing stats")
    p_vlm_gen.set_defaults(handler=_cmd_vlm_generate)

    # vlm chat
    p_vlm_chat = vlm_sub.add_parser(
        "chat",
        help="Interactive multi-turn chat with a vision-language model",
    )
    p_vlm_chat.add_argument("--model", required=True,
                            help="HF repo id or local path")
    p_vlm_chat.add_argument("--max-tokens", type=int, default=512,
                            help="Max tokens per response (default: 512)")
    p_vlm_chat.add_argument("--temperature", type=float, default=0.7,
                            help="Sampling temperature (default: 0.7)")
    p_vlm_chat.add_argument("--top-p", type=float, default=None,
                            help="Nucleus sampling threshold")
    p_vlm_chat.add_argument("--system", default=None, help="System prompt")
    # TurboQuant KV cache options
    p_vlm_chat.add_argument("--kv-bits", type=float, default=None,
                            help="KV cache quantization bits (e.g. 4, 3.5 for TurboQuant)")
    p_vlm_chat.add_argument("--kv-quant-scheme", default=None,
                            choices=("uniform", "turboquant"),
                            help="KV cache quantization backend")
    p_vlm_chat.add_argument("--kv-group-size", type=int, default=None,
                            help="Group size for uniform KV cache quantization")
    p_vlm_chat.add_argument("--quantized-kv-start", type=int, default=None,
                            help="Start token index for quantized KV cache")
    p_vlm_chat.add_argument("--max-kv-size", type=int, default=None,
                            help="Maximum KV cache size in tokens")
    p_vlm_chat.add_argument("--prefill-step-size", type=int, default=None,
                            help="Tokens per prefill step")
    # Thinking
    p_vlm_chat.add_argument("--enable-thinking", action="store_true", default=False,
                            help="Enable chain-of-thought / thinking mode")
    p_vlm_chat.add_argument("--thinking-budget", type=int, default=None,
                            help="Max thinking tokens")
    # Vision feature cache
    p_vlm_chat.add_argument("--vision-cache-size", type=int, default=0,
                            help="Enable VisionFeatureCache with this max size (0=disabled, e.g. 20 for 11x+ speedup)")
    # Model options
    p_vlm_chat.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    p_vlm_chat.add_argument("--trust-remote-code", action="store_true", default=False,
                            help="Trust remote code when loading from HF Hub")
    p_vlm_chat.set_defaults(handler=_cmd_vlm_chat)

    # vlm serve
    p_vlm_serve = vlm_sub.add_parser(
        "serve",
        help="Serve a VLM via OpenAI-compatible API (mlx-vlm server, /v1/chat/completions + /v1/responses)",
    )
    p_vlm_serve.add_argument("--model", required=True,
                             help="HF repo id or local path")
    p_vlm_serve.add_argument("--host", default="127.0.0.1",
                             help="Bind host (default: 127.0.0.1)")
    p_vlm_serve.add_argument("--port", type=int, default=11434,
                             help="Port (default: 11434)")
    # TurboQuant KV cache options
    p_vlm_serve.add_argument("--kv-bits", type=float, default=None,
                             help="KV cache quantization bits (e.g. 4, 3.5 for TurboQuant)")
    p_vlm_serve.add_argument("--kv-quant-scheme", default=None,
                             choices=("uniform", "turboquant"),
                             help="KV cache quantization backend")
    p_vlm_serve.add_argument("--kv-group-size", type=int, default=None,
                             help="Group size for uniform KV cache quantization")
    p_vlm_serve.add_argument("--quantized-kv-start", type=int, default=None,
                             help="Start token index for quantized KV cache")
    p_vlm_serve.add_argument("--max-kv-size", type=int, default=None,
                             help="Maximum KV cache size in tokens")
    p_vlm_serve.add_argument("--prefill-step-size", type=int, default=None,
                             help="Tokens per prefill step (lower = less peak memory; try 512 or 256)")
    # Model options
    p_vlm_serve.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    p_vlm_serve.add_argument("--trust-remote-code", action="store_true", default=False,
                             help="Trust remote code when loading from HF Hub")
    p_vlm_serve.add_argument("--reload", action="store_true", default=False,
                             help="Enable auto-reload on file changes (development only)")
    p_vlm_serve.set_defaults(handler=_cmd_vlm_serve)

    # vlm video-generate
    p_vlm_video = vlm_sub.add_parser(
        "video-generate",
        help="Generate text from video input using a vision-language model",
    )
    p_vlm_video.add_argument("--model", required=True,
                             help="HF repo id or local path (e.g. mlx-community/Qwen2.5-VL-7B-Instruct-4bit)")
    p_vlm_video.add_argument("--video", required=True, action="append",
                             help="Path to video file (can be repeated)")
    p_vlm_video.add_argument("--prompt", default="Describe this video.",
                             help="Text prompt (default: 'Describe this video.')")
    p_vlm_video.add_argument("--system", default=None, help="System prompt")
    p_vlm_video.add_argument("--max-tokens", type=int, default=100,
                             help="Max tokens to generate (default: 100)")
    p_vlm_video.add_argument("--temperature", type=float, default=0.7,
                             help="Sampling temperature (default: 0.7)")
    p_vlm_video.add_argument("--max-pixels", type=int, nargs=2, default=None,
                             help="Maximum pixel dimensions (width height, e.g. 224 224)")
    p_vlm_video.add_argument("--max-frames", type=int, default=None,
                             help="Maximum number of frames to extract")
    p_vlm_video.add_argument("--fps", type=float, default=1.0,
                             help="Frames per second for extraction (default: 1.0)")
    p_vlm_video.add_argument("--verbose", action="store_true", default=True,
                             help="Print verbose output (default: True)")
    p_vlm_video.set_defaults(handler=_cmd_vlm_video_generate)

    # vlm chat-ui
    p_vlm_chat_ui = vlm_sub.add_parser(
        "chat-ui",
        help="Launch Gradio web UI for visual VLM chat (image drag-and-drop)",
    )
    p_vlm_chat_ui.add_argument("--model", required=True,
                               help="HF repo id or local path")
    p_vlm_chat_ui.set_defaults(handler=_cmd_vlm_chat_ui)

    # vlm convert
    p_vlm_convert = vlm_sub.add_parser(
        "convert",
        help="Convert a HuggingFace VLM to MLX format",
    )
    p_vlm_convert.add_argument("--hf-path", required=True,
                               help="HF repo id or local path")
    p_vlm_convert.add_argument("--mlx-path", default="mlx_model",
                               help="Output directory (default: mlx_model)")
    p_vlm_convert.add_argument("--quantize", action="store_true", default=False,
                               help="Enable quantization during conversion")
    p_vlm_convert.add_argument("--q-bits", type=int, default=4,
                               help="Quantization bits (default: 4)")
    p_vlm_convert.add_argument("--q-group-size", type=int, default=64,
                               help="Quantization group size (default: 64)")
    p_vlm_convert.add_argument("--q-mode", default="affine",
                               choices=("affine", "mxfp4", "nvfp4", "mxfp8"),
                               help="Quantization mode (default: affine)")
    p_vlm_convert.add_argument("--dtype", default=None,
                               help="Output dtype (e.g. float16)")
    p_vlm_convert.add_argument("--upload-repo", default=None,
                               help="Upload converted model to HF repo")
    p_vlm_convert.add_argument("--revision", default=None,
                               help="HuggingFace revision (branch/tag/commit)")
    p_vlm_convert.add_argument("--dequantize", action="store_true", default=False,
                               help="Dequantize a quantized model back to full precision")
    p_vlm_convert.add_argument("--trust-remote-code", action="store_true", default=False,
                               help="Trust remote code when loading from HF Hub")
    p_vlm_convert.add_argument("--quant-predicate", default=None,
                               help="Mixed-bit quantization recipe string")
    p_vlm_convert.set_defaults(handler=_cmd_vlm_convert)

    # vlm quantize
    p_vlm_quant = vlm_sub.add_parser(
        "quantize",
        help="Quantize a VLM model (convert with quantization enabled)",
    )
    p_vlm_quant.add_argument("--model", required=True,
                             help="HF repo id or local model path")
    p_vlm_quant.add_argument("--output", default="mlx_model_quantized",
                             help="Output directory (default: mlx_model_quantized)")
    p_vlm_quant.add_argument("--bits", type=int, default=4,
                             help="Quantization bits (default: 4)")
    p_vlm_quant.add_argument("--group-size", type=int, default=64,
                             help="Group size (default: 64)")
    p_vlm_quant.add_argument("--q-mode", default="affine",
                             choices=("affine", "mxfp4", "nvfp4", "mxfp8"),
                             help="Quantization mode (default: affine)")
    p_vlm_quant.add_argument("--dtype", default=None,
                             help="Output dtype (e.g. float16)")
    p_vlm_quant.add_argument("--upload-repo", default=None,
                             help="Upload to HF repo")
    p_vlm_quant.add_argument("--trust-remote-code", action="store_true", default=False,
                             help="Trust remote code when loading from HF Hub")
    p_vlm_quant.set_defaults(handler=_cmd_vlm_quantize)

    # vlm finetune
    p_vlm_ft = vlm_sub.add_parser(
        "finetune",
        help="Fine-tune a VLM using LoRA/QLoRA/full/ORPO (mlx-vlm)",
    )
    p_vlm_ft.add_argument("--model", required=True,
                           help="HF repo id or local path to base model")
    p_vlm_ft.add_argument("--dataset", required=True,
                           help="Path to training data (JSONL or HF dataset)")
    p_vlm_ft.add_argument("--output-path", default="vlm_adapter",
                           help="Output directory for adapter (default: vlm_adapter)")
    # Training hyperparams
    p_vlm_ft.add_argument("--learning-rate", type=float, default=2e-5,
                           help="Learning rate (default: 2e-5)")
    p_vlm_ft.add_argument("--batch-size", type=int, default=4,
                           help="Batch size (default: 4)")
    p_vlm_ft.add_argument("--epochs", type=int, default=None,
                           help="Training epochs (default: None, uses --iters instead)")
    p_vlm_ft.add_argument("--iters", type=int, default=1000,
                           help="Training iterations (default: 1000, ignored if --epochs set)")
    p_vlm_ft.add_argument("--max-seq-length", type=int, default=2048,
                           help="Max sequence length (default: 2048)")
    p_vlm_ft.add_argument("--gradient-accumulation-steps", type=int, default=1,
                           help="Gradient accumulation steps (default: 1)")
    p_vlm_ft.add_argument("--grad-checkpoint", action="store_true", default=False,
                           help="Enable gradient checkpointing")
    p_vlm_ft.add_argument("--grad-clip", type=float, default=None,
                           help="Gradient clipping max norm")
    # LoRA params
    p_vlm_ft.add_argument("--lora-rank", type=int, default=8,
                           help="LoRA rank (default: 8)")
    p_vlm_ft.add_argument("--lora-alpha", type=float, default=16,
                           help="LoRA alpha (default: 16)")
    p_vlm_ft.add_argument("--lora-dropout", type=float, default=0.0,
                           help="LoRA dropout (default: 0.0)")
    # Training mode
    p_vlm_ft.add_argument("--train-vision", action="store_true", default=False,
                           help="Fine-tune vision encoder too")
    p_vlm_ft.add_argument("--full-finetune", action="store_true", default=False,
                           help="Full weight update instead of LoRA")
    p_vlm_ft.add_argument("--train-on-completions", action="store_true", default=False,
                           help="Only train on completion tokens (not prompts)")
    p_vlm_ft.add_argument("--train-mode", default="sft",
                           choices=("sft", "orpo"),
                           help="Training mode: sft (default) or orpo")
    # ORPO-specific
    p_vlm_ft.add_argument("--beta", type=float, default=0.1,
                           help="ORPO beta / odds-ratio weight (default: 0.1)")
    p_vlm_ft.add_argument("--eps", type=float, default=1e-8,
                           help="ORPO epsilon for numerical stability (default: 1e-8)")
    # Dataset options
    p_vlm_ft.add_argument("--split", default="train",
                           help="Dataset split (default: train)")
    p_vlm_ft.add_argument("--dataset-config", default=None,
                           help="HF dataset config name")
    p_vlm_ft.add_argument("--image-resize-shape", type=int, nargs=2, default=None,
                           help="Resize images to this shape (width height)")
    p_vlm_ft.add_argument("--custom-prompt-format", default=None,
                           help="Custom JSON prompt template string")
    p_vlm_ft.add_argument("--assistant-id", type=int, default=77091,
                           help="Assistant token ID for completions-only training (default: 77091)")
    # Reporting and checkpoints
    p_vlm_ft.add_argument("--steps-per-report", type=int, default=10,
                           help="Steps between training reports (default: 10)")
    p_vlm_ft.add_argument("--steps-per-eval", type=int, default=200,
                           help="Steps between evaluations (default: 200)")
    p_vlm_ft.add_argument("--steps-per-save", type=int, default=100,
                           help="Steps between adapter saves (default: 100)")
    p_vlm_ft.add_argument("--val-batches", type=int, default=25,
                           help="Number of validation batches (default: 25)")
    # Resume
    p_vlm_ft.add_argument("--adapter-path", default=None,
                           help="Resume from existing adapter")
    p_vlm_ft.set_defaults(handler=_cmd_vlm_finetune)

    return p_vlm
