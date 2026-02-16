"""Refusal-direction computation for abliterated models (optional deps: torch, transformers)."""

from __future__ import annotations

import random
import re
import shutil
import sys
import tempfile
from pathlib import Path


def _strip_chat_reply(text: str, last_user_content: str | None = None) -> str:
    """Remove leading/trailing role-only lines (user/model/assistant or User:/Model:/Assistant:) and echoed last user message."""
    if not text or not text.strip():
        return text
    role_only = re.compile(r"^(model|user|assistant|User|Model|Assistant)\s*:?\s*$", re.IGNORECASE)
    last_stripped = (last_user_content or "").strip()
    lines = text.split("\n")
    # Strip leading
    while lines:
        line = lines[0].strip()
        if not line:
            lines.pop(0)
            continue
        if role_only.match(line) or (last_stripped and line == last_stripped) or (last_stripped and (line == f"User: {last_stripped}" or line == f"user: {last_stripped}")):
            lines.pop(0)
            continue
        break
    # Strip trailing
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue
        if role_only.match(line):
            lines.pop()
            continue
        break
    return "\n".join(lines)


def _load_model_with_gguf_version_workaround(model_id: str, load_kw: dict):
    """Call AutoModelForCausalLM.from_pretrained; if Invalid version 'N/A' (e.g. from GGUF metadata), patch packaging.version and retry."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    except Exception as e:  # noqa: BLE001
        if "Invalid version" not in str(e) or "N/A" not in str(e):
            raise
        import packaging.version as pkg_version

        _orig = pkg_version.Version

        def _coerce(v: str):
            if not v or v in ("N/A", "n/a") or (v[0].isdigit() is False):
                return _orig("0.0.0")
            return _orig(v)

        pkg_version.Version = _coerce
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        finally:
            pkg_version.Version = _orig


def _model_hidden_size(model):  # noqa: ANN001
    """Return hidden_size from config (supports multimodal Gemma 3 text_config)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if getattr(cfg, "hidden_size", None) is not None:
        return cfg.hidden_size
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None) is not None:
        return text_cfg.hidden_size
    return None


def _model_max_position_embeddings(model):  # noqa: ANN001
    """Return max_position_embeddings from config (supports text_config for Gemma etc.)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    n = getattr(cfg, "max_position_embeddings", None)
    if n is not None and n > 0:
        return n
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        n = getattr(text_cfg, "max_position_embeddings", None)
        if n is not None and n > 0:
            return n
    return None


def get_layers(model):  # noqa: ANN001
    """Return the list of transformer layers (model.model.layers, model.model.language_model.layers, or model.transformer.h)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not find layers: try model.model.layers, model.model.language_model.layers, or model.transformer.h. "
        "See docs/ABLITERATE.md for supported architectures."
    )


def compute_refusal_dir(
    model_id: str,
    harmful_path: str | Path,
    harmless_path: str | Path,
    output_path: str | Path,
    *,
    num_instructions: int = 32,
    layer_fracs: tuple[float, ...] = (0.4, 0.5, 0.6),
    n_directions: int = 1,
    pos: int = -1,
    device: str | None = None,
    load_in_8bit: bool = False,
    gguf_file: str | Path | None = None,
) -> None:
    """
    Compute refusal direction(s) from harmful vs harmless instructions and save to output_path (.pt).
    Tries each layer_frac, picks the layer with largest harmful-harmless gap (direction selection).
    Saves one direction (mean difference) or n_directions (top-k from SVD on difference matrix).
    Requires torch and transformers. Use: uv sync --extra abliterate.
    """
    import torch
    from transformers import AutoTokenizer

    harmful_path = Path(harmful_path)
    harmless_path = Path(harmless_path)
    output_path = Path(output_path)
    if not harmful_path.is_file():
        raise FileNotFoundError(f"harmful file not found: {harmful_path}")
    if not harmless_path.is_file():
        raise FileNotFoundError(f"harmless file not found: {harmless_path}")

    with harmful_path.open() as f:
        harmful_lines = [s for s in (line.strip() for line in f) if s and not s.startswith("#")]
    with harmless_path.open() as f:
        harmless_lines = [s for s in (line.strip() for line in f) if s]

    # Support loading from GGUF (e.g. Ollama blob path) when gguf_file is set or model_id is a .gguf path
    use_gguf = gguf_file is not None or (
        isinstance(model_id, (str, Path)) and str(model_id).lower().endswith(".gguf")
    )
    gguf_path_str = str(gguf_file) if gguf_file else (str(model_id) if use_gguf else None)
    load_from_gguf_kw: dict = {}
    if gguf_path_str:
        load_from_gguf_kw["gguf_file"] = gguf_path_str
        if use_gguf and not gguf_file:
            model_id = gguf_path_str  # use path as identifier for GGUF-only load

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, **load_from_gguf_kw
    )
    if device is None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "cpu"
    load_kw: dict = {
        "trust_remote_code": True,
        "device_map": "auto" if device is None else device,
        "low_cpu_mem_usage": True,
        **load_from_gguf_kw,
    }
    if load_in_8bit and not gguf_path_str:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "load_in_8bit requires bitsandbytes and transformers support. "
                "Install with: pip install bitsandbytes"
            ) from None
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif not gguf_path_str:
        load_kw["dtype"] = torch.bfloat16
    offload_folder: str | None = None
    if load_kw["device_map"] == "auto" and not load_in_8bit and not gguf_path_str:
        offload_folder = tempfile.mkdtemp(prefix="ollama_abliterate_offload_")
        load_kw["offload_folder"] = offload_folder
    try:
        model = _load_model_with_gguf_version_workaround(model_id, load_kw)
    except Exception:
        if offload_folder:
            shutil.rmtree(offload_folder, ignore_errors=True)
        raise
    if offload_folder:
        shutil.rmtree(offload_folder, ignore_errors=True)
    layers = get_layers(model)
    n = min(num_instructions, len(harmful_lines), len(harmless_lines))
    harmful_instructions = random.sample(harmful_lines, n)
    harmless_instructions = random.sample(harmless_lines, n)

    def tokenize(instructions):
        out = []
        use_chat_template = getattr(tokenizer, "chat_template", None) is not None
        for insn in instructions:
            try:
                if use_chat_template:
                    encoded = tokenizer.apply_chat_template(
                        conversation=[{"role": "user", "content": insn}],
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
                else:
                    encoded = tokenizer(
                        insn,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    input_ids = encoded["input_ids"]
            except Exception:
                use_chat_template = False
                encoded = tokenizer(
                    insn,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                input_ids = encoded["input_ids"]
            out.append(input_ids.to(model.device))
        return out

    harmful_toks = tokenize(harmful_instructions)
    harmless_toks = tokenize(harmless_instructions)

    # Direction selection: try each layer_frac, pick layer with largest harmful-harmless gap.
    best_layer_frac = layer_fracs[0]
    best_gap_norm = -1.0
    for layer_frac in layer_fracs:
        layer_idx = int(len(layers) * layer_frac)
        layer_idx = min(layer_idx, len(layers) - 1)
        h_layer = layer_idx + 1
        harmful_h = []
        harmless_h = []
        with torch.inference_mode():
            for toks in harmful_toks:
                out = model(toks, output_hidden_states=True)
                harmful_h.append(out.hidden_states[h_layer][:, pos, :])
            for toks in harmless_toks:
                out = model(toks, output_hidden_states=True)
                harmless_h.append(out.hidden_states[h_layer][:, pos, :])
        harm_mean = torch.cat(harmful_h, dim=0).mean(dim=0).float()
        safe_mean = torch.cat(harmless_h, dim=0).mean(dim=0).float()
        gap = harm_mean - safe_mean
        gnorm = gap.norm().item()
        if gnorm > best_gap_norm:
            best_gap_norm = gnorm
            best_layer_frac = layer_frac

    layer_idx = min(int(len(layers) * best_layer_frac), len(layers) - 1)
    h_layer = layer_idx + 1
    harmful_hidden = []
    harmless_hidden = []
    with torch.inference_mode():
        for toks in harmful_toks:
            out = model(toks, output_hidden_states=True)
            harmful_hidden.append(out.hidden_states[h_layer][:, pos, :])
        for toks in harmless_toks:
            out = model(toks, output_hidden_states=True)
            harmless_hidden.append(out.hidden_states[h_layer][:, pos, :])
    harmful_mat = torch.cat(harmful_hidden, dim=0).float()
    harmless_mat = torch.cat(harmless_hidden, dim=0).float()
    hidden_size = harmful_mat.size(1)

    if n_directions <= 1:
        refusal_dir = (harmful_mat.mean(0) - harmless_mat.mean(0)).unsqueeze(1)
        refusal_dir = refusal_dir / refusal_dir.norm()
    else:
        diff = harmful_mat - harmless_mat
        k = min(n_directions, diff.size(0), hidden_size)
        U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
        D = Vh[:k, :].T
        for j in range(D.size(1)):
            col = D[:, j]
            nrm = col.norm().item()
            if nrm > 1e-8:
                D[:, j] = col / nrm
        refusal_dir = D

    torch.save(refusal_dir, output_path)
    if offload_folder:
        shutil.rmtree(offload_folder, ignore_errors=True)


def apply_refusal_dir_and_save(
    model_id: str,
    refusal_pt_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 1,
    norm_preserving: bool = True,
    verify: bool = True,
    gguf_file: str | Path | None = None,
    strength: float = 1.0,
) -> None:
    """
    Bake the refusal-direction ablation into the model weights and save to output_dir.
    Supports single direction (1D or (H,1)) or multi-direction (H, k) .pt. Uses
    W' = W @ (I - strength * D D^T), optionally norm-preserving. Skips first/last layers.
    strength=1.0 is full ablation; 0 < strength < 1 (e.g. 0.5) softens the effect to reduce
    coherence loss on small models. If verify=True, runs one forward pass and checks loss is finite.
    """
    import torch
    from transformers import AutoTokenizer

    refusal_pt_path = Path(refusal_pt_path)
    output_dir = Path(output_dir)
    if not 0 < strength <= 1:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if not refusal_pt_path.is_file():
        raise FileNotFoundError(f"refusal .pt not found: {refusal_pt_path}")

    d = torch.load(refusal_pt_path, map_location="cpu", weights_only=True)
    d = d.float()
    if d.dim() == 1:
        d = d.unsqueeze(1)
    if d.dim() != 2 or d.size(0) < d.size(1):
        raise ValueError(f"Expected refusal tensor (hidden_size, k), got shape {d.shape}")
    hidden_size, k = d.size(0), d.size(1)
    for j in range(k):
        col = d[:, j]
        nrm = col.norm().item()
        if nrm > 1e-8:
            d[:, j] = col / nrm
    I_minus_DDT = torch.eye(hidden_size) - strength * (d @ d.T)

    use_gguf = gguf_file is not None or (
        isinstance(model_id, (str, Path)) and str(model_id).lower().endswith(".gguf")
    )
    gguf_path_str = str(gguf_file) if gguf_file else (str(model_id) if use_gguf else None)
    load_gguf_kw: dict = {"gguf_file": gguf_path_str} if gguf_path_str else {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, **load_gguf_kw
    )
    load_apply_kw = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
        **load_gguf_kw,
    }
    model = _load_model_with_gguf_version_workaround(model_id, load_apply_kw)
    model_hidden = _model_hidden_size(model)
    if model_hidden is not None and model_hidden != hidden_size:
        raise ValueError(
            f"Refusal vector size {hidden_size} does not match model hidden_size "
            f"({model_hidden}). Use the same model for compute-dir and apply."
        )

    layers = get_layers(model)
    n_layers = len(layers)
    start_idx = min(skip_begin_layers, n_layers - 1)
    end_idx = max(start_idx, n_layers - skip_end_layers)

    # Ablate every linear that has hidden_size as input dim (q_proj, k_proj, v_proj, o_proj)
    # so refusal direction is removed from full attention, not just queries (stronger abliteration).
    def _linears_with_hidden_in(attn):
        out = []
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(attn, name):
                out.append(getattr(attn, name))
        return out

    for layer_idx, layer in enumerate(layers):
        if layer_idx < start_idx or layer_idx >= end_idx:
            continue
        if hasattr(layer, "self_attn"):
            linears = _linears_with_hidden_in(layer.self_attn)
        elif hasattr(layer, "attention"):
            linears = _linears_with_hidden_in(layer.attention)
        else:
            continue
        for linear in linears:
            w = linear.weight.data.float()
            if w.shape[1] != hidden_size:
                continue
            with torch.no_grad():
                new_w = (w @ I_minus_DDT).to(linear.weight.dtype)
                if norm_preserving:
                    orig_norm = torch.linalg.norm(w).item()
                    new_norm = torch.linalg.norm(new_w).item()
                    if new_norm > 1e-8:
                        new_w = new_w * (orig_norm / new_norm)
                linear.weight.data.copy_(new_w)

    if verify:
        with torch.inference_mode():
            try:
                inp = tokenizer("The", return_tensors="pt", add_special_tokens=True)
                input_ids = inp["input_ids"]
                out = model(input_ids, labels=input_ids)
                loss = out.loss.item()
                if not (loss == loss and abs(loss) != float("inf")):
                    print("Warning: verification forward pass produced non-finite loss.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: verification forward pass failed: {e}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except NotImplementedError:
        # Model was loaded with a weight conversion (e.g. MXFP4->bf16) that has no reverse.
        # Save state dict and config manually so convert_hf_to_gguf can still read the checkpoint.
        print("Saving checkpoint manually (large model may take several minutes)...", file=sys.stderr)
        sys.stderr.flush()
        state_dict = model.state_dict()
        # Only tensors (model is already on CPU when device_map='cpu')
        state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        out_path = output_dir / "pytorch_model.bin"
        try:
            # torch.save streams to disk; safetensors may use more peak memory for large models
            torch.save(state_dict, out_path)
        except Exception as e:
            print(f"Error saving state dict: {e}", file=sys.stderr)
            raise
        model.config.save_pretrained(output_dir)
        print("Checkpoint saved.", file=sys.stderr)
        sys.stderr.flush()
    tokenizer.save_pretrained(output_dir)


def run_chat(
    checkpoint_dir: str | Path,
    *,
    max_new_tokens: int | None = None,
    device: str | None = None,
) -> None:
    """
    Load the abliterated checkpoint and run an interactive chat using the Hugging Face
    tokenizer (correct tokenization, no GGUF pre-tokenizer mismatch). Use this when
    the GGUF/Ollama output is garbled; the checkpoint uses the same tokenizer as the
    original model. If max_new_tokens is None, uses the model config max_position_embeddings
    (capped at 8192), else 2048.
    """
    import torch
    from transformers import AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir() or not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Checkpoint dir not found or invalid (no config.json): {checkpoint_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    load_kw: dict = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto" if device is None else device,
        "low_cpu_mem_usage": True,
    }
    model = _load_model_with_gguf_version_workaround(str(checkpoint_dir), load_kw)
    if max_new_tokens is None:
        max_new_tokens = _model_max_position_embeddings(model)
        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = 2048
        else:
            max_new_tokens = min(max_new_tokens, 8192)
    use_chat_template = getattr(tokenizer, "chat_template", None) is not None

    conversation: list[dict[str, str]] = []
    print("Chat with abliterated model (HF tokenizer). Empty line or Ctrl+C to exit.", file=sys.stderr)
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        conversation.append({"role": "user", "content": line})
        if use_chat_template:
            try:
                encoded = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
                toks = {"input_ids": input_ids}
            except Exception:
                use_chat_template = False
                toks = tokenizer(line, return_tensors="pt", add_special_tokens=True)
        else:
            toks = tokenizer(line, return_tensors="pt", add_special_tokens=True)
        if "attention_mask" not in toks:
            toks["attention_mask"] = torch.ones_like(toks["input_ids"], dtype=torch.long)
        toks = {k: v.to(model.device) for k, v in toks.items()}
        with torch.inference_mode():
            out = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][toks["input_ids"].shape[1] :]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)
        reply = _strip_chat_reply(reply, last_user_content=line)
        print(reply)
        conversation.append({"role": "assistant", "content": reply})
    print("Bye.", file=sys.stderr)
