"""Refusal-direction computation for abliterated models (optional deps: torch, transformers)."""

from __future__ import annotations

from pathlib import Path


def get_layers(model):  # noqa: ANN001
    """Return the list of transformer layers (model.model.layers or model.transformer.h)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not find layers: try model.model.layers or model.transformer.h. "
        "See docs/ABLITERATE.md for supported architectures."
    )


def compute_refusal_dir(
    model_id: str,
    harmful_path: str | Path,
    harmless_path: str | Path,
    output_path: str | Path,
    *,
    num_instructions: int = 32,
    layer_frac: float = 0.6,
    pos: int = -1,
    device: str | None = None,
) -> None:
    """
    Compute refusal direction from harmful vs harmless instructions and save to output_path (.pt).
    Requires torch and transformers. Use: uv sync --extra abliterate.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    harmful_path = Path(harmful_path)
    harmless_path = Path(harmless_path)
    output_path = Path(output_path)
    if not harmful_path.is_file():
        raise FileNotFoundError(f"harmful file not found: {harmful_path}")
    if not harmless_path.is_file():
        raise FileNotFoundError(f"harmless file not found: {harmless_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if device is None else device,
    )
    layers = get_layers(model)
    layer_idx = int(len(layers) * layer_frac)

    with harmful_path.open() as f:
        harmful_lines = [line.strip() for line in f if line.strip()]
    with harmless_path.open() as f:
        harmless_lines = [line.strip() for line in f if line.strip()]

    import random
    n = min(num_instructions, len(harmful_lines), len(harmless_lines))
    harmful_instructions = random.sample(harmful_lines, n)
    harmless_instructions = random.sample(harmless_lines, n)

    def tokenize(instructions):
        out = []
        for insn in instructions:
            toks = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": insn}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            out.append(toks.to(model.device))
        return out

    harmful_toks = tokenize(harmful_instructions)
    harmless_toks = tokenize(harmless_instructions)

    # hidden_states from forward: (embed, layer0, layer1, ...); use layer_idx+1
    h_layer = layer_idx + 1
    harmful_hidden = []
    harmless_hidden = []
    with torch.inference_mode():
        for toks in harmful_toks:
            out = model(toks, output_hidden_states=True)
            hidden = out.hidden_states[h_layer][:, pos, :]
            harmful_hidden.append(hidden)
        for toks in harmless_toks:
            out = model(toks, output_hidden_states=True)
            hidden = out.hidden_states[h_layer][:, pos, :]
            harmless_hidden.append(hidden)

    harmful_mean = torch.cat(harmful_hidden, dim=0).mean(dim=0)
    harmless_mean = torch.cat(harmless_hidden, dim=0).mean(dim=0)
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    torch.save(refusal_dir, output_path)
