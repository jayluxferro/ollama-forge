"""TurboQuant packaging pipeline.

Packages a Hugging Face checkpoint into a lightweight `.tqf` directory that
stores runtime metadata plus tokenizer/config files for the working
TurboQuant-style inference path: original HF weights + TurboQuant KV cache.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# torch and turboquant imports are lazy — load_tqf must work without torch
# installed (MLX-only environments).  quantize_model imports them at call time.


@dataclass
class TurboQuantConfig:
    """Quantization configuration."""
    bits: int = 3                       # main channel bits (1-4)
    outlier_channels: int = 32          # number of outlier channels
    outlier_bits: int = 4               # bits for outlier channels
    use_qjl: bool = False               # QJL residual correction
    embed_bits: int = 4                 # bits for embeddings
    kv_bits: int = 3                    # bits for KV cache at inference
    rotation_seed: int = 42
    qjl_seed: int = 137
    # Layer-type overrides
    attn_bits: int | None = None        # None → use main bits
    ffn_bits: int | None = None         # None → use main bits

    def bits_for(self, layer_name: str) -> int:
        """Resolve bit-width for a given layer name."""
        name_lower = layer_name.lower()
        if any(k in name_lower for k in ("embed", "wte", "wpe")):
            return self.embed_bits
        if self.attn_bits and any(k in name_lower for k in (
            "q_proj", "k_proj", "v_proj", "o_proj", "self_attn",
            "query", "key", "value", "qkv",
        )):
            return self.attn_bits
        if self.ffn_bits and any(k in name_lower for k in (
            "gate_proj", "up_proj", "down_proj", "mlp", "ffn",
            "fc1", "fc2", "dense_h_to_4h", "dense_4h_to_h",
        )):
            return self.ffn_bits
        return self.bits


@dataclass
class TurboQuantModel:
    """In-memory representation of a quantized model ready for save/load."""
    config: dict[str, Any]              # HF model config
    quant_config: TurboQuantConfig
    layers: dict[str, Any]              # name → QuantizedTensor
    unquantized: dict[str, Any]         # name → torch.Tensor (fp16)
    source_model: str | None = None
    resolved_model_path: str | None = None
    tokenizer_path: str | None = None
    stats: Any = None                   # CompressionStats (lazy)

    def __post_init__(self):
        if self.stats is None:
            from ollama_forge.turboquant import CompressionStats
            self.stats = CompressionStats()


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------

_SKIP_QUANT_PATTERNS = (
    "layernorm", "layer_norm", "rmsnorm", "rms_norm",
    "norm", "bias", "rotary",
)

_MIN_QUANT_ELEMENTS = 1024  # don't quantize tiny tensors


def _should_quantize(name: str, tensor: torch.Tensor) -> bool:
    """Decide if a tensor should be TurboQuant-compressed."""
    name_lower = name.lower()
    if any(p in name_lower for p in _SKIP_QUANT_PATTERNS):
        return False
    if tensor.dim() != 2:
        return False
    return tensor.numel() >= _MIN_QUANT_ELEMENTS


def quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    config: TurboQuantConfig | None = None,
    *,
    device: str = "auto",
    progress_callback: Any = None,
    source_model: str | None = None,
) -> TurboQuantModel:
    """Package a Hugging Face model for TurboQuant KV-cache inference.

    Args:
        model_path: path to HF checkpoint directory (with safetensors).
        output_path: path for the .tqf output directory.
        config: quantization config (defaults to 3-bit main).
        device: "auto", "cuda", "mps", or "cpu".
        progress_callback: optional callable(step, total, name).
        source_model: original user-provided model identifier (repo ID or path).

    Returns:
        TurboQuantModel with source metadata and model-size statistics.
    """
    from safetensors import safe_open

    from ollama_forge.turboquant import CompressionStats

    if config is None:
        config = TurboQuantConfig()

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model config
    hf_config = _load_hf_config(model_path)

    # Discover safetensors files to compute size statistics. TurboQuant follows
    # the reference implementation: weights stay in the original HF checkpoint
    # and only the KV cache is compressed at inference time.
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    tensor_index: dict[str, Path] = {}
    for sf in st_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                tensor_index[name] = sf

    stats = CompressionStats()
    total = len(tensor_index)
    for step, (name, sf_path) in enumerate(tensor_index.items(), start=1):
        if progress_callback:
            progress_callback(step, total, name)

        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)

        orig_bytes = tensor.numel() * tensor.element_size()
        stats.add_layer(
            name,
            tuple(tensor.shape),
            tensor.element_size() * 8,
            orig_bytes,
            orig_bytes,
        )
        del tensor
        gc.collect()

    stats.finalize()

    model = TurboQuantModel(
        config=hf_config,
        quant_config=config,
        layers={},
        unquantized={},
        source_model=source_model or str(model_path),
        resolved_model_path=str(model_path),
        stats=stats,
    )

    # Save to disk
    _save_tqf(model, output_path)

    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        (output_path / "config.json").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    return model


# ---------------------------------------------------------------------------
# Save / Load .tqf format
# ---------------------------------------------------------------------------

def _save_tqf(model: TurboQuantModel, output_dir: Path):
    """Persist TurboQuantModel to a .tqf directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- metadata.json ---
    meta = {
        "format": "turboquant",
        "version": 2,
        "implementation": "hf-kv-cache",
        "model_config": model.config,
        "source_model": model.source_model,
        "resolved_model_path": model.resolved_model_path,
        "quant_config": asdict(model.quant_config),
        "quantized_layers": {},
        "unquantized_layers": [],
        "stats": {
            "original_params": model.stats.original_params,
            "original_bytes": model.stats.original_bytes,
            "compressed_bytes": model.stats.compressed_bytes,
            "effective_bits_avg": round(model.stats.effective_bits_avg, 3),
            "compression_ratio": round(model.stats.compression_ratio, 2),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def load_tqf(tqf_dir: str | Path) -> TurboQuantModel:
    """Load a .tqf model from disk."""
    from safetensors import safe_open

    from ollama_forge.turboquant import CompressionStats, QuantizedTensor

    tqf_dir = Path(tqf_dir)
    meta = json.loads((tqf_dir / "metadata.json").read_text(encoding="utf-8"))

    quant_cfg = TurboQuantConfig(**meta["quant_config"])

    if meta.get("implementation") == "hf-kv-cache" or meta.get("version", 1) >= 2:
        stats = CompressionStats(**{
            k: v for k, v in meta.get("stats", {}).items()
            if k in ("original_params", "original_bytes", "compressed_bytes",
                     "effective_bits_avg", "compression_ratio")
        })
        return TurboQuantModel(
            config=meta["model_config"],
            quant_config=quant_cfg,
            layers={},
            unquantized={},
            source_model=meta.get("source_model"),
            resolved_model_path=meta.get("resolved_model_path"),
            stats=stats,
        )

    # Load unquantized tensors
    unquantized: dict[str, Any] = {}
    unq_path = tqf_dir / "unquantized.safetensors"
    if unq_path.exists():
        with safe_open(str(unq_path), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                unquantized[name] = f.get_tensor(name)

    # Load quantized tensors
    qt_raw: dict[str, Any] = {}
    for sf in sorted(tqf_dir.glob("quantized_*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                qt_raw[name] = f.get_tensor(name)

    # Reconstruct QuantizedTensor objects
    layers: dict[str, Any] = {}
    for layer_name, layer_meta in meta["quantized_layers"].items():
        safe_name = layer_name.replace(".", "__")
        packed = qt_raw[f"{safe_name}__packed"]
        norms = qt_raw[f"{safe_name}__norms"]
        codebook = qt_raw[f"{safe_name}__codebook"]

        outlier_idx = qt_raw.get(f"{safe_name}__outlier_idx")
        outlier_packed = qt_raw.get(f"{safe_name}__outlier_packed")
        outlier_cb = qt_raw.get(f"{safe_name}__outlier_codebook")
        qjl_signs = qt_raw.get(f"{safe_name}__qjl_signs")
        qjl_gammas = qt_raw.get(f"{safe_name}__qjl_gammas")

        layers[layer_name] = QuantizedTensor(
            shape=tuple(layer_meta["shape"]),
            bits=layer_meta["bits"],
            packed_indices=packed,
            norms=norms,
            rotation_seed=layer_meta["rotation_seed"],
            codebook=codebook,
            use_qjl=layer_meta["use_qjl"],
            qjl_packed_signs=qjl_signs,
            qjl_gammas=qjl_gammas,
            qjl_seed=layer_meta["qjl_seed"],
            outlier_indices=outlier_idx,
            outlier_packed=outlier_packed,
            outlier_bits=layer_meta["outlier_bits"],
            outlier_codebook=outlier_cb,
            dtype=_parse_dtype(s=layer_meta.get("dtype", "torch.float16")),
        )

    stats = CompressionStats(**{
        k: v for k, v in meta.get("stats", {}).items()
        if k in ("original_params", "original_bytes", "compressed_bytes",
                  "effective_bits_avg", "compression_ratio")
    })

    return TurboQuantModel(
        config=meta["model_config"],
        quant_config=quant_cfg,
        layers=layers,
        unquantized=unquantized,
        source_model=meta.get("source_model"),
        resolved_model_path=meta.get("resolved_model_path"),
        stats=stats,
    )


def _load_hf_config(model_path: Path) -> dict[str, Any]:
    """Load config.json from a HF checkpoint directory."""
    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"config.json not found in {model_path}")


def _parse_dtype(s: str) -> Any:
    """Parse a dtype string like 'torch.float16'. Returns torch.dtype."""
    import torch
    mapping = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
    }
    return mapping.get(s, torch.float16)


# ---------------------------------------------------------------------------
# Copy tokenizer files for serving
# ---------------------------------------------------------------------------

def copy_tokenizer(src_dir: str | Path, dst_dir: str | Path):
    """Copy tokenizer files from HF checkpoint to .tqf output."""
    import shutil

    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json",
    ):
        for f in src.glob(pattern):
            shutil.copy2(f, dst / f.name)
