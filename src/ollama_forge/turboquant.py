"""TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

Implements the two-stage quantization algorithm from Zandieh et al. (2025):
  Stage 1 (PolarQuant/MSE): Random rotation → Beta-distributed coordinates →
          optimal Lloyd-Max scalar quantizer per coordinate.
  Stage 2 (QJL residual): 1-bit Quantized Johnson-Lindenstrauss on the residual
          for unbiased inner-product estimation.

Reference: arXiv:2504.19874
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Precomputed Lloyd-Max codebooks for the Beta distribution f_X(x) that arises
# after random rotation on the unit hypersphere.  In high dimensions the Beta
# distribution converges to N(0, 1/d), so the optimal centroids scale as
# c_i / sqrt(d).  We store the *unit-variance* centroids (i.e. for N(0,1))
# and rescale at runtime by 1/sqrt(d).
#
# For b bits we have 2^b centroids.  Computed via continuous 1-D k-means
# (Lloyd-Max) on N(0,1) to 12-digit precision.
# ---------------------------------------------------------------------------

# b=1: 2 centroids — ±sqrt(2/pi) ≈ ±0.7979
_CODEBOOK_1 = [-0.7978845608, 0.7978845608]

# b=2: 4 centroids — optimal for N(0,1)
_CODEBOOK_2 = [-1.510_232_6, -0.452_842_7, 0.452_842_7, 1.510_232_6]

# b=3: 8 centroids
_CODEBOOK_3 = [
    -2.152_174_6, -1.344_171_4, -0.756_421_2, -0.245_340_8,
     0.245_340_8,  0.756_421_2,  1.344_171_4,  2.152_174_6,
]

# b=4: 16 centroids
_CODEBOOK_4 = [
    -2.733_460_0, -2.069_016_0, -1.618_192_0, -1.256_346_0,
    -0.942_082_0, -0.656_532_0, -0.388_378_0, -0.127_961_0,
     0.127_961_0,  0.388_378_0,  0.656_532_0,  0.942_082_0,
     1.256_346_0,  1.618_192_0,  2.069_016_0,  2.733_460_0,
]

_CODEBOOKS = {
    1: _CODEBOOK_1,
    2: _CODEBOOK_2,
    3: _CODEBOOK_3,
    4: _CODEBOOK_4,
}


def _get_codebook(bits: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return Lloyd-Max centroids scaled for dimension *dim*."""
    if bits not in _CODEBOOKS:
        raise ValueError(f"TurboQuant supports 1-4 bits, got {bits}")
    raw = torch.tensor(_CODEBOOKS[bits], dtype=torch.float32, device=device)
    return raw / math.sqrt(dim)


# ---------------------------------------------------------------------------
# Random rotation matrix  Π ∈ R^{d×d}  via QR decomposition of Gaussian.
# For large d, generating and storing a full d×d matrix is expensive.
# We use a structured random rotation (random sign flip + random permutation
# + scaled Hadamard — "fast JL") when d > 2048 to save memory/time.
# For smaller d, full QR rotation is fine.
# ---------------------------------------------------------------------------

# Threshold: use structured (Hadamard) rotation for d above this value.
_STRUCTURED_ROTATION_THRESHOLD = 1024


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """In-place Walsh-Hadamard transform via butterfly operations.

    O(d log d) instead of O(d²) for a full rotation.
    Requires last dimension to be a power of 2.
    The output is scaled by 1/sqrt(d) to make it orthonormal.

    Args:
        x: (..., d) tensor where d is a power of 2.

    Returns:
        Transformed tensor of same shape.
    """
    d = x.shape[-1]
    out = x.clone()
    h = 1
    while h < d:
        # Butterfly operation: pair elements at stride h
        # Split into even/odd halves at current stride
        out_view = out.view(*out.shape[:-1], d // (2 * h), 2, h)
        a = out_view[..., 0, :].clone()
        b = out_view[..., 1, :].clone()
        out_view[..., 0, :] = a + b
        out_view[..., 1, :] = a - b
        h *= 2
    return out / math.sqrt(d)


def fast_hadamard_inverse(x: torch.Tensor) -> torch.Tensor:
    """Inverse Walsh-Hadamard transform.

    Since the normalized WHT is its own inverse (H^{-1} = H for orthonormal H),
    this is the same operation.
    """
    return fast_hadamard_transform(x)


def generate_rotation_matrix(d: int, *, device: torch.device, seed: int | None = None) -> torch.Tensor:
    """Generate a random orthogonal matrix Π ∈ R^{d×d}.

    For d <= _STRUCTURED_ROTATION_THRESHOLD: uses QR decomposition (O(d²) memory).
    For d > threshold: returns None — callers should use structured rotation instead.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    # Generate on CPU, move to device
    G = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Ensure deterministic sign convention: make diagonal of R positive
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs.unsqueeze(0)
    return Q.to(device)


def _generate_structured_rotation_params(d: int, *, device: torch.device,
                                          seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate parameters for a structured random rotation.

    Structured rotation = random signs ⊙ random permutation ⊙ Hadamard.
    O(d) storage instead of O(d²), O(d log d) application instead of O(d²).

    Returns:
        (random_signs, random_perm): sign vector {-1,+1}^d and permutation indices.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    random_signs = torch.where(
        torch.rand(d, generator=gen) > 0.5,
        torch.ones(d), -torch.ones(d),
    ).to(device)
    random_perm = torch.randperm(d, generator=gen).to(device)
    return random_signs, random_perm


def _apply_structured_rotation(x: torch.Tensor, signs: torch.Tensor,
                                perm: torch.Tensor, d_orig: int) -> torch.Tensor:
    """Apply structured rotation: sign flip → permute → Hadamard.

    If d_orig is not a power of 2, x is assumed to be zero-padded on the last dim.
    """
    # Step 1: multiply by random signs
    y = x * signs
    # Step 2: random permutation
    y = y[..., perm]
    # Step 3: Walsh-Hadamard transform
    y = fast_hadamard_transform(y)
    return y


def _apply_structured_rotation_inverse(y: torch.Tensor, signs: torch.Tensor,
                                        perm: torch.Tensor, d_orig: int) -> torch.Tensor:
    """Inverse structured rotation: inverse Hadamard → inverse permute → inverse signs."""
    # Step 1: inverse Hadamard
    x = fast_hadamard_inverse(y)
    # Step 2: inverse permutation
    inv_perm = torch.argsort(perm)
    x = x[..., inv_perm]
    # Step 3: multiply by signs (signs are self-inverse: s * s = 1)
    x = x * signs
    return x


def use_structured_rotation(d: int) -> bool:
    """Decide whether to use structured (Hadamard) rotation for dimension d."""
    return d > _STRUCTURED_ROTATION_THRESHOLD


def generate_rotation_seed(d: int, seed: int) -> dict[str, Any]:
    """Store only the seed + dimension — reconstruct the rotation lazily."""
    return {"d": d, "seed": seed}


def _apply_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate: y = x @ Π^T  (each row of x is a vector)."""
    return x @ rotation.T


def _apply_inverse_rotation(y: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Inverse rotate: x = y @ Π  (since Π is orthogonal, Π^{-1} = Π^T)."""
    return y @ rotation


# ---------------------------------------------------------------------------
# Scalar quantization: nearest-centroid assignment per coordinate.
# ---------------------------------------------------------------------------

def _scalar_quantize(y: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Quantize each element of y to the nearest centroid.

    Args:
        y: (...,) tensor of float values.
        centroids: (2^b,) sorted centroid values.

    Returns:
        Integer tensor of same shape as y, values in [0, 2^b - 1].
    """
    # Boundaries are midpoints between consecutive centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    # searchsorted gives the bucket index
    indices = torch.searchsorted(boundaries, y.contiguous())
    return indices


def _scalar_dequantize(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Look up centroid values from indices."""
    return centroids[indices.long()]


# ---------------------------------------------------------------------------
# QJL: Quantized Johnson-Lindenstrauss 1-bit projection.
#
#   Q_qjl(x) = sign(S · x)     where S ~ N(0,1)^{m×d}
#   Q_qjl^{-1}(z) = sqrt(π/2) / m · S^T · z
#
# For residual correction in TurboQuant_prod, we use m = d (same dimension)
# and store only the sign bits — 1 bit per entry.
# ---------------------------------------------------------------------------

def generate_qjl_matrix(d: int, m: int | None = None, *, device: torch.device,
                        seed: int | None = None) -> torch.Tensor:
    """Generate the random projection matrix S ∈ R^{m×d} for QJL.

    Default m = d for full-dimension projection.
    """
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen, dtype=torch.float32)
    return S.to(device)


def qjl_quantize(residual: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """1-bit QJL: sign(S · r).  Returns {-1, +1}^m packed as int8."""
    proj = residual @ S.T  # (..., m)
    return proj.sign().to(torch.int8)


def qjl_dequantize(qjl_bits: torch.Tensor, S: torch.Tensor, gamma: float) -> torch.Tensor:
    """Dequantize QJL: sqrt(π/2)/m · γ · S^T · z.

    Args:
        qjl_bits: (..., m) sign bits in {-1, +1}.
        S: (m, d) projection matrix.
        gamma: ||r|| — norm of the residual vector.
    """
    m = S.shape[0]
    scale = math.sqrt(math.pi / 2.0) / m * gamma
    return scale * (qjl_bits.float() @ S)


# ---------------------------------------------------------------------------
# Bit packing utilities.
# ---------------------------------------------------------------------------

def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack b-bit indices into uint8 bytes.

    Args:
        indices: integer tensor with values in [0, 2^b - 1].
        bits: bits per value (1-4).

    Returns:
        Packed uint8 tensor.
    """
    flat = indices.flatten().to(torch.uint8)
    vals_per_byte = 8 // bits
    # Pad to multiple of vals_per_byte
    pad = (-len(flat)) % vals_per_byte
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
    flat = flat.view(-1, vals_per_byte)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
    for i in range(vals_per_byte):
        packed |= flat[:, i] << (i * bits)
    return packed


def unpack_indices(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Unpack b-bit indices from uint8 bytes.

    Args:
        packed: uint8 tensor from pack_indices.
        bits: bits per value (1-4).
        numel: original number of elements.

    Returns:
        Integer tensor of shape (numel,).
    """
    vals_per_byte = 8 // bits
    mask = (1 << bits) - 1
    unpacked = []
    for i in range(vals_per_byte):
        unpacked.append((packed >> (i * bits)) & mask)
    result = torch.stack(unpacked, dim=1).flatten()[:numel]
    return result.to(torch.int64)


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack {-1, +1} sign tensor into bits (1 bit per value, stored as uint8)."""
    bits = (signs.flatten() > 0).to(torch.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8, device=bits.device)])
    bits = bits.view(-1, 8)
    packed = torch.zeros(bits.shape[0], dtype=torch.uint8, device=bits.device)
    for i in range(8):
        packed |= bits[:, i] << i
    return packed


def unpack_signs(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack sign bits to {-1, +1} int8 tensor."""
    unpacked = []
    for i in range(8):
        unpacked.append((packed >> i) & 1)
    flat = torch.stack(unpacked, dim=1).flatten()[:numel]
    return (flat.to(torch.int8) * 2 - 1)  # 0→-1, 1→+1


# ---------------------------------------------------------------------------
# Outlier channel detection.
#
# Per TurboQuant paper Section 4.3: split channels into outlier (high
# magnitude) and non-outlier groups.  Outlier channels get higher bit-width.
# Example: 32 outlier channels at 3 bits, remaining at 2 bits → 2.5 effective.
# ---------------------------------------------------------------------------

def detect_outlier_channels(weight: torch.Tensor, n_outliers: int = 32) -> torch.Tensor:
    """Identify outlier channels by column-wise L2 norm.

    Args:
        weight: (out_features, in_features) weight matrix.
        n_outliers: number of outlier channels to select.

    Returns:
        Indices of the top-n_outliers columns by norm.
    """
    col_norms = weight.float().norm(dim=0)
    n_outliers = min(n_outliers, weight.shape[1])
    _, outlier_idx = col_norms.topk(n_outliers)
    return outlier_idx.sort().values


# ---------------------------------------------------------------------------
# High-level quantize / dequantize for a single weight tensor.
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """A TurboQuant-compressed weight tensor."""
    shape: tuple[int, ...]
    bits: int
    packed_indices: torch.Tensor       # packed b-bit centroid indices
    norms: torch.Tensor                # per-row L2 norms (for unit-norm rescaling)
    rotation_seed: int                 # seed to regenerate rotation matrix
    codebook: torch.Tensor             # (2^b,) centroid values (scaled)
    # QJL fields (only for inner-product variant)
    use_qjl: bool = False
    qjl_packed_signs: torch.Tensor | None = None
    qjl_gammas: torch.Tensor | None = None  # per-row residual norms
    qjl_seed: int = 0
    # Outlier fields
    outlier_indices: torch.Tensor | None = None
    outlier_packed: torch.Tensor | None = None
    outlier_bits: int = 0
    outlier_codebook: torch.Tensor | None = None
    # Metadata
    dtype: torch.dtype = torch.float16


def quantize_tensor(
    weight: torch.Tensor,
    *,
    bits: int = 3,
    use_qjl: bool = False,
    rotation_seed: int = 42,
    qjl_seed: int = 137,
    outlier_channels: int = 0,
    outlier_bits: int = 0,
) -> QuantizedTensor:
    """Quantize a 2-D weight matrix using TurboQuant.

    Args:
        weight: (out_features, in_features) weight tensor.
        bits: target bits per coordinate for main channels.
        use_qjl: if True, apply QJL residual correction (Algorithm 2).
        rotation_seed: seed for the rotation matrix.
        qjl_seed: seed for the QJL projection matrix.
        outlier_channels: number of outlier channels to quantize at higher bits.
        outlier_bits: bits for outlier channels (must be > bits).

    Returns:
        QuantizedTensor with all data needed for dequantization.
    """
    orig_dtype = weight.dtype
    W = weight.float()
    out_f, in_f = W.shape
    device = W.device

    # --- Outlier channel split ---
    outlier_idx = None
    outlier_packed = None
    outlier_cb = None
    if outlier_channels > 0 and outlier_bits > bits:
        outlier_idx = detect_outlier_channels(W, outlier_channels)
        normal_mask = torch.ones(in_f, dtype=torch.bool, device=device)
        normal_mask[outlier_idx] = False
        W_outlier = W[:, outlier_idx]          # (out_f, n_outliers)
        W_normal = W[:, normal_mask]           # (out_f, in_f - n_outliers)
    else:
        W_normal = W
        W_outlier = None

    # --- Quantize main channels ---
    packed, norms, codebook = _quantize_block(W_normal, bits, rotation_seed, device)

    # --- Quantize outlier channels (same algorithm, higher bits) ---
    if W_outlier is not None:
        outlier_packed, _, outlier_cb = _quantize_block(
            W_outlier, outlier_bits, rotation_seed + 1, device
        )

    # --- QJL residual correction ---
    qjl_packed_signs = None
    qjl_gammas = None
    if use_qjl and bits > 1:
        # Reconstruct MSE-quantized version to compute residual
        W_recon = _dequantize_block(packed, norms, codebook, bits,
                                     W_normal.shape, rotation_seed, device)
        residual = W_normal - W_recon  # (out_f, d_normal)
        d_normal = W_normal.shape[1]
        S = generate_qjl_matrix(d_normal, device=device, seed=qjl_seed)
        # Compute per-row residual norms and QJL signs
        qjl_gammas = residual.norm(dim=1)  # (out_f,)
        # Normalize residual for QJL
        r_normed = residual / (qjl_gammas.unsqueeze(1) + 1e-10)
        signs = qjl_quantize(r_normed, S)  # (out_f, d_normal) int8
        qjl_packed_signs = pack_signs(signs)
        del S, residual, W_recon, r_normed, signs

    return QuantizedTensor(
        shape=weight.shape,
        bits=bits,
        packed_indices=packed,
        norms=norms,
        rotation_seed=rotation_seed,
        codebook=codebook,
        use_qjl=use_qjl,
        qjl_packed_signs=qjl_packed_signs,
        qjl_gammas=qjl_gammas,
        qjl_seed=qjl_seed,
        outlier_indices=outlier_idx,
        outlier_packed=outlier_packed,
        outlier_bits=outlier_bits,
        outlier_codebook=outlier_cb,
        dtype=orig_dtype,
    )


def dequantize_tensor(qt: QuantizedTensor, *, device: torch.device | None = None) -> torch.Tensor:
    """Reconstruct the full weight matrix from a QuantizedTensor."""
    if device is None:
        device = qt.packed_indices.device
    out_f, in_f = qt.shape

    if qt.outlier_indices is not None:
        n_outliers = len(qt.outlier_indices)
        d_normal = in_f - n_outliers
    else:
        d_normal = in_f
        n_outliers = 0

    # --- Dequantize main channels ---
    W_normal = _dequantize_block(
        qt.packed_indices, qt.norms, qt.codebook, qt.bits,
        (out_f, d_normal), qt.rotation_seed, device,
    )

    # --- QJL correction ---
    if qt.use_qjl and qt.qjl_packed_signs is not None:
        S = generate_qjl_matrix(d_normal, device=device, seed=qt.qjl_seed)
        signs = unpack_signs(qt.qjl_packed_signs.to(device), out_f * d_normal)
        signs = signs.view(out_f, d_normal)
        for row_idx in range(out_f):
            gamma = qt.qjl_gammas[row_idx].item()
            if gamma > 1e-10:
                correction = qjl_dequantize(signs[row_idx:row_idx+1], S, gamma)
                W_normal[row_idx] += correction.squeeze(0)
        del S, signs

    # --- Reassemble with outlier channels ---
    if qt.outlier_indices is not None and qt.outlier_packed is not None:
        W_outlier = _dequantize_block(
            qt.outlier_packed, qt.norms, qt.outlier_codebook, qt.outlier_bits,
            (out_f, n_outliers), qt.rotation_seed + 1, device,
        )
        W = torch.zeros(out_f, in_f, dtype=torch.float32, device=device)
        normal_mask = torch.ones(in_f, dtype=torch.bool, device=device)
        normal_mask[qt.outlier_indices.to(device)] = False
        W[:, normal_mask] = W_normal
        W[:, qt.outlier_indices.to(device)] = W_outlier
    else:
        W = W_normal

    return W.to(qt.dtype)


# ---------------------------------------------------------------------------
# Internal helpers for block quantize/dequantize.
# ---------------------------------------------------------------------------

def _quantize_block(
    W: torch.Tensor, bits: int, rotation_seed: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a (rows, cols) block using TurboQuant_mse.

    Uses structured rotation (Hadamard) for large dims, full QR for small dims.

    Returns:
        packed_indices: packed uint8 tensor.
        norms: (rows,) per-row L2 norms.
        codebook: (2^b,) centroids.
    """
    rows, cols = W.shape
    # Step 1: compute per-row norms and normalize
    norms = W.norm(dim=1)  # (rows,)
    W_normed = W / (norms.unsqueeze(1) + 1e-10)  # unit rows

    # Step 2: random rotation
    if use_structured_rotation(cols):
        d_padded = _next_power_of_2(cols)
        signs, perm = _generate_structured_rotation_params(d_padded, device=device, seed=rotation_seed)
        if d_padded > cols:
            W_padded = torch.zeros(rows, d_padded, dtype=W_normed.dtype, device=device)
            W_padded[:, :cols] = W_normed
        else:
            W_padded = W_normed
        Y = _apply_structured_rotation(W_padded, signs, perm, cols)
        # Use padded dimension for codebook scaling
        codebook = _get_codebook(bits, d_padded, device)
        del signs, perm
    else:
        rotation = generate_rotation_matrix(cols, device=device, seed=rotation_seed)
        Y = _apply_rotation(W_normed, rotation)
        codebook = _get_codebook(bits, cols, device)
        del rotation

    # Step 3: scalar quantize each coordinate
    indices = _scalar_quantize(Y, codebook)  # (rows, d_padded or cols)

    # Step 4: pack indices
    packed = pack_indices(indices, bits)

    return packed, norms, codebook


def _dequantize_block(
    packed: torch.Tensor, norms: torch.Tensor, codebook: torch.Tensor,
    bits: int, shape: tuple[int, int], rotation_seed: int, device: torch.device,
) -> torch.Tensor:
    """Dequantize a packed block back to a float matrix."""
    rows, cols = shape

    if use_structured_rotation(cols):
        d_padded = _next_power_of_2(cols)
        numel = rows * d_padded
        indices = unpack_indices(packed.to(device), bits, numel).view(rows, d_padded)
        Y_hat = _scalar_dequantize(indices, codebook.to(device))
        signs, perm = _generate_structured_rotation_params(d_padded, device=device, seed=rotation_seed)
        W_padded = _apply_structured_rotation_inverse(Y_hat, signs, perm, cols)
        W_normed_hat = W_padded[:, :cols]  # strip padding
        del signs, perm
    else:
        numel = rows * cols
        indices = unpack_indices(packed.to(device), bits, numel).view(rows, cols)
        Y_hat = _scalar_dequantize(indices, codebook.to(device))
        rotation = generate_rotation_matrix(cols, device=device, seed=rotation_seed)
        W_normed_hat = _apply_inverse_rotation(Y_hat, rotation)
        del rotation

    # Rescale by norms
    W_hat = W_normed_hat * norms.to(device).unsqueeze(1)
    return W_hat


# ---------------------------------------------------------------------------
# Effective bits calculation.
# ---------------------------------------------------------------------------

def effective_bits(
    in_features: int,
    bits: int,
    outlier_channels: int = 0,
    outlier_bits: int = 0,
    use_qjl: bool = False,
) -> float:
    """Compute effective bits per parameter.

    Accounts for outlier channels, QJL overhead, and norm storage.
    """
    if outlier_channels > 0 and outlier_bits > 0:
        normal_ch = in_features - outlier_channels
        total_bits = normal_ch * bits + outlier_channels * outlier_bits
        eff = total_bits / in_features
    else:
        eff = float(bits)
    if use_qjl:
        eff += 1.0  # 1 bit per coordinate for QJL signs
    # Norms: 16 bits per row, amortised over in_features
    eff += 16.0 / in_features
    return eff


# ---------------------------------------------------------------------------
# Compression stats.
# ---------------------------------------------------------------------------

@dataclass
class CompressionStats:
    """Summary statistics for a TurboQuant-compressed model."""
    original_params: int = 0
    original_bytes: int = 0
    compressed_bytes: int = 0
    effective_bits_avg: float = 0.0
    compression_ratio: float = 0.0
    layers: list[dict[str, Any]] = field(default_factory=list)

    def add_layer(self, name: str, shape: tuple, bits: float, orig_bytes: int, comp_bytes: int):
        self.layers.append({
            "name": name, "shape": shape, "bits": bits,
            "orig_bytes": orig_bytes, "comp_bytes": comp_bytes,
        })
        self.original_params += math.prod(shape)
        self.original_bytes += orig_bytes
        self.compressed_bytes += comp_bytes

    def finalize(self):
        if self.original_bytes > 0:
            self.compression_ratio = self.original_bytes / max(self.compressed_bytes, 1)
        if self.layers:
            total_params = sum(math.prod(entry["shape"]) for entry in self.layers)
            total_weighted = sum(math.prod(entry["shape"]) * entry["bits"] for entry in self.layers)
            self.effective_bits_avg = total_weighted / max(total_params, 1)


# =========================================================================
# Class-based API — mirrors the reference TurboQuant+ architecture.
#
# PolarQuant (Algorithm 1): random rotation + optimal scalar quantization.
# QJL: 1-bit sign quantization via random projection for residual.
# TurboQuant (Algorithm 2): PolarQuant(b-1) + QJL(1) for inner product.
# TurboQuantMSE: PolarQuant only, optimises MSE (for V cache).
# KVCacheCompressor: asymmetric K/V compression.
# =========================================================================


@dataclass
class CompressedVector:
    """Container for a TurboQuant-compressed vector (or batch)."""
    mse_indices: torch.Tensor     # (d,) or (batch, d) — PolarQuant centroid indices
    vector_norms: torch.Tensor    # scalar or (batch,) — original ||x||_2
    qjl_signs: torch.Tensor       # (d,) or (batch, d) — QJL sign bits {-1,+1}
    residual_norms: torch.Tensor  # scalar or (batch,) — ||residual||_2
    bit_width: int                # total bits per coordinate


class PolarQuant:
    """MSE-optimised vector quantiser via random rotation + scalar quantisation.

    Handles arbitrary-norm vectors by extracting norms before quantisation
    and rescaling after dequantisation.

    Usage::

        pq = PolarQuant(d=128, bit_width=2, seed=42, device=device)
        indices, norms = pq.quantize(x)       # x: (d,) or (batch, d)
        x_hat = pq.dequantize(indices, norms)
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 device: torch.device | None = None):
        if device is None:
            device = torch.device("cpu")
        self.d = d
        self.bit_width = bit_width
        self.device = device
        self.n_centroids = 1 << bit_width
        self.rotation = generate_rotation_matrix(d, device=device, seed=seed)
        self.centroids = _get_codebook(bit_width, d, device)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise vector(s).

        Args:
            x: shape ``(d,)`` or ``(batch, d)``.

        Returns:
            ``(indices, norms)`` — integer indices and L2 norms.
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        x = x.float()

        norms = x.norm(dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_normed = x / safe_norms.unsqueeze(1)

        y = x_normed @ self.rotation.T
        indices = _scalar_quantize(y, self.centroids)

        if single:
            return indices.squeeze(0), norms.squeeze(0)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Reconstruct vector(s) from indices and norms."""
        single = indices.dim() == 1
        if single:
            indices = indices.unsqueeze(0)
            norms = norms.unsqueeze(0)

        y_hat = _scalar_dequantize(indices, self.centroids)
        x_hat_unit = y_hat @ self.rotation
        x_hat = x_hat_unit * norms.unsqueeze(1)

        return x_hat.squeeze(0) if single else x_hat

    def quantize_and_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise and return ``(indices, norms, residual)``."""
        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        residual = x.float() - x_hat.float()
        return indices, norms, residual


class QJLQuantizer:
    """Quantised Johnson-Lindenstrauss 1-bit quantiser.

    Usage::

        qjl = QJLQuantizer(d=128, seed=42, device=device)
        signs, norms = qjl.quantize(residual)
        r_hat = qjl.dequantize(signs, norms)
    """

    def __init__(self, d: int, seed: int = 123, device: torch.device | None = None):
        if device is None:
            device = torch.device("cpu")
        self.d = d
        self.device = device
        self.S = generate_qjl_matrix(d, device=device, seed=seed)

    def quantize(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise residual to sign bits.

        Returns:
            ``(signs, norms)`` — signs in {-1,+1} and L2 norms.
        """
        single = r.dim() == 1
        if single:
            r = r.unsqueeze(0)
        r = r.float()

        norms = r.norm(dim=1)
        signs = qjl_quantize(r, self.S)
        signs[signs == 0] = 1

        if single:
            return signs.squeeze(0), norms.squeeze(0)
        return signs, norms

    def dequantize(self, signs: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Reconstruct approximate residual from signs and norms."""
        single = signs.dim() == 1
        if single:
            signs = signs.unsqueeze(0)
            norms = norms.unsqueeze(0)

        m = self.S.shape[0]
        scale = math.sqrt(math.pi / 2.0) / m
        reconstructed = signs.float() @ self.S
        reconstructed *= (scale * norms).unsqueeze(1)

        return reconstructed.squeeze(0) if single else reconstructed


class TurboQuant:
    """Full TurboQuant quantiser: PolarQuant(b-1 bits) + QJL(1 bit).

    Optimises inner-product preservation — use for K cache.

    Usage::

        tq = TurboQuant(d=128, bit_width=3, seed=42, device=device)
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 device: torch.device | None = None):
        if bit_width < 2:
            raise ValueError("TurboQuant requires bit_width >= 2 (1 PolarQuant + 1 QJL)")
        self.d = d
        self.bit_width = bit_width
        self.polar_quant = PolarQuant(d, bit_width=bit_width - 1, seed=seed, device=device)
        self.qjl = QJLQuantizer(d, seed=seed + 1000, device=device)

    def quantize(self, x: torch.Tensor) -> CompressedVector:
        """Quantise a vector or batch."""
        mse_indices, vector_norms, residual = self.polar_quant.quantize_and_residual(x)
        qjl_signs, residual_norms = self.qjl.quantize(residual)
        return CompressedVector(
            mse_indices=mse_indices,
            vector_norms=vector_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bit_width=self.bit_width,
        )

    def dequantize(self, compressed: CompressedVector) -> torch.Tensor:
        """Reconstruct approximate vector."""
        x_mse = self.polar_quant.dequantize(compressed.mse_indices, compressed.vector_norms)
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)
        return x_mse + x_qjl

    def compression_ratio(self, original_bits: int = 16) -> float:
        """Compression ratio vs original precision."""
        original = self.d * original_bits
        compressed = self.d * self.bit_width + 32  # +32 for residual norm
        return original / compressed


class TurboQuantMSE:
    """MSE-only TurboQuant (Algorithm 1) — no QJL stage.

    Optimises MSE preservation — use for V cache.
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 device: torch.device | None = None):
        self.d = d
        self.bit_width = bit_width
        self.polar_quant = PolarQuant(d, bit_width=bit_width, seed=seed, device=device)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(indices, norms)``."""
        return self.polar_quant.quantize(x)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        return self.polar_quant.dequantize(indices, norms)


@dataclass
class CompressedKVCache:
    """Container for a compressed KV cache."""
    k_compressed: list[list[CompressedVector]] = field(default_factory=list)
    v_indices: list[list[torch.Tensor]] = field(default_factory=list)
    v_norms: list[list[torch.Tensor]] = field(default_factory=list)
    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    k_bit_width: int = 0
    v_bit_width: int = 0


class KVCacheCompressor:
    """Compress and decompress transformer KV cache tensors.

    Uses TurboQuant (Algorithm 2) for K cache — inner product preservation
    for attention scores (Q @ K^T).
    Uses TurboQuantMSE (Algorithm 1) for V cache — MSE preservation for
    value reconstruction (attn_weights @ V).

    Usage::

        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3, device=device)
        compressed = compressor.compress(k_cache, v_cache)
        k_hat, v_hat = compressor.decompress(compressed)
    """

    def __init__(self, head_dim: int, k_bits: int = 3, v_bits: int = 3,
                 seed: int = 42, device: torch.device | None = None):
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_quantizer = TurboQuant(head_dim, bit_width=k_bits, seed=seed, device=device)
        self.v_quantizer = TurboQuantMSE(head_dim, bit_width=v_bits, seed=seed + 500, device=device)

    def compress(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> CompressedKVCache:
        """Compress full KV cache tensors.

        Args:
            k_cache: shape ``(num_layers, num_heads, seq_len, head_dim)``.
            v_cache: same shape.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        assert head_dim == self.head_dim
        assert v_cache.shape == k_cache.shape

        result = CompressedKVCache(
            num_layers=num_layers, num_heads=num_heads,
            seq_len=seq_len, head_dim=head_dim,
            k_bit_width=self.k_bits, v_bit_width=self.v_bits,
        )

        for layer in range(num_layers):
            k_layer: list[CompressedVector] = []
            v_layer_idx: list[torch.Tensor] = []
            v_layer_norms: list[torch.Tensor] = []
            for head in range(num_heads):
                k_vecs = k_cache[layer, head]
                k_layer.append(self.k_quantizer.quantize(k_vecs))

                v_vecs = v_cache[layer, head]
                v_idx, v_n = self.v_quantizer.quantize(v_vecs)
                v_layer_idx.append(v_idx)
                v_layer_norms.append(v_n)

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        return result

    def decompress(self, compressed: CompressedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress back to full KV cache tensors."""
        k_cache = torch.zeros(
            compressed.num_layers, compressed.num_heads,
            compressed.seq_len, compressed.head_dim,
            device=self.k_quantizer.polar_quant.device,
        )
        v_cache = torch.zeros_like(k_cache)

        for layer in range(compressed.num_layers):
            for head in range(compressed.num_heads):
                k_cache[layer, head] = self.k_quantizer.dequantize(
                    compressed.k_compressed[layer][head]
                )
                v_cache[layer, head] = self.v_quantizer.dequantize(
                    compressed.v_indices[layer][head],
                    compressed.v_norms[layer][head],
                )

        return k_cache, v_cache

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict:
        """Compute memory usage statistics."""
        n_vectors = num_layers * num_heads * seq_len
        original_bytes = n_vectors * self.head_dim * 2  # fp16
        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 32)
        v_bits_total = n_vectors * self.head_dim * self.v_bits
        compressed_bytes = (k_bits_total + v_bits_total) / 8
        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
        }


# ---------------------------------------------------------------------------
# Layer-Adaptive KV Cache Compression
#
# The last N layers of a transformer account for most of turbo's quality
# loss.  Layer-adaptive mode protects those layers with higher precision
# (or no compression) while compressing earlier layers aggressively.
#
# Modes (set via TURBO_LAYER_ADAPTIVE env var or constructor):
#   0 — uniform (all layers same bits)
#   2 — last ``n_protected`` layers at ``protected_bits``, rest at base bits
#   7 — "Boundary V": first 2 + last 2 layers protected (V only)
# ---------------------------------------------------------------------------

_LAYER_ADAPTIVE_MODES = {0, 2, 7}


def _layer_adaptive_mode() -> int:
    """Read layer-adaptive mode from environment, defaulting to 0 (uniform)."""
    import os
    raw = os.environ.get("TURBO_LAYER_ADAPTIVE", "0")
    try:
        mode = int(raw)
    except ValueError:
        return 0
    return mode if mode in _LAYER_ADAPTIVE_MODES else 0


class LayerAdaptivePolicy:
    """Per-layer bit-width policy for KV cache compression.

    Determines which layers get full-precision KV and which get compressed,
    based on validated findings that the last ~20% of layers account for
    nearly all of turbo's quality loss.

    Usage::

        policy = LayerAdaptivePolicy(
            num_layers=40, mode=2, base_bits=3,
            protected_bits=0, n_protected=8,
        )
        for layer in range(40):
            bits = policy.kv_bits(layer)  # 0 = uncompressed, >0 = turbo bits
    """

    def __init__(
        self,
        num_layers: int,
        mode: int = 0,
        base_bits: int = 3,
        protected_bits: int = 0,
        n_protected: int = 8,
    ):
        if mode not in _LAYER_ADAPTIVE_MODES:
            raise ValueError(f"Unknown layer-adaptive mode {mode}, expected one of {_LAYER_ADAPTIVE_MODES}")
        self.num_layers = num_layers
        self.mode = mode
        self.base_bits = base_bits
        self.protected_bits = protected_bits
        self.n_protected = n_protected

        # Pre-compute the set of protected layer indices
        if mode == 0:
            self._protected: frozenset[int] = frozenset()
        elif mode == 2:
            # Last n_protected layers
            start = max(0, num_layers - n_protected)
            self._protected = frozenset(range(start, num_layers))
        elif mode == 7:
            # Boundary V: first 2 + last 2 layers
            first = set(range(min(2, num_layers)))
            last = set(range(max(0, num_layers - 2), num_layers))
            self._protected = frozenset(first | last)
        else:
            self._protected = frozenset()

    def kv_bits(self, layer_idx: int) -> int:
        """Return the KV cache bit-width for a given layer index.

        Returns ``protected_bits`` for protected layers, ``base_bits`` otherwise.
        A return value of 0 means no compression (full precision).
        """
        if layer_idx in self._protected:
            return self.protected_bits
        return self.base_bits

    def is_protected(self, layer_idx: int) -> bool:
        """Whether this layer is protected from aggressive compression."""
        return layer_idx in self._protected

    @property
    def protected_layers(self) -> frozenset[int]:
        return self._protected

    def effective_compression(self, original_bits: int = 16) -> float:
        """Average effective compression ratio across all layers."""
        if self.num_layers == 0:
            return 1.0
        total_bits = 0.0
        for i in range(self.num_layers):
            b = self.kv_bits(i)
            total_bits += b if b > 0 else original_bits
        avg_bits = total_bits / self.num_layers
        return original_bits / avg_bits if avg_bits > 0 else 1.0


class LayerAdaptiveKVCacheCompressor:
    """KV cache compressor with per-layer adaptive bit-width.

    Wraps ``KVCacheCompressor`` and ``LayerAdaptivePolicy`` to apply
    different compression to different layers.

    Usage::

        compressor = LayerAdaptiveKVCacheCompressor(
            head_dim=128, num_layers=40, base_bits=3, mode=2,
        )
        compressed = compressor.compress(k_cache, v_cache)
        k_hat, v_hat = compressor.decompress(compressed)
    """

    def __init__(
        self,
        head_dim: int,
        num_layers: int,
        base_bits: int = 3,
        mode: int | None = None,
        protected_bits: int = 0,
        n_protected: int = 8,
        seed: int = 42,
        device: torch.device | None = None,
    ):
        if mode is None:
            mode = _layer_adaptive_mode()
        self.head_dim = head_dim
        self.policy = LayerAdaptivePolicy(
            num_layers=num_layers, mode=mode,
            base_bits=base_bits, protected_bits=protected_bits,
            n_protected=n_protected,
        )
        # Build per-layer quantizers: compressed layers get turbo, protected layers get None
        self._k_quantizers: list[TurboQuant | None] = []
        self._v_quantizers: list[TurboQuantMSE | None] = []
        for layer_idx in range(num_layers):
            bits = self.policy.kv_bits(layer_idx)
            if bits > 0:
                self._k_quantizers.append(
                    TurboQuant(head_dim, bit_width=bits, seed=seed, device=device)
                )
                self._v_quantizers.append(
                    TurboQuantMSE(head_dim, bit_width=bits, seed=seed + 500, device=device)
                )
            else:
                self._k_quantizers.append(None)
                self._v_quantizers.append(None)

    def compress(
        self, k_cache: torch.Tensor, v_cache: torch.Tensor,
    ) -> dict[str, Any]:
        """Compress full KV cache with per-layer adaptive bit-widths.

        Args:
            k_cache: shape ``(num_layers, num_heads, seq_len, head_dim)``.
            v_cache: same shape.

        Returns:
            Dict with ``k_compressed``, ``v_compressed``, ``raw_k``, ``raw_v``,
            and metadata.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape

        result: dict[str, Any] = {
            "k_compressed": [None] * num_layers,
            "v_indices": [None] * num_layers,
            "v_norms": [None] * num_layers,
            "raw_k": [None] * num_layers,
            "raw_v": [None] * num_layers,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "policy": self.policy,
        }

        for layer in range(num_layers):
            kq = self._k_quantizers[layer]
            vq = self._v_quantizers[layer]
            if kq is None:
                # Protected layer: store raw
                result["raw_k"][layer] = k_cache[layer]
                result["raw_v"][layer] = v_cache[layer]
            else:
                k_layer_compressed = []
                v_layer_indices = []
                v_layer_norms = []
                for head in range(num_heads):
                    k_layer_compressed.append(kq.quantize(k_cache[layer, head]))
                    vi, vn = vq.quantize(v_cache[layer, head])
                    v_layer_indices.append(vi)
                    v_layer_norms.append(vn)
                result["k_compressed"][layer] = k_layer_compressed
                result["v_indices"][layer] = v_layer_indices
                result["v_norms"][layer] = v_layer_norms

        return result

    def decompress(self, compressed: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress back to full KV cache tensors."""
        nl = compressed["num_layers"]
        nh = compressed["num_heads"]
        sl = compressed["seq_len"]
        hd = compressed["head_dim"]
        device = None

        # Find a device from any available tensor
        for layer in range(nl):
            if compressed["raw_k"][layer] is not None:
                device = compressed["raw_k"][layer].device
                break
        if device is None:
            for layer in range(nl):
                kq = self._k_quantizers[layer]
                if kq is not None:
                    device = kq.polar_quant.device
                    break
        if device is None:
            device = torch.device("cpu")

        k_cache = torch.zeros(nl, nh, sl, hd, device=device)
        v_cache = torch.zeros(nl, nh, sl, hd, device=device)

        for layer in range(nl):
            if compressed["raw_k"][layer] is not None:
                k_cache[layer] = compressed["raw_k"][layer]
                v_cache[layer] = compressed["raw_v"][layer]
            else:
                kq = self._k_quantizers[layer]
                vq = self._v_quantizers[layer]
                for head in range(nh):
                    k_cache[layer, head] = kq.dequantize(
                        compressed["k_compressed"][layer][head]
                    )
                    v_cache[layer, head] = vq.dequantize(
                        compressed["v_indices"][layer][head],
                        compressed["v_norms"][layer][head],
                    )

        return k_cache, v_cache

    def memory_stats(self, seq_len: int, num_heads: int) -> dict:
        """Memory usage stats accounting for per-layer bit-widths."""
        nl = self.policy.num_layers
        original_bytes = nl * num_heads * seq_len * self.head_dim * 2  # fp16

        compressed_bits = 0
        for layer in range(nl):
            bits = self.policy.kv_bits(layer)
            if bits > 0:
                # K: bits per coord + 32-bit norm, V: bits per coord
                per_head = seq_len * (self.head_dim * bits + 32) + seq_len * self.head_dim * bits
                compressed_bits += num_heads * per_head
            else:
                # Full precision: 16 bits per value for both K and V
                compressed_bits += num_heads * seq_len * self.head_dim * 16 * 2

        compressed_bytes = compressed_bits / 8
        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / max(compressed_bytes, 1),
            "mode": self.policy.mode,
            "n_protected": len(self.policy.protected_layers),
            "effective_compression": self.policy.effective_compression(),
        }


# ---------------------------------------------------------------------------
# Temporal Decay — progressive requantization of old KV cache tokens
#
# Old tokens get requantized from higher to lower bit-width, saving memory
# while keeping recent tokens at full precision.  The approach:
#   1. Track token age (decode steps since insertion)
#   2. Every ``decay_interval`` steps, batch-requantize the oldest tokens
#   3. Exempt attention sinks (positions 0..sink_len-1)
#   4. Respect layer-adaptive policy (protected layers don't decay)
# ---------------------------------------------------------------------------


class TemporalDecayManager:
    """Manages progressive requantization of old KV cache tokens.

    Requantizes from ``source_bits`` to ``target_bits`` by dequantizing
    and re-quantizing to the lower bit-width codebook.

    Usage::

        decay = TemporalDecayManager(
            d=128, source_bits=3, target_bits=2,
            decay_interval=64, batch_size=64,
        )

        # During generation, call every decode step:
        decayed = decay.maybe_decay(indices, norms, step=current_step)
    """

    def __init__(
        self,
        d: int,
        source_bits: int = 3,
        target_bits: int = 2,
        decay_interval: int = 64,
        batch_size: int = 64,
        sink_len: int = 4,
        seed: int = 42,
        device: torch.device | None = None,
    ):
        if target_bits >= source_bits:
            raise ValueError(
                f"target_bits ({target_bits}) must be < source_bits ({source_bits})"
            )
        self.d = d
        self.source_bits = source_bits
        self.target_bits = target_bits
        self.decay_interval = decay_interval
        self.batch_size = batch_size
        self.sink_len = sink_len
        self.device = device or torch.device("cpu")

        # Source and target quantizers for requantization
        self._source_pq = PolarQuant(d, bit_width=source_bits, seed=seed, device=self.device)
        self._target_pq = PolarQuant(d, bit_width=target_bits, seed=seed, device=self.device)

        # Track which positions have been decayed
        self._decayed_positions: set[int] = set()
        self._step_count: int = 0

    def reset(self):
        """Reset decay state for a new sequence."""
        self._decayed_positions.clear()
        self._step_count = 0

    def _requantize(
        self, indices: torch.Tensor, norms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Requantize from source to target bit-width.

        1. Dequantize from source (get float values)
        2. Re-quantize with target (lower bit-width)

        Args:
            indices: ``(batch, d)`` source-bit centroid indices.
            norms: ``(batch,)`` L2 norms.

        Returns:
            ``(new_indices, new_norms)`` at target bit-width.
        """
        # Step 1: reconstruct approximate vectors from source quantizer
        x_hat = self._source_pq.dequantize(indices, norms)

        # Step 2: re-quantize with target (lower) quantizer
        new_indices, new_norms = self._target_pq.quantize(x_hat)
        return new_indices, new_norms

    def maybe_decay(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        total_seq_len: int,
        recent_window: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Conditionally decay old tokens.

        Called every decode step.  Every ``decay_interval`` steps, identifies
        old tokens (outside the recent window and not attention sinks) and
        requantizes a batch of them.

        Args:
            indices: ``(seq_len, d)`` or ``(n_heads*seq_len, d)`` centroid indices.
            norms: ``(seq_len,)`` or ``(n_heads*seq_len,)`` L2 norms.
            total_seq_len: total sequence length (for sink/recent calculation).
            recent_window: number of recent positions to keep at full precision.

        Returns:
            ``(new_indices, new_norms, did_decay)`` — possibly requantized.
        """
        self._step_count += 1

        if self._step_count % self.decay_interval != 0:
            return indices, norms, False

        seq_len = indices.shape[0]
        if seq_len <= self.sink_len + recent_window:
            return indices, norms, False

        # Identify candidates for decay: not sinks, not recent, not already decayed
        decay_end = max(0, seq_len - recent_window)
        candidates = []
        for pos in range(self.sink_len, decay_end):
            if pos not in self._decayed_positions:
                candidates.append(pos)

        if not candidates:
            return indices, norms, False

        # Batch limit
        batch = candidates[:self.batch_size]
        batch_idx = torch.tensor(batch, dtype=torch.long, device=indices.device)

        # Requantize the batch
        batch_indices = indices[batch_idx]
        batch_norms = norms[batch_idx]
        new_idx, new_norms = self._requantize(batch_indices, batch_norms)

        # Update in-place
        indices = indices.clone()
        norms = norms.clone()
        indices[batch_idx] = new_idx
        norms[batch_idx] = new_norms

        self._decayed_positions.update(batch)
        return indices, norms, True

    @property
    def n_decayed(self) -> int:
        """Number of positions that have been decayed so far."""
        return len(self._decayed_positions)

    def memory_savings_ratio(self, total_positions: int) -> float:
        """Estimate memory savings from decayed positions.

        Returns the ratio of compressed size to original (< 1.0 means savings).
        """
        if total_positions == 0:
            return 1.0
        n_full = total_positions - len(self._decayed_positions)
        n_decayed = len(self._decayed_positions)
        full_bits = n_full * self.d * self.source_bits
        decayed_bits = n_decayed * self.d * self.target_bits
        original_bits = total_positions * self.d * self.source_bits
        return (full_bits + decayed_bits) / max(original_bits, 1)


class OutlierTurboQuant:
    """TurboQuant with outlier channel strategy for non-integer bit rates.

    Splits channels into outlier (higher bit-width) and normal (lower bit-width)
    to achieve fractional average bit rates like 2.5 or 3.5 bits per channel.

    Usage::

        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42, device=device)
        compressed = oq.quantize(x)
        x_hat = oq.dequantize(compressed)
    """

    def __init__(self, d: int, target_bits: float, seed: int = 42,
                 device: torch.device | None = None):
        self.d = d
        self.target_bits = target_bits

        low_bits = int(math.floor(target_bits))
        high_bits = low_bits + 1
        frac = target_bits - low_bits

        self.n_outlier = int(round(d * frac))
        self.n_normal = d - self.n_outlier
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.effective_bits = (self.n_outlier * high_bits + self.n_normal * low_bits) / d

        self.outlier_idx = torch.arange(self.n_outlier, device=device or torch.device("cpu"))
        self.normal_idx = torch.arange(self.n_outlier, d, device=device or torch.device("cpu"))

        self.pq_outlier = (
            PolarQuant(self.n_outlier, bit_width=high_bits - 1, seed=seed, device=device)
            if self.n_outlier > 0 else None
        )
        self.pq_normal = (
            PolarQuant(self.n_normal, bit_width=low_bits - 1, seed=seed + 500, device=device)
            if self.n_normal > 0 else None
        )
        self.qjl = QJLQuantizer(d, seed=seed + 1000, device=device)

    def quantize(self, x: torch.Tensor) -> CompressedVector:
        """Quantise with outlier channel split."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        x_outlier = x[:, self.outlier_idx]
        x_normal = x[:, self.normal_idx]

        if self.pq_outlier is not None:
            _, _, out_residual = self.pq_outlier.quantize_and_residual(x_outlier)
        else:
            out_residual = torch.zeros_like(x_outlier)

        if self.pq_normal is not None:
            _, _, norm_residual = self.pq_normal.quantize_and_residual(x_normal)
        else:
            norm_residual = torch.zeros_like(x_normal)

        full_residual = torch.zeros_like(x)
        full_residual[:, self.outlier_idx] = out_residual.float()
        full_residual[:, self.normal_idx] = norm_residual.float()

        qjl_signs, residual_norms = self.qjl.quantize(full_residual)

        out_idx, out_norms = (
            self.pq_outlier.quantize(x_outlier) if self.pq_outlier is not None
            else (torch.tensor([]), torch.tensor([]))
        )
        norm_idx, norm_norms = (
            self.pq_normal.quantize(x_normal) if self.pq_normal is not None
            else (torch.tensor([]), torch.tensor([]))
        )

        # Store combined indices/norms (outlier first, then normal)
        mse_indices = torch.cat([out_idx.flatten(), norm_idx.flatten()])
        vector_norms = torch.cat([out_norms.flatten(), norm_norms.flatten()])

        result = CompressedVector(
            mse_indices=mse_indices,
            vector_norms=vector_norms,
            qjl_signs=qjl_signs.squeeze(0) if single else qjl_signs,
            residual_norms=residual_norms.squeeze(0) if single else residual_norms,
            bit_width=int(round(self.effective_bits)),
        )
        return result

    def compression_ratio(self, original_bits: int = 16) -> float:
        """Compression ratio vs original precision."""
        per_vector_bits = self.d * self.effective_bits + 96  # +32 QJL norm + 64 for outlier/normal norms
        original = self.d * original_bits
        return original / per_vector_bits
