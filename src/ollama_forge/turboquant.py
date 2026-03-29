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
