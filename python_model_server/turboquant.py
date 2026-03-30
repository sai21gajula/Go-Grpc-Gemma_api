"""
TurboQuant: KV Cache Compression for LLM Inference
====================================================
Based on: arXiv:2504.19874 (TurboQuant, ICLR 2026)
          arXiv:2502.02617 (PolarQuant, AISTATS 2026)
          arXiv:2406.03482 (QJL, AAAI 2025)
Authors:  Zandieh, Daliri, Hadian, Mirrokni — Google Research

Algorithm overview
------------------
TurboQuant compresses the attention KV cache at inference time using
two complementary stages:

  Stage 1 — Random Rotation + Lloyd-Max scalar quantization (b-1 bits):
    1. Apply a random orthogonal rotation R to the KV vector x.
       This spreads any structure across all coordinates so the marginal
       distribution of each coordinate becomes approximately N(0, 1/d)
       (technically a scaled Beta distribution, but Gaussian for d ≥ 64).
    2. Store a per-vector L2 norm scale in fp16.
    3. Quantize each rotated-normalised coordinate to the nearest centroid
       from a precomputed Lloyd-Max codebook for N(0,1).

  Stage 2 — QJL residual correction (1 bit, optional):
    The MSE-optimal quantizer introduces bias in inner-product (attention
    score) estimation. To remove this bias:
    1. Compute residual:  r = x_norm - R^T dequant(indices)
    2. Draw random Gaussian projection S ∈ R^(m×d)
    3. Store:  sign(S @ r)  [m bits]  and  γ = ||r||₂  [fp16]
    4. Reconstruct correction:  δ = γ · sqrt(π/2)/m · S^T sign(S@r)

    Note: community benchmarks show MSE-only often outperforms MSE+QJL on
    small head dimensions (d < 128). QJL is enabled only when use_qjl=True.

Memory layout per compressed vector (head_dim=d, bits=b, MSE-only):
  - indices:  d × ceil(b/8) bytes  → stored as uint8 (unpackable to d×b bits)
  - scale:    2 bytes (fp16)
  Net: 4b+1 bits vs 16b for fp16  → ~4× compression at b=3

Usage
-----
  compressor = TurboQuantCompressor(bits=3, use_qjl=False, seed=42)
  cache = TurboQuantCache(compressor)

  # In your generation loop, pass cache= to model.generate():
  outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bit packing utilities
# ---------------------------------------------------------------------------

def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack n-bit integer indices into a uint8 tensor.

    indices : (..., d)  uint8, values in [0, 2^bits)
    returns : (..., ceil(d*bits/8))  uint8

    Example (bits=3, d=8):
      8 values × 3 bits = 24 bits = 3 bytes  →  3x smaller than uint8 storage
    """
    *batch, d = indices.shape
    flat = indices.reshape(-1, d).to(torch.int32)
    B = flat.shape[0]

    total_bytes = math.ceil(d * bits / 8)
    packed = torch.zeros(B, total_bytes, dtype=torch.int32, device=indices.device)

    for i in range(d):
        bit_pos  = i * bits
        byte_idx = bit_pos // 8
        bit_shift = bit_pos % 8
        val = flat[:, i]
        packed[:, byte_idx] |= (val << bit_shift)
        # Handle spillover into the next byte
        overflow_bits = bit_shift + bits - 8
        if overflow_bits > 0 and byte_idx + 1 < total_bytes:
            packed[:, byte_idx + 1] |= (val >> (bits - overflow_bits))

    return packed.to(torch.uint8).reshape(*batch, total_bytes)


def unpack_bits(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """
    Unpack a uint8 tensor back into n-bit integer indices.

    packed  : (..., ceil(d*bits/8))  uint8
    bits    : bits per value
    d       : number of values to reconstruct
    returns : (..., d)  uint8
    """
    *batch, _ = packed.shape
    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
    B = flat.shape[0]

    mask = (1 << bits) - 1
    indices = torch.zeros(B, d, dtype=torch.int32, device=packed.device)

    for i in range(d):
        bit_pos   = i * bits
        byte_idx  = bit_pos // 8
        bit_shift = bit_pos % 8
        val = (flat[:, byte_idx] >> bit_shift) & mask
        overflow_bits = bit_shift + bits - 8
        if overflow_bits > 0 and byte_idx + 1 < flat.shape[1]:
            val = val | (((flat[:, byte_idx + 1]) << (bits - overflow_bits)) & mask)
        indices[:, i] = val

    return indices.to(torch.uint8).reshape(*batch, d)


# ---------------------------------------------------------------------------
# Lloyd-Max codebook centroids for N(0, 1)
# Source: Jayant & Noll "Digital Coding of Waveforms" + numerical derivation
# Boundaries are midpoints between consecutive centroids (uniform assumption).
# For N(0, 1/d): divide all values by sqrt(d).
# ---------------------------------------------------------------------------

_LLOYD_MAX_N01: Dict[int, List[float]] = {
    1: [-0.79788, 0.79788],
    2: [-1.51040, -0.45280, 0.45280, 1.51040],
    3: [
        -2.15200, -1.34390, -0.75600, -0.24510,
         0.24510,  0.75600,  1.34390,  2.15200,
    ],
    4: [
        -2.73260, -2.06900, -1.61800, -1.25650,
        -0.94240, -0.65680, -0.38800, -0.12840,
         0.12840,  0.38800,  0.65680,  0.94240,
         1.25650,  1.61800,  2.06900,  2.73260,
    ],
}


# ---------------------------------------------------------------------------
# Compressed tensor storage
# ---------------------------------------------------------------------------

@dataclass
class CompressedKV:
    """
    Stores one layer's compressed keys or values for all sequence positions.

    Shape conventions (same as HuggingFace KV cache):
      batch, heads, seq, head_dim

    Fields:
      indices   (batch, heads, seq, ceil(head_dim*bits/8)) uint8 — bit-packed centroid indices
      scale     (batch, heads, seq, 1)                     float16 — per-vector L2 norm
      head_dim  int — original head dimension (needed for unpacking)
      qjl_signs (batch, heads, seq, m)                     int8    — QJL sign bits  [if use_qjl]
      qjl_norm  (batch, heads, seq, 1)                     float16 — residual L2 norm [if use_qjl]
    """
    indices:    torch.Tensor
    scale:      torch.Tensor
    head_dim:   int
    qjl_signs:  Optional[torch.Tensor] = None
    qjl_norm:   Optional[torch.Tensor] = None

    def seq_len(self) -> int:
        return self.indices.shape[2]


# ---------------------------------------------------------------------------
# Core compressor
# ---------------------------------------------------------------------------

class TurboQuantCompressor:
    """
    Manages rotation matrices and Lloyd-Max codebooks for TurboQuant
    KV cache compression.

    Parameters
    ----------
    bits : int
        Total bits per coordinate (3 or 4 recommended).
        Stage 1 uses (bits-1) bits; Stage 2 (QJL) uses the last 1 bit.
        With use_qjl=False, all 'bits' go to Stage 1.
    use_qjl : bool
        Whether to add the 1-bit QJL residual correction.
        Recommended False for head_dim < 128 (empirically noisier).
    key_bits : int | None
        Override bits for key tensors. Default: bits + 1 (keys have higher
        norm variance and benefit from extra precision).
    value_bits : int | None
        Override bits for value tensors. Default: bits.
    qjl_proj_dim : int | None
        Projection dimension m for QJL. Default: head_dim.
    seed : int
        Random seed for rotation matrices (must be consistent across
        compress and decompress calls).
    """

    def __init__(
        self,
        bits: int = 3,
        use_qjl: bool = False,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        qjl_proj_dim: Optional[int] = None,
        seed: int = 42,
    ):
        if bits not in (1, 2, 3, 4):
            raise ValueError(f"bits must be 1-4, got {bits}")

        self.bits       = bits
        self.use_qjl    = use_qjl
        self.key_bits   = key_bits   if key_bits   is not None else min(bits + 1, 4)
        self.value_bits = value_bits if value_bits is not None else bits
        self.qjl_proj_dim = qjl_proj_dim
        self.seed       = seed

        # Caches keyed by (head_dim, bits) → rotation matrix / codebook
        self._rotations:  Dict[int, torch.Tensor] = {}   # head_dim → (d, d) fp32
        self._codebooks:  Dict[Tuple[int,int], torch.Tensor] = {}  # (head_dim, bits) → (2^bits,) fp32
        self._qjl_projs:  Dict[int, torch.Tensor] = {}   # head_dim → (m, d) fp32

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def _get_rotation(self, head_dim: int, device: torch.device) -> torch.Tensor:
        """Random orthogonal rotation via QR decomposition (seeded)."""
        if head_dim not in self._rotations:
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            G = torch.randn(head_dim, head_dim, generator=gen)
            R, _ = torch.linalg.qr(G)
            self._rotations[head_dim] = R
        return self._rotations[head_dim].to(device)

    # ------------------------------------------------------------------
    # Lloyd-Max codebook
    # ------------------------------------------------------------------

    def _get_codebook(self, head_dim: int, bits: int, device: torch.device) -> torch.Tensor:
        """
        Lloyd-Max centroids for N(0, 1/d).
        Scales the N(0,1) table by 1/sqrt(d).
        """
        key = (head_dim, bits)
        if key not in self._codebooks:
            centroids_n01 = torch.tensor(_LLOYD_MAX_N01[bits], dtype=torch.float32)
            self._codebooks[key] = centroids_n01 / math.sqrt(head_dim)
        return self._codebooks[key].to(device)

    # ------------------------------------------------------------------
    # QJL projection matrix
    # ------------------------------------------------------------------

    def _get_qjl_proj(self, head_dim: int, device: torch.device) -> torch.Tensor:
        """Random Gaussian projection for QJL correction."""
        if head_dim not in self._qjl_projs:
            m = self.qjl_proj_dim or head_dim
            gen = torch.Generator()
            gen.manual_seed(self.seed + 1)
            S = torch.randn(m, head_dim, generator=gen)
            # Normalise rows so each projection preserves L2 structure
            S = S / S.norm(dim=1, keepdim=True)
            self._qjl_projs[head_dim] = S
        return self._qjl_projs[head_dim].to(device)

    # ------------------------------------------------------------------
    # Encode one batch of vectors
    # ------------------------------------------------------------------

    def _encode(
        self,
        x: torch.Tensor,  # (batch, heads, seq, d)
        bits: int,
    ) -> CompressedKV:
        """
        Compress a KV tensor to (bits) bits per coordinate.

        Steps:
          1. Compute L2 norm scale per vector  →  store fp16
          2. Normalise  →  rotated unit-ish vector
          3. Apply random orthogonal rotation R
          4. Nearest-centroid quantise (Stage 1, uses all 'bits' or bits-1)
          5. Optionally apply QJL residual correction (Stage 2, 1 bit)
        """
        d = x.shape[-1]
        device = x.device
        work_bits = bits - 1 if self.use_qjl else bits

        if work_bits < 1 or work_bits > 4:
            work_bits = max(1, min(4, work_bits))

        R = self._get_rotation(d, device)           # (d, d)
        cb = self._get_codebook(d, work_bits, device)  # (2^work_bits,)

        # 1. L2 norm scale  (batch, heads, seq, 1)
        scale = x.norm(dim=-1, keepdim=True).to(torch.float16)
        scale_f32 = scale.float()
        scale_f32 = scale_f32.clamp(min=1e-8)

        # 2. Normalise
        x_norm = x.float() / scale_f32             # (B, H, S, d)

        # 3. Rotate: y = R @ x_norm  →  coordinates approx N(0, 1/d)
        # matmul over last dim: (..., d) @ (d, d) = (..., d)
        y = x_norm @ R.T                            # (B, H, S, d)

        # 4. Nearest-centroid quantisation
        # y: (B, H, S, d, 1) - cb: (2^b,)
        diff = y.unsqueeze(-1) - cb                 # (B, H, S, d, 2^b)
        raw_indices = diff.abs().argmin(dim=-1).to(torch.uint8)  # (B, H, S, d)

        qjl_signs = None
        qjl_norm  = None

        if self.use_qjl:
            # Stage 2: QJL correction on the quantisation residual
            # Reconstruct Stage-1 output from raw (unpacked) indices
            y_hat = cb[raw_indices.long()]          # (B, H, S, d)
            x_hat = y_hat @ R                       # (B, H, S, d)  (undo rotation)

            # Residual in normalised space
            r = x_norm - x_hat                      # (B, H, S, d)

            S_proj = self._get_qjl_proj(d, device)  # (m, d)
            # Project residual: (B, H, S, d) @ (d, m) = (B, H, S, m)
            proj = r @ S_proj.T
            qjl_signs = proj.sign().to(torch.int8)  # (B, H, S, m)
            qjl_norm  = r.norm(dim=-1, keepdim=True).to(torch.float16)

        # 5. Bit-pack indices: (..., d) uint8 → (..., ceil(d*bits/8)) uint8
        packed_indices = pack_bits(raw_indices, work_bits)

        return CompressedKV(
            indices=packed_indices,
            scale=scale,
            head_dim=d,
            qjl_signs=qjl_signs,
            qjl_norm=qjl_norm,
        )

    # ------------------------------------------------------------------
    # Decode one batch of compressed vectors
    # ------------------------------------------------------------------

    def _decode(self, compressed: CompressedKV, bits: int) -> torch.Tensor:
        """
        Reconstruct fp32 KV tensor from CompressedKV.

        Steps:
          1. Look up centroid values from codebook
          2. Undo rotation: x_hat = R^T @ y_hat
          3. Optionally add QJL correction
          4. Re-scale by stored L2 norm
        """
        scale   = compressed.scale.float()       # (B, H, S, 1)
        d       = compressed.head_dim
        device  = compressed.indices.device

        work_bits = bits - 1 if self.use_qjl else bits
        work_bits = max(1, min(4, work_bits))

        R  = self._get_rotation(d, device)
        cb = self._get_codebook(d, work_bits, device)

        # 1. Unpack bit-packed indices → (B, H, S, d) uint8
        indices = unpack_bits(compressed.indices, work_bits, d)

        # 2. Centroid lookup
        y_hat = cb[indices.long()]               # (B, H, S, d)

        # 3. Undo rotation: x_hat_norm = R^T @ y_hat
        x_hat = y_hat @ R                        # (B, H, S, d)

        # 4. QJL correction
        if self.use_qjl and compressed.qjl_signs is not None:
            S_proj   = self._get_qjl_proj(d, device)      # (m, d)
            m        = S_proj.shape[0]
            signs    = compressed.qjl_signs.float()        # (B, H, S, m)
            res_norm = compressed.qjl_norm.float()         # (B, H, S, 1)

            # δ = γ * sqrt(π/2)/m * S^T * sign(S*r)
            # (B, H, S, m) @ (m, d) = (B, H, S, d)
            delta = (res_norm * math.sqrt(math.pi / 2) / m) * (signs @ S_proj)
            x_hat = x_hat + delta

        # 4. Re-scale
        return x_hat * scale

    # ------------------------------------------------------------------
    # Public API: compress keys / values
    # ------------------------------------------------------------------

    def compress_keys(self, keys: torch.Tensor) -> CompressedKV:
        """Compress a key tensor. keys: (batch, heads, seq, head_dim)"""
        return self._encode(keys, self.key_bits)

    def compress_values(self, values: torch.Tensor) -> CompressedKV:
        """Compress a value tensor. values: (batch, heads, seq, head_dim)"""
        return self._encode(values, self.value_bits)

    def decompress_keys(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress keys to fp32. Returns (batch, heads, seq, head_dim)."""
        return self._decode(compressed, self.key_bits)

    def decompress_values(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress values to fp32. Returns (batch, heads, seq, head_dim)."""
        return self._decode(compressed, self.value_bits)

    # ------------------------------------------------------------------
    # Append along the seq dimension
    # ------------------------------------------------------------------

    @staticmethod
    def _append(existing: Optional[CompressedKV], new: CompressedKV) -> CompressedKV:
        """Concatenate two CompressedKV along the sequence dimension (dim=2)."""
        if existing is None:
            return new
        return CompressedKV(
            indices=torch.cat([existing.indices,    new.indices],    dim=2),
            scale  =torch.cat([existing.scale,      new.scale],      dim=2),
            head_dim=existing.head_dim,
            qjl_signs=(
                torch.cat([existing.qjl_signs, new.qjl_signs], dim=2)
                if existing.qjl_signs is not None and new.qjl_signs is not None
                else None
            ),
            qjl_norm=(
                torch.cat([existing.qjl_norm, new.qjl_norm], dim=2)
                if existing.qjl_norm is not None and new.qjl_norm is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Memory stats
    # ------------------------------------------------------------------

    @staticmethod
    def memory_bytes(compressed: CompressedKV) -> int:
        # indices is already bit-packed: numel() = batch*heads*seq*ceil(d*bits/8)
        total = compressed.indices.numel() * 1   # uint8 = 1 byte each
        total += compressed.scale.numel()   * 2  # fp16  = 2 bytes each
        if compressed.qjl_signs is not None:
            total += compressed.qjl_signs.numel() * 1  # int8 = 1 byte each
        if compressed.qjl_norm is not None:
            total += compressed.qjl_norm.numel()  * 2  # fp16 = 2 bytes each
        return total


# ---------------------------------------------------------------------------
# HuggingFace DynamicCache integration
# ---------------------------------------------------------------------------

try:
    from transformers.cache_utils import DynamicCache as _HFDynamicCache

    class TurboQuantCache(_HFDynamicCache):
        """
        Drop-in replacement for HuggingFace DynamicCache that stores
        KV tensors in TurboQuant-compressed format.

        Usage:
            compressor = TurboQuantCompressor(bits=3)
            cache = TurboQuantCache(compressor)
            output = model.generate(input_ids, past_key_values=cache, use_cache=True)

        Memory savings vs DynamicCache (fp16):
            bits=3 → ~4.3× reduction in KV cache RAM
            bits=4 → ~3.2× reduction in KV cache RAM

        Speed note:
            On CPU the decompress overhead (~2-5ms per layer) is small compared
            to attention computation (~10-50ms). GPU users can integrate the
            Triton fused kernel from dejan.ai/blog/turboquant for additional
            speedup (the algebraic identity: ⟨q, R^T·c[idx]⟩ = ⟨R·q, c[idx]⟩
            avoids materialising the full decompressed tensor).
        """

        def __init__(self, compressor: TurboQuantCompressor):
            super().__init__()
            self.compressor = compressor
            # Compressed storage (replaces parent's key_cache / value_cache lists)
            self._comp_keys:   List[Optional[CompressedKV]] = []
            self._comp_values: List[Optional[CompressedKV]] = []
            self._seen_tokens: int = 0

        # ----------------------------------------------------------------
        # Core override: called by attention module every forward step
        # ----------------------------------------------------------------

        def update(
            self,
            key_states:   torch.Tensor,
            value_states: torch.Tensor,
            layer_idx:    int,
            cache_kwargs: Optional[dict] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compress new key/value states and append to the layer's cache.
            Returns the full (past + new) decompressed tensors for attention.
            """
            # Grow lists if needed
            while len(self._comp_keys) <= layer_idx:
                self._comp_keys.append(None)
                self._comp_values.append(None)

            # Compress new tokens
            new_k_comp = self.compressor.compress_keys(key_states)
            new_v_comp = self.compressor.compress_values(value_states)

            # Append to existing cache
            self._comp_keys[layer_idx]   = TurboQuantCompressor._append(
                self._comp_keys[layer_idx],   new_k_comp)
            self._comp_values[layer_idx] = TurboQuantCompressor._append(
                self._comp_values[layer_idx], new_v_comp)

            # Track token count (only count once, on layer 0)
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]

            # Decompress the full cache for this layer
            all_keys   = self.compressor.decompress_keys(self._comp_keys[layer_idx])
            all_values = self.compressor.decompress_values(self._comp_values[layer_idx])

            # Match dtype of input (model may expect fp16/bf16)
            all_keys   = all_keys.to(key_states.dtype)
            all_values = all_values.to(value_states.dtype)

            return all_keys, all_values

        # ----------------------------------------------------------------
        # Required DynamicCache overrides
        # ----------------------------------------------------------------

        def get_seq_length(self, layer_idx: int = 0) -> int:
            if layer_idx >= len(self._comp_keys) or self._comp_keys[layer_idx] is None:
                return 0
            return self._comp_keys[layer_idx].seq_len()

        def get_max_length(self) -> Optional[int]:
            return None

        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)

        def reorder_cache(self, beam_idx: torch.Tensor) -> None:
            """Support beam search by reordering the batch dimension."""
            for i in range(len(self._comp_keys)):
                if self._comp_keys[i] is not None:
                    ck = self._comp_keys[i]
                    cv = self._comp_values[i]
                    self._comp_keys[i] = CompressedKV(
                        indices=ck.indices[beam_idx],
                        scale=ck.scale[beam_idx],
                        qjl_signs=ck.qjl_signs[beam_idx] if ck.qjl_signs is not None else None,
                        qjl_norm=ck.qjl_norm[beam_idx] if ck.qjl_norm is not None else None,
                    )
                    self._comp_values[i] = CompressedKV(
                        indices=cv.indices[beam_idx],
                        scale=cv.scale[beam_idx],
                        qjl_signs=cv.qjl_signs[beam_idx] if cv.qjl_signs is not None else None,
                        qjl_norm=cv.qjl_norm[beam_idx] if cv.qjl_norm is not None else None,
                    )

        @property
        def seen_tokens(self) -> int:
            return self._seen_tokens

        def memory_summary(self) -> str:
            """Return a human-readable memory usage summary."""
            total_comp = 0
            total_fp16 = 0
            for i, (ck, cv) in enumerate(zip(self._comp_keys, self._comp_values)):
                if ck is not None:
                    comp_bytes = (
                        TurboQuantCompressor.memory_bytes(ck) +
                        TurboQuantCompressor.memory_bytes(cv)
                    )
                    # fp16 equivalent: batch*heads*seq*head_dim * 2 bytes
                    # scale shape is (batch, heads, seq, 1) → numel = batch*heads*seq
                    n_vec = ck.scale.numel()  # batch*heads*seq
                    fp16_bytes = (
                        n_vec * ck.head_dim * 2 +  # keys fp16
                        n_vec * cv.head_dim * 2    # values fp16
                    )
                    total_comp += comp_bytes
                    total_fp16 += fp16_bytes
            ratio = total_fp16 / max(total_comp, 1)
            return (
                f"TurboQuantCache: {total_comp/1e6:.1f} MB compressed "
                f"(vs {total_fp16/1e6:.1f} MB fp16, {ratio:.1f}× savings), "
                f"{self._seen_tokens} tokens cached"
            )

except ImportError:
    logger.warning(
        "transformers not available or DynamicCache not found. "
        "TurboQuantCache will not be available."
    )
    TurboQuantCache = None


# ---------------------------------------------------------------------------
# Validation / smoke test
# ---------------------------------------------------------------------------

def validate_turboquant(bits: int = 3, head_dim: int = 128, seq_len: int = 64) -> dict:
    """
    Validate TurboQuant compression/decompression and measure quality.

    Returns a dict with MSE, cosine similarity, and compression ratio.

    Example:
        results = validate_turboquant(bits=3, head_dim=128)
        print(results)
    """
    torch.manual_seed(0)

    compressor = TurboQuantCompressor(bits=bits, use_qjl=False, seed=42)

    # Simulate KV cache: (batch=1, heads=4, seq, head_dim)
    keys   = torch.randn(1, 4, seq_len, head_dim)
    values = torch.randn(1, 4, seq_len, head_dim)

    # Compress
    comp_k = compressor.compress_keys(keys)
    comp_v = compressor.compress_values(values)

    # Decompress
    recon_k = compressor.decompress_keys(comp_k)
    recon_v = compressor.decompress_values(comp_v)

    # Quality metrics
    mse_k = F.mse_loss(recon_k, keys).item()
    mse_v = F.mse_loss(recon_v, values).item()
    cos_k = F.cosine_similarity(recon_k.flatten(0, -2), keys.flatten(0, -2)).mean().item()
    cos_v = F.cosine_similarity(recon_v.flatten(0, -2), values.flatten(0, -2)).mean().item()

    # Memory ratio
    orig_bytes = keys.numel() * 4 + values.numel() * 4   # fp32
    fp16_bytes = keys.numel() * 2 + values.numel() * 2   # fp16
    comp_bytes = (
        TurboQuantCompressor.memory_bytes(comp_k) +
        TurboQuantCompressor.memory_bytes(comp_v)
    )

    results = {
        "bits": bits,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "key_mse": round(mse_k, 6),
        "value_mse": round(mse_v, 6),
        "key_cosine_sim": round(cos_k, 4),
        "value_cosine_sim": round(cos_v, 4),
        "compression_vs_fp32": round(orig_bytes / comp_bytes, 2),
        "compression_vs_fp16": round(fp16_bytes / comp_bytes, 2),
        "compressed_mb": round(comp_bytes / 1e6, 3),
        "fp16_mb": round(fp16_bytes / 1e6, 3),
    }

    logger.info(
        f"TurboQuant validation: bits={bits} head_dim={head_dim} | "
        f"key_cos={cos_k:.4f} val_cos={cos_v:.4f} | "
        f"{fp16_bytes/1e6:.2f}MB fp16 → {comp_bytes/1e6:.2f}MB compressed "
        f"({fp16_bytes/comp_bytes:.1f}×)"
    )
    return results
