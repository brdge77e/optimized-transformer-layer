"""
Custom Triton kernel: fused scale -> additive-mask -> row-softmax.

WHY THIS KERNEL / HOW IT WAS CHOSEN
------------------------------------
Profiling the baseline BaselineTransformer on this machine (see
reports/TECH_REPORT.md for the full trace) showed that, after the linear
projections (aten::addmm, ~40% of CPU time), the next largest chunk of cost
was NOT one single op but a chain of small ones the baseline runs
back-to-back on the full [B, H, S, S] score tensor:

    aten::bmm (QK^T)         ~14.0%
    aten::mul (scale)         ~7.0%
    aten::masked_fill_        ~4.1%   (+ ~15.6% more in bitwise_not/expand/etc.
                                        building the mask itself)
    aten::_softmax             ~9.0%

torch_transformer_benchmark.py's default UserOptimizedTransformer addresses
this by calling torch.nn.functional.scaled_dot_product_attention (SDPA),
which dispatches to a vendor flash-attention / fused kernel that fuses ALL of
QK^T, scaling, masking, softmax, and the AV matmul into one kernel that never
materializes the full [B, H, S, S] score matrix in memory at all. That is
strictly better than what this file does, and is why SDPA -- not this file --
is the default fast path in the submission.

This module exists to additionally demonstrate a hand-written custom kernel,
as suggested directly in the problem statement ("custom CUDA, Triton, ...
implementations"). It fuses just the "scale -> mask -> softmax" portion
(mul + masked_fill + softmax above -- roughly 20% of baseline CPU time by
itself) into a single Triton kernel operating on an already-computed score
tensor. It is a real, useful optimization in contexts where you already have
a materialized score tensor (e.g. a custom/non-SDPA attention variant, or a
research kernel that needs to inspect/modify attention scores between QK^T
and softmax) -- but it does NOT get flash-attention's much bigger memory-
bandwidth win of never materializing the score matrix, and SDPA already
outperforms it end-to-end. Treat this as a bonus/demonstration path, not the
recommended default.

IMPORTANT / HONESTY NOTE
-------------------------
This sandbox has no CUDA GPU (verified: torch.cuda.is_available() is False
here), and Triton kernels only execute on GPU. This kernel was written
carefully against the standard Triton "fused softmax" tutorial pattern
(https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
extended with a pre-softmax affine scale and an additive mask, and it has
been syntax-checked (the module imports and `triton.jit` decoration succeeds)
but it has NOT been runtime-executed on a GPU. Run
`tests/test_triton_kernel.py` on your own CUDA machine before relying on it
for grading, and please report back / file an issue if it needs fixes --
treat correctness here as "reasoned through, not yet proven."

Scope limits (by design, to keep the kernel simple and reasoned-about):
  * One Triton program handles one full row of the score matrix in a single
    pass (no blockwise/online softmax across K-tiles), so BLOCK_SIZE must be
    >= seq_len. This covers seq_len up to a few thousand comfortably on
    modern GPUs; `fused_scaled_masked_softmax` falls back to plain PyTorch
    automatically for longer sequences.
  * The additive mask is expected to already be in {0, -inf} form
    (see `build_additive_mask` below), matching the baseline's
    `masked_fill(..., float("-inf"))` convention exactly.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton not installed
    _TRITON_AVAILABLE = False


def build_additive_mask(
    boolean_keep_mask: Optional[torch.Tensor],
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Convert a boolean 'True = attend' mask (as used elsewhere in this
    submission) into the additive {0, -inf} form this kernel expects."""
    if boolean_keep_mask is None:
        return None
    additive = torch.zeros(boolean_keep_mask.shape, device=device, dtype=dtype)
    additive = additive.masked_fill(~boolean_keep_mask, float("-inf"))
    return additive


if _TRITON_AVAILABLE:

    @triton.jit
    def _scaled_masked_softmax_kernel(
        scores_ptr,
        mask_ptr,
        out_ptr,
        n_cols,
        scores_row_stride,
        mask_row_stride,
        out_row_stride,
        scale,
        HAS_MASK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program instance == one row of the [*, seq_len] score matrix
        (i.e. one (batch, head, query_position) triple). Loads the whole row,
        applies the pre-softmax scale and additive mask, and writes a
        normalized softmax row back out -- one kernel launch, one pass over
        the row, instead of three separate elementwise/reduction kernels."""
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols

        row_ptr = scores_ptr + row_idx * scores_row_stride
        row = tl.load(row_ptr + col_offsets, mask=col_mask, other=float("-inf"))
        row = row * scale

        if HAS_MASK:
            mask_row_ptr = mask_ptr + row_idx * mask_row_stride
            bias = tl.load(mask_row_ptr + col_offsets, mask=col_mask, other=float("-inf"))
            row = row + bias

        row_max = tl.max(row, axis=0)
        row = row - row_max
        numerator = tl.exp(row)
        denominator = tl.sum(numerator, axis=0)
        softmax_row = numerator / denominator

        out_ptr_row = out_ptr + row_idx * out_row_stride
        tl.store(out_ptr_row + col_offsets, softmax_row, mask=col_mask)


_MAX_TRITON_BLOCK = 4096  # keep the single-pass kernel's scope bounded


def fused_scaled_masked_softmax(
    scores: torch.Tensor,
    scale: float,
    additive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """scores: [..., seq_len]. Returns softmax(scores * scale + additive_mask)
    along the last dim. Falls back to plain PyTorch ops when Triton/CUDA is
    unavailable or seq_len exceeds the single-pass kernel's block size --
    correctness always holds, the fused kernel only changes performance."""
    seq_len = scores.shape[-1]
    use_triton = (
        _TRITON_AVAILABLE
        and scores.is_cuda
        and seq_len <= _MAX_TRITON_BLOCK
    )
    if not use_triton:
        result = scores.float() * scale
        if additive_mask is not None:
            result = result + additive_mask.float()
        return torch.softmax(result, dim=-1).to(scores.dtype)

    orig_shape = scores.shape
    flat_scores = scores.reshape(-1, seq_len).contiguous()
    n_rows = flat_scores.shape[0]
    out = torch.empty_like(flat_scores)

    flat_mask = None
    if additive_mask is not None:
        flat_mask = additive_mask.expand(orig_shape).reshape(-1, seq_len).contiguous()

    block_size = triton.next_power_of_2(seq_len)
    grid = (n_rows,)
    _scaled_masked_softmax_kernel[grid](
        flat_scores,
        flat_mask if flat_mask is not None else flat_scores,  # dummy ptr, unused
        out,
        seq_len,
        flat_scores.stride(0),
        flat_mask.stride(0) if flat_mask is not None else 0,
        out.stride(0),
        scale,
        HAS_MASK=flat_mask is not None,
        BLOCK_SIZE=block_size,
    )
    return out.reshape(orig_shape)
