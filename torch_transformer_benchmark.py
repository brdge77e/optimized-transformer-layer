#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


# Two of the optimizations below -- fusing the q/k/v projections into a single GEMM,
# and replacing the explicit attention math with
# torch.nn.functional.scaled_dot_product_attention -- change the order in
# which floating point additions/reductions happen relative to the baseline
# (a single [3*d, d] GEMM tiles/blocks differently than three separate
# [d, d] GEMMs; SDPA's fused kernel is a different algorithm entirely from
# "matmul -> mask -> softmax -> matmul"). In float32 those reordering effects
# are far below the tolerance and always safe. Measured directly on this
# machine's CPU backend, in bfloat16 they are NOT always safe: for some
# matrix shapes (verified: d_model=512 triggers it, d_model=128 does not) the
# fused-GEMM reduction alone already differs from the unfused projections by
# about one bfloat16 ULP, and SDPA's kernel differs by a similar or larger
# amount; stacked over enough transformer layers this compounds past a fixed
# absolute/relative tolerance. None of this means the fused computation is
# "wrong" -- it is simply a different, and often more accurate, sequence of
# roundings than the reference's -- but this benchmark grades bit-level
# agreement with the reference's own (imprecise) low-precision rounding
# rather than agreement with a continuous ground truth. So by default the
# fused-QKV-GEMM and SDPA fast path is enabled only for float32, falling back to
# separate q/k/v projections plus the baseline's explicit attention formula
# (verified bit-exact across the whole shape sweep in tests/test_shape_matrix.py)
# for every other dtype. Pass `force_fast_path=True` to OptimizedSelfAttention
# (or set TRANSFORMER_FORCE_FAST_PATH=1) to re-enable the fast path for
# fp16/bf16 -- do this only after validating accuracy on your own target GPU,
# since CUDA's cuBLAS/cuDNN and flash-attention kernels accumulate
# differently than the CPU kernels measured here and may not show this drift.
_FAST_PATH_DTYPES = (torch.float32,)

# Above this sequence length, combining a real (non-all-valid) padding mask
# with causal masking is routed to pytorch_flash_attention.py's tiled kernel
# instead of materializing a [batch, 1, seq_len, seq_len] mask -- see
# UserOptimizedTransformer._build_shared_attn_mask. At seq_len=100000
# (the hackathon's own published "Appendix: Test Shapes", row 14) that
# mask would be 320GB as a boolean tensor and 1.28TB once converted to the
# additive float mask SDPA prefers (reports/TECH_REPORT.md SS13) -- a certain
# crash on any GPU, for a shape well beyond anything test_shape_matrix.py
# covers (max seq_len=2048) or that this threshold could ever accidentally
# trigger on. No existing test exercises seq_len > this threshold with real
# padding, so this branch is additive: it cannot regress anything already
# measured in reports/TECH_REPORT.md, only avoid a crash that was previously
# certain.
_TILED_FALLBACK_SEQ_LEN_THRESHOLD = 8192


class OptimizedSelfAttention(nn.Module):
    """Multi-head self-attention that is numerically equivalent to
    ``BaselineSelfAttention`` but is implemented with two optimizations that
    are enabled together on the fast path (see `_FAST_PATH_DTYPES` above):

      1. Fused QKV projection: the three [d_model, d_model] projections are
         concatenated into a single [3*d_model, d_model] GEMM the first time
         the module is used for inference. One large matmul has better
         arithmetic intensity / tensor-core utilization than three small ones
         and cuts three kernel launches down to one.
      2. ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) replaces
         the explicit QK^T -> mask -> softmax -> matmul(V) chain. On CUDA
         this dispatches to a fused flash-attention / memory-efficient kernel
         that never materializes the full [S, S] score matrix in HBM; on CPU
         it dispatches to PyTorch's fused CPU attention kernel. Both avoid
         the extra elementwise mask/softmax kernel launches the baseline
         pays for.

    When the fast path is not enabled for the current dtype, an explicit
    path is used instead: separate (unfused) q/k/v projections plus the
    baseline's exact attention formula, which reproduces the baseline's
    output bit-for-bit (verified in tests/test_shape_matrix.py).

    The submodule keeps the exact parameter names/shapes of
    ``BaselineSelfAttention`` (q_proj, k_proj, v_proj, out_proj), so
    ``copy_model_weights`` / ``load_state_dict`` work completely unmodified.
    """

    def __init__(
        self, d_model: int, num_heads: int, force_fast_path: Optional[bool] = None
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.force_fast_path = force_fast_path

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Lazily-built fused QKV weight/bias cache. Built on first forward()
        # call (after weights have been copied and moved to their final
        # device/dtype) and reused for every subsequent call. This keeps the
        # state_dict identical to the baseline while still giving a single
        # fused GEMM at inference time.
        #
        # _fused_weight/_fused_bias are plain tensor attributes, not
        # registered buffers, so nn.Module.to() does NOT move them -- only
        # rebuilding from q/k/v here keeps them correctly placed. They're
        # kept unregistered on purpose: registering them as buffers would
        # add new keys to state_dict() and break copy_model_weights' default
        # strict=True load_state_dict from a plain BaselineSelfAttention
        # checkpoint (missing_keys). _fused_cache_key guards the
        # cache instead -- rebuilt automatically whenever q_proj.weight's
        # device/dtype no longer matches what the cache was built for (e.g.
        # a second .to() call after the cache already exists). Explicit
        # weight mutation (e.g. a second load_state_dict into an
        # already-used module) still needs reset_fusion_cache() -- see
        # copy_model_weights, which calls it automatically after every load.
        self._fused_weight: Optional[torch.Tensor] = None
        self._fused_bias: Optional[torch.Tensor] = None
        self._fused_cache_key: Optional[Tuple[torch.device, torch.dtype]] = None

    def reset_fusion_cache(self) -> None:
        self._fused_weight = None
        self._fused_bias = None
        self._fused_cache_key = None

    def _fused_qkv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (self.q_proj.weight.device, self.q_proj.weight.dtype)
        if self._fused_weight is None or self._fused_cache_key != cache_key:
            self._fused_weight = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            ).contiguous()
            self._fused_bias = torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
            ).contiguous()
            self._fused_cache_key = cache_key
        return self._fused_weight, self._fused_bias

    def _use_fast_path(self, dtype: torch.dtype) -> bool:
        if self.force_fast_path is not None:
            return self.force_fast_path
        if os.environ.get("TRANSFORMER_FORCE_FAST_PATH") == "1":
            return True
        return dtype in _FAST_PATH_DTYPES

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        use_tiled: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # Fast path only. On CPU/CUDA, SDPA's fused kernels accept
            # strided (transposed) q/k/v directly, so skipping .contiguous()
            # here removes a measurable chunk of the aten::copy_ cost seen
            # when profiling the baseline (~20% of baseline CPU time -- see
            # reports/TECH_REPORT.md). MPS's SDPA path does not share that
            # property: benchmarked at batch=8/seq=128/d_model=512/heads=8,
            # feeding it strided q/k/v is slower than calling .contiguous()
            # up front (1.10ms vs 0.79ms), enough to erase the whole
            # optimization's speedup.
            out = t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            return out.contiguous() if x.device.type == "mps" else out

        def split_heads_exact(t: torch.Tensor) -> torch.Tensor:
            # Exact/fallback path only -- always .contiguous(), matching
            # BaselineSelfAttention._split_heads unconditionally (not just on
            # MPS, unlike the fast path's split_heads above). This isn't a
            # style choice: a strided vs. contiguous matmul is the same
            # mathematical operation but can use a different reduction
            # order/kernel, which rounds differently in low precision.
            # Measured directly (CPU, bfloat16): reusing the fast path's
            # MPS-only .contiguous() here left this "exact" path silently
            # non-bit-exact with the baseline for the "standard" shape --
            # up to ~0.004 max_abs diff from contiguity alone in one layer,
            # compounding past tolerance over 6 stacked layers. Bit-exactness
            # here isn't optional: bf16/fp16 depend on this fallback being a
            # true reproduction of the baseline, not merely an equivalent one
            # (see _FAST_PATH_DTYPES above).
            return (
                t.view(batch, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        if use_tiled:
            # Extreme seq_len + real padding: avoid materializing any
            # [B,H,S,S]-scale tensor at all. attn_mask here is the small
            # per-key padding bias built by _build_shared_attn_mask (not
            # combined with causal); flash_attention_pytorch applies causal
            # masking itself, block by block, so this never touches more
            # than [B,H,seq_len,block_size] at once. See
            # reports/TECH_REPORT.md SS13.
            from pytorch_flash_attention import flash_attention_pytorch

            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = split_heads_exact(q), split_heads_exact(k), split_heads_exact(v)
            context = flash_attention_pytorch(
                q, k, v, self.scale, is_causal=is_causal, attn_bias=attn_mask
            )
        elif self._use_fast_path(x.dtype):
            weight, bias = self._fused_qkv()
            qkv = F.linear(x, weight, bias)  # single fused GEMM instead of 3
            q, k, v = qkv.chunk(3, dim=-1)
            q, k, v = split_heads(q), split_heads(k), split_heads(v)

            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            # Exact path: separate q/k/v projections (matching the baseline's
            # GEMM shapes/tiling exactly) plus the baseline's explicit
            # attention formula. Verified bit-exact across the shape sweep in
            # tests/test_shape_matrix.py for every dtype where the fast path
            # is disabled by default.
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = split_heads_exact(q), split_heads_exact(k), split_heads_exact(v)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if is_causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).triu(diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    # Additive mask (0 / -inf) -- see _build_shared_attn_mask.
                    scores = scores + attn_mask.to(dtype=scores.dtype)
            probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            context = torch.matmul(probs, v)

        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(context)


class OptimizedTransformerBlock(nn.Module):
    """Pre-LN transformer block using OptimizedSelfAttention. Parameter names
    (norm1, attention.{q,k,v,out}_proj, norm2, ffn_in, ffn_out) match
    BaselineTransformerBlock exactly."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = OptimizedSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        use_tiled: bool = False,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attn_mask, is_causal, use_tiled)
        # approximate="none" reproduces the exact erf-based GELU used by the
        # baseline; changing to the tanh approximation would trade accuracy
        # for a small speedup and risks breaking the 1%/1e-3 tolerance.
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    """
    Optimized implementation of BaselineTransformer.

    Requirements satisfied:
      1. forward() signature is unchanged: forward(x, valid_token_mask=None).
      2. Returns a tensor with shape [batch_size, seq_len, d_model].
      3. Parameter names/shapes are identical to the baseline
         (layers.i.norm1, layers.i.attention.{q,k,v,out}_proj, layers.i.norm2,
         layers.i.ffn_in, layers.i.ffn_out, final_norm), so the unmodified
         ``copy_model_weights`` helper / a plain ``load_state_dict`` call
         transfers weights with no customization needed.

    Optimizations applied (see reports/TECH_REPORT.md for details/numbers):
      * Fused QKV projection (3 GEMMs -> 1) per attention call.
      * torch.nn.functional.scaled_dot_product_attention instead of the
        explicit matmul -> mask -> softmax -> matmul chain (fused
        flash-attention / memory-efficient kernel on CUDA, fused kernel on
        CPU) -- avoids ever materializing the full [B, H, S, S] score matrix.
      * The causal/padding attention mask is built ONCE per forward() call
        and shared across every layer, instead of being rebuilt from scratch
        inside every layer (the baseline calls `.triu()` and allocates a new
        mask tensor per layer, i.e. num_layers times per forward pass).
      * When there is no padding, no mask tensor is built at all -- the
        `is_causal=True` flag is passed straight to SDPA so it can use its
        fastest fused causal kernel path (which skips the upper triangle of
        work entirely rather than computing then discarding it).
      * Compatible with `torch.compile` (see --compile-user) and with
        fp16/bf16 autocast; nothing here is Python-control-flow-heavy in a
        way that would break graph capture.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Swap in the optimized block implementation. Parameter names/shapes
        # are identical to BaselineTransformerBlock, so state_dict layout is
        # unchanged (`final_norm`, inherited from BaselineTransformer.__init__,
        # is untouched).
        self.layers = nn.ModuleList(
            [
                OptimizedTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        # The boolean causal keep-pattern depends only on (seq_len, device),
        # never on input values, so it's safe to memoize across forward calls
        # with the same shape (e.g. every call in a serving loop with fixed
        # seq_len, or every timed iteration in this benchmark). Measured on
        # MPS: rebuilding it costs ~0.05ms at seq_len=128 up to ~0.22ms at
        # seq_len=2048 -- small, but pure profile-identified overhead with a
        # free fix once padding is also present (see _build_shared_attn_mask).
        self._causal_keep_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _causal_keep(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        cached = self._causal_keep_cache.get(key)
        if cached is None:
            cached = ~torch.ones(
                seq_len, seq_len, device=device, dtype=torch.bool
            ).triu(diagonal=1)
            self._causal_keep_cache[key] = cached
        return cached

    @staticmethod
    def _no_real_padding(valid_token_mask: Optional[torch.Tensor]) -> bool:
        """True if there's no actual padding to mask (mask is None, or every
        token is valid). Callers pass a real all-True tensor -- rather than
        Python None -- whenever a batch happens to have no padded sequences
        (e.g. every request in this benchmark's default, padding_ratio=0.0
        case). Checking `.all()` costs one sync per forward() call, but it
        lets an all-valid batch skip both SDPA's masked (math-fallback) path
        and every layer's now-unnecessary masked_fill -- see
        reports/TECH_REPORT.md §6.3 for why this matters far more than the
        one extra sync costs."""
        return valid_token_mask is None or bool(valid_token_mask.all())

    def _build_shared_attn_mask(
        self,
        batch: int,
        seq_len: int,
        valid_token_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], bool, bool]:
        """Returns (attn_mask, is_causal, use_tiled) to be reused by every
        layer. use_tiled routes to pytorch_flash_attention.py instead of
        SDPA/the exact fallback -- see _TILED_FALLBACK_SEQ_LEN_THRESHOLD."""
        causal = self.config.causal
        if valid_token_mask is None:
            # No padding: let SDPA use its fastest fused causal/non-causal
            # path with no explicit mask tensor at all.
            return None, causal, False

        if seq_len > _TILED_FALLBACK_SEQ_LEN_THRESHOLD:
            # Real padding at this sequence length would otherwise require
            # materializing a [batch, 1, seq_len, seq_len] mask below --
            # hundreds of GB to TB at the shapes this guards (see the
            # threshold's docstring). Return the small, per-key padding
            # bias unexpanded; OptimizedSelfAttention's tiled path applies
            # causal masking itself, block by block, so it never needs the
            # combined [B,1,S,S] tensor at all.
            padding_bias = torch.zeros(
                batch, 1, 1, seq_len, dtype=torch.float32, device=device
            )
            padding_bias.masked_fill_(~valid_token_mask[:, None, None, :], float("-inf"))
            return padding_bias, causal, True

        # True = "attend to this key". Combine key-padding with causal once.
        allowed = valid_token_mask[:, None, None, :]
        if causal:
            allowed = allowed & self._causal_keep(seq_len, device)

        # SDPA on MPS only reaches for anything faster than its generic
        # "math" fallback when attn_mask is float (additive) or absent --
        # a *boolean* mask forces the slow fallback even when real padding
        # is present and can't just be skipped (measured ~10% faster at this
        # shape; mathematically identical to the boolean mask, verified to
        # max_diff=0.0). This mask is only ever fed to SDPA (never to a
        # masked_fill), so changing its dtype here doesn't affect anything
        # else -- the fp32 fast path is the only caller (_FAST_PATH_DTYPES).
        additive = torch.zeros(allowed.shape, dtype=torch.float32, device=device)
        additive.masked_fill_(~allowed, float("-inf"))
        return additive, False, False

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        # An all-True mask carries no information a masked_fill/masked SDPA
        # call needs to act on -- treat it as None so every downstream mask
        # op (attention masking, per-layer/final masked_fill) is skipped
        # entirely instead of running as an expensive no-op.
        if self._no_real_padding(valid_token_mask):
            valid_token_mask = None
        attn_mask, is_causal, use_tiled = self._build_shared_attn_mask(
            batch, seq_len, valid_token_mask, x.device
        )

        for layer in self.layers:
            x = layer(x, valid_token_mask, attn_mask, is_causal, use_tiled)

        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")
    # load_state_dict copies weight *values* in place, so it wouldn't be
    # caught by OptimizedSelfAttention's device/dtype cache-invalidation
    # check -- explicitly invalidate every fused-QKV cache so a reused
    # module never serves a fusion built from pre-load weights.
    for module in optimized.modules():
        if isinstance(module, OptimizedSelfAttention):
            module.reset_fusion_cache()


def _mps_available() -> bool:
    # torch.backends.mps only exists on torch builds that support Apple
    # Metal (i.e. the standard Mac wheel); guard for other platforms/builds.
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if device.type == "mps" and not _mps_available():
        raise RuntimeError(
            "MPS was requested, but it is not available. Check "
            "torch.backends.mps.is_built() and torch.backends.mps.is_available(), "
            "and make sure you're on Apple Silicon (or an Intel Mac with a "
            "supported AMD GPU) with a recent macOS + PyTorch."
        )
    return device


def synchronize_device(device: torch.device) -> None:
    """Block until all queued work on `device` has finished. Needed before
    trusting any wall-clock timing, because both CUDA and MPS dispatch work
    asynchronously from the host -- without this, time.perf_counter() around
    a call only measures how long it took to *enqueue* the work, not how
    long the GPU actually took to run it, which would make every device
    look unrealistically fast."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def _make_generator(device: torch.device) -> torch.Generator:
    # Not every torch/device build supports a device-local Generator (this
    # has historically been spottier for MPS than for CUDA across PyTorch
    # versions). Fall back to a CPU generator + .to(device) if needed --
    # still fully deterministic per-seed, just generated host-side first.
    try:
        return torch.Generator(device=device)
    except (RuntimeError, TypeError):
        return torch.Generator(device="cpu")


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(device)
    generator.manual_seed(seed)
    gen_device = generator.device

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=gen_device,
        dtype=dtype,
    ).to(device)
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=gen_device,
    ).to(device)
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    synchronize_device(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            # CPU: perf_counter alone is accurate (ops are synchronous).
            # MPS: ops dispatch asynchronously just like CUDA, but PyTorch's
            # MPS backend doesn't expose the same batched multi-Event timing
            # API used above for CUDA across all supported versions, so this
            # takes the simpler (slightly higher-overhead, still correct)
            # approach: synchronizing once per iteration before stopping
            # the clock. Skipping this sync would silently measure only
            # host-side dispatch latency, not real GPU execution time.
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                if device.type == "mps":
                    torch.mps.synchronize()
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")
    elif device.type == "mps":
        print("MPS latency is measured with wall-clock time + torch.mps.synchronize() per call")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")
    if device.type == "mps" and dtype == torch.bfloat16:
        print(
            "[warning] bfloat16 support on the MPS backend varies by macOS/"
            "PyTorch version and may be slow, unsupported for some ops, or "
            "silently upcast -- float16 is generally the safer reduced-"
            "precision choice on Apple GPUs"
        )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    elif device.type == "mps":
        torch.mps.manual_seed(args.seed)
        # TF32 is a CUDA/Ampere-specific tensor-core numeric format and has
        # no MPS equivalent; --allow-tf32 is silently a no-op here.

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("gpu=Apple Metal (MPS) device -- see `system_profiler SPDisplaysDataType` "
              "in Terminal for the exact chip/GPU name")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
