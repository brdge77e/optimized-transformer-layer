#!/usr/bin/env python3
"""
Correctness sweep for pytorch_flash_attention.flash_attention_pytorch against
a direct (non-tiled) reference implementation of the same attention formula
used by BaselineSelfAttention (matmul -> mask -> softmax -> matmul).

Sweeps sequence length (including seq_len=1 and sizes that don't divide
evenly by any block size), block_size (including 1, and values both smaller
and larger than seq_len), causal on/off, and padding on/off. Runs on
whatever device is selected (CPU or MPS/CUDA via --device).

Usage:
    python3 tests/test_flash_attention_pytorch.py [--device cpu|mps|cuda|auto] [--verbose]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytorch_flash_attention import flash_attention_pytorch  # noqa: E402
from torch_transformer_benchmark import resolve_device  # noqa: E402


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    valid_token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Direct, non-tiled attention -- same formula as BaselineSelfAttention."""
    batch, heads, seq_q, _ = q.shape
    seq_kv = k.shape[2]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones(
            seq_q, seq_kv, device=q.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    if valid_token_mask is not None:
        invalid_keys = ~valid_token_mask[:, None, None, :]
        scores = scores.masked_fill(invalid_keys, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(dtype=q.dtype)


def make_case(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    padding_ratio: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, head_dim, generator=generator, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, generator=generator, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, generator=generator, device=device)

    if padding_ratio <= 0:
        return q, k, v, None

    min_valid = max(1, int(round(seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid, high=seq_len + 1, size=(batch,), generator=generator, device=device
    )
    positions = torch.arange(seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    return q, k, v, valid_token_mask


def build_attn_bias(
    valid_token_mask: torch.Tensor | None, heads: int, seq_q: int, device: torch.device
) -> torch.Tensor | None:
    if valid_token_mask is None:
        return None
    allowed = valid_token_mask[:, None, None, :]  # [B,1,1,S]
    bias = torch.zeros(allowed.shape, dtype=torch.float32, device=device)
    bias.masked_fill_(~allowed, float("-inf"))
    return bias.expand(-1, heads, seq_q, -1)


CASES = [
    # (batch, heads, seq_len, head_dim)
    (1, 1, 1, 8),
    (2, 4, 17, 32),
    (3, 2, 33, 16),
    (2, 8, 64, 32),
    (4, 4, 128, 64),
    (1, 2, 333, 32),
    (2, 4, 1024, 64),
]
BLOCK_SIZES = [1, 16, 33, 128, 512]
CAUSAL_OPTIONS = [False, True]
PADDING_OPTIONS = [0.0, 0.3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Running flash_attention_pytorch correctness sweep on device={device}")

    total = 0
    failed = 0
    seed = 0
    for (batch, heads, seq_len, head_dim), causal, padding_ratio in itertools.product(
        CASES, CAUSAL_OPTIONS, PADDING_OPTIONS
    ):
        scale = head_dim**-0.5
        q, k, v, valid_token_mask = make_case(
            batch, heads, seq_len, head_dim, causal, padding_ratio, device, seed
        )
        seed += 1
        ref = reference_attention(q, k, v, scale, causal, valid_token_mask)
        attn_bias = build_attn_bias(valid_token_mask, heads, seq_len, device)

        for block_size in BLOCK_SIZES:
            total += 1
            got = flash_attention_pytorch(
                q, k, v, scale, is_causal=causal, attn_bias=attn_bias, block_size=block_size
            )
            abs_err = (got.float() - ref.float()).abs()
            finite = torch.isfinite(ref)
            max_abs = abs_err[finite].max().item() if finite.any() else 0.0
            ok = max_abs < max(args.atol, args.rtol * ref[finite].abs().max().item() if finite.any() else args.atol)
            if not ok:
                failed += 1
            if args.verbose or not ok:
                status = "PASS" if ok else "FAIL"
                print(
                    f"  [B={batch} H={heads} S={seq_len} D={head_dim} "
                    f"causal={int(causal)} pad={padding_ratio:.2f} block={block_size:<4d}] "
                    f"{status} max_abs={max_abs:.3e}"
                )

    print(f"\n{total - failed}/{total} configurations passed (rtol={args.rtol:g}, atol={args.atol:g}).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
