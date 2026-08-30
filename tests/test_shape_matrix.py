#!/usr/bin/env python3
"""
Correctness sweep across the kind of shape/config matrix described in the
hackathon problem statement: small/large batch size, small/large sequence
length, small/large hidden dimension, causal vs non-causal, padded vs
unpadded, and every supported dtype.

This does NOT replace the official grading harness (torch_transformer_benchmark.py
itself) -- it is an additional, wider sweep used during development to make
sure UserOptimizedTransformer is robust to shapes not otherwise hand-picked.

Usage:
    python3 tests/test_shape_matrix.py [--device cpu|cuda|auto] [--verbose]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_transformer_benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    resolve_device,
)

# (batch_size, seq_len, d_model, num_heads, ffn_dim, num_layers)
SHAPE_CASES = [
    ("tiny", 1, 1, 16, 2, 32, 1),
    ("single_head", 3, 20, 16, 1, 32, 2),
    ("small_batch_large_seq", 1, 1024, 64, 4, 256, 2),
    ("large_batch_small_seq", 64, 8, 64, 4, 128, 2),
    ("standard", 8, 128, 512, 8, 2048, 6),
    ("large_dim", 2, 32, 1024, 16, 4096, 2),
    ("odd_batch", 5, 37, 96, 4, 192, 3),
    ("long_seq", 2, 2048, 128, 8, 256, 2),
    ("deep_stack", 2, 64, 128, 8, 256, 12),
]

CAUSAL_OPTIONS = [False, True]
PADDING_OPTIONS = [0.0, 0.35]
DTYPE_OPTIONS = [torch.float32, torch.bfloat16]

# The official grading tolerance stated in the problem statement.
RTOL = 0.02
ATOL = 0.002


def run_case(
    name: str,
    batch: int,
    seq_len: int,
    d_model: int,
    heads: int,
    ffn_dim: int,
    layers: int,
    causal: bool,
    padding_ratio: float,
    dtype: torch.dtype,
    device: torch.device,
    verbose: bool,
) -> bool:
    config = TransformerConfig(
        batch_size=batch,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        causal=causal,
    )
    config.validate()

    torch.manual_seed(0)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(baseline, optimized)

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=123,
        padding_ratio=padding_ratio,
        input_scale=1.0,
    )

    with torch.inference_mode():
        ref = baseline(x, valid_mask)
        got = optimized(x, valid_mask)

    if ref.shape != (batch, seq_len, d_model):
        print(f"  [{name}] FAIL: unexpected output shape {tuple(ref.shape)}")
        return False

    result = compare_outputs(ref, got, rtol=RTOL, atol=ATOL)
    tag = (
        f"{name:<24} causal={int(causal)} pad={padding_ratio:.2f} "
        f"dtype={str(dtype).split('.')[-1]:<9}"
    )
    status = "PASS" if result.passed else "FAIL"
    if verbose or not result.passed:
        print(
            f"  [{tag}] {status} max_abs={result.max_abs_error:.4g} "
            f"max_rel={result.max_relative_error:.4g} "
            f"failed={result.failed_elements}/{result.total_elements}"
        )
    return result.passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Running shape matrix on device={device}")

    total = 0
    passed = 0
    for (name, batch, seq_len, d_model, heads, ffn_dim, layers), causal, padding, dtype in itertools.product(
        SHAPE_CASES, CAUSAL_OPTIONS, PADDING_OPTIONS, DTYPE_OPTIONS
    ):
        # Skip pathologically slow combinations to keep the sweep fast.
        if seq_len >= 2048 and layers > 2:
            continue
        total += 1
        ok = run_case(
            name, batch, seq_len, d_model, heads, ffn_dim, layers,
            causal, padding, dtype, device, args.verbose,
        )
        passed += int(ok)

    print(f"\n{passed}/{total} configurations passed "
          f"(rtol={RTOL}, atol={ATOL}, matching the problem statement's tolerance).")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
