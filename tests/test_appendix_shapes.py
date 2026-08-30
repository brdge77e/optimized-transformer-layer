#!/usr/bin/env python3
"""
Correctness + memory-safety check for the 14 shapes in the problem
statement's "Appendix: Test Shapes" (added 27 Aug 2026, 6:25PM update).

This is separate from tests/test_shape_matrix.py because row 14
(seq_len=100000) is large enough that a naive causal+padding mask would
try to materialize a [batch, 1, seq_len, seq_len] tensor -- ~320GB at
that shape -- which is expected to OOM, not a bug in the test. Each
config runs in its own try/except so one OOM doesn't take down the rest
of the suite, and CUDA peak memory is reported per config so you have a
concrete number instead of a guess about headroom.

Usage:
    python3 tests/test_appendix_shapes.py --device cuda
    python3 tests/test_appendix_shapes.py --device cuda --with-padding
"""
from __future__ import annotations

import argparse
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

# (#, batch, d_model, heads, seq_len, layers, ffn_dim) -- causal=True for all 14,
# exactly as listed in the appendix table.
APPENDIX_SHAPES = [
    (1, 64, 128, 4, 128, 4, 128),
    (2, 1, 128, 4, 128, 4, 128),
    (3, 4, 128, 4, 128, 4, 128),
    (4, 16, 128, 4, 128, 4, 128),
    (5, 128, 128, 4, 128, 4, 128),
    (6, 10000, 128, 4, 128, 4, 128),
    (7, 64, 32, 4, 128, 4, 32),
    (8, 64, 1024, 4, 128, 4, 1024),
    (9, 64, 128, 1, 128, 4, 128),
    (10, 64, 128, 2, 128, 4, 128),
    (11, 64, 128, 16, 128, 4, 128),
    (12, 64, 128, 4, 32, 4, 128),
    (13, 64, 128, 4, 1024, 4, 128),
    (14, 32, 1024, 16, 100000, 2, 1024),
]

# Rows where a real padding mask would combine with a causal mask to try
# to materialize an O(batch * seq_len^2) tensor. Skipped by default --
# pass --with-padding to attempt them anyway and see exactly where it
# breaks (recommended on a machine you don't mind OOMing).
QUADRATIC_MASK_RISK = {6, 14}

RTOL = 0.02
ATOL = 0.002


def run_one(
    row: tuple, device: torch.device, padding_ratio: float, verbose: bool
) -> tuple[bool, str]:
    idx, batch, d_model, heads, seq_len, layers, ffn_dim = row
    config = TransformerConfig(
        batch_size=batch,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        causal=True,
    )
    config.validate()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        baseline = BaselineTransformer(config)
        optimized = UserOptimizedTransformer(config)
        copy_model_weights(baseline, optimized)
        baseline = baseline.to(device=device, dtype=torch.float32).eval()
        optimized = optimized.to(device=device, dtype=torch.float32).eval()

        x, valid_mask = generate_random_case(
            config=config,
            device=device,
            dtype=torch.float32,
            seed=1234 + idx,
            padding_ratio=padding_ratio,
            input_scale=1.0,
        )
        with torch.inference_mode():
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
        result = compare_outputs(reference, candidate, rtol=RTOL, atol=ATOL)

        peak_mem = ""
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            peak_mem = f", peak_mem={peak_gb:.2f}GB"

        status = "PASS" if result.passed else "FAIL"
        msg = (
            f"max_abs={result.max_abs_error:.3e}, "
            f"max_rel={result.max_relative_error:.3e}{peak_mem}"
        )
        if verbose or not result.passed:
            print(f"    {msg}")
        return result.passed, status

    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        return False, f"OOM: {exc}"
    except RuntimeError as exc:
        # covers CPU OOM / MPS allocation failures, which don't raise
        # torch.cuda.OutOfMemoryError. MPS's actual message for an
        # over-large single allocation is "Invalid buffer size: X GiB" --
        # confirmed directly at row 14 (batch=32, seq_len=100000) on an
        # Apple M3 Pro, not just "MPS backend out of memory" as originally
        # guarded here -- broadened to catch both phrasings.
        message = str(exc).lower()
        if (
            "out of memory" in message
            or "mps backend out of memory" in message
            or "invalid buffer size" in message
        ):
            return False, f"OOM: {exc}"
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--with-padding",
        action="store_true",
        help="also attempt rows 6 and 14 with padding_ratio=0.3 "
        "(expected to OOM at row 14 with the current mask construction)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Running appendix shape sweep on device={device}\n")

    total = 0
    failed = 0
    for row in APPENDIX_SHAPES:
        idx, batch, d_model, heads, seq_len, layers, ffn_dim = row
        label = (
            f"#{idx:>2d}  batch={batch:<6d} d_model={d_model:<5d} heads={heads:<3d} "
            f"seq_len={seq_len:<7d} layers={layers} ffn_dim={ffn_dim}"
        )

        paddings = [0.0]
        if args.with_padding or idx not in QUADRATIC_MASK_RISK:
            if idx in QUADRATIC_MASK_RISK and not args.with_padding:
                pass  # skip padding variant, see note below
            else:
                paddings.append(0.3)

        for padding_ratio in paddings:
            total += 1
            tag = "no-pad" if padding_ratio == 0.0 else "padded"
            print(f"{label} [{tag}]")
            ok, status = run_one(row, device, padding_ratio, args.verbose)
            if not ok:
                failed += 1
                print(f"    -> {status}")
            elif not args.verbose:
                print(f"    -> {status}")

        if idx in QUADRATIC_MASK_RISK and not args.with_padding:
            print(
                "    (padded variant skipped -- causal+padding at this shape "
                "materializes an O(batch*seq_len^2) mask; re-run with "
                "--with-padding to attempt it deliberately)"
            )

    print(f"\n{total - failed}/{total} configurations passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
