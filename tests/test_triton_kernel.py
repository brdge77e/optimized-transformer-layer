#!/usr/bin/env python3
"""
Validate triton_kernels.fused_scaled_masked_softmax against plain PyTorch on
an actual CUDA GPU. This cannot be run in the CPU-only development sandbox
used to build this submission -- already run once on a Colab Tesla T4
(48/48 pass, see reports/TECH_REPORT.md §9); run it again on your own target
GPU to reproduce that directly rather than take it on faith.

Usage: python3 tests/test_triton_kernel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triton_kernels import build_additive_mask, fused_scaled_masked_softmax  # noqa: E402


def check(batch, heads, seq_len, causal, padding, dtype, device):
    torch.manual_seed(0)
    scores = torch.randn(batch, heads, seq_len, seq_len, device=device, dtype=dtype)
    scale = 0.125

    keep = torch.ones(batch, 1, 1, seq_len, dtype=torch.bool, device=device)
    if padding:
        keep[:, :, :, seq_len // 2 :] = False
    if causal:
        causal_keep = ~torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        keep = keep & causal_keep

    add_mask = build_additive_mask(keep, seq_len, device, torch.float32)

    got = fused_scaled_masked_softmax(scores, scale, add_mask)
    ref = torch.softmax(scores.float() * scale + add_mask, dim=-1).to(dtype)

    abs_err = (got.float() - ref.float()).abs()
    finite = torch.isfinite(ref)
    max_abs = abs_err[finite].max().item()
    ok = max_abs < 1e-3
    print(
        f"batch={batch} heads={heads} seq_len={seq_len} causal={causal} "
        f"padding={padding} dtype={dtype}: max_abs_err={max_abs:.3e} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("No CUDA device available -- this test must be run on a GPU machine.")
        return 1

    device = torch.device("cuda")
    all_ok = True
    for seq_len in (16, 128, 512, 2048):
        for causal in (False, True):
            for padding in (False, True):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    all_ok &= check(4, 8, seq_len, causal, padding, dtype, device)

    print("\nALL PASS" if all_ok else "\nSOME FAILED -- do not use the Triton path yet")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
