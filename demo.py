#!/usr/bin/env python3
"""
End-to-end demo for the "demonstrates your solution working end-to-end"
deliverable (inference results + a small results dashboard).

This is deliberately separate from torch_transformer_benchmark.py's own
CLI: that script is the *grading* harness (accuracy gate + full benchmark
statistics). This script is a short, narratable demo you can screen-record:
it builds both models, runs one real forward pass, prints actual input/
output tensors (the "model predictions" the rubric asks for), reports
accuracy + speedup, and saves a small PNG chart you can show on screen.

Usage:
    python3 demo.py                 # auto device (mps/cuda/cpu)
    python3 demo.py --device mps    # force a specific device
"""
from __future__ import annotations

import argparse
import time

import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    resolve_device,
    synchronize_device,
)


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def time_calls(model, x, mask, device, iterations=20) -> float:
    with torch.inference_mode():
        for _ in range(5):  # warmup
            model(x, mask)
        synchronize_device(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(x, mask)
        synchronize_device(device)
        end = time.perf_counter()
    return (end - start) / iterations * 1000.0  # ms/call


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end optimized transformer demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--no-chart", action="store_true", help="skip saving the PNG dashboard")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.float32

    banner("1. SETUP")
    print(f"Device selected : {device}")
    print(f"PyTorch version  : {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("GPU              : Apple Metal (MPS)")

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=True,
    )
    print(f"Model config     : {config}")

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(baseline, optimized)  # identical weights -> fair, direct comparison
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    banner("2. INFERENCE — real forward pass, real tensors")
    x, valid_mask = generate_random_case(
        config, device=device, dtype=dtype, seed=7, padding_ratio=0.0, input_scale=1.0
    )
    print(f"Input  shape : {tuple(x.shape)}  (batch, seq_len, d_model)")

    with torch.inference_mode():
        baseline_out = baseline(x, valid_mask)
        optimized_out = optimized(x, valid_mask)

    print(f"Output shape : {tuple(optimized_out.shape)}  (batch, seq_len, d_model)")
    print("\nSample prediction -- first sequence, first 6 output features, first 3 tokens:")
    print("  baseline :\n", baseline_out[0, :3, :6].cpu())
    print("  optimized:\n", optimized_out[0, :3, :6].cpu())

    banner("3. ACCURACY — optimized vs. baseline (reference)")
    result = compare_outputs(baseline_out, optimized_out, rtol=0.02, atol=0.002)
    status = "PASS" if result.passed else "FAIL"
    print(f"Tolerance   : rtol=0.02, atol=0.002 (the problem statement's own criterion)")
    print(f"Result      : {status}")
    print(f"Max abs err : {result.max_abs_error:.6g}")
    print(f"Max rel err : {result.max_relative_error:.6g}")
    print(f"Mismatches  : {result.failed_elements}/{result.total_elements}")

    banner("4. PERFORMANCE — baseline vs. optimized latency")
    baseline_ms = time_calls(baseline, x, valid_mask, device)
    optimized_ms = time_calls(optimized, x, valid_mask, device)
    speedup = baseline_ms / optimized_ms
    print(f"Baseline  : {baseline_ms:.3f} ms/call")
    print(f"Optimized : {optimized_ms:.3f} ms/call")
    print(f"Speedup   : {speedup:.2f}x")

    if not args.no_chart:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["Baseline", "Optimized"], [baseline_ms, optimized_ms],
                           color=["#94a3b8", "#2563eb"])
            ax.set_ylabel("Latency (ms/call)")
            ax.set_title(
                f"Transformer layer latency — {device.type.upper()}\n"
                f"batch={config.batch_size}, seq_len={config.seq_len}, "
                f"d_model={config.d_model}, layers={config.num_layers}"
            )
            for bar, value in zip(bars, [baseline_ms, optimized_ms]):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f} ms",
                        ha="center", va="bottom")
            ax.text(0.5, 0.95, f"{speedup:.2f}x speedup", transform=ax.transAxes,
                     ha="center", va="top", fontsize=12, fontweight="bold", color="#16a34a")
            fig.tight_layout()
            out_path = "demo_dashboard.png"
            fig.savefig(out_path, dpi=150)
            print(f"\nDashboard chart saved to: {out_path}")
        except ImportError:
            print("\n(matplotlib not installed -- skipping chart; `pip install matplotlib` to enable)")

    banner("DONE")
    print(f"Summary: accuracy={status}, speedup={speedup:.2f}x on {device}")


if __name__ == "__main__":
    main()
