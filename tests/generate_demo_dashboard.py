#!/usr/bin/env python3
"""
Run the benchmark suite on whatever device is available and produce a small
visual "dashboard": a bar chart of speedup-per-config plus a markdown table,
suitable for screen-recording or dropping into a Devpost post.

This is meant to be run by YOU, on your own GPU (MPS on Mac, CUDA on
Linux/Windows) -- it was written and syntax-checked in a CPU-only sandbox,
but every number it reports comes from actually executing
torch_transformer_benchmark.py, so it's real end-to-end inference output,
not a mockup.

Usage:
    python3 tests/generate_demo_dashboard.py [--device auto|cpu|cuda|mps]

Outputs (written next to this script's parent directory, in reports/):
    reports/dashboard.png           bar chart of speedup by config
    reports/dashboard_results.md    the same data as a markdown table
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "torch_transformer_benchmark.py"
OUT_DIR = ROOT / "reports"

CONFIGS = [
    ("small\n(bs=4,seq=64,d=128)",
     ["--batch-size", "4", "--seq-len", "64", "--d-model", "128", "--heads", "8",
      "--ffn-dim", "512", "--layers", "3"]),
    ("default\n(bs=8,seq=128,d=512)",
     ["--batch-size", "8", "--seq-len", "128", "--d-model", "512", "--heads", "8",
      "--ffn-dim", "2048", "--layers", "6"]),
    ("large batch\n(bs=64,seq=32,d=256)",
     ["--batch-size", "64", "--seq-len", "32", "--d-model", "256", "--heads", "8",
      "--ffn-dim", "1024", "--layers", "4"]),
    ("long seq\n(bs=2,seq=1024,d=256)",
     ["--batch-size", "2", "--seq-len", "1024", "--d-model", "256", "--heads", "8",
      "--ffn-dim", "1024", "--layers", "4"]),
    ("causal+padding\n(bs=8,seq=128,d=512)",
     ["--batch-size", "8", "--seq-len", "128", "--d-model", "512", "--heads", "8",
      "--ffn-dim", "2048", "--layers", "6", "--causal", "--padding-ratio", "0.3"]),
]

SPEEDUP_RE = re.compile(r"speedup\s*:\s*([\d.]+)x")
SUMMARY_RE = re.compile(r"summary: (PASS|FAIL)")
DEVICE_RE = re.compile(r"device=(\S+),")


def run_one(label: str, args: list[str], device: str) -> dict:
    cmd = [
        sys.executable, str(SCRIPT),
        "--device", device,
        "--rtol", "0.02", "--atol", "0.002",
        "--accuracy-trials", "5", "--warmup", "10", "--repeats", "20",
        "--benchmark-rounds", "2",
        *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = result.stdout
    status = SUMMARY_RE.search(out)
    speedup = SPEEDUP_RE.search(out)
    resolved_device = DEVICE_RE.search(out)
    return {
        "label": label,
        "passed": bool(status and status.group(1) == "PASS"),
        "speedup": float(speedup.group(1)) if speedup else None,
        "device": resolved_device.group(1) if resolved_device else device,
        "raw": out,
        "stderr": result.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    results = []
    for label, cli_args in CONFIGS:
        print(f"Running: {label.splitlines()[0]} ...")
        r = run_one(label, cli_args, args.device)
        results.append(r)
        if not r["passed"]:
            print(f"  WARNING: accuracy did not pass for this config.")
            print(r["raw"][-2000:])
            print(r["stderr"][-2000:])
        else:
            print(f"  device={r['device']} speedup={r['speedup']}x")

    resolved_device = results[0]["device"] if results else args.device

    # --- markdown table ---
    md_lines = [
        f"# Benchmark dashboard (device={resolved_device})",
        "",
        "| Config | Accuracy | Speedup |",
        "|---|---|---|",
    ]
    for r in results:
        label_oneline = r["label"].replace("\n", " ")
        acc = "PASS" if r["passed"] else "FAIL"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
        md_lines.append(f"| {label_oneline} | {acc} | {speedup_str} |")
    (OUT_DIR / "dashboard_results.md").write_text("\n".join(md_lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'dashboard_results.md'}")

    # --- bar chart ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["label"] for r in results]
        speedups = [r["speedup"] if r["speedup"] else 0.0 for r in results]
        colors = ["#4C72B0" if r["passed"] else "#C44E52" for r in results]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, speedups, color=colors)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Speedup (baseline / optimized median latency)")
        ax.set_title(f"Optimized Transformer speedup on {resolved_device}")
        for bar, s in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{s:.2f}x", ha="center", va="bottom", fontsize=9)
        plt.xticks(fontsize=8)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "dashboard.png", dpi=150)
        print(f"Wrote {OUT_DIR / 'dashboard.png'}")
    except ImportError:
        print("matplotlib not installed -- skipping chart image "
              "(pip install matplotlib to enable it). Markdown table was "
              "still written.")

    all_passed = all(r["passed"] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
