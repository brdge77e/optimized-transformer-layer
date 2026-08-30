#!/usr/bin/env python3
"""
Run the official torch_transformer_benchmark.py accuracy+speed harness across
a representative set of shapes and print a summary table (used to generate
the numbers in reports/TECH_REPORT.md).

This just shells out to torch_transformer_benchmark.py once per config and
parses its stdout, so the numbers are exactly what the official grading
script would report.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "torch_transformer_benchmark.py"

# (label, extra_cli_args)
CONFIGS = [
    ("small (bs=4, seq=64, d=128, L=3)",
     ["--batch-size", "4", "--seq-len", "64", "--d-model", "128", "--heads", "8",
      "--ffn-dim", "512", "--layers", "3"]),
    ("default (bs=8, seq=128, d=512, L=6)",
     ["--batch-size", "8", "--seq-len", "128", "--d-model", "512", "--heads", "8",
      "--ffn-dim", "2048", "--layers", "6"]),
    ("large batch (bs=64, seq=32, d=256, L=4)",
     ["--batch-size", "64", "--seq-len", "32", "--d-model", "256", "--heads", "8",
      "--ffn-dim", "1024", "--layers", "4"]),
    ("long seq (bs=2, seq=1024, d=256, L=4)",
     ["--batch-size", "2", "--seq-len", "1024", "--d-model", "256", "--heads", "8",
      "--ffn-dim", "1024", "--layers", "4"]),
    ("causal + padding (bs=8, seq=128, d=512, L=6)",
     ["--batch-size", "8", "--seq-len", "128", "--d-model", "512", "--heads", "8",
      "--ffn-dim", "2048", "--layers", "6", "--causal", "--padding-ratio", "0.3"]),
    ("default + torch.compile",
     ["--batch-size", "8", "--seq-len", "128", "--d-model", "512", "--heads", "8",
      "--ffn-dim", "2048", "--layers", "6", "--compile-user"]),
]

SPEEDUP_RE = re.compile(r"speedup\s*:\s*([\d.]+)x")
BASELINE_RE = re.compile(r"baseline : median=([\d.]+) ms")
OPTIMIZED_RE = re.compile(r"optimized: median=([\d.]+) ms")
SUMMARY_RE = re.compile(r"summary: (PASS|FAIL)")


def run_one(label: str, args: list[str], device: str) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--device", device,
        "--rtol", "0.02", "--atol", "0.002",
        # No --accuracy-trials/--warmup/--repeats/--benchmark-rounds override
        # here on purpose: this script's whole point is reproducing the
        # official script's own numbers, so it must run with the official
        # script's own full-rigor defaults (5 accuracy trials, 20 warmup,
        # 100 repeats, 3 rounds), not a lighter subset. An earlier version
        # of this script overrode these to a much lighter setting (3/5/10/2)
        # -- likely the actual source of several numbers in
        # reports/TECH_REPORT.md that didn't reproduce under full rigor and
        # had to be corrected (see §6.3). Do not reintroduce lighter
        # overrides here; use tests/generate_demo_dashboard.py instead if a
        # quick, explicitly-lower-fidelity snapshot is what's wanted.
        *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = result.stdout
    status = SUMMARY_RE.search(out)
    speedup = SPEEDUP_RE.search(out)
    base_ms = BASELINE_RE.search(out)
    opt_ms = OPTIMIZED_RE.search(out)
    print(
        f"| {label:<42} | "
        f"{status.group(1) if status else 'N/A':<4} | "
        f"{base_ms.group(1) if base_ms else '-':>10} | "
        f"{opt_ms.group(1) if opt_ms else '-':>10} | "
        f"{speedup.group(1) + 'x' if speedup else '-':>7} |"
    )
    if not status or status.group(1) != "PASS":
        print("---- FULL OUTPUT (failure) ----")
        print(out)
        print(result.stderr)


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(f"Device: {device}\n")
    header = (
        f"| {'config':<42} | {'acc':<4} | {'base ms':>10} | {'opt ms':>10} | {'speedup':>7} |"
    )
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for label, args in CONFIGS:
        run_one(label, args, device)


if __name__ == "__main__":
    main()
