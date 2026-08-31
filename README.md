# Optimized Transformer Layer — GPU Kernel Hackathon Submission

**The claim this repo tests:** most transformer-optimization advice — fuse
QKV, call `scaled_dot_product_attention`, drop unnecessary `.contiguous()`
calls — is developed and validated once, on CUDA, then applied everywhere
else on faith that it transfers. That faith doesn't always hold. Profiling
and re-validating the same "safe" optimizations on a second real backend
(Apple Silicon/MPS) found three cases where it was measurably wrong, and
profiling *again* on real CUDA hardware — rather than assuming the MPS
findings would carry over either — confirmed which of those held and which
didn't (§6). That's the hackathon's own framing question — how
AI-assisted profiling finds a workload's real bottlenecks and produces an
implementation tuned to specific GPU hardware — answered across three
backends, not assumed from one. Full analysis: `reports/TECH_REPORT.md`.

## Why this matters beyond the benchmark

This is validated end-to-end on real CUDA hardware (a Colab Tesla T4) —
the actual grading target: the hackathon's own updated benchmark script
resolves its device as `"cuda" if torch.cuda.is_available() else "cpu"`,
with no other backend in the loop (tech report §13). Every optimization
here holds on CUDA, with real, GPU-measured speedups up to 3.22x (§4).

The more differentiated part of this submission is *how* those
optimizations were found and validated: this was originally built and
profiled on Apple Silicon (a MacBook Pro, via PyTorch's MPS backend) before
CUDA hardware was available, on the premise that most transformer-
optimization advice — fuse QKV, use `scaled_dot_product_attention`, skip
unnecessary `.contiguous()` calls — is written and validated against CUDA
and rarely double-checked elsewhere. That MPS pass surfaced two standard,
CUDA-safe optimizations that were measurably *wrong* on MPS: skipping
`.contiguous()` before attention (helps on CPU/CUDA, hurts on MPS — §6.1)
and passing an all-valid mask through as-is (silently forces a slow,
unfused attention path on MPS — §6.3). Both were found by profiling the
actual device, not by reasoning from CPU numbers or general knowledge —
and both turned out to generalize to CUDA, where the mask fix is an even
*bigger* win than it was on MPS (§4). The one place the two backends gave
opposite verdicts: a hand-written flash-attention kernel
(`pytorch_flash_attention.py`) is ~2x slower than the library's attention
op on MPS but 25x–300x slower on CUDA, because CUDA's SDPA has a real
fused kernel to lose to and MPS's doesn't (§10) — same code, two backends,
both measured, not one assumed from the other.

## Who this is useful to beyond this submission

Self-attention isn't specific to this hackathon's fixed transformer layer —
it's the mechanism the problem statement's own background section names as
the core of NLP, computer vision, speech, recommendation, and LLM systems
(§3.1).

- **Anyone deploying transformer inference outside a CUDA datacenter**
  (on-device Apple Silicon inference is real and growing — MLX, llama.cpp's
  Metal backend, on-device Apple Intelligence) gets two concrete,
  reproducible examples of standard, CUDA-safe optimizations that are
  actively wrong on MPS, plus the general lesson: verify per backend,
  don't assume an optimization transfers. In capacity terms, not just
  x-factors: the 2.08x–3.22x speedups measured on CUDA (§4) mean roughly
  half to over two-thirds fewer GPU-hours for the same serving throughput,
  or more than double to over triple the request volume on the same fleet,
  at this workload's shape range.
- **A portable pre-deployment checklist**, extracted from §6, for anyone
  serving attention-based models across more than one backend:
  - `.contiguous()` before an attention call is cheap on CPU/CUDA but can be
    actively slower on MPS (§6.1) — profile per backend, don't assume.
  - A boolean `attn_mask` vs. an additive float mask can select different
    SDPA kernel paths entirely (§6.3) — check what your backend's SDPA
    actually dispatches to, not just what the API accepts.
  - An "all valid" padding mask that's a real tensor rather than Python
    `None` can silently defeat an `is_causal` fast path (§6.3) — verify
    what your data pipeline actually emits when there's no padding.
  - `torch.compile` regressed 3x on MPS in this PyTorch version but helped
    on CUDA (§6.2) — benchmark it per backend before enabling by default.
- **Anyone deciding whether to hand-write a CUDA/Triton kernel instead of
  calling a library op** gets a real, measured answer rather than a rule
  of thumb: on CUDA, SDPA's existing fused kernel wins by 25x–300x over a
  correct hand-written alternative; on a backend with no fused kernel at
  all (MPS), the gap narrows to ~2x–2.5x. That's a reusable data point for
  the actual question most engineers face, not a hackathon-specific result.
- **The hackathon's own organizers** get two concrete findings about their
  published grading materials: the newly-added Appendix's row 14 cannot be
  executed by the fixed baseline reference on any existing hardware
  (~20.5 TB for one tensor, §13), and row 6 (batch=10,000) exercises a real,
  reproducible non-determinism bug in PyTorch's MPS backend that would
  affect grading fairness for any participant tested on Apple Silicon —
  both documented precisely enough to act on.
- **The bonus Triton kernel's 48/48 result is independently reproducible
  in about a minute** via a free Colab GPU
  (`notebooks/verify_triton_kernel.ipynb`).

## Project overview

**How this addresses the problem statement:** the assignment asks for GPU
kernels that implement a fixed transformer layer, pass the provided
correctness tests (`rtol<0.02, atol<0.002`), and improve runtime.

- Profiled `BaselineTransformer` with `torch.profiler` to find the real
  bottlenecks (linear projections, redundant `.contiguous()` copies, a
  4-kernel attention chain — tech report §2), then fused QKV into one GEMM
  and replaced the explicit attention chain with
  `torch.nn.functional.scaled_dot_product_attention`.
- Found and fixed a bf16 precision issue where the fused path drifts from
  the reference's exact rounding trace; the fast path is gated to fp32 only,
  with a bit-exact fallback for bf16/fp16 (§5). That fallback's own
  "bit-exact" claim was itself wrong until a later pass caught it: it was
  silently reusing the fast path's MPS-only `.contiguous()` skip, so on
  CPU/CUDA it ran on strided tensors and rounded differently in bf16 —
  failing 4/72 shape-matrix configs on this Mac's CPU backend specifically,
  despite passing cleanly on every GPU backend already validated. Fixed
  with a dedicated always-contiguous fallback path, re-verified 72/72 (§5).
- Validated end-to-end on real GPU hardware (Apple M3 Pro via MPS) and found
  two backend-specific regressions this way: the `.contiguous()` optimization
  above, which is backwards on MPS (§6.1), and `torch.compile`, which is a 3x
  regression on MPS in this PyTorch version, confirmed via
  `torch._dynamo.explain` to be a backend codegen issue rather than a bug in
  my own caching (§6.2).
- The largest win: SDPA on MPS was spending 44% of total time in an unfused
  fallback because the benchmark's data generator passes an all-`True`
  tensor for "no padding" instead of Python `None`, defeating the fast path.
  Fixed by detecting an all-valid mask and routing it through the real fast
  path, plus a smaller additive-vs-boolean mask fix (§6.3).
- Tested three further optimization ideas after that and rejected two:
  forcing bf16/fp16 independently failed on MPS too (§5); `torch.jit.trace`
  gave a real 4.5% speedup but was proven unsafe — a traced graph reused on
  a differently-masked input silently corrupted output
  (`max_abs_diff=4.06`) — and was not shipped (§7).
- **Re-validated everything above on a real CUDA GPU (Colab Tesla T4)** —
  not just MPS, and not just once. Every fix holds, and most shapes do
  substantially better on CUDA than MPS: **2.08x–2.11x** on small shapes
  (vs. 1.20x on MPS), **3.16x–3.22x** on long sequences (vs. 1.84x–1.89x).
  Both were re-checked twice after regenerating the results dashboard
  turned up a mismatch with the headline table — the originally published
  1.93x and 2.06x both understated the real result and have been corrected
  (§6.3). The one exception is large batch, which lands at 1.13x–1.15x on
  both backends rather than improving on CUDA — an earlier 1.49x reading
  for that config also didn't reproduce, in the opposite direction, and was
  corrected the same way (§6.3). Separately, `torch.compile` regresses 3x
  on MPS but is a genuine 1.16x win on CUDA, confirming that MPS result is
  an Inductor/MPS-backend issue, not a flaw in the design (§6.2).
- Added a hand-written Triton kernel (fused scale+mask+softmax) as a bonus.
  It requires CUDA, which neither dev environment has — written and
  syntax-checked only during development, then **verified on a real CUDA
  GPU** via `notebooks/verify_triton_kernel.ipynb`: **48/48 configurations
  pass** on a Colab Tesla T4 (§9).
- Wrote a second, independent custom kernel that doesn't have that gap:
  `pytorch_flash_attention.py` implements tiled attention with online
  softmax (the flash-attention algorithm) in plain PyTorch, so it's
  verifiable on my own hardware without waiting on CUDA access.
  Correctness: **140/140 configurations pass on CPU, MPS, and CUDA**
  (`tests/test_flash_attention_pytorch.py`). Performance is the clearest
  finding in this submission for *when custom kernel-writing is worth it*:
  ~2x–2.5x slower than SDPA on MPS after six rounds of profiling-driven
  fixes (§10), but **25x–300x slower on CUDA** — because CUDA's SDPA
  dispatches to a real, hardware-optimized fused flash-attention kernel
  that no hand-written composition can approach, while MPS's SDPA has no
  fused kernel at all, so the MPS comparison was always "this vs. an equally
  unoptimized fallback." Same kernel, same correctness, two backends, two
  honestly different verdicts (§10).
- The hackathon published an official "Appendix: Test Shapes" (27 Aug 2026)
  mid-submission — 14 disclosed grading configurations. 12/14 pass cleanly
  on MPS (13/14 on CUDA — row 6's non-determinism below is MPS-specific);
  the other two surfaced real findings, not gaps: row 14
  (seq_len=100,000) is mathematically infeasible for the *fixed baseline
  formula* to run at all (~20.5 TB for one tensor), confirmed directly, not
  just calculated; and combining that row with padding would have crashed
  my *own* optimized code too, via a mask-construction bug that
  reintroduced the exact `O(seq_len²)` memory cost flash-attention exists
  to avoid — real on CUDA (the confirmed grading backend), moot on MPS/CPU
  where baseline crashes first anyway. Fixed with a threshold-gated route
  to the tiled kernel above, verified correct and non-regressive against
  the full existing test suite (§13). Row 6 (batch=10,000) independently
  surfaced a **non-determinism bug in PyTorch's MPS backend** — reproduced
  in the unmodified baseline reference, confirmed absent on CPU, and
  re-verified in a second, separate run designed to rule out a simpler
  alternative explanation (§13).

**Development tools:** Claude (Anthropic) via an agentic coding sandbox with
bash/Python execution, used interactively to write, profile, test, and
iterate — plus a MacBook Pro (Apple M3 Pro) for MPS GPU validation and a
free Google Colab (Tesla T4) session for CUDA GPU validation.

**APIs used:** none — pure local PyTorch compute, no external network APIs.

**Libraries and frameworks:** PyTorch (`torch`, `torch.nn`,
`torch.nn.functional`, `torch.profiler`, `torch.compile`), Triton (bonus
kernel), matplotlib (charts), Python standard library.

**Datasets and assets:** none — correctness and performance are evaluated on
synthetically generated random tensors (`generate_random_case`), per the
problem statement.

## Repository layout

```
torch_transformer_benchmark.py   # official benchmark script with
                                  # UserOptimizedTransformer replaced by the
                                  # optimized implementation
demo.py                          # short end-to-end demo: inference, sample
                                  # predictions, accuracy + speedup, PNG chart
triton_kernels.py                # bonus custom Triton kernel (CUDA-only,
                                  # see reports/TECH_REPORT.md §9)
pytorch_flash_attention.py       # second bonus kernel: hand-written
                                  # flash-attention in plain PyTorch, runnable
                                  # on CPU/MPS/CUDA (see §10)
tests/
  test_shape_matrix.py           # 72-config correctness sweep
  test_appendix_shapes.py        # the hackathon's own published Appendix
                                  # test shapes, with OOM-safe handling and
                                  # per-config peak-memory reporting (§13)
  test_triton_kernel.py          # Triton kernel validation (CUDA-only)
  test_flash_attention_pytorch.py  # 140-config sweep for the flash-attention kernel
  run_benchmark_suite.py         # benchmark across representative shapes
  generate_demo_dashboard.py     # regenerates reports/dashboard.png
reports/
  TECH_REPORT.md                 # full analysis, results, limitations
  dashboard.png, dashboard_results.md  # generated benchmark summary
notebooks/
  verify_cuda_benchmarks.ipynb   # one-click Colab check for the §4 headline
                                  # speedup numbers
  verify_triton_kernel.ipynb     # one-click Colab check for the Triton kernel
requirements.txt
```

## Setup and installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to reproduce results

```bash
# Official grading-style run:
python3 torch_transformer_benchmark.py --device auto

# With causal masking + padding:
python3 torch_transformer_benchmark.py --device auto --causal --padding-ratio 0.3

# Full correctness sweep across shapes/dtypes:
python3 tests/test_shape_matrix.py --device auto --verbose

# Benchmark table across representative shapes:
python3 tests/run_benchmark_suite.py mps    # or cuda / cpu

# The hackathon's own published Appendix shapes, run verbatim
# (OOM-safe: row 14 reports a clean error instead of crashing):
python3 tests/test_appendix_shapes.py --device auto --verbose

# Second bonus kernel: correctness sweep + benchmark against SDPA
python3 tests/test_flash_attention_pytorch.py --device auto
python3 pytorch_flash_attention.py --device auto

# CUDA-only: validate the bonus Triton kernel:
python3 tests/test_triton_kernel.py
```

**Don't have a CUDA machine?** Everything above (plus the Triton kernel) has
already been verified on a free Colab GPU — see `reports/TECH_REPORT.md` §4,
§6, §9, §10 for the results — but you can reproduce that yourself in about a
minute:

- **Main benchmark table (§4's headline 1.13x–3.22x speedups):**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brdge77e/optimized-transformer-layer/blob/main/notebooks/verify_cuda_benchmarks.ipynb)
  — clones this repo and runs `tests/run_benchmark_suite.py cuda` at the
  official script's full-rigor defaults; six configs, each should print
  `PASS` with a speedup close to what's in §4.
- **Bonus Triton kernel:**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brdge77e/optimized-transformer-layer/blob/main/notebooks/verify_triton_kernel.ipynb)
  — clones this repo and runs `tests/test_triton_kernel.py`; every row
  should print `PASS`, matching the 48/48 already recorded in the tech
  report.

**Environment:** built in a CPU-only sandbox, then validated end-to-end on
two independent real GPUs — a MacBook Pro (Apple M3 Pro) via PyTorch's MPS
backend, and a free Colab session (Tesla T4) via CUDA. Every number in
`reports/TECH_REPORT.md` §4, §5, §6, §9, and §10 is measured on one or both
of those, not estimated from CPU or reasoned to generalize. What's still
unverified: genuine tensor-core dispatch specifically (the T4 is Turing
architecture and doesn't support TF32 — an Ampere-or-newer GPU is needed to
check that particular claim) — see tech report §8 and §11.

## Limitations and what I'd improve with more time

- **Genuine tensor-core dispatch is still unverified** — the Colab Tesla T4
  used for CUDA validation is Turing architecture, which doesn't support
  TF32 at all (Ampere-or-newer only), so this one claim needs different
  hardware to check, not just "any CUDA GPU" (§8).
- **The fp16/bf16 fast path is disabled by default, confirmed necessary on
  all three tested backends** — forcing it (`TRANSFORMER_FORCE_FAST_PATH=1`)
  now fails independently on CPU, MPS, *and* CUDA (40/72 shape-matrix
  configs on both MPS and the T4, §5). Correctness-first choice; leaves a
  reduced-precision speedup on the table, but it isn't a gap left
  unverified — it's a safety gate proven necessary everywhere it's been
  checked.
- **The Triton kernel fuses only scale+mask+softmax**, not the full
  attention op, so it doesn't get flash-attention's memory-bandwidth
  advantage. A full blockwise flash-attention kernel is the natural next
  step — now that the fusion kernel itself is CUDA-verified (§9), and given
  that a compiled Triton kernel doesn't carry the eager-mode
  PyTorch dispatch penalty that sank the hand-written version in §10.
- **Measured speedups: 1.15x–1.20x (MPS) / 1.13x–1.16x (CUDA) on
  short/default shapes, up to 1.84x–1.89x (MPS) / 3.16x–3.22x (CUDA) on
  long sequences** (§4/§6.3) — real, GPU-measured numbers on two
  independent backends, not a single-backend result assumed to generalize.
- **The hackathon's own published Appendix (row 14, seq_len=100,000)
  cannot be run end-to-end by the fixed baseline formula on any hardware
  that exists** (~20.5 TB for one tensor, confirmed directly, §13) — not a
  limitation of this submission specifically, but worth knowing before
  attempting to reproduce every appendix row literally. If reproducing on
  Apple Silicon: row 6 (batch=10,000) intermittently fails accuracy due to
  a non-determinism bug in PyTorch's MPS backend itself, reproduced in the
  unmodified baseline reference — re-running it, or using CUDA/CPU instead,
  is the correct response, not assuming a defect (§13).

## Team member contributions

Solo submission; implementation, profiling, testing, and this report were
developed interactively with Claude (Anthropic), as described in
`reports/TECH_REPORT.md` §12.
