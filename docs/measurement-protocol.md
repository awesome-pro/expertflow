# Measurement protocol (frozen before Phase 2)

Any number that appears in `research/baseline.md`, `results/benchmarks.jsonl`, the README, or the
final writeup is produced by this protocol. Changes require a dated entry in §7.

## 1. What is timed

The call to `moe_decode(...)` (see [operator-spec.md](operator-spec.md) §1) — the complete
pre-routed operator — for candidate and baseline alike. Sub-kernel timings are diagnostic only and
are always reported as such.

## 2. Harness

- One Python entry point, `benchmarks/bench.py`, with `--impl {baseline,candidate}` and
  `--tokens/--dist/--seed`. Both implementations receive the *same tensor objects* (same storage),
  so no materialization or cast sits inside one but outside the other.
- Inputs are generated once from a fixed seed and saved to `fixtures/<shape>-<dist>-<seed>.pt`
  (gitignored, regenerated deterministically). A SHA-256 of each tensor is recorded in the result row.
- Warmup: 50 iterations outside the timed region.
- Timed region: **CUDA Graph replay** of the operator call, 1000 replays per trial, 30 trials.
  The graph is captured once per (impl, shape) after warmup, on the default stream.
- Between trials: L2/weight-cache invalidation by zeroing a 256 MB buffer (outside the timed region).
  **Cold-L2 is the primary metric**; a warm-cache number is also recorded and labelled.
- Per-trial latency = elapsed / 1000 (one operator call). Reported: median, p10, p90, min, std.
- Kernel launch count per operator call is recorded (CUDA profiler / CUPTI) as a diagnostic, plus
  a separate host-side "launch overhead" run without graph capture.

## 3. GPU state

- GPU: H100 SXM 80GB, exclusive, TP=1.
- Clocks: attempt `nvidia-smi -lgc <max>`; if not permitted, record actual SM/HBM clocks before and
  after each trial. Do not compare runs taken at materially different clocks; re-run instead.
- Record per session: driver, CUDA runtime, image digest, `nvidia-smi -q` excerpt, GPU UUID/serial,
  measured HBM bandwidth (`bandwidthTest`), GPU and HBM temperature during the run.

## 4. Workloads

| Dimension | Values |
|---|---|
| Tokens | 1, 2, 4 (primary); 8 (regression) |
| Routing distributions | `uniform` (iid categorical over 128), `skew80` (Zipf-like 80/20 over 8 hot experts), `single` (all tokens → same 8 experts), `repeat` (all tokens → one expert) |
| Seeds | 3 independent seeds per cell |

`T=1` combined with `repeat` and `single` produces the extreme few-distinct-expert cases; `uniform`
at T=8 is the most distinct-expert case. Report every cell; do not cherry-pick.

## 5. Comparison discipline

- Interleave baseline and candidate within a session (A/B/A/B) to bound clock drift.
- Report absolute latency in µs, never only a ratio.
- Report the ratio with its distribution across trials, not a single best-case ratio.
- If a cell regresses, it is reported in the same table with the same prominence.
- Any speedup claim must be reproducible in an independent session; otherwise it is labelled
  "not reproduced".

## 6. Environment block (must accompany every published number)

```
GPU            H100 SXM 80GB <uuid>
Driver / CUDA  <versions>
Image          vllm/vllm-openai:<tag>@sha256:<digest>
vLLM commit    <sha>   FlashInfer commit <sha>   CUTLASS tag <tag>
Python/PyTorch <versions>
Clocks         SM <MHz>  MEM <MHz>   (locked: yes/no)
Workload       T=<n> dist=<name> seed=<n>
Metric         median of 30 trials x 1000 graph replays, cold-L2
```

## 7. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-09 | Protocol frozen | Pre-registration before any GPU spend |
