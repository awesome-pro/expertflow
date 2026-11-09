# ExpertFlow

A from-scratch CUDA FP8 MoE decode kernel for `Qwen/Qwen3-30B-A3B-FP8` on NVIDIA H100 (SM90a),
with a measured comparison against the best applicable production backend.

**Status: Phase 0 complete — go/no-go is `PROCEED_TO_BASELINE` on a narrowed hypothesis.
No kernel exists yet, and no GPU time has been spent.**

Phase 0 found that the default decode path on H100 is vLLM's Triton fused-MoE — six launches,
`BLOCK_M=16`, TMA weight loads, no alignment kernel at T≤4 — that the routed weights impose an
11.27 µs HBM floor at T=1, that four of those six launches move under 0.5 % of the bytes, and that the
pair-wise dispatch streams an expert's weights once per selecting token (up to 4× the necessary bytes
at T=4, exactly 0× waste at T=1). The selected optimization is one kernel with distinct-expert
dispatch and fused quantize/SwiGLU/combine.

## The operator

One pre-routed mixture-of-experts layer step, the part of decode that is neither attention nor
routing:

```
y[t] = Σ_{k<K} topk_w[t,k] · ( SiLU(x[t]·W_gate[e]) ⊙ (x[t]·W_up[e]) ) · W_down[e],   e = topk_id[t,k]
```

`H=2048`, `I=768`, `E=128`, `K=8`, FP8 E4M3 weights with 128×128 block scales, dynamic FP8 activation
quantization, BF16 I/O, `T ∈ {1,2,4}` tokens per step. Timed end to end including quantization,
expert dispatch, both grouped GEMMs, activation, and combine.

## Why

At `T=1` this operator must stream ~38 MB of routed expert weights per layer, which is ~11 µs of pure
HBM time on H100. Any latency far above that floor is overhead — padding, launch cost, scheduling, or
wasted MMA work — and that gap is the project's target. The specific bottleneck is selected from
measurement, not assumed: see [research/feasibility.md](research/feasibility.md) (Phase 0 deliverable).

## Plan of record

- [PROJECT.md](PROJECT.md) — scope, pre-registered success thresholds, cost policy, phase gates
- [docs/operator-spec.md](docs/operator-spec.md) — exact contract and weight layout
- [docs/measurement-protocol.md](docs/measurement-protocol.md) — how every number is produced
- [docs/environment.md](docs/environment.md) — toolchain and remote session runbook
- [PLAN.md](PLAN.md) — original brief

## Constraints

- One GPU: H100 SXM 80GB, `sm_90a` only.
- One model shape, one precision configuration.
- Budget target of $30–40 of GPU rental for the whole project; every session is scripted, batched,
  and destroyed afterwards ([results/spend.csv](results/spend.csv)).

## Layout

```
csrc/ include/        CUDA kernel sources (Phase 3)
python/               reference implementation, test-data generators, baseline adapters
tests/                CPU reference tests + GPU correctness tests
benchmarks/           sweep driver and JSON result export
scripts/gpu/          one script per billable GPU session
research/             feasibility.md, baseline.md
results/              benchmarks.jsonl, spend.csv, plots, logs
integration/vllm/     opt-in vLLM integration (Phase 5 only)
```

## Local development

```bash
uv sync --extra dev --extra reference
make test-cpu
```

CUDA execution requires a remote H100; see [docs/environment.md](docs/environment.md) §4.

## License

Apache-2.0.
