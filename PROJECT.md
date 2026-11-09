# PROJECT.md — Plan of Record

**Status:** Phase 0 complete — gate decision **PROCEED_TO_BASELINE** on H1: single-launch fusion
with distinct-expert dispatch (2026-11-09). Phase 1 not started.
**Source brief:** [PLAN.md](PLAN.md). This file refines it; where the two disagree, this file wins.

ExpertFlow builds one custom CUDA FP8 MoE decode operator for `Qwen/Qwen3-30B-A3B-FP8` on H100 SXM,
finds a real bottleneck in the existing backend, fixes it, and proves the fix at operator level
(with an optional vLLM integration).

---

## 1. Decisions locked with the owner

| Date | Decision | Choice |
|---|---|---|
| 2026-11-09 | Final deliverable | Portfolio-grade writeup **and** an upstream vLLM PR |
| 2026-11-09 | Phase gates | Stop for owner review at **every** gate; never auto-advance |
| 2026-11-09 | If the hypothesis is already solved upstream | **STOP**, report the evidence, do not auto-pivot |
| 2026-11-09 | GPU budget | Minimize. Target **$30–40**, hard ceiling **$45**, no idle hardware, H100 only when a measurement needs it |
| 2026-11-09 | Repo workflow | Commit and push to `origin/main` as we go; Apache-2.0 |
| 2026-11-09 | Stack | Python 3.12 via `uv` (locked); CUTLASS/CuTe via CMake `FetchContent` at a pinned tag; pinned Docker image for GPU work |
| 2026-11-09 | **Phase 0 gate** | **PROCEED_TO_BASELINE** on H1 — one kernel, distinct-expert dispatch. Bottleneck identified structurally: 4 of 6 launches move <0.5% of the bytes, and pair-wise dispatch moves up to 4x the necessary weight bytes at T=4. M-padding hypothesis retired (Triton uses M=16). Upstream overlap is a coordination risk, not a gate |

**Working mode:** the agent works autonomously inside a phase and stops at each gate. Anything that
changes the target operator, doubles the spend estimate, or invalidates a pre-registered threshold
is a gate-level decision and must be raised immediately, not absorbed.

---

## 2. Operator boundary (the thing being measured)

Full definition: [docs/operator-spec.md](docs/operator-spec.md). Summary:

```
out[T,H] BF16  =  Σ_{k<K} topk_w[t,k] · fp8_moe(x[t], w13[e], w2[e], e = topk_id[t,k])
```

**Included in the timed region:** dynamic per-token-group activation quantization (1×128),
token→expert dispatch/alignment, gate+up grouped GEMM with 128×128 weight block scales,
SiLU(gate)·up, down grouped GEMM, weighted combine over K, BF16 cast,
and any workspace traffic the backend needs between those steps.

**Excluded:** router logits/softmax/top-k, attention, RMSNorm, embedding, sampling.

Both the candidate and the baseline are wrapped by the *same* Python entry point taking the same
tensors, so the boundary cannot drift between them.

---

## 3. Pre-registered success criteria

Fixed **now**, before any measurement exists. Changing one requires a written justification in
this file and owner approval.

| Gate | Metric | Threshold |
|---|---|---|
| P2 PROCEED | Share of complete-operator latency attributable to the chosen bottleneck, T=1 | **≥ 20%** |
| P2 PROCEED | Same share, T=4 | **≥ 15%** |
| P2 PROCEED | Upstream solved? | No merged commit and no funded active PR implementing it |
| P4 COMPETITIVE | Median cold-L2 operator latency vs best compatible baseline, T=1 and T=2 | **≤ 0.85×** |
| P4 COMPETITIVE | Same, T=4 | **≤ 0.90×** |
| P4 COMPETITIVE | T=8 regression guard | **≤ 1.05×** |
| P4 COMPETITIVE | RMS relative error vs FP32-dequantized reference | **≤ 1e-2** |
| P4 COMPETITIVE | Max relative deviation vs vLLM FP8 baseline | **≤ 1e-2**, zero nonfinite values |
| P4 COMPETITIVE | Result reproducible | Same verdict across 3 independent sessions/seed sets |
| P5 GO | End-to-end decode TPOT improvement, fixed workload | **≥ 2%** with no quality regression |

Measurement methodology is pinned in [docs/measurement-protocol.md](docs/measurement-protocol.md)
(graph replay, cold-L2 flush, 1000 iterations, median/p10/p90, clock reporting, launch counts).

---

## 4. Cost policy

The budget is the binding constraint, so it is designed for rather than hoped for.

| Item | Policy |
|---|---|
| Rate assumptions | H100 SXM $2.00–3.00/hr (spot where available); RTX 4090 / L4 Ada ~$0.35/hr; CPU container ~$0.06/hr |
| Session rule | A pod starts **only** after a committed `scripts/gpu/<session>.sh` exists and passes `--dry-run` locally. One session = provision → run → rsync artifacts → destroy |
| Idle time | Zero. No pod is held open for interactive exploration; no pod survives the end of a work block |
| Compilation | Expect no compiled artifacts on the GPU box: use the pinned image, and compile locally-checkable code without a GPU where possible |
| Model weights | Operator work uses synthetic data plus a single-layer weight extract (~604 MB). The 30 GB checkpoint is downloaded **only** if/when Phase 5 starts |
| Baseline kernel | Use the pinned official vLLM image instead of building vLLM from source (source builds cost 30–60 min ≈ $2 per session) |
| Artifacts | Every session rsyncs results back and commits them; nothing important lives only on the pod |
| Ledger | [results/spend.csv](results/spend.csv) updated the same day as the spend |

### Spend plan

| Phase | H100 hours | Est. cost | Notes |
|---|---|---|---|
| P0 Research | 0 | $0 | Local + GitHub read-only |
| P1 Reference/harness | 0 | $0 | Local CPU only; optional $0.10 compile check |
| P2 Baseline + go/no-go | 2.5–3.5 | $6–11 | One session; baseline sweep + Nsight |
| P3 Kernel correctness | 3–4 | $7–12 | Batched build/validate sessions; 1h Ada smoke if useful ($0.35) |
| P4 Optimize | 3.5–4.5 | $8–14 | Profiling-driven, ablation retained |
| P5 vLLM integration | 2.5–3 | $6–9 | Full model + serving benchmark |
| **Total** | **11.5–15** | **$27–46** | Expected **$30–38** at spot/community rates |

The plan absorbs one full re-do of a single session (~$3) without breaching the ceiling.

---

## 5. Environment and toolchain

Local machine facts and the version-pinning checklist live in
[docs/environment.md](docs/environment.md). Key consequences:

- **No local CUDA.** Local work is Python/CPU reference, test data, harness, docs, and review of
  upstream source. CUDA compile/run happens only on a remote pod.
- **Python 3.12 (not the local 3.14)** via `uv`, lockfile committed.
- **Target arch:** `sm_90a` only. No portability work outside H100 until the result is validated.
- **GPU image:** official `vllm/vllm-openai:<pinned tag>` digest, extended with our extension.
  Pinned in Phase 0 after the backend selection is known.

---

## 6. Repo layout

```
expertflow/
├── PROJECT.md              # this file: plan of record, thresholds, budget
├── PLAN.md                 # original brief (immutable)
├── README.md               # public entry point
├── docs/                   # operator spec, measurement protocol, environment
├── research/               # feasibility.md, baseline.md
├── csrc/, include/         # CUDA sources (Phase 3)
├── python/                 # reference.py, test_data.py, baselines/
├── tests/                  # pytest: CPU reference + GPU correctness
├── benchmarks/             # sweep driver, result export
├── integration/vllm/       # Phase 5 only
├── scripts/gpu/            # one script per billable session
└── results/                # spend.csv, benchmarks.jsonl, plots, logs
```

---

## 7. Phase plan and gates

| Phase | Artifact | Gate decision | Est. spend |
|---|---|---|---|
| 0 Research & feasibility | `research/feasibility.md` | PROCEED_TO_BASELINE / REVISE / STOP | $0 |
| 1 Reference, data, harness | `python/`, `tests/`, `benchmarks/`, `results/spend.csv` | Ready for GPU work with no exploratory setup | $0 |
| 2 Hopper baseline | `research/baseline.md` | PROCEED / REVISE / STOP per §3 | $6–11 |
| 3 CUDA kernel | `csrc/`, `include/`, correctness tests | Correct FP8, graph-safe, repeatable | $7–12 |
| 4 Optimize & compare | `results/benchmarks.jsonl`, plots, profiler notes | Competitive per §3 | $8–14 |
| 5 vLLM integration | `integration/vllm/`, serving results | End-to-end effect per §3 | $6–9 |
| 6 Final artifacts | README, technical writeup, traces | Publishes only tied-to-environment numbers | $0 |

Phase 5 runs only if Phase 4 clears its threshold; a negative result is reported as a negative
result, not reframed.

---

## 8. Phase 0 execution plan (complete — outcome below)

**Outcome (2026-11-09):** [research/feasibility.md](research/feasibility.md). Default path on H100 is
Triton — 6 launches, `BLOCK_M=16`, `BLOCK_N=64`, TMA weight loads, **no** alignment kernel at `T ≤ 4`.
Findings: (1) four of the six launches move <0.5 % of the bytes and cost 2-4 µs; (2) pair-wise dispatch
streams an expert's weights once per selecting token, so `T=4` moves up to 151 MB against a 37.75 MB
minimum, while `T=1` gains nothing from dedup by construction; (3) the GEMMs already use TMA and have
~18× compute margin at `M=16`, so raw MMA efficiency is probably not the limiter — a testable
prediction for Phase 2, not an assumption. H1: one kernel, distinct-expert dispatch, fused
quant/SwiGLU/combine. Phase 2 (≈$8) measures the split and carries pre-registered kill criteria.



**Objective:** choose one worthwhile optimization, with evidence, before writing performance code.
**Time-box:** 4–6 focused hours. **Cost:** $0.

### Sources

1. `Qwen/Qwen3-30B-A3B-FP8` — `config.json`, `model.safetensors.index.json`, one layer's tensor
   headers (dtypes/shapes/scales). *Partially done; recorded in operator-spec.*
2. vLLM at a pinned tag — MoE modular-kernel backend selection, the FP8 block-scaled MoE path,
   `moe_align_block_size`, its CUDA/CUTLASS sources, and the Triton fallback.
3. FlashInfer at a pinned tag — CUTLASS fused MoE and TRTLLM-gen MoE paths for small batch.
4. CUTLASS — only the primitives we would reuse (blockwise/grouped GEMM, TMA, wgmma).
5. Merged and open work in all three repos touching small-batch FP8 MoE.

### Questions that must be answered

| # | Question |
|---|---|
| Q1 | Which vLLM tag do we pin, and which backend does it select for `Qwen3MoE` + `fp8` + block scales on H100 at T≤8? |
| Q2 | Exact call sequence and kernel launches for one pre-routed decode step. |
| Q3 | Does the selected path pad M to 128 (or 64)? Where, and how much MMA work does that add? |
| Q4 | Exact activation quantization formula/granularity used by that path, and the scale tensor layout. |
| Q5 | What is the memory floor for the routed weights, and what does the measured split say is winnable? |
| Q6 | Which merged PRs already implement each candidate optimization? Which are open? |
| Q7 | Three candidate bottlenecks, each with operation, hypothesized cost, CUDA remedy, and measurement. |
| Q8 | The single primary hypothesis, its baseline definition, test shapes, and the evidence needed. |

### Stop rule

If, after the overlap audit, the leading hypothesis is already merged upstream or carried by a
funded active PR, do not pivot. Report the data and stop, per §1.

### Deliverable

`research/feasibility.md` with the sections PLAN.md §Phase 0 requires, the pinned commit hashes,
the Q1–Q8 answers, and one explicit decision line.

---

## 9. Assumptions to validate (risk if wrong)

| # | Assumption | Consequence if wrong |
|---|---|---|
| A1 | Runtime fuses gate/up into `w13 = [gate; up]` (rows 0:I gate, I:2I up) | **Confirmed** — Triton consumes `w1 [E, 2I, H]`; the fused tensor is a loader artifact |
| A2 | No shared expert in this checkpoint | **Confirmed** — absent from `config.json` |
| A3 | `topk_weights` are FP32 and pre-normalized (sum ≈ 1) | **Confirmed** — `norm_topk_prob: true` |
| A4 | Activation quant is dynamic per-token-group (1×128) | **Confirmed** — `kFp8Dynamic128Sym` + `per_token_group_quant_fp8` |
| A5 | Baseline accepts pre-computed top-k ids/weights | **Confirmed** — the operator boundary takes `topk_ids`/`topk_weights` |
| A6 | The operator is latency-bound at T≤4 | **Open** — Phase 2 profiler decides |

Each is closed in Phase 0 (A1–A5) or Phase 2 (A6).

---

## 10. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | Concurrent upstream work on the same Triton path | Medium | Treated as coordination risk, not a gate: the claim is structural (distinct-expert dispatch + single launch); related work recorded at Phase 2 and Phase 4 |
| R2 | Kernel complexity exceeds budget/timeline | Medium | Ship simplest-correct version first; optimize only profiled factors; M=16→wgmma incrementally |
| R3 | vLLM API churn breaks integration | Medium | Pin image digest; thin opt-in adapter; Phase 5 is not a prerequisite for the result |
| R4 | FP8 numerics disagree with vLLM semantics | Medium | Replicate the exact quant formula; compare against an FP32-dequantized reference as well |
| R5 | CUDA-graph capture hazards (allocations, syncs) | Medium | Preallocated workspace, no host sync in hot path, capture test in the harness |
| R6 | Spot preemption / pod loss | Medium | Idempotent session scripts, short sessions, artifacts committed off-box |
| R7 | Remote debugging burns paid hours | High | Pre-flight checklist, local numeric emulation, batched `compute-sanitizer` runs |
| R8 | Clock/thermal variance corrupts comparisons | Low-Med | Lock clocks where permitted, record clocks per run, interleave baseline/candidate |

---

## 11. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-09 | Plan of record created; thresholds, cost policy, and Phase 0 plan fixed | Kickoff review with owner |
| 2026-11-09 | Phase 0 closed: hypothesis replaced with single-launch/distinct-expert fusion; PR-overlap demoted to a risk | Kernel-level inspection of vLLM `b6ed63d2ac` |
