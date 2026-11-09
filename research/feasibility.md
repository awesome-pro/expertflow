# Phase 0 — Feasibility

**Date:** 2026-11-09
**Scope:** identify the bottleneck and select one optimization before writing performance code.
**Spend:** $0 (all findings from source inspection and static analysis).
**Decision:** **PROCEED_TO_BASELINE** on H1 — single-launch fusion of the operator (§7).

---

## 0. Summary

The operator has a hard physical floor: at `T=1` with `K=8` distinct experts it must read **37.75 MB**
of expert weights from HBM, which is **11.27 µs** at the H100 SXM peak of 3.35 TB/s. Everything above
that line is overhead or inefficiency, and the job of Phase 0 is to say which of the two dominates.

Three findings, in order of importance:

1. **The kernel that runs is a six-launch pipeline in which four launches move < 0.5 % of the bytes.**
   Dynamic activation quant, `silu_and_mul`, the intermediate quant, and `moe_sum` each touch tens of
   kilobytes and exist only because the operator was decomposed into library-op-shaped pieces. They
   are ~20 % of an estimated 15-18 µs operator at `T=1`.
2. **The dispatch format destroys expert reuse.** For `T ≤ 4` vLLM deliberately skips expert
   alignment and emits one 16-row block per *`(token, expert)` pair*. When two tokens select the same
   expert, that expert's 4.72 MB is streamed twice. At `T=4` the pair-wise path moves 151 MB against a
   minimum of 37.75 MB (single-distribution) — a 4× traffic multiplier that is invisible at `T=1` and
   grows with batch size and routing skew.
3. **Raw GEMM efficiency is probably *not* the problem, and that is a useful result.** The kernel
   already uses TMA tensor descriptors for the weight loads, `BLOCK_N=64` N-tiles and a swap-AB
   `tl.dot` to get a 64-row tensor-core tile out of a 16-row problem. At `M=16` the compute:memory
   ratio is 16 MACs per weight byte against a machine peak of ~295 MACs/byte, so compute has ~18× of
   margin and cannot be the limiter unless Triton's serialized `acc += dot*scale` introduces bubbles.
   The floor is therefore memory-side, and the removable costs are structure, not arithmetic.

The optimization that follows is not "a faster GEMM". It is **one kernel that does the whole operator,
streaming each distinct expert's weights exactly once, with the quantize / SwiGLU / quantize / combine
work absorbed into its prologue and epilogue**. Predicted effect by cell (falsifiable, see §8):

| Cell | Mechanism | Predicted latency win |
|---|---|---|
| T=1 | removes 4 launches + 2 round trips | 15-20 % |
| T=2 uniform | fixed cost shrinks relative to 2× traffic; dedup is only 6 % | 13-20 % |
| T=4 uniform | fixed cost is 5 % of a 45 µs operator; dedup 12 % | 13-20 % |
| T=4 skew / single | dedup removes 55-75 % of the weight traffic | 40-60 % |
| T=8 uniform | must not regress (aligned path already dedups) | 0-8 % |

---

## 1. Model and quantization details (verified 2026-11-09)

| Item | Value | Source |
|---|---|---|
| Repo / revision | `Qwen/Qwen3-30B-A3B-FP8` @ `d206ba732169f29bb77fbf80fc2c4b81d4d30782` | HF API |
| Architecture | `Qwen3MoeForCausalLM`, 48 layers | `config.json` |
| Hidden / MoE intermediate | 2048 / 768 | `config.json` |
| Experts / per token | 128 / 8 | `config.json` |
| Router normalization | `norm_topk_prob: true` (weights pre-normalized) | `config.json` |
| Shared expert | none | `config.json` |
| Quantization | `fp8`, `fmt: e4m3`, `weight_block_size: [128,128]`, `activation_scheme: dynamic` | `config.json` |
| Not quantized | 145 modules: `lm_head`, norms, all `mlp.gate` routers | `config.json` |
| Scale tensors | `weight_scale_inv` (inverse, multiply to dequantize), per expert per projection | index |
| Shards | 7 files, ~30 GB, 37 491 tensors | index |

On disk the two GEMM weights are **three separate projections per expert**, not a fused tensor:

```
experts.E.gate_proj.weight           [768, 2048]  e4m3   + weight_scale_inv [6, 16] f32
experts.E.up_proj.weight             [768, 2048]  e4m3   + weight_scale_inv [6, 16] f32
experts.E.down_proj.weight           [2048, 768]  e4m3   + weight_scale_inv [16, 6] f32
```

Per-expert weight bytes = `2·768·2048 + 2048·768` = **4 718 592 B (4.72 MB)**; scales add 1.15 KB.
The fused `w13 [E,1536,2048]` / `w2 [E,2048,768]` are loader artifacts.

Activation quantization is **dynamic, per token, per 128-element group** (`kFp8Dynamic128Sym`),
produced by a Triton kernel `_per_token_group_quant_fp8`
(`quantization/utils/fp8_utils.py:83`). Weight key is `kFp8Static128BlockSym`.

---

## 2. Pinned versions

| Component | Pin | Note |
|---|---|---|
| vLLM | `b6ed63d2acbe306ff465b979c603d4800c664818` (`main`, 2026-11-09) | local checkout == `origin/main` |
| PyTorch | `2.14.0` | `requirements/cuda.txt:11` |
| CUDA | `13.0.3` | `docker/Dockerfile:25` |
| FlashInfer | `flashinfer-python==0.7.3` | `requirements/cuda.txt:13` |
| CUTLASS | `v4.7.1` | `CMakeLists.txt:517` |
| CuTe DSL | `nvidia-cutlass-dsl[cu13]==4.8.0` | `requirements/cuda.txt:26` |
| GPU | H100 SXM 80GB, sm_90a, 132 SMs, 3.35 TB/s, 50 MB L2 | target |

---

## 3. Which backend actually runs

`select_fp8_moe_backend` (`oracle/fp8.py:415`) builds a priority list and then applies a platform
override (`oracle/fp8.py:129-141`):

```python
if (current_platform.is_device_capability(90)
    and activation_key == kFp8Dynamic128Sym
    and weight_key == kFp8Static128BlockSym):
    if moe_config.moe_parallel_config.ep_size > 1:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.FLASHINFER_CUTLASS)
    else:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.TRITON)
```

Hopper + block-FP8 + **TP=1/EP=1** → **TRITON**, i.e. `TritonExperts` driving
`fused_moe_kernel` (`fused_moe.py:298`). The other block-FP8-capable backends
(`flashinfer_cutlass`, `deep_gemm`, `cutlass`, `marlin`) are reachable via `--moe-backend`, so the
*measured* best backend is still an open question for Phase 2 — but the default, and therefore the
thing to beat, is Triton at six launches.

---

## 4. Execution path at `T ≤ 4` — six launches, two of them doing the work

`_prepare_expert_assignment` (`fused_moe.py:1667`) contains a decode shortcut:

```python
naive_block_assignment = (
    expert_map is None
    and not weights_host_resident
    and num_tokens * top_k_num * 4 <= global_num_experts   # 8, 16, 32 ≤ 128 for T = 1, 2, 4
)
```

So for `T ≤ 4`: **no `moe_align_block_size`**, no sort, no expert grouping. `expert_ids` is the
flattened `topk_ids`, `sorted_token_ids` is `None`, and `EM = T·K·BLOCK_SIZE_M`
(`fused_moe.py:871`). Each `(token, expert)` pair occupies its own `BLOCK_SIZE_M = 16` row block.

| # | Kernel | Work | Where |
|---|---|---|---|
| 1 | `per_token_group_quant_fp8` | 4 KB → 2 KB | `fp8_utils.py:632` |
| 2 | `fused_moe_kernel` GEMM1 (gate‖up) | **25.2 MB weights** | `fused_moe.py:298`, grid `fused_moe.py:872` |
| 3 | `silu_and_mul` | 24 KB → 24 KB | `triton_moe.py:649` |
| 4 | `per_token_group_quant_fp8` (intermediate) | 24 KB → 12 KB | `triton_moe.py:663` |
| 5 | `fused_moe_kernel` GEMM2 (down) | **12.6 MB weights** | `triton_moe.py:686` |
| 6 | `moe_sum` (top-k combine) | 32 KB → 4 KB | `triton_moe.py:770` |

Kernels 3 and 4 are not fused because `VLLM_USE_DEEP_GEMM_E8M0` defaults to 1 (`envs.py:1668`), which
makes `is_deep_gemm_e8m0_used()` true and disables the fused
`silu_and_mul_per_block_quant` alternative (`triton_moe.py:641`). That same default forces the
Triton path onto power-of-two activation scales (an accuracy cost, measured upstream at 0.069 → 0.062
relative error when removed).

**Tiles** come from `get_default_config` (`fused_moe.py:1388-1442`) because no tuned JSON exists for
`E=128, N=768, NVIDIA_H100_80GB_HBM3, block_shape=[128,128]`:

```
BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, GROUP_M=1, num_warps=4,
num_stages=4 (M≤4) / 3 (M=8), SWAP_AB=true (sm90, M<64, N≥64 → utils.py:838)
```

**Grid at T=1**: GEMM1 `8 pairs × 24 N-tiles = 192` CTAs, each streaming 128 KB;
GEMM2 `8 × 32 = 256` CTAs, each 48 KB. With ~40 KB SMEM per CTA all CTAs are co-resident, so the
kernels are bandwidth-bound, not occupancy-bound.

**Weight loads use TMA.** `use_tma_b` (`fused_moe.py:883-899`) is true for both GEMMs
(`B.size(1) % 64 == 0`, `B.size(2) % 128 == 0`, sm90, contiguous), so `b_desc.load(...)` issues bulk
tensor loads. This is important: the obvious "add TMA" improvement is already present.

**At T=8** the heuristic flips (64 pairs × 4 = 256 > 128), and `moe_align_block_size` runs with
`block_size = BLOCK_SIZE_M = 16` (`fused_moe.py:1711`), which *does* group an expert's tokens into one
block. The same operator therefore has two different traffic laws on either side of `T=4`.

---

## 5. Roofline and traffic model

### 5.1 The floor

| Quantity | Value |
|---|---|
| Routed weights, `T=1`, 8 distinct experts | 37.75 MB |
| HBM time at 3.35 TB/s | **11.27 µs** |
| GEMM1 / GEMM2 split | 25.2 MB (7.52 µs) / 12.6 MB (3.76 µs) |
| All non-weight traffic (activations, intermediates, output) | < 100 KB (0.03 µs) |

So at `T=1` the operator is 99.7 % weight streaming. There is no algorithmic way below 11.27 µs; the
only questions are how much of the 11.27 µs the kernel achieves and what is added on top.

### 5.2 The traffic the default path actually generates

The pair-wise dispatch streams full expert weights **once per `(token, expert)` pair**, whereas the
minimum is once per *distinct* expert:

| T | pairs | default traffic | uniform: distinct / ideal | single: distinct / ideal |
|---|---|---|---|---|
| 1 | 8 | 37.75 MB (11.3 µs) | 8 / 37.75 MB (0 %) | 8 / 37.75 MB (0 %) |
| 2 | 16 | 75.5 MB (22.5 µs) | 15.0 / 70.8 MB (6 %) | 8 / 37.75 MB (50 %) |
| 4 | 32 | 151.0 MB (45.1 µs) | 28.3 / 133.6 MB (12 %) | 8 / 37.75 MB (75 %) |
| 8 | 64 | 238 MB (71 µs) | 50.4 / 237.7 MB — *aligned path, already deduped* | 8 / 37.75 MB |

`E[distinct] = E·(1 − (1 − 1/E)^pairs)` for uniform routing. Two consequences:

- At `T=1` **dedup is worth exactly zero** — the 8 selected experts are distinct by construction.
  Any claim that a decode kernel wins at `T=1` by dedup is wrong.
- At `T=4` the pair-wise path moves up to **4× the necessary bytes**, and even under uniform routing it
  moves 13 % more than necessary. Under a realistic 80/20 router the distinct count sits between the
  two columns, so 1.5-3× is the realistic range.

Caveat that only a profiler can settle: the 50 MB L2 may absorb some duplicate streams, so DRAM bytes
saved can be less than bytes saved. §8 measures actual DRAM counters.

### 5.3 Fixed cost

Each launch — even captured in a CUDA graph — costs a start/teardown and a gap. Upstream measured the
gap at **~0.45 µs** for these specific Triton MoE launches. Four kernels move < 0.5 % of the bytes
(quant ×2, SiLU, sum) and carry 4 gaps plus their own minimal execution: **2-4 µs of dead time**, i.e.
15-25 % of a 15-18 µs operator. This is the single largest removable block at `T=1`.

### 5.4 Why the GEMMs are not obviously the problem

At `M=16` (the tensor-core minimum for FP8; `mma.sync` is `m16n8k32` and `wgmma` starts at `m64`),
the kernel performs 16 MACs per weight byte. H100 peak is ~3 784 MAC/cycle/SM against ~12.8 B/cycle/SM
of HBM, i.e. ~295 MAC/byte. Compute therefore needs only **5.4 % of peak tensor throughput** to keep
pace with the memory stream, and swap-AB already reshapes the dot to `64×16×128` to avoid a
`16×64` MMA. Unless Triton's block-scale pattern — `accumulator += tl.dot(b, a) * b_scale * a_scale`
(`fused_moe.py:576`), which forces a wgmma wait and a 1024-element scale-multiply-add **per one of the
16 K-blocks** rather than chaining `acc=` — creates a bubble, the GEMM is memory-bound and its
remaining inefficiency is modest.

That is a testable prediction, not an assumption: it says the operator should measure close to
`11.27/eff + fixed`, and that effort spent on MMA shape is misdirected. Phase 2 measures the tensor
pipe and the DRAM counters separately to confirm or kill it.

---

## 6. The three candidate bottlenecks

### B1 — Decomposed operator: four of six launches are pure overhead
- **Operation:** kernels 1, 3, 4, 6 (activation quant, SiLU·mul, intermediate quant, top-k combine).
- **Why slow:** each touches ≤ 32 KB but pays a full launch, a graph node, and a dependent-kernel
  tail; the pipeline is strictly serial, so their latency adds instead of overlapping.
- **How CUDA fixes it:** do the quantization in the loading CTA's prologue, SwiGLU in the GEMM1
  epilogue, and the weighted combine in the GEMM2 epilogue. One launch, no intermediate in HBM.
- **Upper bound:** ~2-4 µs at every `T` (15-25 % at `T=1`).
- **How to measure:** nsys per-kernel durations + gap histogram; sum of the four kernels' spans.

### B2 — Pair-wise dispatch discards expert reuse across tokens
- **Operation:** `_prepare_expert_assignment`'s `naive_block_assignment` branch
  (`fused_moe.py:1688`) and the resulting per-pair grid (`fused_moe.py:872`).
- **Why slow:** one 16-row block per pair means an expert selected by `m` tokens is streamed `m`
  times. At `T=4` that is up to 4× the DRAM traffic; under uniform routing 13 %; under skew 1.5-3×.
- **How CUDA fixes it:** a 128-entry histogram over `topk_ids` in shared memory, then a work queue
  over `(distinct expert, N-tile)` so each distinct expert's weights are read exactly once, with all
  of its token rows gathered into one 16-row MMA tile.
- **Upper bound:** 0 % at `T=1`; up to 75 % at `T=4` single, 12 % at `T=4` uniform.
- **How to measure:** ncu `dram__bytes_read` per GEMM vs `pairs × 4.72 MB`; and a runtime monkey-patch
  that forces the aligned path (`moe_align_block_size`) at `T ≤ 4` to isolate the dispatch effect
  without writing a kernel (this is a pure-Python patch — no rebuild, §8 E4).

### B3 — Streaming efficiency of the two GEMMs
- **Operation:** `fused_moe_kernel` at `M=16`, `N=64`, `K=128` with TMA weight loads.
- **Why slow (hypothesis):** unknown achieved DRAM fraction; possible contributors are the per-K-block
  scale epilogue serialization and 128-byte-per-row weight segments (`W` rows are contiguous in `K`,
  so a 64×128 tile is 64 disjoint 128 B segments).
- **How CUDA fixes it:** deeper K-pipelining with larger TMA boxes, and folding the scale product into
  the epilogue of a wider accumulator so 16 separate wgmma waits become 2-4.
- **Upper bound:** `11.27/eff`; at 70 % efficiency it is 16.1 µs (+4.8 µs), at 85 % it is 13.3 µs.
- **How to measure:** ncu `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tensor*`,
  and kernel duration vs 7.52/3.76 µs.

**Ranking:** B1 is certain and worth ~15-25 % at `T=1`; B2 is certain but only material at `T ≥ 2` with
collisions; B3 is unknown and probably small. They are not exclusive — a single fused kernel with
expert-major scheduling addresses all three, and the writeup can attribute per cell. That is what §7
selects, with B1 as the headline mechanism and B2 as the scaling mechanism.

---

## 7. Selected optimization

> **H1 — Replace the six-launch, pair-wise operator with one kernel that dispatches over distinct
> experts and does quantize / GEMM1 / SwiGLU / quantize / GEMM2 / weighted-combine inside it, so each
> distinct expert's weights are read exactly once and no intermediate touches HBM.**

Why this one:
- It is the only option whose win does not depend on an unknown (B3) and scales with the workload
  (B2), while still paying at `T=1` (B1).
- It changes *structure*, not numerics: quantization stays dynamic per-token-group FP8, weights stay
  128×128 block-scaled, accumulation stays FP32. Accuracy must land within the same tolerance as the
  baseline, and the pow2-scale rounding of the default path is not inherited (that is a free
  accuracy improvement, not a speed claim).
- It is measured honestly per cell: at `T=1` it is a fixed-cost win; at `T=4` skewed it is a traffic
  win; at `T=8` it must at least tie the aligned path or the result is reported as a regression.

Design sketch (refined in Phase 3, not frozen here):

```
launch 1 (single kernel, persistent CTAs = SM count or 2×SM)
  prologue  : quantize x[T,2048] -> fp8 + per-128 scales        (one CTA, <1 µs)
              histogram topk_ids -> distinct experts + row lists (shared/global scratch)
  GEMM1     : work queue over (expert, N-tile of 1536), TMA weight boxes 128x128,
              m16n8k32 MMA, FP32 accumulate per 128-K block with (w_scale*a_scale) applied
              per block, epilogue = SiLU(gate)*up and quantize to fp8 group-128
  GEMM2     : work queue over (expert, N-tile of 2048) consuming GEMM1's output,
              epilogue = multiply topk_weights and accumulate into Y via deterministic
              per-expert partials (no atomics on the critical path)
```

Two-launch fallback (GEMM1 and GEMM2 with fused epilogues) keeps the design simple if a device-wide
barrier proves fragile inside a CUDA graph; it loses only ~0.45 µs against the single-launch variant,
so the ablation can settle it cheaply.

**Pre-registered predictions** (used as the Phase 2/4 go/no-go in §8):

| Cell | Predicted median latency vs default baseline |
|---|---|
| T=1 uniform | 0.80-0.85× |
| T=2 uniform | 0.80-0.87× |
| T=4 uniform | 0.80-0.87× |
| T=4 skew80 | 0.40-0.60× |
| T=8 uniform | 0.92-1.00× (no regression) |

If the `T=1` prediction fails while `T=4` holds, H1 is refuted as stated (it was a fixed-cost claim)
and the writeup must say so.

---

## 8. Phase 2 measurement plan (≈2.5-3.5 h H100, $6-11)

One batched session, one script, digest-pinned image, artifacts committed before teardown.

| # | Experiment | Answers | Cost |
|---|---|---|---|
| E1 | Environment capture: GPU UUID, driver, clocks, `nvidia-smi -q`, image digest, versions, measured HBM bandwidth via `bandwidthTest` | what the numbers are tied to | 2 min |
| E2 | Default-path operator latency, `T ∈ {1,2,4,8}` × `{uniform, skew80, single, repeat}` × 3 seeds, cold-L2, 1000 graph replays × 30 trials | the denominator | 20 min |
| E3 | nsys + ncu on the `T=1` and `T=4` default path: per-kernel span, start-to-start gaps, `dram__bytes_read`, DRAM % of peak, tensor-pipe active % | tests B1 and B3 | 40 min |
| E4 | **Monkey-patch A/B:** force `moe_align_block_size` at `T ≤ 4` (pure Python patch of `triton_moe._prepare_expert_assignment`, no rebuild) and re-run E2 | isolates B2 without writing a kernel | 20 min |
| E5 | Forced backends `{flashinfer_cutlass, cutlass, deep_gemm}` at the same cells | fixes the "best applicable baseline" | 20 min |
| E6 | Empty-chain micro-benchmark: 6 no-op launches in a graph | converts the 0.45 µs estimate into a number for this image | 5 min |

**Pre-registered go/no-go (no post-hoc changes):**

| Observation | Action |
|---|---|
| E4 shows ≥ 20 % gain from forced alignment at `T=4` | B2 confirmed — kernel justified |
| E3 shows the four tiny kernels + gaps ≥ 2.5 µs at `T=1` | B1 confirmed |
| E3 shows both GEMMs ≥ 85 % of measured peak BW **and** B1 < 1.5 µs | **STOP** — no headroom; publish roofline negative result |
| E5 finds a non-default backend ≥ 10 % faster than Triton | re-baseline against it before any kernel claim |
| operator ≤ 13 µs at `T=1` on the best baseline | **STOP** — within 15 % of the floor |

---

## 9. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Concurrent upstream work on the same Triton path (launch fusion, scale formats, SM90 fused MoE). Treated as a coordination risk, not a gate: our claim is structural (distinct-expert dispatch + single launch), and any upstream overlap is recorded in the writeup as related work | re-check at Phase 2 and Phase 4 |
| R2 | L2 absorbs duplicate streams, shrinking B2's real effect | E3 measures DRAM bytes; E4 measures the dispatch A/B directly |
| R3 | Single-kernel device-wide barrier is fragile under CUDA graph capture | two-launch fallback with fused epilogues; ablation decides |
| R4 | Numerics diverge from the baseline's pow2 scales | exact reference in `docs/operator-spec.md` §6; compare against R2 (quantized FP32 reference) |
| R5 | The `T=8` aligned path already dedups, so our kernel must at least tie it | T=8 is an explicit regression gate |
| R6 | `M=16` padding is the tensor-core minimum; no further MMA win is available | accepted; compute margin analysis (§5.4) says it does not matter |

---

## 10. Decision

**PROCEED_TO_BASELINE** on H1 (single-launch, distinct-expert fusion), with the Phase 2 kill criteria
of §8 pre-registered.

The bottleneck is identified by construction, not by analogy: four of six launches move < 0.5 % of the
bytes (B1), and the dispatch format moves up to 4× the necessary weight bytes on colliding routes (B2).
Both are structural properties of the current implementation that I can point at in the source, and
both are removable by one kernel. The remaining unknown (B3, GEMM streaming efficiency) is measured
before any kernel work is funded.

If Phase 2 shows the operator is already within 15 % of the 11.27 µs + irreducible-overhead estimate,
the project stops and publishes the measurement instead of a kernel — that outcome is pre-agreed, not
improvised.
