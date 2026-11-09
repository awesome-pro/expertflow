# Phase 0 — Feasibility

**Date:** 2026-11-09
**Scope:** select one worthwhile optimization before writing performance code; spend $0 GPU.
**Decision:** **PROCEED_TO_BASELINE**, with the hypothesis revised (see §10).

---

## 0. Summary in one page

The default single-GPU path for this model on Hopper is **not** CUTLASS and **not** DeepGEMM. It is
vLLM's **Triton** fused-MoE: six kernel launches, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`,
`BLOCK_SIZE_K=128`, and a per-`(token, expert)`-pair grid with no alignment kernel at `T ≤ 4`.

Three findings drive the decision:

1. The plan's implicit assumption that decode pads `M` to 128 is **false for the default path**.
   Triton runs `M=16` tiles and skips `moe_align_block_size` entirely for `T*K*4 ≤ E` (i.e. T ≤ 4).
   That hypothesis is dead on arrival.
2. The measured cost of a kernel launch gap on this path is **~0.45 µs, even inside a CUDA graph**
   (vLLM PR #64846), and six launches exist. Launch overhead is therefore a real 2.7 µs, ~15-20 % of
   a ~15 µs operator — **but it is already carried by two open PRs** (#64846 PDL, #65794 fused
   SiLU+quant). Building a fused kernel to win that same 2.7 µs is not a differentiated project.
3. What remains unclaimed is **GEMM streaming efficiency at M ≤ 16**: whether the Triton pair
   actually reaches the 11.3 µs HBM floor for this operator on H100, or sits at 50-70 % of peak
   bandwidth because of 16-row MMA tiles and a 16-iteration per-K-block scale epilogue.

The operator's physics is fixed and not in dispute: at `T=1` with 8 distinct experts it must read
**37.7 MB** of expert weights, which is **11.3 µs at 3.35 TB/s**. GEMM1 accounts for 25.2 MB (7.5 µs)
and GEMM2 for 12.6 MB (3.8 µs). Every microsecond above that floor is overhead, and the size of that
overhead is the only question worth paying for. Phase 2 answers it for ~$8 and carries a hard kill
criterion.

---

## 1. Model and quantization details (verified 2026-11-09)

| Item | Value | Source |
|---|---|---|
| Repo | `Qwen/Qwen3-30B-A3B-FP8` | HF |
| Revision | `d206ba732169f29bb77fbf80fc2c4b81d4d30782` (2025-07-26) | HF API |
| Architecture | `Qwen3MoeForCausalLM`, 48 layers | `config.json` |
| Hidden / MoE intermediate | 2048 / 768 | `config.json` |
| Experts / per token | 128 / 8 | `config.json` |
| Attention | 32 heads, 4 KV heads, head_dim 128, `rope_theta` 1e6 | `config.json` |
| Router normalization | `norm_topk_prob: true` (weights pre-normalized) | `config.json` |
| Shared expert | none | `config.json` |
| Quantization | `fp8`, `fmt: e4m3`, `weight_block_size: [128,128]`, `activation_scheme: dynamic` | `config.json` |
| Not quantized | 145 modules: `lm_head`, all norms, all `mlp.gate` (router) | `config.json` |
| Shards / size | 7 `.safetensors` (~30 GB), 37 491 tensors | index |
| Scale tensors | `weight_scale_inv`, per-expert per-projection, inverse scales | index |

On-disk layout (per layer, per expert) — **three separate projections, not a fused tensor**:

```
experts.E.gate_proj.weight          [768, 2048]  e4m3
experts.E.gate_proj.weight_scale_inv [6, 16]     f32
experts.E.up_proj.weight            [768, 2048]  e4m3
experts.E.down_proj.weight          [2048, 768]  e4m3
experts.E.down_proj.weight_scale_inv [16, 6]     f32
```

Per-expert weight bytes = `2·768·2048 + 2048·768` = **4 718 592 B (4.72 MB)**; scales add 1.15 KB.
The fused runtime `w13 [E, 1536, 2048]` / `w2 [E, 2048, 768]` is created by the loader.

Activation quant is **dynamic, per token, per 128-element group** (`kFp8Dynamic128Sym`), implemented by
`per_token_group_quant_fp8` (`utils.py:320-345`). Weight key is `kFp8Static128BlockSym`.

---

## 2. Pinned versions

| Component | Pin | Note |
|---|---|---|
| vLLM | `b6ed63d2acbe306ff465b979c603d4800c664818` (`main`, 2026-11-09) | local checkout == `origin/main` |
| PyTorch | `2.14.0` | `requirements/cuda.txt:11` |
| CUDA | `13.0.3` | `docker/Dockerfile:25` |
| FlashInfer | `flashinfer-python==0.7.3` | `requirements/cuda.txt:13` |
| CUTLASS | `v4.7.1` (`CUTLASS_REVISION`) | `CMakeLists.txt:517` |
| CuTe DSL | `nvidia-cutlass-dsl[cu13]==4.8.0` | `requirements/cuda.txt:26` |
| Model | revision `d206ba7…` | §1 |

The pod image must be digest-pinned against these; exact digest is captured in Phase 2.

---

## 3. Which backend runs this model on one H100

`select_fp8_moe_backend` (`oracle/fp8.py:415`) builds a priority list
(`oracle/fp8.py:95-119`) and then applies platform overrides. The override that hits us
(`oracle/fp8.py:129-141`):

```python
# On Hopper for Block Fp8, prefer Triton for TP and FI CUTLASS for EP.
if (current_platform.is_device_capability(90)
    and activation_key == kFp8Dynamic128Sym
    and weight_key == kFp8Static128BlockSym):
    if moe_config.moe_parallel_config.ep_size > 1:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.FLASHINFER_CUTLASS)
    else:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.TRITON)
```

Our case is Hopper, block-FP8 128×128, dynamic 128 activation, **TP=1 / EP=1** → **TRITON**.

Alternatives exist and are forceable with `--moe-backend`, which matters because the default is a
*heuristic*, not a measurement:

| Backend | Reachable | Block-FP8 on sm90 | Form |
|---|---|---|---|
| `triton` (default) | yes | yes | 6 launches |
| `flashinfer_cutlass` | yes | yes (`flashinfer_cutlass_moe.py:175`) | one fused FlashInfer call (SM90 block-scale path) |
| `deep_gemm` | yes (env) | yes | grouped GEMMs |
| `cutlass` (vLLM) | yes (`allow_vllm_cutlass`) | yes | grouped GEMMs |
| `marlin`, `humming` | yes | yes | weight-only / alt |
| `hpc` | needs Tencent `hpc-ops` | — | third-party fused kernel |
| MegaMoE (DeepGEMM) | no | SM100-only | — |

**Consequence for the benchmark:** "best applicable backend" cannot be assumed to be TRITON. If
FlashInfer's fused block-scale kernel is faster at `T ≤ 4`, it becomes the baseline, and the M=128
padding question returns for *that* path (FlashInfer/CUTLASS align `M` to a 128-row tile, unlike
Triton). Phase 2 measures all three.

---

## 4. Exact execution path (T ≤ 4) and its six launches

`_prepare_expert_assignment` (`fused_moe.py:1667`) takes a **low-latency shortcut**:

```python
naive_block_assignment = (
    expert_map is None
    and not weights_host_resident
    and num_tokens * top_k_num * 4 <= global_num_experts   # 8, 16, 32 ≤ 128 for T = 1, 2, 4
)
if naive_block_assignment:
    return (None, topk_ids.view(-1),
            torch.full((1,), topk_ids.numel() * config["BLOCK_SIZE_M"], ...))
```

So for `T ≤ 4` there is **no `moe_align_block_size` launch**, no sorting, no padding to 128:
each `(token, expert)` pair becomes its own `BLOCK_SIZE_M = 16` row block, and `expert_ids` is the
flattened `topk_ids`. At `T = 8` the condition fails (256 > 128) and the aligned path with an extra
kernel is used — a genuine behavioural discontinuity at the T=8 "regression" shape.

Kernel sequence per operator call (`T ≤ 4`, default env):

| # | Kernel | Bytes touched | Source |
|---|---|---|---|
| 1 | `per_token_group_quant_fp8` (input) | 4 KB → 2 KB + scales | `utils.py:320` |
| 2 | `fused_moe_kernel` (GEMM1, gate+up, swap-AB) | **25.2 MB** weights | `triton_moe.py:526`, `fused_moe.py:826` |
| 3 | `silu_and_mul` | 24 KB → 24 KB | `triton_moe.py:649` |
| 4 | `per_token_group_quant_fp8` (intermediate) | 24 KB → 12 KB + scales | `triton_moe.py:663` |
| 5 | `fused_moe_kernel` (GEMM2, down, router-weighted) | **12.6 MB** weights | `triton_moe.py:686` |
| 6 | `moe_sum` (top-k combine) | 32 KB → 4 KB | `triton_moe.py:770` |

Kernels 3 and 4 stay separate because `VLLM_USE_DEEP_GEMM_E8M0` defaults to **1**
(`envs.py:1668`), so `is_deep_gemm_e8m0_used()` is true on this image and the fused
`silu_and_mul_per_block_quant` alternative is disabled (`triton_moe.py:641`). That default also means
the Triton path rounds activation scales to powers of two — an accuracy cost, not just a launch cost
(measured by PR #65794: 0.069 → 0.062 relative error when fixed).

**Distribution dependence (an unclaimed structural finding).** The naive path creates one row block
per `(token, expert)` pair, so when tokens collide on the same experts the expert's weights are
streamed once *per pair*. At `T=4` with all four tokens on the same 8 experts that is 32 blocks over
8 experts: **151 MB instead of 37.7 MB, i.e. 45 µs instead of 11.3 µs**. At `T=4` with iid uniform
routing all ~32 pairs are distinct and there is no duplication. At `T=8` the heuristic flips to the
aligned path with `block_size = config["BLOCK_SIZE_M"] = 16` (`fused_moe.py:1711`), which groups an
expert's tokens into one block and does **not** duplicate weights. So the same operator has two
different traffic laws either side of `T=4`, and which one is better depends on the routing
distribution — a real serving property (hot experts), and measurable.

Chosen tiles: no tuned JSON exists for `E=128, N=768, NVIDIA_H100_80GB_HBM3, block_shape=[128,128]`
(the `N=768` H100 file is absent; `N=768` exists only for H200/H20/B200), so
`get_default_config` (`fused_moe.py:1409-1442`) applies:

```
BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128,
GROUP_SIZE_M=1, SPLIT_K=1, num_warps=4,
num_stages=4 (M ≤ 4) / 3 (M = 8),  SWAP_AB = sm90 and M<64 and N>=64 → true
```

Grid at `T=1` for GEMM1 = 8 pairs × (1536/64) = **192 CTAs**, each streaming 128 KB;
GEMM2 = 8 × (2048/64) = **256 CTAs**, each 48 KB. With ~41 KB SMEM per CTA and 228 KB/SM, all CTAs
are co-resident — the kernel is bandwidth-bound, not occupancy- or wave-bound. That removes the
"wave quantization" explanation and leaves achieved-bandwidth efficiency as the open question.

---

## 5. Cost arithmetic (the only physics that matters)

| Quantity | Value |
|---|---|
| Expert weights, routed (T=1, 8 distinct) | 37.7 MB |
| H100 SXM HBM3 peak | 3.35 TB/s |
| **Operator memory floor** | **11.3 µs** |
| GEMM1 / GEMM2 split | 25.2 MB (7.5 µs) / 12.6 MB (3.8 µs) |
| Padded MMA work at M=16 | 1.21 GFLOP → 1.5-6 µs depending on achieved TFLOPS |
| M=128 padding (CUTLASS/FI paths) | 9.7 GFLOP → ~10 µs at 1 PFLOP/s |
| Launch gaps (6 kernels × 0.45 µs) | 2.7 µs |
| Tiny-kernel work (kernels 1,3,4,6) | < 0.2 µs of real work, 4 launches |

Two implications:

- At `M=16` the padding waste is no longer 128×; it is 16× and worth 1.5-6 µs of MMA, which can hide
  under the 11.3 µs of loads. It is a margin issue, not a ceiling.
- The ceiling on any *launch-elimination* work is ~2.7 µs ≈ 15-20 % of a ~15 µs operator, and that
  is exactly what #64846 and #65794 already deliver (2.5-3.5 % TPOT each, on 48 layers).

So the project's viability reduces to one measurable number: **the achieved fraction of peak HBM
bandwidth in the two GEMMs at `T ≤ 4`.** If it is ≥ 75 %, there is no meaningful headroom and the
honest result is a negative one. If it is ≤ 60 %, a 25-35 % operator win is physically available.

---

## 6. Candidate bottlenecks and overlapping work

### B1 — Fixed launch overhead (6 dependent launches)
- **Why slow:** each launch starts ~0.45 µs after its predecessor *even in a CUDA graph*; four of the
  six kernels do negligible work.
- **CUDA remedy:** fuse the whole operator into one persistent kernel, or use PDL between launches.
- **How to measure:** Nsight Systems gap histogram; wall time of a 6-launch empty chain.
- **Overlap:** **#64846** (PDL, open, +31 lines, claims −3.2 % TPOT at conc 1) and **#65794**
  (fused SiLU+quant, open, claims 2.5-3.5 % TPOT on Qwen3-30B-A3B-FP8). **Claimed.**

### B2 — Redundant intermediate round trips
- **Why slow:** `intermediate_cache1` (24 KB) and `intermediate_cache2` (12 KB) are written to HBM and
  re-read, and two quant kernels exist where one could.
- **CUDA remedy:** fuse activation+quant into the GEMM1 epilogue and the weighted combine into the
  GEMM2 epilogue.
- **How to measure:** ncu sector counts on kernels 3/4/6; total bytes vs the 37.7 MB weight traffic.
- **Overlap:** #65794 fuses exactly this; #64978 trims another load. **Claimed.**

### B3 — Small-M GEMM streaming efficiency (unclaimed)
- **Why slow (hypothesis):** with `BLOCK_M=16` and `BLOCK_N=64` on Hopper, `tl.dot` runs at low
  arithmetic intensity per CTA, and the block-scaled kernel must accumulate FP32 partials per 128-wide
  K block (16 iterations at `K=2048`) before applying the scale — extra epilogue work per tile that no
  amount of L2 reuse removes. Whether the resulting kernel reaches 3.35 TB/s is unknown.
- **CUDA remedy:** keep the same traffic but stream it better — TMA bulk loads (`cp.async.bulk`) of
  contiguous 128×128 weight blocks, `M=16` `mma.sync` (or `wgmma.m64` with the pair's 16 rows padded to
  64 only if it wins), per-128-K FP32 scale accumulation in registers, and a persistent CTA layout that
  reads each selected expert's weights exactly once. Optionally one launch (which picks up B1 for free).
- **How to measure:** ncu `dram__throughput.avg.pct_of_peak_sustained_elapsed` and kernel duration for
  kernels 2 and 5 at T=1,2,4; compare against 7.5 µs / 3.8 µs.
- **Overlap:** no PR targets Triton GEMM bandwidth at TP=1. The nearest is **#62603**, which wires
  FlashInfer's SM90 block-FP8 MegaMoE (fused dispatch+GEMMs+combine) into vLLM — but it is gated behind
  `--moe-backend flashinfer_moe_ep_cutedsl` (EP, multi-GPU) and reports only **−7 %** at 8 tokens/rank
  on one EP8 layer. FlashInfer also merged Hopper block-FP8 low-latency decode kernels
  (#7174, #7212). **Partially overlapping, not covering TP=1.**

### B4 — Per-pair weight duplication on colliding routes (`T = 4`, skewed routing)
- **Why slow:** see §4 — the naive path re-streams an expert's weights for every token that selects it;
  4× traffic on the `single` distribution, ~1.2-1.3× on a realistic 80/20 skew.
- **CUDA remedy:** dispatch in expert-major order (group pairs by expert, one weight stream per expert),
  which is what the aligned path already does at `T=8`; a fused kernel gets it for free.
- **How to measure:** operator latency for `single`/`skew80` vs `uniform` at `T=4`; ncu DRAM bytes.
- **Overlap:** none found. Secondary target: it does not help `T=1`, which is the primary shape, so it
  cannot carry the project alone.

### Overlap audit table

| Work | State | Covers | Verdict |
|---|---|---|---|
| #64846 PDL for fused-MoE launches | open, +31 | B1 | Kills "fuse for launch overhead" |
| #65794 fused SiLU+quant, fp32 scales | open, +503 | B1, B2 | Kills intermediate-quant fusion |
| #64978 skip token-count load | open | B1 (micro) | Kills token-count micro-opt |
| Per-pair duplication on skewed routes (B4) | none | — | Unclaimed, but T=4-only |
| #62603 SM90 block-FP8 MegaMoE (FlashInfer, EP) | open, +675 | B3 shape, EP only | Does not cover TP=1 |
| FlashInfer #7174/#7212 Frost block-FP8 decode | merged | B3 for FI callers | Must be benchmarked as a baseline |
| DeepGEMM MegaMoE | merged | — | SM100 only |
| Triton decode tuning (swap-AB, N=64, stages=4) | merged | B3 partial | Baseline already tuned; headroom undetermined |

---

## 7. Selected hypothesis

> **H1 — At `T ≤ 4` the Triton block-FP8 expert GEMM pair runs at ≤ 60 % of peak HBM bandwidth, and a
> purpose-built Hopper kernel that streams each selected expert's weights exactly once with TMA bulk
> loads and 16-row tensor-core tiles, accumulating per-128-K FP32 scale products in registers, reaches
> ≥ 75 % — recovering ≥ 20 % of complete-operator latency without changing quantization semantics.**

It is deliberately narrow:
- It does **not** claim launch-overhead reduction as its own contribution (#64846, #65794 already do).
- It does **not** claim a faster quantization formula (that changes numerics and is #65794's fight).
- It **is** allowed to ship as one fused launch, because that is free once the kernel exists — but the
  claimed win must come from bandwidth, and the ablation must show the GEMM-only difference.

**What would falsify it:** measured GEMM efficiency ≥ 75 % of peak, or the operator already within
~15 % of 11.3 µs + irreducible overhead.

---

## 8. Phase 2 measurement plan (≈2.5-3.5 h H100, $6-11)

One session, one script (`scripts/gpu/p2_baseline.sh`), fixed image digest.

1. **Environment capture** — GPU UUID, driver, clocks, `nvidia-smi -q`, image digest, torch/vLLM/
   FlashInfer/CUTLASS versions, `bandwidthTest` measured HBM bandwidth (do not trust the datasheet).
2. **Baseline sweep** — forced backends `{triton, flashinfer_cutlass, cutlass, deep_gemm}` ×
   `T ∈ {1,2,4,8}` × distributions `{uniform, skew80, single, repeat}` × 3 seeds, cold-L2, graph
   replay, per §docs/measurement-protocol.md. This also settles whether the default is the best
   applicable backend.
3. **Roofline attribution** — ncu on the default path: duration, DRAM throughput %, and sectors for
   each of the six kernels; nsys gap histogram between them; achieved-BW fraction for the two GEMMs
   against 7.5 µs / 3.8 µs.
4. **Repeat** the winner and the default back-to-back (A/B/A/B) to bound clock drift.

**Pre-registered kill/continue criteria (no post-hoc adjustment):**

| Measurement | Action |
|---|---|
| Best baseline operator latency ≤ 13 µs at T=1 | **STOP** — within 15 % of the floor |
| GEMM DRAM throughput ≥ 75 % of measured peak | **STOP** — H1 falsified |
| GEMM DRAM throughput ≤ 60 % of measured peak and ≥ 20 % of operator latency | **PROCEED** to kernel |
| 60-75 % | **REVISE** — smaller target (e.g. T=4 skewed) or accept a negative result |
| FlashInfer/cutlass backend ≥ 15 % faster than Triton | Re-baseline against it before proceeding |

A negative outcome here is still a publishable artifact: a roofline attribution of where a
`Qwen3-30B-A3B-FP8` decode step spends its time on H100, with the kernels measured, not inferred.

---

## 9. Residual uncertainty

| # | Uncertainty | Resolution |
|---|---|---|
| U1 | Achieved bandwidth of the Triton pair is unknown; §7's 60 % is an inference from tile geometry, not a measurement | Phase 2 ncu |
| U2 | FlashInfer's fused block-scale path may already beat Triton at T≤4, changing the baseline | Phase 2 sweep |
| U3 | The 0.45 µs gap figure is measured on a *different* model (Qwen3.6-35B) | Phase 2 nsys |
| U4 | Whether the `T=8` aligned path changes the picture (different kernel, extra launch) | Phase 2 sweep |
| U5 | Upstream may merge #65794/#64846 before our result lands, further shrinking the delta | Re-check at Phase 2 and Phase 4 |
| U6 | `weight_scale_inv` → Triton's e8m0/pow2 scale conversion must be replicated exactly or accuracy comparisons are void | Phase 3 numerics test |

---

## 10. Decision

**PROCEED_TO_BASELINE**, on the revised hypothesis H1 (§7), with the Phase 2 kill criteria of §8
pre-registered.

Rationale:
- The original hypothesis (M padded to 128) is factually wrong on the default path and is retired.
- The two remaining "easy" wins (launch overhead, intermediate quant fusion) are already implemented
  in open PRs with measured TPOT results; pursuing them would duplicate work and fail the
  "not already implemented" test of PLAN.md §Phase 0.
- H1 is unclaimed at TP=1, is physically bounded, and is decided by one cheap profiler session.

**If the owner prefers the strict reading of "already claimed → stop":** the evidence is B1/B2 are
claimed and H1 is an inference, and the project would stop here with this document as the artifact.
I recommend the $8 measurement first, because it converts an inference into a number and either
justifies the kernel or produces a credible negative result — which is the outcome the plan asks for
when headroom is absent.

---

## Appendix A — Commands used for verification

```bash
# model facts (no 30 GB download; config only)
curl -sL https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/raw/main/config.json
curl -sL https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/raw/main/model.safetensors.index.json
curl -sL https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B-FP8            # sha

# backend selection and kernel path
grep -n "On Hopper for Block Fp8" -A 14 vllm/model_executor/layers/fused_moe/oracle/fp8.py
grep -n "naive_block_assignment" -A 22 vllm/model_executor/layers/fused_moe/fused_moe.py
grep -n "def get_default_config" -A 55 vllm/model_executor/layers/fused_moe/fused_moe.py

# overlap audit
gh api -X GET search/issues -f q='repo:vllm-project/vllm type:pr is:open moe fp8 hopper'
gh pr view 62603 --repo vllm-project/vllm
gh pr view 65794 --repo vllm-project/vllm
gh pr view 64846 --repo vllm-project/vllm
```
