# ExpertFlow — CUDA FP8 MoE Decode Kernel

> **Note:** this is the original project brief, kept as written. The plan of record —
> decisions, pre-registered thresholds, cost policy, and phase gates — is
> [PROJECT.md](PROJECT.md). Where they differ, PROJECT.md wins.

## 1. Project

**Goal:** Develop a CUDA C++ implementation of a low-latency FP8 Mixture-of-Experts (MoE) operator for small-batch LLM decoding. Find and optimize a specific bottleneck in current implementations, then validate the improvement against an optimized inference backend.

| Item | Specification |
|---|---|
| Language | CUDA C++ (`.cu`, `.cpp`) |
| GPU | NVIDIA H100 SXM 80GB (Hopper, SM90a) |
| Precision | FP8 E4M3, W8A8, checkpoint-compatible scaling |
| Model | `Qwen/Qwen3-30B-A3B-FP8` |
| Workload | Single-GPU decode; 1, 2, and 4 tokens per step |
| Main operator | Pre-routed MoE expert computation |
| Target runtime | vLLM, if integration is justified |

**Desired result:** A correct, original CUDA kernel with reproducible performance measurements against the best applicable existing backend, plus an explanation of where its improvement comes from.

The specific optimization will be selected after inspecting and profiling the current implementation; it is not assumed in advance.

## 2. Operator definition

### Model configuration

- Hidden size `H = 2048`
- Expert intermediate size `I = 768`
- Routed experts `E = 128`
- Experts per token `K = 8`
- Primary token counts `T ∈ {1, 2, 4}`; `T = 8` for regression testing
- Expert weights: FP8 E4M3 with 128 × 128 block scaling
- Activations: dynamically quantized to FP8 for W8A8 computation
- Accumulation: FP32 where required
- Operator input/output: BF16, consistent with the model/runtime interface

Verify weight layout, packing, scale tensor shapes, activation scale granularity, and routing-weight normalization against the pinned model configuration and selected backend before implementing kernels.

### Inputs

```text
X                [T, H]       BF16 activations
expert_ids       [T, K]       selected expert indices
expert_weights   [T, K]       normalized routing weights
W_gate_up        expert FP8 weights + scales
W_down           expert FP8 weights + scales
```

### Computation

For each token `t` and selected expert `e`:

```text
gate = X[t] @ W_gate[e]
up   = X[t] @ W_up[e]
h    = SiLU(gate) * up
z    = h @ W_down[e]
Y[t] = sum_over_selected_experts(expert_weight[t, e] * z)
```

The CUDA implementation should perform real FP8 Tensor Core arithmetic, including required activation quantization and scaling. Use this expression as the mathematical reference, not as a prescription for separate kernel launches.

### Output

```text
Y                [T, H]       BF16 MoE output
```

Routing/top-k selection happens before the operator and is not part of this project.

## 3. Scope

**Implement:** One complete MoE expert-computation path, native CUDA kernels, correctness tests, benchmark harness, one targeted optimization, and an optional vLLM integration.

**Exclude:** A new router, full inference engine, attention, prefill, multi-GPU expert parallelism, general-purpose GEMM library, alternative GPU architectures, and multiple runtime integrations.

Keep the project on one model shape and one GPU type until the primary result is validated.

## 4. Implementation phases

Each phase produces an artifact and ends with a decision or verification check. Do not start GPU-intensive development before Phase 0 and the first baseline measurement are complete.

### Phase 0 — Research and feasibility

**Objective:** Select one worthwhile optimization before writing the performance kernel.

**Tasks**

1. Read the model configuration and inspect its FP8 weight/activation formats.
2. Identify the existing single-GPU MoE execution paths in vLLM and FlashInfer that support this model and Hopper.
3. Trace the selected backend from operator entry point through expert GEMMs, activation, quantization, and combine.
4. Inspect relevant open and recently merged issues/PRs. Record any work that overlaps the intended optimization.
5. Identify up to three possible bottlenecks. For each, state the exact operation involved, why it might be slow, how CUDA could improve it, and how to measure it.
6. Choose the most promising bottleneck. Specify the proposed change and how it differs from existing implementations.
7. Define the baseline, test shapes, measurement procedure, and minimum evidence needed to proceed.

**Deliverable:** `research/feasibility.md` containing:

```text
Model and quantization details
Pinned repository versions / commit hashes
Existing supported kernels and backend selection
Potential bottlenecks and relevant overlapping work
One selected optimization hypothesis
Proposed measurement to confirm or reject it
Decision: PROCEED_TO_BASELINE / REVISE / STOP
```

**Completion check:** The target operation, its existing implementation, the proposed difference, and a credible test are all specific. If the work is already implemented or covered by an active PR, revise the hypothesis. If no meaningful gap can be identified, stop or change the operator.

**Resources:** Local machine; no paid GPU required.

### Phase 1 — Reference, test data, and benchmark harness

**Objective:** Prepare the entire experiment before renting a GPU.

**Tasks**

1. Implement a PyTorch reference MoE operator for the exact model shapes.
2. Generate reproducible activations, expert IDs, routing weights, FP8 expert weights, and scales.
3. Include random, repeated-expert, and skewed-routing test cases.
4. Add comparisons for output errors (`max_abs`, `mean_abs`, relative/RMS error) and checks for nonfinite values.
5. Add adapters to run the applicable optimized backend with matching inputs and quantization semantics.
6. Implement CUDA Graph replay benchmarking, warmup, repeated measurements, and JSON result export.
7. Prepare an Nsight Systems profiling command, a minimal environment check, and a cost ledger.

**Deliverables:** `python/reference.py`, `python/test_data.py`, `tests/`, `benchmarks/`, and `results/spend.csv`.

**Completion check:** Scripts and expected outputs are ready; GPU work requires no substantial exploratory setup or model download.

**Resources:** Mac for source code and CPU reference tests; CUDA execution and final correctness checks need an NVIDIA GPU.

### Phase 2 — Hopper baseline and go/no-go

**Objective:** Confirm the bottleneck on actual hardware.

**Tasks**

1. Rent H100 SXM and record GPU, driver, CUDA, library versions, clocks, and utilization.
2. Run the existing optimized baseline for `T = 1, 2, 4, 8`.
3. Measure complete operator latency under CUDA Graph replay.
4. Profile the relevant quantization, expert GEMM, activation, combine, launch, and memory costs.
5. Check whether the selected bottleneck represents enough of total latency to make the proposed change useful.
6. Recheck the corresponding current implementation and active PRs before proceeding.

**Deliverable:** `research/baseline.md` with results, profiling evidence, selected implementation target, and decision.

**Decision:**

- **PROCEED:** Measured bottleneck, feasible CUDA approach, no substantial overlap, and enough remaining budget.
- **REVISE:** Bottleneck exists but the proposed optimization or benchmark needs adjustment.
- **STOP:** No credible performance headroom, the work already exists, or the cost/complexity is unjustified.

### Phase 3 — CUDA kernel implementation

**Objective:** Implement the chosen operator optimization in native CUDA C++.

**Tasks**

1. Define the kernel interface and any preallocated workspace.
2. Implement the simplest CUDA version that correctly handles expert selection, FP8 scales, GEMM outputs, activation, and weighted reduction.
3. Introduce native Hopper FP8 Tensor Core computation; CUDA C++ CuTe primitives may be used for low-level MMA/layout operations.
4. Handle thread/warp/CTA mapping, memory access, synchronization, alignment, and boundary conditions.
5. Support `T = 1, 2, 4, 8` and CUDA Graph replay.
6. Validate numerical behavior against the reference and compatible FP8 backend before performance tuning.

**Deliverables:** `csrc/` and CUDA correctness/replay tests.

**Completion check:** Correct FP8 execution, no unsupported synchronization or allocations on the captured hot path, repeatable outputs, and passing tests.

### Phase 4 — Optimize and compare

**Objective:** Improve the selected bottleneck and prove the effect.

**Tasks**

1. Profile the CUDA kernel with Nsight Systems/Compute.
2. Optimize only the factors supported by profiling: work mapping, Tensor Core utilization, weight traffic, memory coalescing, shared-memory use, occupancy, instruction/launch overhead, or necessary fusion.
3. Keep an unoptimized/previous version for an ablation comparison.
4. Compare **complete operator latency** against the strongest compatible baseline with matching FP8 semantics and CUDA Graph settings.
5. Repeat tests for different token counts and routing distributions; report regressions as well as improvements.
6. Record detailed environment information and numerical error.

**Deliverables:** `results/benchmarks.jsonl`, plots, profiler notes, and a documented kernel implementation.

**Completion check:** Reproducible and meaningful operator-level improvement, numerically valid results, and a clear performance explanation. If the kernel is not competitive, record the result rather than presenting a synthetic speedup as a serving improvement.

### Phase 5 — vLLM integration

**Objective:** Test whether the optimized operator improves real inference.

**Prerequisite:** Phase 4 demonstrates a competitive operator-level result.

**Tasks**

1. Integrate through an existing vLLM MoE backend/interface, preferably behind an opt-in setting.
2. Load the actual model checkpoint and verify outputs/correctness.
3. Test CUDA Graph capture/replay and decode behavior.
4. Measure TPOT and output throughput under fixed workloads, alongside the unmodified runtime.
5. Keep changes narrow and prepare an upstream PR if the improvement is applicable and not duplicated.

**Deliverables:** Integration code/patch, reproducible serving benchmark, and test results.

**Completion check:** Measured end-to-end impact with no unexplained correctness or stability regression.

### Phase 6 — Final project artifacts

**Objective:** Make the result easy to inspect and reproduce.

**Deliver**

- README: problem, implementation, supported environment, usage, and measured result.
- Technical write-up: bottleneck, kernel design, relevant GPU behavior, experiments, and limitations.
- Benchmark scripts and result files.
- Correctness tests and installation/build instructions.
- Nsight screenshots/traces or summarized findings.
- Upstream PR link, if submitted.

Only publish performance numbers tied to an exact GPU, model, backend version, workload, and test method.

## 5. Benchmark requirements

| Property | Requirement |
|---|---|
| Primary metric | Complete pre-routed MoE operator latency (µs) |
| Comparison | Best applicable vLLM/FlashInfer optimized implementation |
| GPU | Same H100 SXM for candidate and baseline |
| Precision | Equivalent FP8 quantization/scaling behavior |
| Workloads | `T = 1, 2, 4`; `T = 8` regression check |
| Execution | Same eager/graph mode; prefer CUDA Graph replay |
| Routing | Same expert IDs and weights; multiple distributions |
| Quality | Numerical error plus real-model correctness if integrated |
| Report | Median, variation, repetitions, GPU/software configuration |

Include required quantization, scaling, activation, GEMMs, and combine operations in primary latency measurements. Isolated subkernel measurements are diagnostic only.

## 6. Hardware and budget

- **Final measurement GPU:** H100 SXM 80GB.
- **Optional cheaper GPU:** RTX 4090 for portable FP8 tests; use only when it saves more than the additional setup costs. Hopper-specific code should run on H100.
- **Local work:** Model/backend research, implementation planning, CPU reference, test fixtures, documentation, plotting, and benchmark scripts.
- **Full model download and serving benchmarks:** Only when Phase 5 is reached.

Maintain `results/spend.csv` with session, provider, GPU, hours, effective rate, storage/other costs, cumulative spend, and output. Delete/stop billable resources after each experiment.

## 7. Repository layout

```text
expertflow/
├── PROJECT.md
├── README.md
├── research/
│   ├── feasibility.md
│   └── baseline.md
├── csrc/
├── include/
├── python/
│   ├── reference.py
│   └── test_data.py
├── tests/
├── benchmarks/
├── integration/
│   └── vllm/                # phase 5 only
└── results/
    ├── spend.csv
    └── benchmarks.jsonl
```

## 8. First implementation task

**Start with Phase 0 only.**

Create the repository and `research/feasibility.md`. Inspect the model FP8 configuration and existing single-GPU Hopper MoE implementations in vLLM/FlashInfer. Identify the current execution path for the target shape, review ongoing overlapping work, and propose one specific optimization and measurement plan.

Finish with a clear `PROCEED_TO_BASELINE`, `REVISE`, or `STOP` decision and the evidence supporting it.

**Do not write the optimized kernel or rent GPU hardware during this task.**

### References for implementation

- Model: https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8
- vLLM: https://github.com/vllm-project/vllm
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- CUTLASS/CuTe C++: https://github.com/NVIDIA/cutlass
- NVIDIA CUDA documentation: https://docs.nvidia.com/cuda/
