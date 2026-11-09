# Environment

## 1. Local workstation (verified 2026-11-09)

| Item | Value | Consequence |
|---|---|---|
| Machine | Apple M4, 24 GB RAM, macOS (arm64) | No CUDA. Local work is CPU-only |
| CUDA toolkit | **absent** (`nvcc` not found) | Any CUDA compile/run happens on a remote pod |
| Python | 3.14.5 (Homebrew) | **Unusable** for torch/vLLM; project env is 3.12 via `uv` |
| `uv` | 0.11.26 | Creates the pinned 3.12 env |
| Compilers | Apple clang 21 | Host-side C++ sanity builds only |
| Tools | cmake, make, clang-format, gh, git | Present |
| Docker | **absent** | Local containerization is not required; the GPU image is pulled on the pod |
| Disk | 345 GB free | Enough for a single-layer extract, not for repeated 30 GB downloads |
| Network | GitHub/HF reachable | Research and weight-header inspection work locally |

## 2. Version pinning checklist (must be complete before Phase 2)

| Thing | Pin | Where recorded |
|---|---|---|
| Project Python | 3.12 | `pyproject.toml` + `uv.lock` |
| PyTorch | matching the GPU image's torch | `docs/environment.md` §4, Phase 2 |
| vLLM | release tag **and** image digest | `research/feasibility.md` |
| FlashInfer | commit / wheel version | `research/feasibility.md` |
| CUTLASS | git tag used by `FetchContent` | `CMakeLists.txt` |
| CUDA toolkit | image's CUDA version | image digest |
| GPU driver | recorded per session | `results/benchmarks.jsonl` |

## 3. Why a pinned image instead of a source build

A vLLM source build costs 30–60 minutes of GPU-pod time (≈ $2/session) and is the single easiest way
to blow the budget. The official `vllm/vllm-openai` image already contains compiled FP8 MoE kernels
and is digest-pinnable, which gives reproducibility without the compile cost. Our own extension is
small and builds in minutes.

Phase 0 must confirm that the chosen backend's kernels are actually the ones in that image (not
runtime-JIT-compiled Triton only), otherwise the baseline is measuring the wrong thing.

## 4. Remote session runbook

1. `scripts/gpu/<session>.sh --dry-run` passes locally.
2. Owner provides SSH target; pod is started fresh from the pinned image.
3. One command runs: environment capture → build → correctness → benchmark → artifact rsync.
4. `results/logs/<session>.txt` is committed; `results/spend.csv` gets a row; pod is destroyed.

Pre-flight checklist lives in `scripts/gpu/README.md` (added in Phase 1).

## 5. Local development commands

```bash
uv sync --extra dev --extra reference   # CPU env, Python 3.12
make test-cpu                           # fast local checks, no GPU
make lint                               # ruff
```
