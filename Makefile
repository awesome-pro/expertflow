# ExpertFlow developer entry points.
#
# CPU targets work on the local Mac. CUDA targets require a Linux + CUDA host
# (the H100 pod) and are added in Phase 3.

PY        := uv run
SRC_PY    := python tests benchmarks
SRC_CU    := csrc include

.DEFAULT_GOAL := help
.PHONY: help install test-cpu lint fmt fmt-check clean guard-gpu

help:  ## show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## create the pinned Python 3.12 env (CPU + reference deps)
	uv sync --extra dev --extra reference

test-cpu:  ## run fast local tests (no GPU)
	$(PY) pytest

lint:  ## ruff
	$(PY) ruff check $(SRC_PY)
	$(PY) ruff format --check $(SRC_PY)

fmt:  ## auto-format Python and (if present) CUDA sources
	$(PY) ruff format $(SRC_PY)
	$(PY) ruff check --fix $(SRC_PY)
	@if [ -d $(SRC_CU) ] && command -v clang-format >/dev/null 2>&1; then \
		find $(SRC_CU) -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' | xargs clang-format -i; \
	fi

fmt-check:  ## verify formatting without writing
	$(PY) ruff format --check $(SRC_PY)
	@if [ -d $(SRC_CU) ] && command -v clang-format >/dev/null 2>&1; then \
		find $(SRC_CU) -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \
			| xargs clang-format --dry-run --Werror; \
	fi

guard-gpu:  ## fail fast if this host cannot run CUDA work
	@command -v nvcc >/dev/null 2>&1 || { echo "no nvcc: this target needs the GPU pod"; exit 1; }
	@nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible"; exit 1; }

clean:  ## remove build artifacts and caches
	rm -rf build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
