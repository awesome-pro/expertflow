"""ExpertFlow: FP8 MoE decode operator for Qwen3-30B-A3B on Hopper.

This package holds the CPU-side reference implementation, test-data generators,
baseline adapters, and benchmark driver. The CUDA kernel lives in ``csrc/``.
"""

from dataclasses import dataclass

# Verified against Qwen/Qwen3-30B-A3B-FP8 config.json on 2026-11-09.
# See docs/operator-spec.md for the contract these describe.
H = 2048  # hidden size
INTER = 768  # moe intermediate size (I in docs/operator-spec.md)
E = 128  # routed experts
K = 8  # experts per token
N_LAYERS = 48
BLOCK = 128  # FP8 weight block size (both dims)

PRIMARY_TOKENS = (1, 2, 4)
REGRESSION_TOKENS = (8,)
TOKENS = PRIMARY_TOKENS + REGRESSION_TOKENS


@dataclass(frozen=True)
class ModelConfig:
    """Shape and quantization description of the operator's model."""

    hidden: int = H
    intermediate: int = INTER
    experts: int = E
    top_k: int = K
    block: int = BLOCK
    fp8_dtype: str = "e4m3"
    weight_block_shape: tuple[int, int] = (BLOCK, BLOCK)  # (rows, cols) of a scale block
    act_quant_group: int = 128  # per-token, per-128-element activation group
    act_dynamic: bool = True
    accumulate_dtype: str = "float32"
    io_dtype: str = "bfloat16"


def expert_weight_bytes(block: int = BLOCK) -> int:
    """Bytes of one expert's FP8 weights (gate, up, down)."""
    return (2 * INTER * H) + (H * INTER)


def expert_scale_scalars(block: int = BLOCK) -> int:
    """Number of FP32 scale entries for one expert (gate, up, down)."""
    gu = (INTER // block) * (H // block)
    return 2 * gu + (H // block) * (INTER // block)


def routed_weight_bytes(tokens: int, distinct_experts: int, block: int = BLOCK) -> int:
    """Weight bytes that must be read from HBM for one operator call."""
    return distinct_experts * expert_weight_bytes(block)


__all__ = [
    "H",
    "INTER",
    "E",
    "K",
    "BLOCK",
    "N_LAYERS",
    "TOKENS",
    "PRIMARY_TOKENS",
    "REGRESSION_TOKENS",
    "ModelConfig",
    "expert_weight_bytes",
    "expert_scale_scalars",
    "routed_weight_bytes",
]
