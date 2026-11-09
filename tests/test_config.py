"""Checks that the code's notion of the model matches the frozen operator spec."""

import pytest
from expertflow import (
    BLOCK,
    INTER,
    N_LAYERS,
    E,
    H,
    K,
    ModelConfig,
    expert_scale_scalars,
    expert_weight_bytes,
)

H100_HBM_BYTES_PER_S = 3.35e12  # SXM 80GB, datasheet peak


def test_shapes_match_spec():
    assert (H, INTER, E, K, N_LAYERS) == (2048, 768, 128, 8, 48)


def test_fp8_config_matches_checkpoint():
    cfg = ModelConfig()
    assert cfg.fp8_dtype == "e4m3"
    assert cfg.weight_block_shape == (128, 128)
    assert cfg.act_quant_group == 128
    assert cfg.act_dynamic is True


def test_dims_divide_into_scale_blocks():
    # Block-scaled GEMM requires both dimensions to tile cleanly.
    assert H % BLOCK == 0
    assert INTER % BLOCK == 0


def test_per_expert_weight_bytes():
    # gate 768x2048 + up 768x2048 + down 2048x768, one byte per FP8 element.
    assert expert_weight_bytes() == 4_718_592
    assert expert_scale_scalars() == 3 * 6 * 16


@pytest.mark.parametrize("distinct", [1, 8, 16, 32, 64, 128])
def test_memory_floor_is_plausible(distinct):
    """T=1 with 8 distinct experts should need ~11 us of HBM time on H100."""
    seconds = distinct * expert_weight_bytes() / H100_HBM_BYTES_PER_S
    if distinct == 8:
        assert 10e-6 < seconds < 13e-6
    assert seconds > 0
