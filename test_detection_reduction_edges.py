"""
Edge-case unit tests for Detection Track reduction classes in last_stage_capacity/_detection.py.

Classes covered:
  - LinearProjectionReduction
  - SEReduction
  - ConditionalCapacityBlock
  - CapacityReductionStack

Uses synthetic tensors only.
"""
import pytest
import torch

from last_stage_capacity._detection import (
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)


def test_linear_projection_reduction_invalid_channels():
    """LinearProjectionReduction should raise ValueError when out_channels >= in_channels."""
    with pytest.raises(ValueError, match="must be < in_channels"):
        LinearProjectionReduction(64, 64)

    with pytest.raises(ValueError, match="must be < in_channels"):
        LinearProjectionReduction(64, 128)


def test_linear_projection_reduction_invalid_dim():
    """LinearProjectionReduction should raise ValueError when input is not 2D, 3D, or 4D."""
    m = LinearProjectionReduction(64, 32)
    # 1D tensor
    x1 = torch.randn(64)
    with pytest.raises(ValueError, match="Expected 2-4D input"):
        m(x1)

    # 5D tensor
    x5 = torch.randn(1, 64, 8, 8, 8)
    with pytest.raises(ValueError, match="Expected 2-4D input"):
        m(x5)


def test_linear_projection_reduction_extra_repr():
    """LinearProjectionReduction extra_repr output check."""
    m = LinearProjectionReduction(128, 64)
    repr_str = m.extra_repr()
    assert "in_channels=128" in repr_str
    assert "out_channels=64" in repr_str


def test_se_reduction_invalid_ratio():
    """SEReduction should raise ValueError when reduction_ratio is not in (0, 1)."""
    with pytest.raises(ValueError, match="reduction_ratio must be in"):
        SEReduction(in_channels=128, reduction_ratio=0.0)

    with pytest.raises(ValueError, match="reduction_ratio must be in"):
        SEReduction(in_channels=128, reduction_ratio=1.0)

    with pytest.raises(ValueError, match="reduction_ratio must be in"):
        SEReduction(in_channels=128, reduction_ratio=-0.5)

    with pytest.raises(ValueError, match="reduction_ratio must be in"):
        SEReduction(in_channels=128, reduction_ratio=1.5)


def test_se_reduction_activations_and_repr():
    """SEReduction supports 'swish' and 'gelu' activations, and provides extra_repr."""
    x = torch.randn(2, 64, 8, 8)

    m_swish = SEReduction(in_channels=64, reduction_ratio=0.5, activation="swish")
    out_swish = m_swish(x)
    assert out_swish.shape == (2, 32, 8, 8)

    m_gelu = SEReduction(in_channels=64, reduction_ratio=0.5, activation="gelu")
    out_gelu = m_gelu(x)
    assert out_gelu.shape == (2, 32, 8, 8)

    repr_str = m_swish.extra_repr()
    assert "in_channels=64" in repr_str
    assert "out_channels=32" in repr_str
    assert "reduction_ratio=0.5" in repr_str


def test_conditional_capacity_block_invalid_out_channels():
    """ConditionalCapacityBlock raises ValueError if computed out_channels >= in_channels."""
    with pytest.raises(ValueError, match="must be < in_channels"):
        ConditionalCapacityBlock(in_channels=16, reduction_ratio=1.0, min_channels=16)

    with pytest.raises(ValueError, match="must be < in_channels"):
        ConditionalCapacityBlock(in_channels=8, reduction_ratio=0.5)  # default min_channels is max(16, ...) = 16 >= 8


def test_conditional_capacity_block_health_and_repr():
    """ConditionalCapacityBlock get_gate_health and extra_repr."""
    m = ConditionalCapacityBlock(in_channels=128, reduction_ratio=0.25)
    health = m.get_gate_health()
    assert "note" in health

    repr_str = m.extra_repr()
    assert "in_channels=128" in repr_str
    assert "out_channels=32" in repr_str
    assert "reduction_ratio=0.25" in repr_str


def test_capacity_reduction_stack_validation():
    """CapacityReductionStack validation of channels, ratios, and block types."""
    # Fewer than 2 stages
    with pytest.raises(ValueError, match="Need at least 2 stages"):
        CapacityReductionStack(channels=[64])

    # Mismatched list of reduction ratios
    with pytest.raises(ValueError, match="Need len\\(channels\\) - 1 values"):
        CapacityReductionStack(channels=[128, 64, 32], reduction_ratios=[0.5])

    # Unknown block_type
    with pytest.raises(ValueError, match="Unknown block_type"):
        CapacityReductionStack(channels=[128, 64], block_type="unknown")


def test_capacity_reduction_stack_linear_and_conditional():
    """CapacityReductionStack with block_type='linear' and block_type='conditional'."""
    channels = [256, 128, 64]
    x = torch.randn(2, 256, 16, 16)

    # Linear block type with list of ratios
    stack_linear = CapacityReductionStack(
        channels=channels,
        reduction_ratios=[0.5, 0.5],
        block_type="linear",
    )
    outs_linear = stack_linear(x)
    assert len(outs_linear) == 2
    assert outs_linear[0].shape == (2, 128, 16, 16)
    assert outs_linear[1].shape == (2, 64, 16, 16)

    # Conditional block type
    stack_cond = CapacityReductionStack(
        channels=channels,
        reduction_ratios=0.5,
        block_type="conditional",
    )
    outs_cond = stack_cond(x)
    assert len(outs_cond) == 2
    assert outs_cond[0].shape == (2, 128, 16, 16)
    assert outs_cond[1].shape == (2, 64, 16, 16)

    assert "stages=2" in stack_cond.extra_repr()
    assert "block_type=conditional" in stack_cond.extra_repr()


def test_capacity_reduction_stack_list_inputs():
    """CapacityReductionStack with a list of tensors as input."""
    channels = [256, 128, 64]
    stack = CapacityReductionStack(channels=channels, block_type="linear")

    t1 = torch.randn(2, 256, 16, 16)
    t2 = torch.randn(2, 128, 16, 16)

    outs = stack([t1, t2])
    assert len(outs) == 2
    assert outs[0].shape == (2, 128, 16, 16)
    assert outs[1].shape == (2, 64, 16, 16)
