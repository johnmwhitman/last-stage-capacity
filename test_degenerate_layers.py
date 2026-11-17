"""Edge-case tests for capacity reduction on degenerate layers.

Degenerate cases tested:
1. 1-channel convs (1-channel input/output, residual bottlenecks, depthwise final, SE)
2. Already-minimal width (boundary channel counts, clamp to 1 channel, invalid configurations)
3. Depthwise convs (DepthwiseFinal module, depthwise separable blocks in WidthScaler, isolated scaling error)

All tests use synthetic tensors only — no pretrained weights.
"""

import pytest
import torch
import torch.nn as nn

from last_stage_capacity import (
    BottleneckBlock,
    ProgressiveNarrowing,
    WidthScaler,
    DepthwiseFinal,
    SqueezeExcitation,
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)


# ===================================================================
# 1. Edge-cases for 1-channel convolutions
# ===================================================================

def test_width_scaler_one_channel_conv():
    """WidthScaler correctly handles 1-channel Conv2d without collapsing width."""
    # 1 -> 1 channel conv
    conv = nn.Conv2d(1, 1, 3, padding=1)
    scaled = WidthScaler(conv, 0.5)
    assert scaled.module.in_channels == 1
    assert scaled.module.out_channels == 1  # max(1, int(1 * 0.5)) == 1

    x = torch.randn(2, 1, 16, 16)
    out = scaled(x)
    assert out.shape == (2, 1, 16, 16)

    # 1 -> 4 channel conv
    conv_expand = nn.Conv2d(1, 4, 3, padding=1)
    scaled_expand = WidthScaler(conv_expand, 0.5)
    assert scaled_expand.module.in_channels == 1
    assert scaled_expand.module.out_channels == 2

    out_expand = scaled_expand(x)
    assert out_expand.shape == (2, 2, 16, 16)


def test_width_scaler_one_channel_sequential():
    """WidthScaler on a Sequential chain with 1-channel input and output."""
    chain = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 1, 3, padding=1),
        nn.BatchNorm2d(1),
    )
    scaled = WidthScaler(chain, 0.5)
    # conv1: in=1, out=2
    # conv2: in=2, out=1
    assert scaled.module[0].in_channels == 1
    assert scaled.module[0].out_channels == 2
    assert scaled.module[1].num_features == 2
    assert scaled.module[3].in_channels == 2
    assert scaled.module[3].out_channels == 1
    assert scaled.module[4].num_features == 1

    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    out = scaled(x)
    assert out.shape == (2, 1, 16, 16)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_bottleneck_block_one_channel():
    """BottleneckBlock with 1-channel input and output."""
    block = BottleneckBlock(1, 1, bottleneck_ratio=0.25)
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    out = block(x)
    assert out.shape == (2, 1, 16, 16)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_depthwise_final_one_channel():
    """DepthwiseFinal with 1-channel conv."""
    dw = DepthwiseFinal(1, 1, expansion_ratio=1.0)
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    out = dw(x)
    assert out.shape == (2, 1, 16, 16)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_squeeze_excitation_one_channel():
    """SqueezeExcitation channel attention with 1 channel."""
    se = SqueezeExcitation(channels=1, reduction=16)
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    out = se(x)
    assert out.shape == (2, 1, 16, 16)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_linear_projection_reduction_one_channel():
    """LinearProjectionReduction on 1-channel: assert clear error when no reduction is possible."""
    with pytest.raises(ValueError, match="out_channels .* must be < in_channels"):
        LinearProjectionReduction(in_channels=1, out_channels=1)

    # Correct reduction from 2 channels to 1 channel
    proj = LinearProjectionReduction(in_channels=2, out_channels=1)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    out = proj(x)
    assert out.shape == (2, 1, 8, 8)
    out.sum().backward()
    assert x.grad is not None


def test_se_reduction_one_channel():
    """SEReduction on 1-channel: asserts clear error since out_channels cannot be < 1."""
    with pytest.raises(ValueError, match="out_channels .* must be < in_channels"):
        SEReduction(in_channels=1, reduction_ratio=0.5)

    # Correct reduction from 2 channels to 1 channel
    ser = SEReduction(in_channels=2, reduction_ratio=0.5)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    out = ser(x)
    assert out.shape == (2, 1, 8, 8)
    out.sum().backward()
    assert x.grad is not None


def test_conditional_capacity_block_one_channel():
    """ConditionalCapacityBlock on 1-channel: asserts clear error."""
    with pytest.raises(ValueError, match="Computed out_channels .* must be < in_channels"):
        ConditionalCapacityBlock(in_channels=1, reduction_ratio=0.5)


# ===================================================================
# 2. Edge-cases for already-minimal width
# ===================================================================

def test_width_scaler_already_minimal_width():
    """WidthScaler on layer that is already minimal width (out_channels=1)."""
    conv = nn.Conv2d(8, 1, 3, padding=1)
    scaled = WidthScaler(conv, 0.5)
    assert scaled.module.out_channels == 1

    x = torch.randn(2, 8, 16, 16)
    out = scaled(x)
    assert out.shape == (2, 1, 16, 16)


def test_width_scaler_aggressive_clamping():
    """WidthScaler clamps to at least 1 channel under extreme reduction ratios."""
    conv = nn.Conv2d(16, 2, 3, padding=1)
    # 2 * 0.001 = 0.002 -> int is 0, must be clamped to 1
    scaled = WidthScaler(conv, 0.001)
    assert scaled.module.out_channels == 1

    x = torch.randn(2, 16, 8, 8)
    out = scaled(x)
    assert out.shape == (2, 1, 8, 8)


def test_width_scaler_invalid_scale():
    """WidthScaler asserts clear ValueError when width_scale <= 0."""
    conv = nn.Conv2d(8, 8, 3)
    with pytest.raises(ValueError, match="width_scale must be positive"):
        WidthScaler(conv, 0.0)

    with pytest.raises(ValueError, match="width_scale must be positive"):
        WidthScaler(conv, -0.5)


def test_linear_projection_reduction_invalid_channels():
    """LinearProjectionReduction asserts clear ValueError for degenerate channel specs."""
    # out_channels >= in_channels
    with pytest.raises(ValueError, match="out_channels .* must be < in_channels"):
        LinearProjectionReduction(in_channels=4, out_channels=4)

    with pytest.raises(ValueError, match="out_channels .* must be < in_channels"):
        LinearProjectionReduction(in_channels=4, out_channels=8)

    # out_channels <= 0
    with pytest.raises(ValueError, match="out_channels .* must be > 0"):
        LinearProjectionReduction(in_channels=4, out_channels=0)


def test_se_reduction_invalid_ratio():
    """SEReduction asserts clear ValueError for non-(0, 1) reduction ratios."""
    with pytest.raises(ValueError, match="reduction_ratio must be in \\(0, 1\\)"):
        SEReduction(in_channels=16, reduction_ratio=1.0)

    with pytest.raises(ValueError, match="reduction_ratio must be in \\(0, 1\\)"):
        SEReduction(in_channels=16, reduction_ratio=0.0)

    with pytest.raises(ValueError, match="reduction_ratio must be in \\(0, 1\\)"):
        SEReduction(in_channels=16, reduction_ratio=-0.25)


def test_conditional_capacity_block_minimal_width():
    """ConditionalCapacityBlock: clear error on invalid min_channels, success on minimal width."""
    # Cannot reduce when out_channels >= in_channels
    with pytest.raises(ValueError, match="Computed out_channels .* must be < in_channels"):
        ConditionalCapacityBlock(in_channels=8, reduction_ratio=0.5, min_channels=8)

    # Reducing to minimal width (1 channel)
    ccb = ConditionalCapacityBlock(in_channels=2, reduction_ratio=0.5, min_channels=1)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    out = ccb(x)
    assert out.shape == (2, 1, 8, 8)
    out.sum().backward()
    assert x.grad is not None


def test_progressive_narrowing_minimal_stages():
    """ProgressiveNarrowing assertions and reduction to minimal width."""
    # Under-specified stage list
    with pytest.raises(AssertionError, match="Need at least input and output channels"):
        ProgressiveNarrowing(channels=[4])

    # Reducing down to minimal width 1
    pn = ProgressiveNarrowing(channels=[8, 4, 2, 1], bottleneck_ratio=0.5, use_se=False)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    out = pn(x)
    assert out.shape == (2, 1, 16, 16)
    out.sum().backward()
    assert x.grad is not None


def test_capacity_reduction_stack_minimal_width():
    """CapacityReductionStack assertions and reduction to minimal width."""
    # Less than 2 stages
    with pytest.raises(ValueError, match="Need at least 2 stages"):
        CapacityReductionStack(channels=[8])

    # Non-reducing stages (out >= in)
    with pytest.raises(ValueError, match="out_channels .* must be < in_channels"):
        CapacityReductionStack(channels=[8, 8], block_type='linear')

    # Progressive reduction down to minimal width 1
    stack = CapacityReductionStack(channels=[8, 4, 2, 1], block_type='linear')
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    outs = stack(x)
    assert len(outs) == 3
    assert outs[-1].shape == (1, 1, 16, 16)
    outs[-1].sum().backward()
    assert x.grad is not None


# ===================================================================
# 3. Edge-cases for depthwise convolutions
# ===================================================================

def test_depthwise_final_reduction():
    """DepthwiseFinal correctly reduces channels using depthwise separable convolutions."""
    # Standard reduction
    dw_block = DepthwiseFinal(in_channels=32, out_channels=16, expansion_ratio=1.0)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    out = dw_block(x)
    assert out.shape == (2, 16, 16, 16)
    out.sum().backward()
    assert x.grad is not None

    # Reduction with low expansion ratio
    dw_contract = DepthwiseFinal(in_channels=16, out_channels=8, expansion_ratio=0.5)
    x2 = torch.randn(2, 16, 16, 16, requires_grad=True)
    out2 = dw_contract(x2)
    assert out2.shape == (2, 8, 16, 16)
    out2.sum().backward()
    assert x2.grad is not None


def test_width_scaler_depthwise_separable_block():
    """WidthScaler on a depthwise-separable block preserves depthwise grouping and reduces params."""
    # Standard depthwise separable block: pointwise -> depthwise 3x3 -> BN -> ReLU -> pointwise
    block = nn.Sequential(
        nn.Conv2d(32, 16, kernel_size=1),
        nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=1),
        nn.BatchNorm2d(8),
    )
    scaled = WidthScaler(block, 0.5)

    # First conv: in=32, out=8
    assert scaled.module[0].in_channels == 32
    assert scaled.module[0].out_channels == 8

    # Depthwise conv: in=8, out=8, groups=8 (preserved depthwise grouping!)
    dw_layer = scaled.module[1]
    assert dw_layer.in_channels == 8
    assert dw_layer.out_channels == 8
    assert dw_layer.groups == 8

    # BatchNorm: 8 features
    assert scaled.module[2].num_features == 8

    # Final conv: in=8, out=4
    assert scaled.module[4].in_channels == 8
    assert scaled.module[4].out_channels == 4
    assert scaled.module[5].num_features == 4

    # Parameter reduction verification
    orig_params = sum(p.numel() for p in block.parameters())
    scaled_params = sum(p.numel() for p in scaled.parameters())
    assert scaled_params < orig_params

    # Forward and backward with synthetic tensor
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    out = scaled(x)
    assert out.shape == (2, 4, 16, 16)
    out.sum().backward()
    assert x.grad is not None


def test_width_scaler_isolated_depthwise_error():
    """WidthScaler asserts clear ValueError when scaling isolated depthwise conv with mismatched channels."""
    # Standalone depthwise conv preserves in_channels=16, but scales out_channels to 8.
    # Because in_channels (16) != out_channels (8), depthwise convolution constraint fails.
    dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
    with pytest.raises(ValueError, match="Cannot scale depthwise Conv2d .* requires in_channels == out_channels == groups"):
        WidthScaler(dw, 0.5)


def test_width_scaler_grouped_conv():
    """WidthScaler handles grouped conv when divisible, raises clear ValueError when not."""
    # 16 in, 16 out, groups=4. When scaled to 8 out (prev is 8): 8 is divisible by 4.
    block_valid = nn.Sequential(
        nn.Conv2d(16, 16, 1),
        nn.Conv2d(16, 16, 3, padding=1, groups=4),
    )
    scaled_valid = WidthScaler(block_valid, 0.5)
    assert scaled_valid.module[1].groups == 4
    assert scaled_valid.module[1].in_channels == 8
    assert scaled_valid.module[1].out_channels == 8

    x = torch.randn(2, 16, 8, 8)
    out = scaled_valid(x)
    assert out.shape == (2, 8, 8, 8)

    # Indivisible grouped conv: groups=5, channels scale to 8 (not divisible by 5)
    block_invalid = nn.Sequential(
        nn.Conv2d(20, 20, 1),
        nn.Conv2d(20, 20, 3, padding=1, groups=5),
    )
    # 20 * 0.4 = 8. 8 is not divisible by 5.
    with pytest.raises(ValueError, match="Cannot scale grouped Conv2d .* channels must be divisible by groups"):
        WidthScaler(block_invalid, 0.4)
