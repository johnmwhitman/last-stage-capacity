"""Pytest edge-case tests for capacity reduction on grouped convolution (groups > 1),
3D convolution, and stride-2 convolution layers.

All tests use synthetic tensors only — no pretrained weights.
Tests verify that layers/modules either perform correct capacity reduction
or raise clear asserted errors.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest
import torch
import torch.nn as nn

from last_stage_capacity import (
    BottleneckBlock,
    ProgressiveNarrowing,
    WidthScaler,
    DepthwiseFinal,
    SqueezeExcitation,
    CapacityReductionHead,
    EmbeddingCompressor,
)
from last_stage_capacity._detection import (
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)


# =============================================================================
# 1. Grouped Convolution (groups > 1) Tests
# =============================================================================

def test_depthwise_final_grouped_conv_reduction():
    """DepthwiseFinal inherently uses grouped convolution (depthwise conv: groups == hidden_dim > 1).
    Verify that capacity reduction operates correctly and gradients flow.
    """
    in_channels, out_channels = 64, 32
    dw_block = DepthwiseFinal(in_channels, out_channels, kernel_size=3, stride=1)

    # Assert internal depthwise conv has groups > 1
    assert dw_block.dw.groups > 1, f"Expected dw.groups > 1, got {dw_block.dw.groups}"
    assert dw_block.dw.groups == dw_block.dw.in_channels
    assert dw_block.out_channels < dw_block.in_channels, "Capacity reduction not configured"

    x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
    out = dw_block(x)
    assert out.shape == (2, out_channels, 16, 16), f"Expected (2, {out_channels}, 16, 16), got {out.shape}"

    # Backward pass check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients failed to propagate through DepthwiseFinal"
    assert dw_block.dw.weight.grad is not None, "Gradients missing on grouped conv weights"


def test_width_scaler_on_grouped_conv_sequential():
    """WidthScaler applied to a Sequential containing a grouped convolution (groups > 1).
    Verify that out_channels are scaled down and forward/backward execute properly on synthetic tensors.
    """
    in_channels, out_channels = 64, 128
    groups = 4
    conv_grouped = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False)
    bn = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()
    seq = nn.Sequential(conv_grouped, bn, relu)

    scaler = WidthScaler(seq, width_scale=0.5)

    # Output channels of the conv should be reduced from 128 to 64
    scaled_conv = scaler.module[0]
    scaled_bn = scaler.module[1]
    assert scaled_conv.in_channels == in_channels
    assert scaled_conv.out_channels == 64
    assert scaled_bn.num_features == 64

    x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
    out = scaler(x)
    assert out.shape == (2, 64, 16, 16)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_grouped_conv_channel_mismatch_error():
    """When a grouped convolution has in_channels or out_channels not divisible by groups,
    PyTorch raises a clear ValueError. Assert this error behavior explicitly.
    """
    in_channels = 64
    invalid_out_channels = 35  # not divisible by groups=4
    groups = 4

    with pytest.raises(ValueError, match="divisible by groups"):
        _ = nn.Conv2d(in_channels, invalid_out_channels, kernel_size=3, groups=groups)


# =============================================================================
# 2. 3D Convolution / 5D Tensor Tests
# =============================================================================

def test_linear_projection_reduction_3d_conv_unsupported_error():
    """LinearProjectionReduction expects 2D, 3D (B, C, T), or 4D (B, C, H, W) tensors.
    For 3D convolution feature maps (5D tensor: B, C, D, H, W), it should assert a clear ValueError.
    """
    reducer = LinearProjectionReduction(in_channels=32, out_channels=16)
    x_5d = torch.randn(2, 32, 4, 8, 8)  # 5D synthetic feature map from Conv3d

    with pytest.raises(ValueError, match="Expected 2-4D input, got 5D"):
        reducer(x_5d)


def test_conditional_capacity_block_3d_conv_unsupported_error():
    """ConditionalCapacityBlock delegates reduction to LinearProjectionReduction and uses
    Conv2d gating. When fed a 5D tensor (3D conv output), it must raise an asserted error.
    """
    ccb = ConditionalCapacityBlock(in_channels=32, reduction_ratio=0.5)
    x_5d = torch.randn(2, 32, 4, 8, 8)

    with pytest.raises(ValueError, match="Expected 2-4D input, got 5D"):
        ccb(x_5d)


def test_se_reduction_3d_conv_unsupported_error():
    """SEReduction uses AdaptiveAvgPool2d and expects 4D inputs (B, C, H, W).
    When given a 5D tensor (Conv3d feature map), it should raise a clear RuntimeError or ValueError.
    """
    se = SEReduction(in_channels=32, reduction_ratio=0.5)
    x_5d = torch.randn(2, 32, 4, 8, 8)

    with pytest.raises((RuntimeError, ValueError)):
        se(x_5d)


def test_bottleneck_block_3d_conv_unsupported_error():
    """BottleneckBlock uses Conv2d layers and expects 4D inputs.
    When given a 5D tensor from 3D convolution, it should raise a clear RuntimeError.
    """
    block = BottleneckBlock(in_channels=32, out_channels=16)
    x_5d = torch.randn(2, 32, 4, 8, 8)

    with pytest.raises(RuntimeError, match="conv2d"):
        block(x_5d)


def test_depthwise_final_3d_conv_unsupported_error():
    """DepthwiseFinal uses Conv2d and expects 4D inputs.
    When given a 5D tensor from 3D convolution, it should raise a clear RuntimeError.
    """
    dw = DepthwiseFinal(in_channels=32, out_channels=16)
    x_5d = torch.randn(2, 32, 4, 8, 8)

    with pytest.raises(RuntimeError, match="conv2d"):
        dw(x_5d)


def test_squeeze_excitation_3d_conv_unsupported_error():
    """SqueezeExcitation expects 4D tensors (b, c, h, w).
    Unpacking 5D tensor shapes raises a clear ValueError.
    """
    se = SqueezeExcitation(channels=32)
    x_5d = torch.randn(2, 32, 4, 8, 8)

    with pytest.raises(ValueError, match="too many values to unpack"):
        se(x_5d)


def test_width_scaler_preserves_conv3d_unscaled():
    """WidthScaler operates on 2D vision models (Conv2d, BatchNorm2d, Linear, LayerNorm).
    Verify that an isolated nn.Conv3d or a module containing nn.Conv3d remains intact without
    corrupting the layer.
    """
    conv3d = nn.Conv3d(16, 32, kernel_size=3, padding=1)
    scaler = WidthScaler(conv3d, width_scale=0.5)

    # Conv3d is preserved as-is
    assert isinstance(scaler.module, nn.Conv3d)
    assert scaler.module.in_channels == 16
    assert scaler.module.out_channels == 32

    x = torch.randn(2, 16, 4, 8, 8)
    out = scaler(x)
    assert out.shape == (2, 32, 4, 8, 8)


# =============================================================================
# 3. Stride-2 Convolution Tests
# =============================================================================

def test_bottleneck_block_stride_2_reduction():
    """BottleneckBlock with stride=2 simultaneously performs channel capacity reduction
    and spatial downsampling (stride-2). Verify shortcut matching and output shape.
    """
    in_channels, out_channels = 64, 32
    block = BottleneckBlock(in_channels, out_channels, bottleneck_ratio=0.25, stride=2)

    # Shortcut should downsample to match stride=2 and project channels
    assert not isinstance(block.shortcut, nn.Identity)

    x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
    out = block(x)

    # Spatial dimensions should halve (16 -> 8), channels reduced (64 -> 32)
    assert out.shape == (2, out_channels, 8, 8)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_bottleneck_block_stride_2_with_se():
    """BottleneckBlock with stride=2 and use_se=True.
    Verify combined SE channel attention, capacity reduction, and stride-2 downsampling.
    """
    in_channels, out_channels = 128, 64
    block = BottleneckBlock(in_channels, out_channels, bottleneck_ratio=0.25, stride=2, use_se=True)

    assert block.se is not None

    x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
    out = block(x)
    assert out.shape == (2, out_channels, 8, 8)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_depthwise_final_stride_2_reduction():
    """DepthwiseFinal with stride=2 reduces channel capacity and spatial dimensions (stride-2)
    using depthwise separable convolutions.
    """
    in_channels, out_channels = 64, 32
    dw = DepthwiseFinal(in_channels, out_channels, stride=2)

    assert dw.stride == 2
    assert not isinstance(dw.shortcut, nn.Identity)

    x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
    out = dw(x)

    # Spatial dimensions halved, channels reduced to out_channels
    assert out.shape == (2, out_channels, 8, 8)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_width_scaler_on_stride_2_conv_sequential():
    """WidthScaler on a Sequential containing a stride-2 Conv2d layer.
    Verify that both channel capacity reduction (via width_scale) and spatial downsampling
    are preserved properly.
    """
    in_channels, out_channels = 32, 64
    seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    scaled = WidthScaler(seq, width_scale=0.5)

    assert scaled.module[0].in_channels == 32
    assert scaled.module[0].out_channels == 32
    assert scaled.module[0].stride == (2, 2)
    assert scaled.module[1].num_features == 32

    x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
    out = scaled(x)
    assert out.shape == (2, 32, 8, 8)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_progressive_narrowing_with_stride_2_block():
    """Test ProgressiveNarrowing stage compatibility when custom stride-2 bottleneck block
    is included in a multi-stage reduction pipeline.
    """
    # ProgressiveNarrowing default stages have stride=1, but individual stages can be replaced
    pn = ProgressiveNarrowing([128, 64, 32], bottleneck_ratio=0.25)

    # Replace the first stage with a stride-2 BottleneckBlock
    pn.stages[0] = BottleneckBlock(128, 64, bottleneck_ratio=0.25, stride=2)

    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    out = pn(x)

    # 128 -> 64 (spatial / 2) -> 32 (spatial unchanged)
    assert out.shape == (2, 32, 8, 8)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
