"""Test suite for last_stage_capacity library."""
import os, sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SCRIPT_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
import torch.nn as nn
from last_stage_capacity import (
    BottleneckBlock, ProgressiveNarrowing, WidthScaler,
    EmbeddingCompressor, DepthwiseFinal, SqueezeExcitation,
    CapacityReductionHead
)


def test_bottleneck_block():
    block = BottleneckBlock(512, 256, bottleneck_ratio=0.25)
    x = torch.randn(2, 512, 32, 32)
    out = block(x)
    assert out.shape == (2, 256, 32, 32), f"Expected (2, 256, 32, 32), got {out.shape}"
    print(f'BottleneckBlock: {x.shape} -> {out.shape} ✓')


def test_progressive_narrowing():
    pn = ProgressiveNarrowing([512, 256, 128, 64], bottleneck_ratio=0.25)
    x = torch.randn(1, 512, 16, 16)
    out = pn(x)
    assert out.shape == (1, 64, 16, 16), f"Expected (1, 64, 16, 16), got {out.shape}"
    print(f'ProgressiveNarrowing [512,256,128,64]: {x.shape} -> {out.shape} ✓')


def test_depthwise_final():
    dw = DepthwiseFinal(256, 256, stride=1)
    x = torch.randn(2, 256, 16, 16)
    out = dw(x)
    assert out.shape == (2, 256, 16, 16), f"Expected (2, 256, 16, 16), got {out.shape}"
    print(f'DepthwiseFinal: {x.shape} -> {out.shape} ✓')


def test_embedding_compressor():
    ec = EmbeddingCompressor(768, 512, hidden_ratio=0.5)
    x = torch.randn(4, 768)
    out = ec(x)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"
    print(f'EmbeddingCompressor: {x.shape} -> {out.shape} ✓')


def test_squeeze_excitation():
    se = SqueezeExcitation(512, reduction=16)
    x = torch.randn(2, 512, 16, 16)
    out = se(x)
    assert out.shape == (2, 512, 16, 16), f"Expected (2, 512, 16, 16), got {out.shape}"
    print(f'SqueezeExcitation: {x.shape} -> {out.shape} ✓')


def test_width_scaler():
    """WidthScaler scales OUT channels only, preserving in_channels."""
    m = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    scaled = WidthScaler(m, 0.5)
    # in_channels should stay 64, out_channels should become 64 (from 128)
    assert scaled.module[0].in_channels == 64
    assert scaled.module[0].out_channels == 64
    assert scaled.module[1].num_features == 64
    x = torch.randn(2, 64, 16, 16)
    out = scaled(x)
    assert out.shape == (2, 64, 16, 16), f"Expected (2, 64, 16, 16), got {out.shape}"
    # Verify parameter reduction
    orig_params = sum(p.numel() for p in m.parameters())
    scaled_params = sum(p.numel() for p in scaled.parameters())
    assert scaled_params < orig_params, f"Expected param reduction: {orig_params} -> {scaled_params}"
    print(f'WidthScaler: {x.shape} -> {out.shape}, params {orig_params} -> {scaled_params} ✓')


def test_capacity_reduction_head():
    head = CapacityReductionHead(768, 1000)
    x = torch.randn(8, 768)
    out = head(x)
    assert out.shape == (8, 1000), f"Expected (8, 1000), got {out.shape}"
    print(f'CapacityReductionHead: {x.shape} -> {out.shape} ✓')


def test_backward_pass():
    """Ensure gradients flow through all modules.
    
    Fixed: each module type gets its correct input dimensionality.
    - 2D modules (EmbeddingCompressor, CapacityReductionHead): 2D tensor (B, C)
    - 4D modules (BottleneckBlock, ProgressiveNarrowing, DepthwiseFinal, SqueezeExcitation): 4D tensor (B, C, H, W)
    """
    # 4D input modules — each needs correct in_channels for first block
    # BottleneckBlock(128,64): 128 in
    # ProgressiveNarrowing([256,128,64]): first block needs 256 in
    # DepthwiseFinal(64,64): 64 in
    # SqueezeExcitation(128): 128 in
    mod_4d_cases = [
        ("BottleneckBlock(128->64)", BottleneckBlock(128, 64, bottleneck_ratio=0.25), (2, 128, 16, 16)),
        ("ProgressiveNarrowing([256,128,64])", ProgressiveNarrowing([256, 128, 64], bottleneck_ratio=0.25), (2, 256, 16, 16)),
        ("DepthwiseFinal(64->64)", DepthwiseFinal(64, 64), (2, 64, 16, 16)),
        ("SqueezeExcitation(128)", SqueezeExcitation(128, reduction=8), (2, 128, 16, 16)),
    ]
    # 2D input modules
    mod_2d_cases = [
        ("EmbeddingCompressor(512->256)", EmbeddingCompressor(512, 256, hidden_ratio=0.5)),
        ("CapacityReductionHead(384->1000)", CapacityReductionHead(384, 1000)),
    ]

    for name, mod, shape in mod_4d_cases:
        mod.train()
        x = torch.randn(*shape).requires_grad_(True)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, f"Gradient missing for {name}"
        print(f'  4D [{name}]: input={list(shape)}, output={list(out.shape)} ✓')

    for name, mod in mod_2d_cases:
        mod.train()
        # EmbeddingCompressor(512, 256) needs (B, 512); CapacityReductionHead(384, 1000) needs (B, 384)
        in_features = 512 if "Embedding" in name else 384
        x = torch.randn(2, in_features).requires_grad_(True)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, f"Gradient missing for {name}"
        print(f'  2D [{name}]: input={list(x.shape)}, output={list(out.shape)} ✓')

    print(f'Backward pass: all {len(mod_4d_cases) + len(mod_2d_cases)} modules ✓')


if __name__ == "__main__":
    print("=" * 50)
    print("last_stage_capacity library — test suite")
    print("=" * 50)
    test_bottleneck_block()
    test_progressive_narrowing()
    test_depthwise_final()
    test_embedding_compressor()
    test_squeeze_excitation()
    test_width_scaler()
    test_capacity_reduction_head()
    test_backward_pass()  # was missing!
    print("=" * 50)
    print("All tests passed.")
