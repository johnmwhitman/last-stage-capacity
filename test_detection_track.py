"""
Tests for the Detection Track — last-stage capacity reduction for detection necks.

Tests: LinearProjectionReduction, SEReduction, ConditionalCapacityBlock, CapacityReductionStack

Run: python test_detection_track.py
"""
import torch
import torch.nn as nn
from last_stage_capacity._detection import (
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)

def test_linear_projection_reduction():
    """LinearProjectionReduction: 4D/3D/2D tensor support."""
    # 4D: BCHW
    m = LinearProjectionReduction(256, 64)
    x4 = torch.randn(2, 256, 32, 32)
    out4 = m(x4)
    assert out4.shape == (2, 64, 32, 32), f"4D: expected (2,64,32,32), got {out4.shape}"
    print(f"  LinearProjectionReduction 4D: {x4.shape} → {out4.shape} ✓")

    # 3D: BCT (e.g., sequence features)
    m3 = LinearProjectionReduction(512, 128)
    x3 = torch.randn(3, 512, 50)
    out3 = m3(x3)
    assert out3.shape == (3, 128, 50), f"3D: expected (3,128,50), got {out3.shape}"
    print(f"  LinearProjectionReduction 3D: {x3.shape} → {out3.shape} ✓")

    # 2D: BC (e.g., per-token features)
    m2 = LinearProjectionReduction(768, 192)
    x2 = torch.randn(4, 768)
    out2 = m2(x2)
    assert out2.shape == (4, 192), f"2D: expected (4,192), got {out2.shape}"
    print(f"  LinearProjectionReduction 2D: {x2.shape} → {out2.shape} ✓")

    # Gradient check
    out4.mean().backward()
    assert m.proj.weight.grad is not None
    print("  LinearProjectionReduction backward: ✓")


def test_se_reduction():
    """SEReduction: Squeeze-and-Excitation channel attention + channel reduction."""
    m = SEReduction(in_channels=256, reduction_ratio=0.25)
    x = torch.randn(2, 256, 32, 32)
    out = m(x)
    assert out.shape == (2, 64, 32, 32), f"expected (2,64,32,32), got {out.shape}"
    print(f"  SEReduction: {x.shape} → {out.shape} ✓")

    # Check SE has the right sub-modules
    assert hasattr(m, 'squeeze') and hasattr(m, 'excite') and hasattr(m, 'reduce'), "Missing SE layers"
    print("  SEReduction has squeeze/excite/reduce layers ✓")

    # Backward
    out.mean().backward()
    # The excite module (Sequential) contains Linear layers
    assert any(p.grad is not None for p in m.excite.parameters())
    print("  SEReduction backward: ✓")


def test_conditional_capacity_block():
    """ConditionalCapacityBlock: spatial gating — learns WHERE reduction is safe."""
    m = ConditionalCapacityBlock(in_channels=128, reduction_ratio=0.5)
    x = torch.randn(2, 128, 16, 16)
    out = m(x)
    assert out.shape == (2, 64, 16, 16), f"expected (2,64,16,16), got {out.shape}"
    print(f"  ConditionalCapacityBlock: {x.shape} → {out.shape} ✓")

    # Gate should be spatial (H,W) or per-position
    assert hasattr(m, 'gate_conv') and hasattr(m, 'reduce'), "Missing gate components"
    print("  ConditionalCapacityBlock has spatial gate ✓")

    # Backward
    out.mean().backward()
    # The gate_conv is a Sequential with a Conv2d at index 2
    assert any(p.grad is not None for p in m.gate_conv.parameters())
    print("  ConditionalCapacityBlock backward: ✓")


def test_capacity_reduction_stack():
    """CapacityReductionStack: progressive narrowing across multiple detection stages."""
    # Simulate an FPN-style neck with actual channel reduction: 256→128→64→32
    channels = [256, 128, 64, 32]
    stack = CapacityReductionStack(
        channels=channels,
        block_type='se',
    )

    x = torch.randn(1, 256, 32, 32)
    outs = stack(x)
    # Returns a list: [stage0_out, stage1_out, stage2_out]
    # stage0: 256→128 (ratio 0.5), stage1: 128→64 (ratio 0.5), stage2: 64→32 (ratio 0.5)
    assert len(outs) == 3, f"expected 3 stage outputs, got {len(outs)}"
    assert outs[0].shape == (1, 128, 32, 32), f"stage0: expected (1,128,32,32), got {outs[0].shape}"
    assert outs[1].shape == (1, 64, 32, 32), f"stage1: expected (1,64,32,32), got {outs[1].shape}"
    assert outs[2].shape == (1, 32, 32, 32), f"stage2: expected (1,32,32,32), got {outs[2].shape}"
    print(f"  CapacityReductionStack: {x.shape} → {outs[2].shape} across 3 stages ✓")

    # Gradient check on final stage
    outs[-1].mean().backward()
    print("  CapacityReductionStack backward: ✓")


def test_detection_e2e():
    """E2E: detection neck (simplified FPN) with all reduction strategies."""
    # Stage 1: FPN lateral connection → 256 channels
    lateral = torch.randn(2, 256, 32, 32)

    # Apply SE reduction at top of FPN
    se = SEReduction(256, reduction_ratio=0.5)
    reduced = se(lateral)
    assert reduced.shape == (2, 128, 32, 32)
    print(f"  E2E SE in FPN: {lateral.shape} → {reduced.shape} ✓")

    # Apply ConditionalCapacityBlock at a later stage
    cond = ConditionalCapacityBlock(128, reduction_ratio=0.5)
    cond_out = cond(reduced)
    assert cond_out.shape == (2, 64, 32, 32)
    print(f"  E2E Conditional in FPN: {reduced.shape} → {cond_out.shape} ✓")

    # Final projection to 32 output channels
    proj = LinearProjectionReduction(64, 32)
    final = proj(cond_out)
    assert final.shape == (2, 32, 32, 32)
    print(f"  E2E LinearProjection: {cond_out.shape} → {final.shape} ✓")

    print("  Detection E2E: all stages ✓")


def main():
    print("=" * 55)
    print("Detection Track — Test Suite")
    print("=" * 55)

    tests = [
        ("LinearProjectionReduction", test_linear_projection_reduction),
        ("SEReduction", test_se_reduction),
        ("ConditionalCapacityBlock", test_conditional_capacity_block),
        ("CapacityReductionStack", test_capacity_reduction_stack),
        ("E2E detection neck", test_detection_e2e),
    ]

    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            raise

    print("\n" + "=" * 55)
    print("All detection track tests passed.")
    print("=" * 55)


if __name__ == '__main__':
    main()
