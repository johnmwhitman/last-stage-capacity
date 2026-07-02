"""Test ViT and ConvNeXt integration with capacity reduction.

ViT is a transformer — attach_final_stage_reduction should raise ValueError.
ConvNeXt is a CNN — should work normally.
"""
import sys

import torch
import timm

from last_stage_capacity.timm_integration import (
    describe_timm_model,
    attach_final_stage_reduction,
    replace_classifier_head,
)


def test_describe_vit():
    """ViT classifier is model.head (Linear)."""
    print('=== ViT tiny describe_timm_model ===')
    info = describe_timm_model('vit_tiny_patch16_224')
    assert info['num_params'] > 0, f"Expected params > 0, got {info['num_params']}"
    assert info['classifier'] is not None, "ViT classifier should not be None"
    assert info['classifier']['name'] == 'head', f"Expected 'head', got {info['classifier']['name']}"
    assert info['classifier']['type'] == 'Linear', f"Expected 'Linear', got {info['classifier']['type']}"
    assert info['classifier']['out_features'] == 1000, f"Expected 1000 classes, got {info['classifier']['out_features']}"
    print(f"  params: {info['num_params']:,}")
    print(f"  classifier: {info['classifier']}")
    print('  PASS')


def test_describe_convnext():
    """ConvNeXt classifier is model.head.fc (Linear inside NormMlpClassifierHead)."""
    print('\n=== ConvNeXt tiny describe_timm_model ===')
    info = describe_timm_model('convnext_tiny')
    assert info['num_params'] > 0, f"Expected params > 0, got {info['num_params']}"
    assert info['classifier'] is not None, "ConvNeXt classifier should not be None"
    assert info['classifier']['name'] == 'head.fc', f"Expected 'head.fc', got {info['classifier']['name']}"
    assert info['classifier']['type'] == 'Linear', f"Expected 'Linear', got {info['classifier']['type']}"
    assert info['classifier']['out_features'] == 1000, f"Expected 1000 classes, got {info['classifier']['out_features']}"
    print(f"  params: {info['num_params']:,}")
    print(f"  classifier: {info['classifier']}")
    print('  PASS')


def test_vit_raises_on_attach():
    """ViT should raise ValueError for attach_final_stage_reduction (no pooling layer)."""
    print('\n=== ViT attach_final_stage_reduction raises ValueError ===')
    try:
        attach_final_stage_reduction('vit_tiny_patch16_224', reduction_ratio=0.5, block_type='se')
        assert False, "Expected ValueError for ViT"
    except ValueError as e:
        assert 'transformer' in str(e).lower(), f"Error should mention 'transformer', got: {e}"
        print(f"  Raised ValueError (expected): {e}")
        print('  PASS')


def test_convnext_attach_reduction():
    """ConvNeXt + SE reduction forward pass."""
    print('\n=== ConvNeXt tiny + SE reduction (ratio=0.5) ===')
    conv_red = attach_final_stage_reduction('convnext_tiny', reduction_ratio=0.5, block_type='se')
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = conv_red(x)
    assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"
    print(f"  output: {out.shape}")
    print('  PASS')


def test_convnext_conditional_reduction():
    """ConvNeXt + conditional reduction forward pass."""
    print('\n=== ConvNeXt tiny + conditional reduction (ratio=0.5) ===')
    conv_red = attach_final_stage_reduction('convnext_tiny', reduction_ratio=0.5, block_type='conditional')
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = conv_red(x)
    assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"
    print(f"  output: {out.shape}")
    print('  PASS')


def test_vit_replace_head():
    """ViT supports replace_classifier_head (different from attach_final_stage_reduction)."""
    print('\n=== ViT tiny + replace_classifier_head ===')
    reheaded = replace_classifier_head('vit_tiny_patch16_224', num_classes=10, hidden_ratio=0.5)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = reheaded(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    print(f"  output: {out.shape}")
    print('  PASS')


if __name__ == '__main__':
    test_describe_vit()
    test_describe_convnext()
    test_vit_raises_on_attach()
    test_convnext_attach_reduction()
    test_convnext_conditional_reduction()
    test_vit_replace_head()
    print('\n' + '=' * 50)
    print('ALL TESTS PASSED')
    print('=' * 50)
