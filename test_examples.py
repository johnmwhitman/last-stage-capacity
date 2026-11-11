"""Pytest smoke tests for last_stage_capacity/examples.py.

Verifies construction on CPU with random initialization (no pretrained weights
or network downloads), forward-pass output shapes on small synthetic inputs,
and last-stage capacity reduction in ResNetWithLastStageReduction.
"""
import torch
import torch.nn as nn
import pytest

from last_stage_capacity import ProgressiveNarrowing
from last_stage_capacity.examples import (
    ResNetWithLastStageReduction,
    ImageClassifierWithCompressor,
)


def test_resnet_with_last_stage_reduction_forward():
    """Smoke test forward pass of ResNetWithLastStageReduction on CPU."""
    batch_size = 2
    num_classes = 10
    model = ResNetWithLastStageReduction(num_classes=num_classes)
    model.eval()

    # Small synthetic input on CPU
    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, num_classes), (
        f"Expected output shape {(batch_size, num_classes)}, got {out.shape}"
    )


def test_resnet_with_last_stage_reduction_default_classes():
    """Test forward pass with default num_classes=1000."""
    model = ResNetWithLastStageReduction()
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1000), f"Expected output shape (1, 1000), got {out.shape}"


def test_resnet_last_stage_width_actually_reduced():
    """Verify that the ResNet demo's last-stage width is actually reduced."""
    model = ResNetWithLastStageReduction(num_classes=10)

    # layer4 should be ProgressiveNarrowing
    assert isinstance(model.layer4, ProgressiveNarrowing)

    # Channel progression in layer4: [256 -> 192 -> 128 -> 512]
    stage_in_channels = [stage.in_channels for stage in model.layer4.stages]
    stage_out_channels = [stage.out_channels for stage in model.layer4.stages]

    assert stage_in_channels == [256, 192, 128], (
        f"Expected input channels [256, 192, 128], got {stage_in_channels}"
    )
    assert stage_out_channels == [192, 128, 512], (
        f"Expected output channels [192, 128, 512], got {stage_out_channels}"
    )

    # Intermediate stages are narrowed below the final 512 width
    assert stage_out_channels[0] < 512, "Stage 0 out_channels should be narrowed (< 512)"
    assert stage_out_channels[1] < 512, "Stage 1 out_channels should be narrowed (< 512)"
    assert stage_out_channels[1] < stage_out_channels[0], "Stage 1 should be progressively narrower than stage 0"

    # Internal bottleneck channels (hidden dimensions) are compressed via bottleneck_ratio=0.25
    for stage in model.layer4.stages:
        min_dim = min(stage.in_channels, stage.out_channels)
        expected_hidden = max(4, int(min_dim * 0.25))
        actual_hidden = stage.conv1.out_channels
        assert actual_hidden == expected_hidden, (
            f"Expected hidden channels {expected_hidden}, got {actual_hidden}"
        )
        assert actual_hidden < min_dim, (
            f"Hidden channels {actual_hidden} should be strictly less than min(in, out) {min_dim}"
        )

    # Compare total parameter count of layer4 vs an unnarrowed baseline [256, 512, 512, 512]
    reduced_layer4_params = sum(p.numel() for p in model.layer4.parameters())
    unnarrowed_layer4 = ProgressiveNarrowing([256, 512, 512, 512], bottleneck_ratio=0.25, use_se=True)
    unnarrowed_layer4_params = sum(p.numel() for p in unnarrowed_layer4.parameters())

    assert reduced_layer4_params < unnarrowed_layer4_params, (
        f"Expected reduced layer4 params ({reduced_layer4_params}) to be less than "
        f"unnarrowed params ({unnarrowed_layer4_params})"
    )


def test_image_classifier_with_compressor_forward():
    """Smoke test forward pass of ImageClassifierWithCompressor on CPU."""
    batch_size = 4
    embed_dim = 64
    num_classes = 5

    # Small synthetic CNN backbone on CPU
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, embed_dim, kernel_size=3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )

    classifier = ImageClassifierWithCompressor(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
    )
    classifier.eval()

    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        out = classifier(x)

    assert out.shape == (batch_size, num_classes), (
        f"Expected output shape {(batch_size, num_classes)}, got {out.shape}"
    )


def test_models_backward_pass():
    """Verify gradient flow through both demo models on synthetic data."""
    # Test ResNetWithLastStageReduction backward
    resnet = ResNetWithLastStageReduction(num_classes=4)
    resnet.train()
    x_res = torch.randn(2, 3, 32, 32, requires_grad=True)
    out_res = resnet(x_res)
    loss_res = out_res.sum()
    loss_res.backward()
    assert x_res.grad is not None, "Gradients should flow to ResNet input"

    # Test ImageClassifierWithCompressor backward
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    clf = ImageClassifierWithCompressor(backbone=backbone, embed_dim=16, num_classes=3)
    clf.train()
    x_clf = torch.randn(2, 3, 16, 16, requires_grad=True)
    out_clf = clf(x_clf)
    loss_clf = out_clf.sum()
    loss_clf.backward()
    assert x_clf.grad is not None, "Gradients should flow to Classifier input"


if __name__ == "__main__":
    test_resnet_with_last_stage_reduction_forward()
    test_resnet_with_last_stage_reduction_default_classes()
    test_resnet_last_stage_width_actually_reduced()
    test_image_classifier_with_compressor_forward()
    test_models_backward_pass()
    print("All tests in test_examples.py passed.")
