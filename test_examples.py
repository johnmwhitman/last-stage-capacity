"""Test suite for last_stage_capacity/examples.py."""
import pytest
import torch
import torch.nn as nn

from last_stage_capacity.examples import (
    ResNetWithLastStageReduction,
    ImageClassifierWithCompressor,
    demo,
)


@pytest.mark.parametrize(
    "batch_size,height,width,num_classes",
    [
        (2, 224, 160, 1000),
        (3, 128, 256, 10),
        (4, 192, 160, 100),
    ],
)
def test_resnet_with_last_stage_reduction_forward_shapes(
    batch_size: int, height: int, width: int, num_classes: int
):
    """Verify ResNetWithLastStageReduction produces correct output shape on non-square inputs."""
    assert batch_size > 1, "Batch size must be greater than one"
    assert height != width, "Input dimensions must be non-square"

    # Randomly initialized model (no pretrained weights, no network calls)
    model = ResNetWithLastStageReduction(num_classes=num_classes)
    model.eval()

    # Synthetic non-square input tensor with batch size > 1
    x = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (
        batch_size,
        num_classes,
    ), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"


@pytest.mark.parametrize(
    "batch_size,height,width,embed_dim,num_classes",
    [
        (2, 160, 224, 64, 10),
        (4, 256, 192, 128, 100),
        (5, 96, 128, 32, 5),
    ],
)
def test_image_classifier_with_compressor_forward_shapes(
    batch_size: int, height: int, width: int, embed_dim: int, num_classes: int
):
    """Verify ImageClassifierWithCompressor produces correct output shape on non-square inputs."""
    assert batch_size > 1, "Batch size must be greater than one"
    assert height != width, "Input dimensions must be non-square"

    # Define a simple synthetic randomly initialized CNN backbone (no pretrained weights, no network calls)
    backbone = nn.Sequential(
        nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(embed_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )

    model = ImageClassifierWithCompressor(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
    )
    model.eval()

    # Synthetic non-square input tensor with batch size > 1
    x = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (
        batch_size,
        num_classes,
    ), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_demo_execution(capsys):
    """Verify the demo function in examples.py executes without error and prints output."""
    demo()
    captured = capsys.readouterr()
    assert "Last-Stage Capacity Reduction — demo" in captured.out
    assert "ResNetWithLastStageReduction:" in captured.out
    assert "Bottleneck efficiency" in captured.out
    assert "WidthScaler" in captured.out
