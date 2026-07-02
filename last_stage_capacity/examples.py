"""
Demo: Attaching last-stage capacity reduction to a ResNet backbone.
Run: python examples.py
"""
import os, sys
# Allow running as: python capacity_reduction/examples.py (from echo/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SCRIPT_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
import torch.nn as nn
from last_stage_capacity import (
    BottleneckBlock, ProgressiveNarrowing, DepthwiseFinal,
    EmbeddingCompressor, CapacityReductionHead, WidthScaler
)


class ResNetWithLastStageReduction(nn.Module):
    """
    ResNet50 where only the final stage (layer4) uses capacity reduction.
    The bottleneck blocks replace the original layer4 convolutions.
    
    This demonstrates the key use case: keep early layers at full capacity
    (they capture low-level features), reduce only in the final stage
    where compression into task-specific representations happens.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # Early layers stay at full capacity
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stage 1-3: full capacity (simplified)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)

        # Stage 4: LAST-STAGE CAPACITY REDUCTION
        # Replace the standard conv with bottleneck blocks that narrow
        # from 256 -> 512 (typical ResNet50 layer4) using a reduced bottleneck
        self.layer4 = ProgressiveNarrowing(
            [256, 192, 128, 512],  # narrowed from [256, 512, 512]
            bottleneck_ratio=0.25,
            use_se=True,
        )

        # Final head with embedding compression
        self.head = CapacityReductionHead(512, num_classes, hidden_ratio=0.5)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BottleneckBlock(in_ch, out_ch, stride=stride, bottleneck_ratio=0.25)]
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_ch, out_ch, bottleneck_ratio=0.25))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=[2, 3])  # GAP
        return self.head(x)


class ImageClassifierWithCompressor(nn.Module):
    """
    Generic classifier: any CNN backbone + EmbeddingCompressor head.
    Demonstrates using EmbeddingCompressor as a drop-in head upgrade.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.compressor = EmbeddingCompressor(
            in_features=embed_dim,
            out_features=embed_dim,
            hidden_ratio=0.5,
            dropout=0.2,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        compressed = self.compressor(features)
        return self.classifier(compressed)


def demo():
    print("=" * 60)
    print("Last-Stage Capacity Reduction — demo")
    print("=" * 60)

    # 1. ResNet with ProgressiveNarrowing on final stage
    model = ResNetWithLastStageReduction(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"\nResNetWithLastStageReduction:")
    print(f"  Input:    {x.shape}")
    print(f"  Output:   {out.shape} (logits for 1000 classes)")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params:   {total:,}")

    # 2. Parameter efficiency comparison
    print("\nBottleneck efficiency (same in/out channels):")
    standard = BottleneckBlock(256, 256, bottleneck_ratio=1.0)  # no reduction
    reduced = BottleneckBlock(256, 256, bottleneck_ratio=0.25)  # bottleneck
    std_params = sum(p.numel() for p in standard.parameters())
    red_params = sum(p.numel() for p in reduced.parameters())
    ratio = red_params / std_params
    print(f"  Standard block params:  {std_params:,}")
    print(f"  Bottleneck block params: {red_params:,}")
    print(f"  Reduction ratio:         {ratio:.1%}")

    # 3. Width scaling demo
    print("\nWidthScaler (50% width reduction):")
    # WidthScaler wraps a module and scales all channel dimensions by 0.5
    # Input Conv2d(64, 128) becomes Conv2d(32, 64)
    original_module = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    scaled_module = WidthScaler(original_module, 0.5)
    x = torch.randn(1, 64, 32, 32)
    out_scaled = scaled_module(x)
    print(f"  WidthScale: Conv2d(64,128) -> scaled Conv2d(32,64)")
    print(f"  Output shape: {out_scaled.shape}")
    print(f"  Param reduction: {sum(p.numel() for p in original_module.parameters()):,} -> {sum(p.numel() for p in scaled_module.parameters()):,}")


if __name__ == "__main__":
    demo()
