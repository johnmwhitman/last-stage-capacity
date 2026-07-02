# Last-Stage Capacity Reduction

Patterns for progressively narrowing neural network representations in the final stages of a model — where compression into task-specific representations happens.

## Core Principle

Early layers capture low-level features; later layers must compress these into task-specific representations. Strategic narrowing at this stage forces beneficial compression, improves regularization, and reduces inference cost without significant accuracy loss.

Grounded in: Huang et al., "Exploring Architectural Ingredients of Adversarially Robust DNNs" (NeurIPS 2021).

## Two Tracks

**Classification track** — for backbones (ResNet, ConvNeXt, ViT, EfficientNet), embedding compression, and classification heads:

| Block | Use case |
|-------|----------|
| `BottleneckBlock` | ResNet-style residual bottleneck with configurable ratio |
| `ProgressiveNarrowing` | Stepwise width reduction over N stages (512→256→128→64) |
| `WidthScaler` | Uniform width multiplier for any Sequential module |
| `EmbeddingCompressor` | Pre-classifier embedding dimension reduction |
| `DepthwiseFinal` | MobileNet-style final stage (lowest FLOPs) |
| `SqueezeExcitation` | Standalone channel attention (plug-in) |
| `CapacityReductionHead` | Drop-in classification head with compression |

**Detection track** — for object detection necks (FPN, PAN, BiFPN), segmentation decoders, and multi-scale feature fusion:

| Block | Use case |
|-------|----------|
| `LinearProjectionReduction` | Simple learnable channel projection (4D/3D/2D) |
| `SEReduction` | Squeeze-and-Excitation channel attention + reduction |
| `ConditionalCapacityBlock` | Spatial gating: network learns where reduction is safe |
| `CapacityReductionStack` | Progressive narrowing across multiple stages |

**timm integration** — apply capacity reduction to pretrained models from the [timm](https://github.com/huggingface/pytorch-image-models) library:

| Function | What it does |
|----------|-------------|
| `replace_classifier_head` | Swap the classifier for CapacityReductionHead |
| `timm_feature_extractor` | Create a feature extractor with optional compression |
| `scale_timm_model_width` | Uniformly scale channel width of any timm model |
| `attach_final_stage_reduction` | Attach SE/conditional reduction before global pooling |
| `describe_timm_model` | Inspect parameter count and head structure |

## Installation

Install from source:

```bash
git clone https://github.com/johnmwhitman/last-stage-capacity.git
cd last-stage-capacity
pip install -e .
```

Dependencies: `torch`, `timm`, `numpy`

## Quick Start

### Classification: ResNet with progressive narrowing on the final stage

```python
import torch
import torch.nn as nn
from last_stage_capacity import (
    BottleneckBlock, ProgressiveNarrowing,
    CapacityReductionHead, WidthScaler,
)

# Replace ResNet50's layer4 with progressive narrowing
# Standard: 256 -> 512 (no compression)
# Reduced:  256 -> 192 -> 128 -> 512 (bottleneck at each step)
final_stage = ProgressiveNarrowing(
    [256, 192, 128, 512],
    bottleneck_ratio=0.25,
    use_se=True,
)

# Compress features before the classifier
head = CapacityReductionHead(512, num_classes=1000, hidden_ratio=0.5)
```

### Width scaling: uniformly reduce channel widths

```python
import torch.nn as nn
from last_stage_capacity import WidthScaler

original = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)

# Scale all channel dimensions by 0.5
# Conv2d(64, 128) -> Conv2d(32, 64), BN tracks correctly
scaled = WidthScaler(original, width_scale=0.5)
```

`WidthScaler` handles channel propagation through arbitrary module trees, including ResNet `BasicBlock` and `Bottleneck` blocks, with correct BatchNorm pairing after conv scaling.

### Detection: channel reduction for FPN/PAN necks

```python
import torch
from last_stage_capacity import (
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)

# Simple projection: 256 -> 128 channels
reducer = LinearProjectionReduction(256, 128)
x = torch.randn(2, 256, 32, 32)  # BCHW
out = reducer(x)  # (2, 128, 32, 32)

# SE attention + reduction
se_reducer = SEReduction(256, 128, se_reduction=16)

# Spatial gating: network learns WHERE reduction is safe
conditional = ConditionalCapacityBlock(256, 128)

# Progressive narrowing across multiple FPN levels
stack = CapacityReductionStack([256, 192, 128, 64])
```

### timm integration: attach reduction to pretrained models

```python
from last_stage_capacity.timm_integration import (
    describe_timm_model,
    attach_final_stage_reduction,
    scale_timm_model_width,
    replace_classifier_head,
)

# Inspect a model
info = describe_timm_model('resnet18')
print(f"Parameters: {info['num_params']:,}")

# Attach SE reduction to the final stage
reduced = attach_final_stage_reduction('resnet18', reduction_ratio=0.5, block_type='se')

# Scale the entire model's width by 50%
scaled = scale_timm_model_width('resnet18', width_scale=0.5)

# Replace the classifier with a compressed head
reheaded = replace_classifier_head('resnet18', num_classes=10, hidden_ratio=0.5)
```

## Parameter Reduction Example

Running the included demo (`python examples.py`):

```
Bottleneck efficiency (same in/out channels):
  Standard block params:  590,080
  Bottleneck block params: 148,480
  Reduction ratio:         25.2%

WidthScaler (50% width reduction):
  Conv2d(64,128) -> scaled Conv2d(32,64)
  Param reduction: 75,008 -> 18,944 (74.7%)
```

## Project Structure

```
last-stage-capacity/
├── __init__.py              # Classification track (BottleneckBlock, ProgressiveNarrowing, WidthScaler, etc.)
├── _detection.py            # Detection track (LinearProjectionReduction, SEReduction, ConditionalCapacityBlock, etc.)
├── timm_integration.py      # timm integration (replace_classifier_head, scale_timm_model_width, etc.)
├── examples.py              # Classification demos (ResNet with reduction, width scaling)
├── examples/
│   └── timm_demo.py         # timm integration demo
├── test_*.py                # Test suite
├── pyproject.toml           # Package config
└── README.md
```

## Tests

```bash
# Core library tests (no timm required)
python test_capacity_reduction.py

# Detection track tests
python test_detection_track.py

# timm integration tests (requires timm)
python test_vit_convnext.py
python test_timm_integration.py
```

## Research Context

This library implements patterns from Huang et al., "Exploring Architectural Ingredients of Adversarially Robust DNNs" (NeurIPS 2021), which showed that last-stage capacity reduction improves robustness with fewer parameters. The key insight: early layers capture low-level features that need full capacity, but later layers compress into task-specific representations where strategic narrowing forces beneficial regularization.

## License

Apache-2.0