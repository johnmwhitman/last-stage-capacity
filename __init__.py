"""
Last-Stage Capacity Reduction Library
=====================================
Patterns for progressively narrowing neural network representations
in the later stages of a model.

Core principle: Early layers capture low-level features; later layers
must compress these into task-specific representations. Strategic narrowing
at this stage forces beneficial compression, improves regularization,
and reduces inference cost without significant accuracy loss.

Grounded in: Huang et al., "Exploring Architectural Ingredients of
Adversarially Robust DNNs" (NeurIPS 2021) — verified paper showing
last-stage reduction improves robustness with fewer parameters.

=====================================================================
Two complementary submodules:
=====================================================================

DETECTION TRACK (last_stage_capacity_reduction.py)
---------------------------------------------------
For object detection necks (FPN, PAN, BiFPN), segmentation decoders,
and multi-scale feature fusion stages.

Classes:
  LinearProjectionReduction — simple learnable channel projection (4D/3D/2D)
  SEReduction              — Squeeze-and-Excitation channel attention + reduction
  ConditionalCapacityBlock  — spatial gating: network learns where reduction is safe
  CapacityReductionStack    — progressive narrowing across multiple stages

CLASSIFICATION TRACK (capacity_reduction/*.py)
----------------------------------------------
For classification backbones, embedding compression, and classification heads.

Classes:
  BottleneckBlock          — ResNet-style residual bottleneck
  ProgressiveNarrowing     — stepwise width reduction over N stages
  WidthScaler              — uniform width multiplier for any Sequential module
  EmbeddingCompressor      — pre-classifier embedding dimension reduction
  DepthwiseFinal           — Mobilenet-style final stage (lowest FLOPs)
  SqueezeExcitation        — standalone channel attention (plug-in)
  CapacityReductionHead    — drop-in classification head with compression

TIMM INTEGRATION (timm_integration.py)
-------------------------------------
Apply capacity reduction to pretrained models from the timm library.

Functions:
  replace_classifier_head       — swap the classifier for CapacityReductionHead
  timm_feature_extractor        — create a feature extractor with optional compression
  scale_timm_model_width        — uniformly scale channel width of any timm model
  attach_final_stage_reduction  — attach SE/conditional reduction before global pooling
  describe_timm_model           — inspect parameter count and head structure

=====================================================================
Quick reference — which block for which use case:

Use case                                      | Best block
---------------------------------------------|---------------------------
Fixed channel reduction, minimal params        | LinearProjectionReduction
Need channel attention before reducing         | SEReduction
Different spatial locations need different cap | ConditionalCapacityBlock
Replacing ResNet/VGG final blocks              | BottleneckBlock
Gradual narrowing (512→256→128→64)            | ProgressiveNarrowing
Uniform experiment across width budgets        | WidthScaler
High-dim features → compact classifier        | EmbeddingCompressor
Mobile deployment, lowest compute              | DepthwiseFinal
Adding attention to any existing block        | SqueezeExcitation
Drop-in replacement for nn.Linear head        | CapacityReductionHead
Applying reduction to timm pretrained models  | timm_integration module

=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal

# ------------------------------------------------------------------
# Classification track (this file)
# ------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """
    Residual bottleneck: expand -> reduce -> expand.
    Final stage capacity reduction through a narrow bottleneck.
    
    Architecture: x -> 1x1conv(in, hidden) -> 3x3conv(hidden, hidden) -> 1x1conv(hidden, out) -> +x
    The 'hidden' dimension is the bottleneck. Making it smaller than in/out
    is what achieves last-stage capacity reduction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: float = 0.25,
        stride: int = 1,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = max(4, int(min(in_channels, out_channels) * bottleneck_ratio))

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = None
        if use_se:
            se_channels = max(4, hidden_dim // se_reduction)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.shortcut = nn.Identity() if (in_channels == out_channels and stride == 1) else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = out * self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ProgressiveNarrowing(nn.Module):
    """
    Apply stepwise width reduction over N stages.
    Each stage reduces channels by a narrowing factor.
    
    Example: [512, 256, 128, 64] over 3 narrowing steps.
    Useful for gradually compressing feature maps before a classification head.
    """

    def __init__(
        self,
        channels: List[int],
        bottleneck_ratio: float = 0.25,
        use_se: bool = True,
        se_reduction: int = 16,
    ):
        """
        Args:
            channels: List of channel dimensions, e.g. [512, 256, 128, 64]
            bottleneck_ratio: How narrow the bottleneck is relative to min(in, out)
            use_se: Whether to use Squeeze-Excitation in each block
            se_reduction: SE reduction ratio
        """
        super().__init__()
        assert len(channels) >= 2, "Need at least input and output channels"
        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.stages.append(
                BottleneckBlock(
                    channels[i],
                    channels[i + 1],
                    bottleneck_ratio=bottleneck_ratio,
                    stride=1,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
            )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class WidthScaler(nn.Module):
    """
    Uniformly scale channel widths of a module by a fixed factor.
    Useful for scaling down the final classification head or
    a set of final layers without redesigning the architecture.

    Rebuilds the module graph with scaled channel dimensions for
    Conv2d, Linear, BatchNorm1d, BatchNorm2d, and LayerNorm layers.
    Preserves all other layer types (ReLU, Dropout, MaxPool, etc.).
    """

    def __init__(self, module: nn.Module, width_scale: float):
        super().__init__()
        import copy
        # Work on a deep copy so the original module is preserved.
        # This allows callers to compare original.parameters() vs scaled.parameters().
        self.module = self._rebuild_with_scaling(copy.deepcopy(module), width_scale)
        self.width_scale = width_scale

    def _rebuild_with_scaling(self, module: nn.Module, scale: float,
                               prev_out_channels: int | None = None) -> nn.Module:
        """Rebuild a module tree with channel widths scaled by `scale`.

        prev_out_channels is the output-channel count of the layer that feeds
        into this module. For the first child of a Sequential chain, this is
        the input channels of the Sequential itself. For isolated modules, None.
        """
        # Known container types — delegate with channel tracking
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            return self._rebuild_container(module, scale, prev_out_channels)

        # Single channel layer
        if self._is_channel_layer(module):
            rebuilt, _ = self._build_scaled_layer(module, scale, prev_out_channels)
            return rebuilt

        # Plain nn.Module root (e.g. ResNet): check for explicit block types first.
        # BasicBlock and Bottleneck must be handled by _rebuild_block to preserve
        # the internal channel flow (conv1→bn1→act→conv2→bn2, plus skip connection).
        name = type(module).__name__
        if name in ('BasicBlock', 'Bottleneck'):
            new_block, eff_out = self._rebuild_block(module, scale, prev_out_channels)
            return new_block

        # For other plain modules (e.g. ResNet root itself), walk named children.
        prev = prev_out_channels
        for name, child in list(module.named_children()):
            new_child = self._rebuild_with_scaling(child, scale, prev)
            if new_child is not child:
                module.register_module(name, new_child)
            child_eff = self._get_output_channels(new_child)
            if child_eff is not None:
                prev = child_eff
            # If child_eff is None (channel-preserving layer like pooling/flatten):
            # prev is unknown, but we don't reset it — the layer preserves channels
            # from the previous layer. The last known channel count remains valid.
        return module

    def _is_channel_layer(self, layer: nn.Module) -> bool:
        """Return True for layers that carry channel dimensions we scale."""
        return isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d,
                                   nn.BatchNorm1d, nn.LayerNorm))

    def _build_scaled_layer(self, layer: nn.Module, scale: float, prev_out_channels: int | None = None) -> tuple[nn.Module, int | None]:
        """
        Rebuild a layer with scaled channel dimensions.

        Returns (rebuilt_layer, effective_out_channels) so the caller can
        pass effective_out_channels as prev_out_channels to the next call.
        This ensures intermediate layers get correct in_channels after scaling.
        """
        if isinstance(layer, nn.Conv2d):
            in_c = prev_out_channels if prev_out_channels is not None else layer.in_channels
            out_c = max(1, int(layer.out_channels * scale))
            return (
                nn.Conv2d(
                    in_c, out_c,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias is not None,
                ),
                out_c,
            )
        elif isinstance(layer, nn.Linear):
            # Linear layers: in_features = prev (scaled channel count from preceding layer).
            # out_features is preserved (e.g. num_classes = 1000 for classifier head).
            in_c = prev_out_channels if prev_out_channels is not None else layer.in_features
            return (
                nn.Linear(in_c, layer.out_features, bias=layer.bias is not None),
                layer.out_features,
            )
        elif isinstance(layer, nn.BatchNorm2d):
            # BN follows a conv that was JUST rebuilt (prev_out_channels is already
            # scaled from that conv). BN.num_features must match the conv output —
            # DO NOT scale again. BN just normalizes the scaled channels.
            num_features = prev_out_channels if prev_out_channels is not None else layer.num_features
            return (
                nn.BatchNorm2d(num_features, eps=layer.eps, momentum=layer.momentum, affine=layer.affine, track_running_stats=layer.track_running_stats),
                num_features,
            )
        elif isinstance(layer, nn.BatchNorm1d):
            num_features = prev_out_channels if prev_out_channels is not None else layer.num_features
            return (
                nn.BatchNorm1d(num_features, eps=layer.eps, momentum=layer.momentum, affine=layer.affine, track_running_stats=layer.track_running_stats),
                num_features,
            )
        elif isinstance(layer, nn.LayerNorm):
            normalized_shape = layer.normalized_shape
            if isinstance(normalized_shape, int):
                normalized_shape = max(1, int(normalized_shape * scale))
            ln = nn.LayerNorm(normalized_shape, eps=layer.eps, elementwise_affine=layer.elementwise_affine)
            return (ln, normalized_shape)
        else:
            return (layer, None)

    def _rebuild_sequential(self, module: nn.Sequential, scale: float,
                             prev_out_channels: int | None = None) -> nn.Sequential:
        """Rebuild a Sequential with channel propagation through the chain."""
        new_children = []
        prev = prev_out_channels
        # Track when the previous child was a Conv2d (so subsequent BN should use its output directly)
        prev_was_conv = False
        for child in module.children():
            if isinstance(child, (nn.Sequential, nn.ModuleList)):
                rebuilt = self._rebuild_container(child, scale, prev)
                eff_out = self._get_output_channels(rebuilt)
                prev_was_conv = False
            elif isinstance(child, nn.Conv2d):
                rebuilt, eff_out = self._build_scaled_layer(child, scale, prev)
                prev_was_conv = True
            elif isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BN following a Conv2d: use conv's output channels directly, no extra scaling.
                if prev_was_conv and prev is not None:
                    num_f = prev
                else:
                    num_f = max(1, int((prev if prev is not None else child.num_features) * scale))
                if isinstance(child, nn.BatchNorm2d):
                    rebuilt = nn.BatchNorm2d(num_f, eps=child.eps, momentum=child.momentum,
                                           affine=child.affine, track_running_stats=child.track_running_stats)
                else:
                    rebuilt = nn.BatchNorm1d(num_f, eps=child.eps, momentum=child.momentum,
                                           affine=child.affine, track_running_stats=child.track_running_stats)
                eff_out = num_f
                prev_was_conv = False
            elif self._is_channel_layer(child):
                rebuilt, eff_out = self._build_scaled_layer(child, scale, prev)
                prev_was_conv = False
            else:
                rebuilt, eff_out = self._rebuild_block(child, scale, prev)
                prev_was_conv = False
            new_children.append(rebuilt)
            prev = eff_out
        return nn.Sequential(*new_children)

    def _rebuild_block(self, block: nn.Module, scale: float,
                       prev_out_channels: int | None = None) -> tuple[nn.Module, int]:
        """Rebuild a BasicBlock/Bottleneck with channel propagation.

        Returns (rebuilt_block, block_out_channels).
        Key insight: a BatchNorm that follows a Conv2d uses the conv's output channels
        (which we've already scaled). BN should NOT apply additional scaling.
        We track prev_conv_output separately to handle this.
        """
        new_block = block.__class__.__new__(block.__class__)
        nn.Module.__init__(new_block)

        for k, v in block.__dict__.items():
            if not isinstance(v, nn.Module):
                new_block.__dict__[k] = v

        # Track channel state through the block:
        # - block_input: the block's own input channels (identity path, for downsample)
        # - prev_conv_output: output channels of the most recent Conv2d (for BN pairing)
        # - block_prev_out: general channel tracker for passthrough layers
        block_input = prev_out_channels
        prev_conv_output = None
        block_prev_out = prev_out_channels

        children_list = list(block.named_children())
        i = 0
        while i < len(children_list):
            name, child = children_list[i]

            if isinstance(child, nn.Conv2d):
                in_c = block_prev_out if block_prev_out is not None else child.in_channels
                out_c = max(1, int(child.out_channels * scale))
                new_block.register_module(name, nn.Conv2d(
                    in_c, out_c, kernel_size=child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    bias=child.bias is not None))
                prev_conv_output = out_c
                block_prev_out = out_c

            elif isinstance(child, nn.BatchNorm2d):
                # BN that follows a Conv2d: use the conv's output directly, no extra scaling.
                # If prev_conv_output is None, BN doesn't follow a recent conv — treat as passthrough.
                if prev_conv_output is not None:
                    new_block.register_module(name, nn.BatchNorm2d(
                        prev_conv_output, eps=child.eps, momentum=child.momentum,
                        affine=child.affine, track_running_stats=child.track_running_stats))
                    block_prev_out = prev_conv_output
                    prev_conv_output = None
                else:
                    # BN without preceding conv (edge case): use block_prev_out
                    num_f = block_prev_out if block_prev_out is not None else child.num_features
                    new_block.register_module(name, nn.BatchNorm2d(
                        num_f, eps=child.eps, momentum=child.momentum,
                        affine=child.affine, track_running_stats=child.track_running_stats))
                    block_prev_out = num_f

            elif isinstance(child, nn.BatchNorm1d):
                num_f = block_prev_out if block_prev_out is not None else child.num_features
                new_block.register_module(name, nn.BatchNorm1d(
                    num_f, eps=child.eps, momentum=child.momentum,
                    affine=child.affine, track_running_stats=child.track_running_stats))
                block_prev_out = num_f

            elif isinstance(child, nn.Linear):
                # Linear layer: use prev as in_features (this is the scaled channel count).
                # out_features is preserved (num_classes for classifier, or original dim for hidden).
                in_c = block_prev_out if block_prev_out is not None else child.in_features
                new_block.register_module(name, nn.Linear(in_c, child.out_features, bias=child.bias is not None))
                block_prev_out = child.out_features

            elif isinstance(child, (nn.Sequential, nn.ModuleList)):
                ds_in = block_input if block_input is not None else prev_out_channels
                rebuilt = self._rebuild_container(child, scale, ds_in)
                new_block.register_module(name, rebuilt)
                block_prev_out = self._get_output_channels(rebuilt)

            elif isinstance(child, nn.LayerNorm):
                ns = child.normalized_shape
                if isinstance(ns, int):
                    ns = max(1, int(ns * scale))
                new_block.register_module(name, nn.LayerNorm(ns, eps=child.eps, elementwise_affine=child.elementwise_affine))
                block_prev_out = ns

            else:
                # Non-channel layer: keep as-is, preserve prev_conv_output
                new_block.register_module(name, child)

            i += 1

        final_out = block_prev_out if block_prev_out is not None else self._get_output_channels(new_block)
        return new_block, final_out if final_out is not None else 1

    def _rebuild_layernorm(self, block: nn.Module, ln: nn.LayerNorm, scale: float,
                           name: str, prev_out: int | None) -> int:
        ns = ln.normalized_shape
        if isinstance(ns, int):
            ns = max(1, int(ns * scale))
        new_block = nn.LayerNorm(ns, eps=ln.eps, elementwise_affine=ln.elementwise_affine)
        block.register_module(name, new_block)
        return ns

    def _get_output_channels(self, module: nn.Module) -> int | None:
        """Return out_channels / out_features if this module carries channels.

        Returns None only for pure passthrough layers (Identity, ReLU, Dropout,
        avg/max pooling with no learned weights) where we cannot infer channel count.
        For pooling layers that preserve channels (AdaptiveAvgPool, etc.), returns
        the input channel count via the first Conv2d/Linear found in the module.
        """
        # nn.Identity, Dropout, etc.: no Conv2d/Linear inside
        # nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d: preserve channels
        #   but don't contain Conv2d — return None (channel count unchanged from input)
        if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                               nn.MaxPool2d, nn.AvgPool2d, nn.Identity,
                               nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                               nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU)):
            return None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m.out_channels
            if isinstance(m, nn.Linear):
                return m.out_features
        return None

    def _rebuild_container(self, module: nn.Module, scale: float,
                           prev_out_channels: int | None = None) -> nn.Module:
        if isinstance(module, nn.Sequential):
            # Rebuild the Sequential in-place by replacing each child.
            # prev is the input channel count to this Sequential (from parent layer's output).
            prev = prev_out_channels
            for name, child in module.named_children():
                new_child = self._rebuild_with_scaling(child, scale, prev)
                if new_child is not child:
                    module.register_module(name, new_child)
                child_eff = self._get_output_channels(new_child)
                if child_eff is not None:
                    prev = child_eff
            return module
        elif isinstance(module, nn.ModuleList):
            rebuilt_list = []
            prev = prev_out_channels
            for m in module:
                new_child = self._rebuild_with_scaling(m, scale, prev)
                child_eff = self._get_output_channels(new_child)
                if child_eff is not None:
                    prev = child_eff
                rebuilt_list.append(new_child)
            return nn.ModuleList(rebuilt_list)
        else:
            # Plain nn.Module: check if it's a residual block type we handle explicitly.
            # BasicBlock and Bottleneck need _rebuild_block (not child-level iteration).
            name = type(module).__name__
            if name in ('BasicBlock', 'Bottleneck'):
                new_block, eff_out = self._rebuild_block(module, scale, prev_out_channels)
                return new_block
            # For other plain modules, walk named children (channel propagation).
            prev = prev_out_channels
            for name, child in list(module.named_children()):
                new_child = self._rebuild_with_scaling(child, scale, prev)
                if new_child is not child:
                    module.register_module(name, new_child)
                child_eff = self._get_output_channels(new_child)
                if child_eff is not None:
                    prev = child_eff
            return module

    def forward(self, x):
        return self.module(x)


class EmbeddingCompressor(nn.Module):
    """
    Reduce embedding dimension before a classification/regression head.
    Projects high-dimensional representations to a compact form.

    Architecture: x -> LayerNorm -> Linear(in, hidden) -> ReLU -> Linear(hidden, out)
    Forces compression through the narrow hidden dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.1,
        norm: Literal["layernorm", "batchnorm", "none"] = "layernorm",
    ):
        super().__init__()
        hidden_dim = max(4, int(min(in_features, out_features) * hidden_ratio))
        self.norm = (
            nn.LayerNorm(in_features) if norm == "layernorm"
            else nn.BatchNorm1d(in_features) if norm == "batchnorm"
            else nn.Identity()
        )
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        return x


class DepthwiseFinal(nn.Module):
    """
    Replace standard convolutions in the final stage with depthwise convolutions.
    Depthwise separable convolutions dramatically reduce parameters and FLOPs
    by splitting spatial and channel mixing.
    
    Final stage architecture:
    pointwise 1x1 (channels already compressed) + depthwise 3x3 + pointwise 1x1
    
    This is the 'mobilenet-style' final stage — low compute cost for the last
    layers where spatial pattern refinement happens, not channel mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_ratio: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        hidden_dim = max(4, int(in_channels * expansion_ratio))

        self.pw1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dw = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.pw2 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity() if (in_channels == out_channels and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.pw1(x)))
        out = self.bn2(self.dw(out))
        out = self.bn3(self.pw2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SqueezeExcitation(nn.Module):
    """
    Lightweight channel attention mechanism with reduced complexity.
    Applies global context across channels, then re-weights them.
    
    Unlike standard SE which operates at full channel count, this version
    uses an intermediate reduction ratio for efficiency at high channel counts.
    
    This is a standalone SE module that can be plugged into any existing block.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CapacityReductionHead(nn.Module):
    """
    Drop-in classification head that applies last-stage capacity reduction
    before the final linear classifier.
    
    Architecture: features -> EmbeddingCompressor -> dropout -> classifier
    Compresses the feature representation before projecting to class logits.
    Useful when features come from a high-dimensional backbone.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.2,
        norm: Literal["layernorm", "batchnorm", "none"] = "layernorm",
    ):
        super().__init__()
        self.compressor = EmbeddingCompressor(
            in_features=in_features,
            out_features=in_features,
            hidden_ratio=hidden_ratio,
            dropout=dropout,
            norm=norm,
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        compressed = self.compressor(x)
        return self.classifier(compressed)


# ------------------------------------------------------------------
# Detection track (inside this package)
# ------------------------------------------------------------------
from ._detection import (
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)

__all__ = [
    # Classification track
    "BottleneckBlock",
    "ProgressiveNarrowing",
    "WidthScaler",
    "EmbeddingCompressor",
    "DepthwiseFinal",
    "SqueezeExcitation",
    "CapacityReductionHead",
    # Detection track
    "LinearProjectionReduction",
    "SEReduction",
    "ConditionalCapacityBlock",
    "CapacityReductionStack",
]
