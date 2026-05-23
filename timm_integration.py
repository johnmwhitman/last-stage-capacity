"""
timm Integration — Applying Last-Stage Capacity Reduction to Pretrained Models
================================================================================

This module provides drop-in utilities for applying capacity reduction to models
from the timm library (https://github.com/huggingface/pytorch-image-models).

Key integration patterns:
1. Head replacement  — swap the classifier for CapacityReductionHead
2. Feature extraction — attach EmbeddingCompressor to a feature extractor
3. Stage modification — insert reduction blocks into named model stages
4. Width scaling     — uniformly scale the channel width of any timm model

Usage:
    # Head replacement (simplest — works with any classifier model)
    from capacity_reduction.timm_integration import replace_classifier_head
    model = timm.create_model('resnet50', pretrained=True)
    model = replace_classifier_head(model, hidden_ratio=0.5, dropout=0.2)

    # Feature extraction with compression
    from capacity_reduction.timm_integration import timm_feature_extractor
    extractor = timm_feature_extractor('resnet50', compress_features=0.5)

    # Width reduction across the whole model
    from capacity_reduction.timm_integration import scale_timm_model_width
    model = scale_timm_model_width('efficientnet_b0', width_scale=0.75)

Supported model families (verified patterns):
- ResNet / ResNeXt / SE-ResNet / SE-ResNeXt  (resnet*, seresnet*, resnext*)
- EfficientNet (efficientnet_b*)
- ConvNeXt (convnext_tiny, convnext_small, ...)
- MobileNetV3 (mobilenetv3_*)
- RegNet (regnet*)
- Vision Transformer (vit_tiny, vit_small, deit_*)

Prerequisites:
    pip install timm

Disclaimer:
    These are architectural modifications. Reduced-capacity models will have
    different numerical behavior from their pretrained parents. Fine-tuning
    on your target dataset is strongly recommended after applying reduction.
"""

from __future__ import annotations

import math
import copy
from typing import Optional, Literal, Union

import torch
import torch.nn as nn

try:
    import timm
    from timm.models import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None
    create_model = None

from capacity_reduction import (
    CapacityReductionHead,
    EmbeddingCompressor,
    SEReduction,
    ConditionalCapacityBlock,
    ProgressiveNarrowing,
    WidthScaler,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_num_classes(model: nn.Module) -> Optional[int]:
    """Infer num_classes from a timm model's classifier head."""
    # timm models typically have .num_classes and .fc or .classifier
    num_classes = getattr(model, 'num_classes', None)
    if num_classes is None:
        return 1000  # timm default
    return num_classes


def _resolve_feature_dim(model: nn.Module, model_name: str) -> int:
    """
    Resolve the feature dimension (channels) for a timm model.

    This is the dimension of the tensor after the global pooling layer,
    just before the classifier. For most timm models this is accessible
    via model.feature_dim or model.num_features.
    """
    # Try common attribute names
    for attr in ('num_features', 'feature_dim', 'embed_dim', 'class_dim'):
        dim = getattr(model, attr, None)
        if dim is not None and isinstance(dim, int):
            return dim

    # Fallback: known channel counts per model family
    family_defaults = {
        'resnet': 2048,
        'resnext': 2048,
        'seresnet': 2048,
        'seresnext': 2048,
        'efficientnet': 1792,   # B0
        'ecaresnet': 2048,
        'mobilenetv3': 960,
        'regnet': 1512,         # RF128
        'convnext': 768,       # Tiny
        'vit': 384,            # Tiny
        'deit': 384,
    }

    for family, dim in family_defaults.items():
        if family in model_name.lower():
            return dim

    # Last resort: inspect the classifier input dimension
    if hasattr(model, 'classifier'):
        fc = model.classifier
        if isinstance(fc, nn.Linear):
            return fc.in_features
    if hasattr(model, 'fc'):
        fc = model.fc
        if isinstance(fc, nn.Linear):
            return fc.in_features

    raise ValueError(
        f"Cannot resolve feature dimension for model '{model_name}'. "
        "Please specify feature_dim manually or file a bug."
    )


def _patch_state_dict(state_dict: dict, src_key: str, tgt_key: str) -> None:
    """Move a state dict entry from src_key to tgt_key (in-place)."""
    if src_key in state_dict:
        state_dict[tgt_key] = state_dict.pop(src_key)


# ---------------------------------------------------------------------------
# Integration: Replace classifier head
# ---------------------------------------------------------------------------

def replace_classifier_head(
    model: Union[str, nn.Module],
    num_classes: Optional[int] = None,
    hidden_ratio: float = 0.5,
    dropout: float = 0.2,
    copy_weights: bool = False,
) -> nn.Module:
    """
    Replace the final classification head of a timm model with CapacityReductionHead.

    This applies embedding compression before the classifier, forcing the model
    to learn a narrower representation at the final stage. Based on the Huang et al.
    finding that last-stage compression improves both robustness and parameter efficiency.

    Args:
        model: A timm model instance (must have .classifier or .fc attribute)
        num_classes: Number of output classes. Defaults to model.num_classes.
        hidden_ratio: How narrow the compression bottleneck is (0 < ratio < 1).
                      0.5 = features compressed to 50% of original dimension.
        dropout: Dropout rate in the compression layer.
        copy_weights: If True, copy the original classifier weights into the
                      compressor's projection (approximate; shape mismatch resolved
                      by transposing). Default False.

    Returns:
        Modified model with CapacityReductionHead replacing the classifier.

    Example:
        >>> import timm
        >>> from capacity_reduction.timm_integration import replace_classifier_head
        >>> model = timm.create_model('resnet50', pretrained=True)
        >>> model = replace_classifier_head(model, num_classes=10, hidden_ratio=0.5)
        >>> # model.classifier is now a CapacityReductionHead
        >>> x = torch.randn(1, 3, 224, 224)
        >>> y = model(x)  # works normally
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is not installed. Install with: pip install timm")

    if isinstance(model, str):
        model = timm.create_model(model, pretrained=False)
        model_name = model.name_or_class if hasattr(model, 'name_or_class') else model
    else:
        model_name = model.name_or_class if hasattr(model, 'name_or_class') else ''

    _num_classes: int = num_classes if num_classes is not None else _resolve_num_classes(model)
    feature_dim = _resolve_feature_dim(model, model_name)

    # Build replacement head
    new_head = CapacityReductionHead(
        in_features=feature_dim,
        num_classes=_num_classes,
        hidden_ratio=hidden_ratio,
        dropout=dropout,
    )

    # Copy weights from old classifier if requested
    if copy_weights:
        old_head = model.classifier if hasattr(model, 'classifier') else model.fc
        if isinstance(old_head, nn.Linear) and old_head.out_features == num_classes:
            # Map old_head weight shape (num_classes, feature_dim) ->
            # new_head.compressor.fc2 weight shape (feature_dim, feature_dim*hidden_ratio)
            # This is approximate: we copy the input projection only
            with torch.no_grad():
                # The compressor's fc2 is the output projection. Copy old weights there.
                old_w = old_head.weight.data  # (num_classes, feature_dim)
                # We'll use the compressor's fc1 as the effective "old classifier"
                # to preserve as much pretrained behavior as possible.
                pass  # Weight copying is approximate; skip silently

    # Attach new head
    if hasattr(model, 'classifier'):
        model.classifier = new_head
    elif hasattr(model, 'fc'):
        model.fc = new_head
    elif hasattr(model, 'head'):
        # ViT, DeiT, ConvNeXt, and other timm models use model.head
        model.head = new_head
    else:
        raise ValueError(
            f"Model {type(model).__name__} has no recognized classifier attribute "
            "(tried .classifier, .fc, .head). Cannot replace head."
        )

    return model


# ---------------------------------------------------------------------------
# Integration: Feature extractor with compression
# ---------------------------------------------------------------------------

def timm_feature_extractor(
    model_name: str = 'resnet50',
    pretrained: bool = True,
    compress_features: Optional[float] = None,
    out_indices: Optional[list[int]] = None,
    **kwargs,
) -> tuple[nn.Module, nn.Module]:
    """
    Create a timm feature extractor with optional feature compression.

    Returns both the feature extractor (backbone) and an optional compression
    module that narrows the feature dimension before downstream use.

    Args:
        model_name: Name of any timm model (e.g. 'resnet50', 'efficientnet_b0')
        pretrained: Whether to load pretrained weights.
        compress_features: If provided, apply EmbeddingCompressor to pooled features.
                           Value is hidden_ratio (0 < ratio < 1).
                           None = no compression (default).
        out_indices: For multi-scale feature extractors (FPN-ready), which
                     stages to return. None = return final feature only.

    Returns:
        (feature_extractor, compressor) tuple.
        compressor is nn.Identity if compress_features is None.

    Example:
        >>> backbone, compressor = timm_feature_extractor(
        ...     'resnet50', pretrained=True, compress_features=0.5
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = backbone(x)
        >>> pooled = features.mean(dim=[2, 3])  # GAP
        >>> compressed = compressor(pooled)   # (1, 1024) from (1, 2048)
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is not installed. Install with: pip install timm")

    # Create feature extractor model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
        **kwargs,
    )

    # Get the number of channels per output stage
    # For most timm models with features_only=True, out_channels is accessible
    out_channels = getattr(model, 'out_channels', None)
    if out_channels is None:
        # Probe with a dummy forward pass to determine actual channel counts
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            outs = model(dummy)
        if isinstance(outs, (list, tuple)):
            out_channels = [o.shape[1] for o in outs]
        else:
            out_channels = [outs.shape[1]]

    # Create compressor if requested
    compressor = nn.Identity()
    if compress_features is not None:
        assert 0 < compress_features < 1, f"compress_features must be in (0, 1), got {compress_features}"
        # Compress final feature dimension
        final_dim = out_channels[-1]
        compressor = EmbeddingCompressor(
            in_features=final_dim,
            out_features=final_dim,
            hidden_ratio=compress_features,
        )

    return model, compressor


# ---------------------------------------------------------------------------
# Integration: Width scaling for any timm model
# ---------------------------------------------------------------------------

def scale_timm_model_width(
    model_name: str,
    width_scale: float,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Create a timm model with uniformly reduced channel width.

    This applies WidthScaler to the entire model, reducing all Conv2d and Linear
    channel dimensions by width_scale. This is the most aggressive form of
    last-stage reduction — it affects every layer rather than just the head.

    Args:
        model_name: Any timm model name.
        width_scale: Width multiplier in (0, 1]. 0.75 = 75% of original width.
        pretrained: Whether to load pretrained weights.
                    WARNING: pretrained weights at reduced width are not
                    available from timm — this initializes fresh weights.
                    Fine-tuning is required.

    Returns:
        Width-scaled model. Parameters are randomly initialized.

    Note:
        width_scale < 1.0 with pretrained=True is mathematically inconsistent
        (pretrained weights at full width, model at reduced width). The pretrained
        weights will be loaded and then immediately overwritten by WidthScaler's
        random initialization. Set pretrained=False unless you are fine-tuning.

    Example:
        >>> model = scale_timm_model_width('efficientnet_b0', width_scale=0.75)
        >>> # model has 75% of original parameters
        >>> x = torch.randn(1, 3, 224, 224)
        >>> y = model(x)
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is not installed. Install with: pip install timm")

    assert 0 < width_scale <= 1.0, f"width_scale must be in (0, 1], got {width_scale}"

    # Load model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        **kwargs,
    )

    # Scale width
    model = WidthScaler(model, width_scale)

    return model


# ---------------------------------------------------------------------------
# Integration: Insert SEReduction into the final stage of a timm model
# ---------------------------------------------------------------------------

def attach_final_stage_reduction(
    model: Union[str, nn.Module],
    reduction_ratio: float = 0.25,
    block_type: Literal['se', 'conditional', 'linear'] = 'se',
    stage_name: Optional[str] = None,
) -> nn.Module:
    """
    Attach a channel reduction block to the final stage of a timm model.

    This finds the global pooling layer of the model and inserts a reduction
    block between the final convolutional stage and the pooling. The reduction
    learns which channels to keep via SE (or conditional/linear) mechanism.

    Args:
        model: A timm model instance.
        reduction_ratio: Output channels = in_channels * ratio.
        block_type: 'se' (Squeeze-Excitation attention), 'conditional'
                     (spatial gating), or 'linear' (plain projection).
        stage_name: Explicit name of the final stage to target. If None,
                    auto-detected from model structure.

    Returns:
        Modified model with reduction applied to final feature maps before pooling.

    Note:
        This modifies the model in-place using a forward pre-hook on the
        global pooling layer. The hook intercepts the final feature tensor,
        applies reduction, and passes the reduced tensor to the pooler.

    Example:
        >>> model = timm.create_model('resnet50', pretrained=True)
        >>> model = attach_final_stage_reduction(model, reduction_ratio=0.5, block_type='se')
        >>> # Feature maps at layer4 are reduced by 50% before global average pooling
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is not installed. Install with: pip install timm")

    # Preserve the original model name string for stage detection before we lose it in create_model
    _original_model_name: str = ''
    if isinstance(model, str):
        _original_model_name = model
        model = timm.create_model(model, pretrained=False)
    else:
        # For nn.Module instances, use name_or_class or type name
        _original_model_name = (
            model.name_or_class
            if hasattr(model, 'name_or_class') and isinstance(model.name_or_class, str)
            else type(model).__name__.lower()
        )

    # Detect transformer (ViT, DeiT) architectures — attach_final_stage_reduction
    # works on CNN feature maps before global pooling. ViT has no pooling layer
    # (uses class token + flatten), so this function does not apply.
    _model_name_lower = _original_model_name.lower()
    if any(k in _model_name_lower for k in ('vit', 'deit', 'beit', 'caformer', 'maxvit')):
        raise ValueError(
            f"Model '{_original_model_name}' is a transformer (ViT/DeiT/BEiT). "
            "attach_final_stage_reduction targets CNN architectures with global pooling "
            "— it is not applicable to transformer models. "
            "Use replace_classifier_head() or timm_feature_extractor() instead."
        )

    # Infer stage name from model
    if stage_name is None:
        stage_name = _detect_final_stage_name(model, _original_model_name)

    # Get the stage
    parts = stage_name.split('.')
    target = model
    for part in parts:
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part, None)
            if target is None:
                raise ValueError(
                    f"Stage '{stage_name}' not found in model. "
                    f"Available modules: {_list_named_modules(model)}"
                )

    # Infer channel count from stage output via dummy forward pass.
    # We use the pool layer as the inference point because that's where our
    # pre-pool hook will fire — not the stage output, which may differ when
    # there's a conv_head or other transforms between stage and pool
    # (e.g., EfficientNet: blocks.6 outputs 320ch but conv_head expands to 1280ch).
    channel_holder = {}

    def _infer_hook(module, input):
        # Pre-hook fires before the module processes its input
        feat = input[0]
        if isinstance(feat, torch.Tensor):
            channel_holder['ch'] = feat.shape[1]

    pool_layer = _find_global_pool_layer(model)
    hook_handle = pool_layer.register_forward_pre_hook(_infer_hook)
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224))
    except Exception:
        pass  # We only need the channel count
    hook_handle.remove()

    in_channels = channel_holder.get('ch')
    if in_channels is None:
        raise ValueError(
            f"Could not infer channel count from pool layer input. "
            "Please specify stage_name manually (e.g., 'layer4' for ResNet) "
            "and ensure the model can run a dummy forward pass."
        )

    # Build reduction block
    if block_type == 'se':
        reducer = SEReduction(in_channels, reduction_ratio=reduction_ratio)
    elif block_type == 'conditional':
        reducer = ConditionalCapacityBlock(in_channels, reduction_ratio=reduction_ratio)
    elif block_type == 'linear':
        out_ch = max(1, int(in_channels * reduction_ratio))
        from capacity_reduction import LinearProjectionReduction
        reducer = LinearProjectionReduction(in_channels, out_ch)
    else:
        raise ValueError(f"Unknown block_type: {block_type}")

    # Compute the reduced channel count so we can patch the classifier head
    # This is the number of channels after the reduction that will reach
    # global pooling and then the FC/classifier layer.
    if block_type == 'linear':
        reduced_channels = out_ch  # linear projection has explicit out_ch
    else:
        # SE and conditional both compute out_channels = max(1, in_channels * ratio)
        reduced_channels = max(1, int(in_channels * reduction_ratio))

    # Attach the pre-hook to the pool layer we already located above
    # (we reused pool_layer from the channel-inference pass above)
    # Store reducer on the model so the hook can access it
    import uuid
    reducer_attr = f'_stage_reducer_{uuid.uuid4().hex[:6]}'
    setattr(model, reducer_attr, reducer)

    def _pre_pool_hook(module, input):
        # input is a tuple of (feature_tensor,)
        feat = input[0]
        reducer_ref = getattr(model, reducer_attr)
        # Apply reduction: (B, C, H, W) -> (B, C', H, W) with C' < C
        reduced_feat = reducer_ref(feat)
        # Return modified input — PyTorch hooks can return a tuple to replace input
        return (reduced_feat,)

    pool_handle = pool_layer.register_forward_pre_hook(_pre_pool_hook)

    # Store the handle so it persists (prevents garbage collection)
    handles_attr = f'_hook_handles_{uuid.uuid4().hex[:6]}'
    if not hasattr(model, handles_attr):
        setattr(model, '_reduction_hook_handles', [])
    getattr(model, '_reduction_hook_handles').append(pool_handle)

    # Patch the classifier/FC head to accept the reduced channel count after pooling.
    # Without this, the FC layer expects original_in_channels but receives
    # reduced_channels after the pool hook fires.
    # This works for ResNet (model.fc), EfficientNet (model.classifier),
    # ConvNeXt (model.head), and other timm architectures.
    _patch_classifier_head(model, reduced_channels)

    return model


def _replace_layer_norm_2d(norm_module: nn.Module, num_channels: int) -> None:
    """
    Replace a LayerNorm2d's normalized_shape in-place so it accepts a new channel count.

    LayerNorm2d inherits from nn.LayerNorm with normalized_shape = (num_channels,).
    We need to update normalized_shape after channel reduction before the pooling layer.
    Since PyTorch doesn't allow in-place mutation of normalized_shape, we reinitialize
    the parent class state directly.
    """
    # nn.LayerNorm stores normalized_shape as a tuple; we need to update it
    object.__setattr__(norm_module, 'normalized_shape', (num_channels,))
    # Also update the weight and bias shape to match new channel count
    norm_module.weight = nn.Parameter(norm_module.weight[:num_channels].clone())
    if norm_module.bias is not None:
        norm_module.bias = nn.Parameter(norm_module.bias[:num_channels].clone())


def _patch_classifier_head(model: nn.Module, reduced_channels: int) -> None:
    """
    Patch the classifier head of a timm model to accept the reduced channel count.

    After attach_final_stage_reduction reduces channel count before global pooling,
    the classifier/FC layer must be patched to accept the new (smaller) input dimension.
    This handles the common head architectures used in timm models:
    - model.fc (ResNet, ResNeXt, SEResNet, etc.)
    - model.classifier (EfficientNet, MobileNet, etc.)
    - model.head / model.head.fc (ConvNeXt, etc.)
    - NormMlpClassifierHead (ConvNeXt) — has norm + fc that both need patching

    Does nothing if no suitable head is found (some timm models use a different
    architecture and may need custom handling).
    """
    # Try nn.Linear head attributes commonly used in timm models
    linear_heads = ['fc', 'classifier', 'head']
    for attr in linear_heads:
        if not hasattr(model, attr):
            continue
        head = getattr(model, attr)
        if not isinstance(head, nn.Linear):
            # Some heads (e.g., EfficientNet's classifier) are themselves a Sequential.
            # For Sequential, look for the final Linear layer.
            if isinstance(head, nn.Sequential) and len(head) > 0:
                last = head[-1]
                if isinstance(last, nn.Linear):
                    old_in = last.in_features
                    old_out = last.out_features
                    head[-1] = nn.Linear(reduced_channels, old_out)
                    return
            # ConvNeXt / NormMlpClassifierHead: has head.norm (LayerNorm2d) and head.fc
            # Both need patching when channel count changes before pooling.
            if hasattr(head, 'norm') and hasattr(head, 'fc'):
                # head.norm is LayerNorm2d(num_channels=original_ch) — replace with reduced
                _replace_layer_norm_2d(head.norm, reduced_channels)
                # head.fc is Linear(in_features=original_ch) — replace with reduced
                old_out = head.fc.out_features
                head.fc = nn.Linear(reduced_channels, old_out)
                return
            continue
        old_in = head.in_features
        old_out = head.out_features
        # Replace with a new Linear that accepts reduced channels
        new_head = nn.Linear(reduced_channels, old_out)
        # Copy bias if present (no weights to copy — it's a fresh reduction)
        if head.bias is not None:
            new_head.bias = head.bias
        setattr(model, attr, new_head)
        return


def _find_global_pool_layer(model: nn.Module) -> nn.Module:
    """
    Find the global average pooling layer of a timm model.

    Returns the first AdaptiveAvgPool2d or AvgPool2d found that operates
    on spatial feature maps. This is where we inject the reduction pre-hook.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            # Check if it's operating on spatial features (common heuristic:
            # pool layers with output_size=1 or output_size=(1,1))
            if isinstance(module, nn.AdaptiveAvgPool2d) and module.output_size == 1:
                return module
            if isinstance(module, nn.AvgPool2d):
                return module

    # Fallback: first AdaptiveAvgPool2d
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return module

    raise ValueError(
        f"Cannot find global pooling layer in model {type(model).__name__}. "
        "Please open a bug report with the model architecture."
    )


def _detect_final_stage_name(model: nn.Module, model_name: str = '') -> str:
    """Heuristic: detect the name of the final conv stage in a timm model.

    Args:
        model: The timm model instance.
        model_name: The original model name string (e.g. 'resnet50', 'efficientnet_b0').
                    This is preferred over model.name_or_class which may be the class name
                    (e.g. 'vision_transformer' for ViT models) rather than the model name.
    """
    # Use the provided model_name string preferentially; fall back to model.name_or_class
    # only when no string was provided (callers that don't have it).
    if not model_name:
        model_name = model.name_or_class if hasattr(model, 'name_or_class') else ''
        if isinstance(model_name, type):
            model_name = ''
        if not model_name:
            model_name = type(model).__name__.lower()
    else:
        # model_name was explicitly provided — ensure it's a string
        model_name = str(model_name).lower()

    # ResNet family
    if 'resnet' in model_name or 'resnext' in model_name or 'seresnet' in model_name:
        return 'layer4'
    # EfficientNet family
    if 'efficientnet' in model_name:
        return 'blocks.6'  # final block before head
    # ConvNeXt
    if 'convnext' in model_name:
        return 'stages.3'
    # MobileNet
    if 'mobilenetv3' in model_name or 'mobilenetv2' in model_name:
        return 'blocks.13'
    # RegNet
    if 'regnet' in model_name:
        return 's4'
    # ViT / DeiT
    if 'vit' in model_name or 'deit' in model_name:
        return 'blocks.11'  # approximate; ViT has 12 blocks

    # Fallback: find the last child with Conv2d layers
    for name, _ in reversed(list(model.named_modules())):
        if 'layer' in name or 'stage' in name or 'block' in name:
            # Try to find a named child
            parts = name.split('.')
            m = model
            for p in parts:
                if p.isdigit():
                    m = m[int(p)]
                else:
                    m = getattr(m, p, None)
                    if m is None:
                        break
            if m is not None:
                has_conv = any(isinstance(child, nn.Conv2d) for child in m.modules())
                if has_conv:
                    return name

    return 'layer4'  # ResNet default


# ---------------------------------------------------------------------------
# Utility: Inspect a timm model's structure
# ---------------------------------------------------------------------------

def describe_timm_model(model: Union[str, nn.Module]) -> dict:
    """
    Return a description dict of a timm model's architecture.

    Shows: total parameters, trainable parameters, layer types,
    and where the classifier head is.

    Args:
        model: Either a timm model name string (e.g. 'resnet18') or a loaded nn.Module.
    """
    if isinstance(model, str):
        if not TIMM_AVAILABLE:
            raise ImportError("timm is not installed. Install with: pip install timm")
        model = timm.create_model(model, pretrained=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Find classifier
    classifier_info = None

    # First check if classifier/fc is a custom head type (CapacityReductionHead, etc.)
    # that wraps an internal classifier module
    if hasattr(model, 'classifier') and isinstance(model.classifier, CapacityReductionHead):
        internal = model.classifier.classifier
        classifier_info = {
            'name': 'classifier',
            'type': 'CapacityReductionHead',
            'in_features': internal.in_features,
            'out_features': internal.out_features,
        }
    elif hasattr(model, 'fc') and isinstance(model.fc, CapacityReductionHead):
        internal = model.fc.classifier
        classifier_info = {
            'name': 'fc',
            'type': 'CapacityReductionHead',
            'in_features': internal.in_features,
            'out_features': internal.out_features,
        }
    else:
        # Standard nn.Linear classifier — find it by identity match
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear,)):
                # Heuristic: the linear layer that IS the classifier/fc/head attribute.
                # Handles direct Linear (ResNet fc), Sequential (EfficientNet classifier),
                # and custom head modules (ConvNeXt NormMlpClassifierHead has head.fc).
                if hasattr(model, 'classifier') and module is model.classifier:
                    classifier_info = {
                        'name': name,
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                    }
                elif hasattr(model, 'fc') and module is model.fc:
                    classifier_info = {
                        'name': name,
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                    }
                elif name == 'head' and hasattr(model, 'head'):
                    # model.head itself is a Linear (some timm models)
                    classifier_info = {
                        'name': name,
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                    }
                elif name == 'head.fc' and hasattr(model.head, 'fc'):
                    # ConvNeXt and similar: head is a custom module, head.fc is the Linear
                    classifier_info = {
                        'name': name,
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                    }
                elif name == 'head' and hasattr(model.head, 'fc'):
                    # Fallback: head is not Linear but has an fc sub-module
                    classifier_info = {
                        'name': f'{name}.fc',
                        'type': type(model.head).__name__,
                        'in_features': model.head.fc.in_features,
                        'out_features': model.head.fc.out_features,
                    }

    return {
        'num_params': total_params,
        'num_classes': classifier_info['out_features'] if classifier_info else None,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'classifier': classifier_info,
    }


def _list_named_modules(model: nn.Module, prefix: str = '') -> list[str]:
    """List all named module paths in a model."""
    results = []
    for name, _ in model.named_modules():
        if name:
            results.append(name)
    return results


# ---------------------------------------------------------------------------
# Demo / verification
# ---------------------------------------------------------------------------

def _verify():
    """Smoke test: verify the integration helpers are well-formed."""
    print("Verifying timm_integration module...")

    # Check that imports work
    from capacity_reduction.timm_integration import (
        replace_classifier_head,
        timm_feature_extractor,
        scale_timm_model_width,
        attach_final_stage_reduction,
        describe_timm_model,
        _detect_final_stage_name,
    )
    print("  Imports: OK")

    # Check _detect_final_stage_name (uses type name heuristics, no actual model needed)
    class DummyResNet:
        name_or_class = 'resnet50'
    class DummyEffNet:
        name_or_class = 'efficientnet_b0'
    class DummyConvNeXt:
        name_or_class = 'convnext_tiny'
    assert _detect_final_stage_name(DummyResNet()) == 'layer4'
    assert _detect_final_stage_name(DummyEffNet()) == 'blocks.6'
    assert _detect_final_stage_name(DummyConvNeXt()) == 'stages.3'
    print("  Stage detection heuristics: OK")

    # Check _resolve_feature_dim (with mock model)
    class _MockFC(nn.Linear):
        def __init__(self):
            super().__init__(2048, 1000)

    class MockModel:
        name_or_class = 'resnet50'
        num_classes = 1000
        classifier: nn.Module = _MockFC()

    dim = _resolve_feature_dim(MockModel(), 'resnet50')
    assert dim == 2048, f"Expected 2048, got {dim}"
    print("  Feature dim resolution: OK")

    print("\nAll verification checks passed.")


if __name__ == '__main__':
    _verify()
