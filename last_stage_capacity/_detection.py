"""
Detection Track - Last-Stage Capacity Reduction for Detection Necks
===================================================================

Channel reduction strategies for object detection necks (FPN, PAN, BiFPN),
segmentation decoders, and multi-scale feature fusion stages.

Classes:
  LinearProjectionReduction - simple learnable channel projection (BCHW, BCT, BC)
  SEReduction              - Squeeze-and-Excitation channel attention + reduction
  ConditionalCapacityBlock  - spatial gating: network learns where reduction is safe
  CapacityReductionStack    - progressive narrowing across multiple stages

Moved from last_stage_capacity_reduction.py (repo root) into the last_stage_capacity
package on 2026-05-17 to fix broken import chain for timm_integration.py.
"""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProjectionReduction(nn.Module):
    """
    Channel reduction via learnable linear projection.
    
    Simplest form: y = Wx + b where W is [out_channels, in_channels].
    No activation function - appropriate for final stages where we want
    a pure dimension change without non-linearity overhead.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (must be < in_channels)
        bias: Whether to include bias term (default: True)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super().__init__()
        if out_channels >= in_channels:
            raise ValueError(
                f"out_channels ({out_channels}) must be < in_channels ({in_channels}). "
                "Use identity or a different block if no reduction is needed."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)
        self._init_weights()
    
    def _init_weights(self):
        # Kaiming-like init for projections
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        if self.proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.proj.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W) or (B, C)
        if x.dim() == 4:
            # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C) -> project -> reshape
            B, C, H, W = x.shape
            x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x_flat = x_perm.reshape(-1, C)    # (B*H*W, C)
            out = self.proj(x_flat)           # (B*H*W, out_channels)
            out = out.reshape(B, H, W, self.out_channels)
            return out.permute(0, 3, 1, 2)   # (B, out_channels, H, W)
        elif x.dim() == 3:
            # (B, C, T) - e.g. temporal features
            B, C, T = x.shape
            x_flat = x.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
            out = self.proj(x_flat)                     # (B*T, out_channels)
            return out.reshape(B, T, self.out_channels).permute(0, 2, 1)  # (B, out_channels, T)
        elif x.dim() == 2:
            # (B, C) - already flat
            return self.proj(x)
        else:
            raise ValueError(f"Expected 2-4D input, got {x.dim()}D: {x.shape}")
    
    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


class SEReduction(nn.Module):
    """
    Squeeze-and-Excitation channel reduction block.
    
    SE blocks originally (Hu et al., 2018) were used for channel excitation.
    Here we adapt them for capacity reduction: the SE mechanism learns which
    channels are worth keeping, then a projection reduces dimension.
    
    The squeeze (global avg pool) produces channel-wise statistics.
    The excitation (two FC layers) learns channel importance.
    The reduction projection applies the actual dimension change.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: How much to reduce (out = in * ratio). Default: 0.25
        bias: Whether excitation layers have bias. Default: False
        activation: Activation after reduction projection. Default: 'relu'
    """
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: float = 0.25,
        bias: bool = False,
        activation: Literal['relu', 'swish', 'gelu'] = 'relu',
    ):
        super().__init__()
        if not 0 < reduction_ratio < 1:
            raise ValueError(f"reduction_ratio must be in (0, 1), got {reduction_ratio}")
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.out_channels = max(1, int(in_channels * reduction_ratio))
        
        # Squeeze: global average pooling (produces (B, C, 1, 1) per spatial loc averaged)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: learns channel importance weights
        squeeze_channels = max(1, in_channels // 16)
        act_fn = {'relu': nn.ReLU, 'swish': nn.SiLU, 'gelu': nn.GELU}[activation]
        self.excite = nn.Sequential(
            nn.Linear(in_channels, squeeze_channels, bias=bias),
            act_fn(),
            nn.Linear(squeeze_channels, in_channels, bias=bias),
            nn.Sigmoid(),
        )
        
        # Reduction projection
        self.reduce = LinearProjectionReduction(in_channels, self.out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        original_shape = x.shape
        
        # Squeeze + excite for channel weights
        w = self.squeeze(x)                      # (B, C, 1, 1)
        w = w.view(w.size(0), -1)                 # (B, C)
        w = self.excite(w)                        # (B, C) - channel-wise weights in [0,1]
        w = w.view(original_shape[0], original_shape[1], 1, 1)  # (B, C, 1, 1)
        
        # Reweight channels before reduction
        x = x * w                                 # (B, C, H, W) - reweighted
        
        # Project to reduced dimension
        return self.reduce(x)
    
    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"reduction_ratio={self.reduction_ratio}")


class ConditionalCapacityBlock(nn.Module):
    """
    Learnable capacity reduction with spatial gating.
    
    This block reduces channel capacity but learns a per-spatial-position gate
    that controls how much the reduction is applied. The gate learns to suppress
    the reduction (gate -> 1) or apply it fully (gate -> 0) at each spatial
    location independently.
    
    This addresses the common case where naive reduction hurts performance because
    the network needed certain spatial locations at full capacity. The gating
    mechanism lets the network selectively reduce where it's safe to do so.
    
    Architecture:
        reduced = proj(x)                        # (B, out_channels, H, W)
        gate = sigmoid(conv(gate_conv(x)))       # (B, 1, H, W) - per-spatial scalar
        output = reduced * gate                  # (B, out_channels, H, W)
    
    The gate has shape (B, 1, H, W) so it applies uniformly across all
    output channels at each spatial position.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Target reduction ratio. Default: 0.25
        min_channels: Minimum output channels. Default: auto
        gate_bias_init: Initial bias for gate conv (negative = reduction-on by default).
                        Default: -2.0 (gate starts near 0.12, meaning reduction applied)
    """
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: float = 0.25,
        min_channels: Optional[int] = None,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.out_channels = min_channels or max(16, int(in_channels * reduction_ratio))
        
        if self.out_channels >= in_channels:
            raise ValueError(
                f"Computed out_channels ({self.out_channels}) must be < in_channels ({in_channels}). "
                "Increase reduction_ratio or set min_channels manually."
            )
        
        # Reduction projection
        self.reduce = LinearProjectionReduction(in_channels, self.out_channels)
        
        # Gate network: learns per-spatial-position scalar in [0, 1]
        # Takes original input, outputs a single scalar per spatial position
        interleave_channels = max(4, in_channels // 16)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, interleave_channels, 1, bias=False),
            nn.BatchNorm2d(interleave_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(interleave_channels, 1, 1, bias=True),  # single channel out
        )
        # Initialize last conv bias negative so gate starts in "reduce" mode
        nn.init.constant_(self.gate_conv[3].bias, gate_bias_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        # Compute reduction
        reduced = self.reduce(x)                          # (B, out_channels, H, W)
        
        # Compute per-spatial-position gate
        gate_logit = self.gate_conv(x)                   # (B, 1, H, W)
        gate = torch.sigmoid(gate_logit)                 # (B, 1, H, W) - in [0, 1]
        
        # Apply gate: when gate -> 1: full reduced output; gate -> 0: suppressed
        # This means gate=1 means "use reduction", gate=0 means "suppress"
        # (init bias=-2 means gate≈0.12 at start = some suppression)
        out = reduced * gate                             # (B, out_channels, H, W)
        return out
    
    def get_gate_health(self) -> dict:
        """
        Return statistics about gate values - useful for visualizing
        where the network is applying vs. suppressing reduction.
        """
        return {"note": "call after forward pass to get actual gate stats"}
    
    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"reduction_ratio={self.reduction_ratio}")


class CapacityReductionStack(nn.Module):
    """
    Stack of capacity reduction blocks for deep necks.
    
    Useful when you need to reduce capacity progressively across multiple
    stages of a detection decoder or segmentation head, rather than all at once.
    
    Example for a 4-stage FPN-style neck:
        stack = CapacityReductionStack(
            channels=[256, 128, 64, 32],
            reduction_ratios=[0.5, 0.5],  # 3 ratios for 4 channels
            block_type='se',
        )
    
    Args:
        channels: List of channel counts per stage [C0, C1, C2, ...]
        reduction_ratios: Reduction ratio per stage or single value for all
        block_type: 'linear', 'se', or 'conditional'
    """
    def __init__(
        self,
        channels: list[int],
        reduction_ratios: float | list[float] = 0.5,
        block_type: Literal['linear', 'se', 'conditional'] = 'se',
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("Need at least 2 stages for a reduction stack")
        
        if isinstance(reduction_ratios, float):
            ratios_list: list[float] = [reduction_ratios] * (len(channels) - 1)
        else:
            ratios_list = list(reduction_ratios)
        
        if len(ratios_list) != len(channels) - 1:
            raise ValueError(
                f"Got {len(channels)} channels but {len(ratios_list)} reduction_ratios. "
                "Need len(channels) - 1 values."
            )
        
        self.stages = nn.ModuleList()
        self.block_type = block_type
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            if block_type == 'linear':
                block = LinearProjectionReduction(in_ch, out_ch)
            elif block_type == 'se':
                block = SEReduction(in_ch, reduction_ratio=(out_ch / in_ch))
            elif block_type == 'conditional':
                block = ConditionalCapacityBlock(
                    in_ch, reduction_ratio=(out_ch / in_ch), min_channels=out_ch
                )
            else:
                raise ValueError(f"Unknown block_type: {block_type}")
            
            self.stages.append(block)
    
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply progressive reduction.
        Input: single tensor or list of tensors
        Output: list of tensors, one per stage
        """
        if not isinstance(x, list):
            x = [x]
        
        outputs = []
        for i, stage in enumerate(self.stages):
            if i < len(x):
                inp = x[i]
            else:
                inp = outputs[-1]
            outputs.append(stage(inp))
        return outputs
    
    def extra_repr(self):
        return f"stages={len(self.stages)}, block_type={self.block_type}"
