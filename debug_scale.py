"""Debug WidthScaler."""
import os, sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SCRIPT_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
import torch.nn as nn
from last_stage_capacity import WidthScaler

# Minimal case
m = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)
print("Before scaling:")
for name, child in m.named_children():
    print(f"  {name}: {child}")

scaled = WidthScaler(m, 0.5)
print("\nAfter scaling:")
for name, child in scaled.module.named_children():
    print(f"  {name}: {child}")
    if hasattr(child, 'weight'):
        print(f"    weight shape: {child.weight.shape}")

x = torch.randn(1, 64, 32, 32)
print(f"\nInput: {x.shape}")
try:
    out = scaled(x)
    print(f"Output: {out.shape}")
except Exception as e:
    print(f"ERROR: {e}")
