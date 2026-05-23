"""Inspect ConvNeXt NormMlpClassifierHead structure."""
import timm, torch

m = timm.create_model('convnext_tiny', pretrained=False)
head = m.head
print(f'head type: {type(head).__name__}')
print(f'head attributes: {[a for a in dir(head) if not a.startswith("_")]}')
print(f'head.norm: {type(head.norm).__name__} with normalized_shape={getattr(head.norm, "normalized_shape", None)}')
print(f'head.flatten: {type(head.flatten).__name__}')
print(f'head.pre_logits: {type(head.pre_logits).__name__}')
print(f'head.fc: {type(head.fc).__name__} in_features={head.fc.in_features} out_features={head.fc.out_features}')

# Patch head.fc and see what happens
print('\n--- After patching head.fc ---')
# Simulate patching
import torch.nn as nn
new_fc = nn.Linear(384, head.fc.out_features)  # 768 * 0.5 = 384
with torch.no_grad():
    new_fc.weight.zero_()  # dummy
    new_fc.bias.zero_()
head.fc = new_fc
print(f'head.fc now: in_features={head.fc.in_features}')

# Try a forward pass with 384 channels
x = torch.randn(2, 384, 1, 1)  # After SE reduction, 384 channels
try:
    with torch.no_grad():
        out = head(x)
    print(f'Forward with 384ch succeeded: {out.shape}')
except Exception as e:
    print(f'Forward with 384ch failed: {type(e).__name__}: {e}')
