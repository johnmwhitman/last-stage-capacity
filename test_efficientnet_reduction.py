"""Verify attach_final_stage_reduction works on EfficientNet (Sequential head)."""
import sys
sys.path.insert(0, '.')
import torch
import timm

from last_stage_capacity.timm_integration import attach_final_stage_reduction

print('Testing EfficientNet-B0...')
model = timm.create_model('efficientnet_b0', pretrained=False)
print('Original classifier type:', type(model.classifier))

modified = attach_final_stage_reduction(model, reduction_ratio=0.5, block_type='se')
print('After patch, classifier type:', type(modified.classifier))
print('After patch, classifier[-1]:', modified.classifier[-1] if hasattr(modified.classifier, '__getitem__') else modified.classifier)

out = modified(torch.randn(1, 3, 224, 224))
print('Forward pass succeeded, output shape:', out.shape)
print('SUCCESS')
