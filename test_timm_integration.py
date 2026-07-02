"""Test attach_final_stage_reduction for shape mismatch."""
import sys
sys.path.insert(0, '.')

import torch

try:
    import timm
    print('timm available:', timm.__version__)
except ImportError:
    print('timm NOT available — skipping timm integration test')
    sys.exit(0)

from last_stage_capacity.timm_integration import attach_final_stage_reduction

# Test with resnet18
print('\n--- Testing attach_final_stage_reduction on ResNet18 ---')

# Test 1: reduction_ratio=0.5 with SE block
print('\nTest 1: reduction_ratio=0.5 with SE block')
model1 = timm.create_model('resnet18', pretrained=False)
modified1 = attach_final_stage_reduction(model1, reduction_ratio=0.5, block_type='se')
try:
    out1 = modified1(torch.randn(1, 3, 224, 224))
    print('Forward pass succeeded, output shape:', out1.shape)
except Exception as e:
    print('FORWARD PASS FAILED:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()

# Test 2: reduction_ratio=0.5 with conditional
print('\nTest 2: reduction_ratio=0.5 with conditional block')
model2 = timm.create_model('resnet18', pretrained=False)
modified2 = attach_final_stage_reduction(model2, reduction_ratio=0.5, block_type='conditional')
try:
    out2 = modified2(torch.randn(1, 3, 224, 224))
    print('Forward pass succeeded, output shape:', out2.shape)
except Exception as e:
    print('FORWARD PASS FAILED:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()

# Test 3: reduction_ratio=0.5 with linear projection
print('\nTest 3: reduction_ratio=0.5 with linear projection')
model3 = timm.create_model('resnet18', pretrained=False)
modified3 = attach_final_stage_reduction(model3, reduction_ratio=0.5, block_type='linear')
try:
    out3 = modified3(torch.randn(1, 3, 224, 224))
    print('Forward pass succeeded, output shape:', out3.shape)
except Exception as e:
    print('FORWARD PASS FAILED:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()

# Test 4: resnet50
print('\n--- Testing ResNet50 ---')
model4 = timm.create_model('resnet50', pretrained=False)
modified4 = attach_final_stage_reduction(model4, reduction_ratio=0.5, block_type='se')
try:
    out4 = modified4(torch.randn(1, 3, 224, 224))
    print('Forward pass succeeded, output shape:', out4.shape)
except Exception as e:
    print('FORWARD PASS FAILED:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()

print('\n--- All tests complete ---')
