"""Check ConvNeXt stage structure."""
import timm, torch

m = timm.create_model('convnext_tiny', pretrained=False)

# Find pool layer
print('=== Pool layers ===')
for name, module in m.named_modules():
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        print(f'  pool: {name}')

# Print top-level children
print('\n=== Top-level children ===')
for name, _ in m.named_children():
    print(f'  {name}')

# Try to access stages.3
print('\n=== stages.3 accessible? ===')
try:
    s = m.stages
    print(f'  stages exists: {type(s).__name__}')
    if hasattr(s, '__len__'):
        print(f'  len(stages): {len(s)}')
    for i in range(4):
        try:
            print(f'  stages.{i}: {type(m.stages[i]).__name__}')
        except Exception as e:
            print(f'  stages.{i}: ERROR {e}')
except AttributeError as e:
    print(f'  stages: NOT FOUND - {e}')

# Check what the final conv stage is
print('\n=== Searching for final conv stage ===')
for name, module in reversed(list(m.named_modules())):
    if isinstance(module, torch.nn.Conv2d):
        print(f'  {name}: {module.out_channels}ch')
        break
