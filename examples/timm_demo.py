"""
Demo: timm integration with real pretrained model.
Shows attaching last-stage reduction to a pretrained ResNet18.
"""
import sys
sys.path.insert(0, "C:/AI/agents/echo")

from capacity_reduction.timm_integration import (
    describe_timm_model,
    attach_final_stage_reduction,
    scale_timm_model_width,
    replace_classifier_head,
)

def demo():
    import torch
    
    model_name = "resnet18"
    
    # 1. Describe the model
    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    info = describe_timm_model(model_name)
    print(f"  Parameters: {info['num_params']:,}")
    print(f"  Classifier: {info['classifier']}")
    
    # 2. Attach SE reduction to final stage
    print("\n[1] Attaching SE reduction (ratio=0.5)...")
    reduced = attach_final_stage_reduction(model_name, reduction_ratio=0.5, block_type="se")
    info2 = describe_timm_model(reduced)
    print(f"  Parameters: {info2['num_params']:,}")
    print(f"  Reduction: {info['num_params'] - info2['num_params']:,} params saved")
    
    # 3. Test forward pass
    print("\n[2] Forward pass test...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = reduced(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    
    # 4. Scale width by 0.5
    print("\n[3] Scaling model width by 0.5...")
    scaled = scale_timm_model_width(model_name, width_scale=0.5)
    info3 = describe_timm_model(scaled)
    print(f"  Parameters: {info3['num_params']:,}")
    print(f"  vs original: {info['num_params'] - info3['num_params']:,} params saved")
    
    # 5. Replace classifier head
    print("\n[4] Replacing classifier head (hidden_ratio=0.5)...")
    reheaded = replace_classifier_head(model_name, num_classes=10, hidden_ratio=0.5)
    info4 = describe_timm_model(reheaded)
    print(f"  Parameters: {info4['num_params']:,}")
    print(f"  Classifier: {info4['classifier']}")
    
    # Test that it still works
    with torch.no_grad():
        out4 = reheaded(x)
    print(f"  Output: {out4.shape}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    demo()
