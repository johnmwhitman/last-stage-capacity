"""Test attach_final_stage_reduction and timm integration."""
import sys
sys.path.insert(0, '.')

import pytest
import torch
import torch.nn as nn

try:
    import timm
    from timm.models._registry import _model_entrypoints
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    _model_entrypoints = {}

from last_stage_capacity.timm_integration import (
    attach_final_stage_reduction,
    replace_classifier_head,
)


def test_attach_final_stage_reduction_resnet18():
    """Verify attach_final_stage_reduction works on ResNet18."""
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    # SE block
    model1 = timm.create_model('resnet18', pretrained=False)
    modified1 = attach_final_stage_reduction(model1, reduction_ratio=0.5, block_type='se')
    out1 = modified1(torch.randn(1, 3, 224, 224))
    assert out1.shape == (1, 1000)

    # Conditional block
    model2 = timm.create_model('resnet18', pretrained=False)
    modified2 = attach_final_stage_reduction(model2, reduction_ratio=0.5, block_type='conditional')
    out2 = modified2(torch.randn(1, 3, 224, 224))
    assert out2.shape == (1, 1000)

    # Linear projection
    model3 = timm.create_model('resnet18', pretrained=False)
    modified3 = attach_final_stage_reduction(model3, reduction_ratio=0.5, block_type='linear')
    out3 = modified3(torch.randn(1, 3, 224, 224))
    assert out3.shape == (1, 1000)


def test_attach_final_stage_reduction_resnet50():
    """Verify attach_final_stage_reduction works on ResNet50."""
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    model = timm.create_model('resnet50', pretrained=False)
    modified = attach_final_stage_reduction(model, reduction_ratio=0.5, block_type='se')
    out = modified(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)


def test_unsupported_architecture_attach_final_stage_reduction_missing_stage():
    """
    Test attach_final_stage_reduction with unsupported architecture via fake registry entry.

    Model creation is fully mocked with no downloads. The architecture has no conv stages
    matching stage heuristics and lacks 'layer4', raising the documented ValueError
    rather than a raw KeyError or unhandled exception.
    """
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    class FakeUnsupportedNoStageNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 5)

        def forward(self, x):
            return self.fc(x)

    fake_model_name = 'fake_unsupported_no_stage_model'
    _model_entrypoints[fake_model_name] = lambda pretrained=False, **kwargs: FakeUnsupportedNoStageNet()

    try:
        # 1. Instantiated model path: model creation fully mocked via fake registry
        mock_model = timm.create_model(fake_model_name, pretrained=False)
        with pytest.raises(ValueError) as exc_info_inst:
            attach_final_stage_reduction(mock_model)
        assert "Stage 'layer4' not found in model" in str(exc_info_inst.value)

        # 2. String model_name path: created through timm registry lookup
        with pytest.raises(ValueError) as exc_info_str:
            attach_final_stage_reduction(fake_model_name)
        assert "Stage 'layer4' not found in model" in str(exc_info_str.value)

    finally:
        _model_entrypoints.pop(fake_model_name, None)


def test_unsupported_architecture_attach_final_stage_reduction_missing_pool():
    """
    Test attach_final_stage_reduction with unsupported architecture that has a stage
    named 'layer4' but no global pooling layer.

    Raises the documented ValueError from _find_global_pool_layer.
    """
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    class FakeNoPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(nn.Conv2d(3, 16, 3))

        def forward(self, x):
            return self.layer4(x)

    fake_model_name = 'fake_unsupported_no_pool_model'
    _model_entrypoints[fake_model_name] = lambda pretrained=False, **kwargs: FakeNoPoolNet()

    try:
        mock_model = timm.create_model(fake_model_name, pretrained=False)
        with pytest.raises(ValueError) as exc_info:
            attach_final_stage_reduction(mock_model)
        assert "Cannot find global pooling layer in model" in str(exc_info.value)
    finally:
        _model_entrypoints.pop(fake_model_name, None)


def test_replace_classifier_head_model_instance_unresolvable_feature_dim():
    """
    Test replace_classifier_head with a model instance whose feature dimension cannot
    be resolved (no standard feature dim attributes, unknown family name, no linear head).

    Verifies the documented ValueError is raised instead of a raw KeyError.
    """
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    class FakeUnresolvableDimNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)

        def forward(self, x):
            return self.conv(x)

    fake_model_name = 'fake_unresolvable_dim_model'
    _model_entrypoints[fake_model_name] = lambda pretrained=False, **kwargs: FakeUnresolvableDimNet()

    try:
        mock_model = timm.create_model(fake_model_name, pretrained=False)
        with pytest.raises(ValueError) as exc_info:
            replace_classifier_head(mock_model)
        assert "Cannot resolve feature dimension for model" in str(exc_info.value)
    finally:
        _model_entrypoints.pop(fake_model_name, None)


def test_replace_classifier_head_model_instance_unrecognized_classifier():
    """
    Test replace_classifier_head with a model instance that has a resolved feature dimension
    but no recognized classifier attribute (.classifier, .fc, or .head).

    Verifies the documented ValueError is raised.
    """
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    class FakeNoClassifierNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_features = 64
            self.custom_head = nn.Linear(64, 10)

        def forward(self, x):
            return self.custom_head(x)

    fake_model_name = 'fake_no_classifier_model'
    _model_entrypoints[fake_model_name] = lambda pretrained=False, **kwargs: FakeNoClassifierNet()

    try:
        mock_model = timm.create_model(fake_model_name, pretrained=False)
        with pytest.raises(ValueError) as exc_info:
            replace_classifier_head(mock_model)
        assert "has no recognized classifier attribute" in str(exc_info.value)
    finally:
        _model_entrypoints.pop(fake_model_name, None)


def test_replace_classifier_head_model_instance_fallback_resolution():
    """
    Test replace_classifier_head with a model instance where feature dimension is resolved
    via inspection of the classifier input dimension (fallback path).
    """
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    class FakeFallbackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            # No num_features/feature_dim attribute, unknown family name
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            return self.fc(self.conv(x).mean([-2, -1]))

    fake_model_name = 'fake_fallback_fc_model'
    _model_entrypoints[fake_model_name] = lambda pretrained=False, **kwargs: FakeFallbackNet()

    try:
        mock_model = timm.create_model(fake_model_name, pretrained=False)
        # Should succeed using fallback inspection of self.fc.in_features (32)
        modified = replace_classifier_head(mock_model, num_classes=10)
        assert hasattr(modified, 'fc')
        out = modified(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 10)
    finally:
        _model_entrypoints.pop(fake_model_name, None)


@pytest.mark.xfail(
    reason=(
        "replace_classifier_head(model: str) reassigns model_name = model.name_or_class "
        "after model creation, which evaluates to the nn.Module instance rather than a str "
        "when name_or_class is not set on the model. This causes an AttributeError: "
        "'...' object has no attribute 'lower' in _resolve_feature_dim instead of raising "
        "the documented ValueError."
    ),
    strict=True,
)
def test_replace_classifier_head_string_name_unresolvable_documented_contract():
    """
    Test replace_classifier_head called with a string model name for an unresolvable architecture.

    Contract: It should raise the documented ValueError:
    'Cannot resolve feature dimension for model ...'
    rather than a raw KeyError or unhandled AttributeError.
    """
    if not TIMM_AVAILABLE:
        pytest.skip("timm not available")

    class FakeUnresolvableNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)

        def forward(self, x):
            return self.conv(x)

    fake_model_name = 'fake_unresolvable_str_model'
    _model_entrypoints[fake_model_name] = lambda pretrained=False, **kwargs: FakeUnresolvableNet()

    try:
        with pytest.raises(ValueError) as exc_info:
            replace_classifier_head(fake_model_name)
        assert "Cannot resolve feature dimension for model" in str(exc_info.value)
    finally:
        _model_entrypoints.pop(fake_model_name, None)


if __name__ == '__main__':
    test_attach_final_stage_reduction_resnet18()
    test_attach_final_stage_reduction_resnet50()
    test_unsupported_architecture_attach_final_stage_reduction_missing_stage()
    test_unsupported_architecture_attach_final_stage_reduction_missing_pool()
    test_replace_classifier_head_model_instance_unresolvable_feature_dim()
    test_replace_classifier_head_model_instance_unrecognized_classifier()
    test_replace_classifier_head_model_instance_fallback_resolution()
    print("Direct execution passed!")
