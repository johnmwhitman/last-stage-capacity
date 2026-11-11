"""Pytest tests for the supported-architecture path in last_stage_capacity/timm_integration.py.

Verifies that a fake registry entry resolving to a fully mocked model factory is used
successfully instead of the fallback/error path. Model creation is fully mocked without
downloads or external network access.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

try:
    import timm
    import timm.models._registry as timm_registry
    from timm.models import register_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from last_stage_capacity.timm_integration import (
    replace_classifier_head,
    attach_final_stage_reduction,
    describe_timm_model,
    timm_feature_extractor,
    scale_timm_model_width,
    CapacityReductionHead,
    EmbeddingCompressor,
    WidthScaler,
)


class DummyMockResNet(nn.Module):
    """Mock ResNet model for testing supported architecture paths."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        feature_dim: int = 64,
        features_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.features_only = features_only
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes
        self.num_features = feature_dim
        self.out_channels = [feature_dim]
        self.name_or_class = "mock_resnet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        features = self.layer4(x)
        if self.features_only:
            return features
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        return self.fc(flattened)


class DummyMockConvNeXt(nn.Module):
    """Mock ConvNeXt model for testing supported architecture paths."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        feature_dim: int = 64,
        features_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.features_only = features_only
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(32, 48, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(48, feature_dim, kernel_size=3, padding=1)),
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        class MockHead(nn.Module):
            def __init__(self, in_f: int, num_c: int):
                super().__init__()
                self.norm = nn.LayerNorm(in_f)
                self.fc = nn.Linear(in_f, num_c)

            def forward(self, feat: torch.Tensor) -> torch.Tensor:
                return self.fc(self.norm(feat))

        self.head = MockHead(feature_dim, num_classes)
        self.num_classes = num_classes
        self.num_features = feature_dim
        self.out_channels = [feature_dim]
        self.name_or_class = "mock_convnext"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for stage in self.stages:
            x = stage(x)
        if self.features_only:
            return x
        pooled = self.global_pool(x)
        flattened = torch.flatten(pooled, 1)
        return self.head(flattened)


@pytest.fixture(autouse=True)
def guard_no_network_downloads(monkeypatch):
    """Ensure no real downloads occur during test execution."""
    def _blocked_download(*args, **kwargs):
        raise RuntimeError("Network download was attempted during mock test!")

    monkeypatch.setattr("urllib.request.urlopen", _blocked_download)


@pytest.fixture
def fake_timm_registry():
    """Register fake models in timm registry with mock factories and clean up after test."""
    if not TIMM_AVAILABLE:
        pytest.skip("timm is not installed")

    registered_names: list[str] = []
    factories: dict[str, MagicMock] = {}

    def _register(model_name: str, model_class=DummyMockResNet, **default_kwargs):
        mock_factory = MagicMock(side_effect=lambda **kwargs: model_class(**{**default_kwargs, **kwargs}))

        # Create a dynamic function with the desired __name__ so register_model picks it up
        def entrypoint_fn(**kwargs):
            return mock_factory(**kwargs)

        entrypoint_fn.__name__ = model_name
        register_model(entrypoint_fn)

        registered_names.append(model_name)
        factories[model_name] = mock_factory
        return mock_factory

    yield _register

    # Teardown: unregister fake models safely
    for name in registered_names:
        if hasattr(timm_registry, "_model_entrypoints"):
            timm_registry._model_entrypoints.pop(name, None)
        if hasattr(timm_registry, "_model_to_module"):
            old_mod = timm_registry._model_to_module.pop(name, None)
            if old_mod and hasattr(timm_registry, "_module_to_models"):
                mod_models = timm_registry._module_to_models.get(old_mod)
                if mod_models:
                    mod_models.discard(name)
        if hasattr(timm_registry, "_clear_model_entries"):
            timm_registry._clear_model_entries(name)


def test_registry_resolution_vs_fallback_error(fake_timm_registry):
    """Verify that a fake registered model resolves via the registry rather than falling back to error."""
    model_name = "test_fake_resnet_res"
    mock_factory = fake_timm_registry(model_name, DummyMockResNet)

    # 1. Non-existent model fails with RuntimeError in timm.create_model
    with pytest.raises(RuntimeError, match="Unknown model"):
        attach_final_stage_reduction("unknown_nonexistent_architecture_xyz")

    # 2. Registered model succeeds and factory is called
    assert not mock_factory.called
    model = attach_final_stage_reduction(model_name, reduction_ratio=0.5, block_type="se")
    assert mock_factory.called
    assert model is not None

    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_replace_classifier_head_supported_resnet(fake_timm_registry):
    """Verify replace_classifier_head using fake registry entry for ResNet."""
    model_name = "test_fake_resnet_head"
    mock_factory = fake_timm_registry(model_name, DummyMockResNet, feature_dim=64, num_classes=10)

    model = replace_classifier_head(model_name, num_classes=5, hidden_ratio=0.5, dropout=0.1)

    assert mock_factory.called
    assert isinstance(model.fc, CapacityReductionHead)
    assert model.fc.classifier.out_features == 5

    # Run forward pass through the replaced head
    dummy_input = torch.randn(2, 64)
    out = model.fc(dummy_input)
    assert out.shape == (2, 5)


def test_replace_classifier_head_supported_convnext(fake_timm_registry):
    """Verify replace_classifier_head using fake registry entry for ConvNeXt (.head attribute)."""
    model_name = "test_fake_convnext_head"
    mock_factory = fake_timm_registry(model_name, DummyMockConvNeXt, feature_dim=64, num_classes=10)

    model = replace_classifier_head(model_name, num_classes=7, hidden_ratio=0.5)

    assert mock_factory.called
    assert isinstance(model.head, CapacityReductionHead)
    assert model.head.classifier.out_features == 7

    # Run forward pass through the replaced head
    dummy_input = torch.randn(2, 64)
    out = model.head(dummy_input)
    assert out.shape == (2, 7)


@pytest.mark.parametrize("block_type", ["se", "conditional", "linear"])
def test_attach_final_stage_reduction_supported_blocks(fake_timm_registry, block_type):
    """Verify attach_final_stage_reduction with 'se', 'conditional', and 'linear' blocks."""
    model_name = f"test_fake_resnet_{block_type}"
    mock_factory = fake_timm_registry(model_name, DummyMockResNet, feature_dim=64, num_classes=10)

    model = attach_final_stage_reduction(model_name, reduction_ratio=0.5, block_type=block_type)

    assert mock_factory.called
    assert hasattr(model, "_reduction_hook_handles")
    assert len(model._reduction_hook_handles) > 0

    # Ensure forward pass runs successfully through the hook
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_describe_timm_model_supported_architecture(fake_timm_registry):
    """Verify describe_timm_model resolves model via registry and accurately describes architecture."""
    model_name = "test_fake_resnet_describe"
    mock_factory = fake_timm_registry(model_name, DummyMockResNet, feature_dim=64, num_classes=20)

    desc = describe_timm_model(model_name)

    assert mock_factory.called
    assert desc["num_classes"] == 20
    assert desc["num_params"] > 0
    assert desc["classifier"] is not None
    assert desc["classifier"]["name"] == "fc"
    assert desc["classifier"]["out_features"] == 20
    assert desc["classifier"]["in_features"] == 64


def test_timm_feature_extractor_supported_architecture(fake_timm_registry):
    """Verify timm_feature_extractor resolution and feature compression on fake registry model."""
    feature_dim = 64
    model_name = "test_fake_resnet_featextract"
    mock_factory = fake_timm_registry(model_name, DummyMockResNet, feature_dim=feature_dim, num_classes=10)

    extractor, compressor = timm_feature_extractor(
        model_name,
        pretrained=False,
        compress_features=0.5,
    )

    assert mock_factory.called
    assert isinstance(compressor, EmbeddingCompressor)

    # Test feature extraction forward and compressor narrowing
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = extractor(x)
        assert features.shape == (2, feature_dim, 32, 32)
        pooled = features.mean(dim=[2, 3])
        assert pooled.shape == (2, feature_dim)
        compressed = compressor(pooled)
        assert compressed.shape == (2, feature_dim)


def test_scale_timm_model_width_supported_architecture(fake_timm_registry):
    """Verify scale_timm_model_width resolves model from registry and applies WidthScaler."""
    model_name = "test_fake_resnet_widthscale"
    mock_factory = fake_timm_registry(model_name, DummyMockResNet, feature_dim=64, num_classes=10)

    scaled_model = scale_timm_model_width(
        model_name,
        width_scale=0.5,
        pretrained=False,
    )

    assert mock_factory.called
    assert isinstance(scaled_model, WidthScaler)
    assert scaled_model.width_scale == 0.5
