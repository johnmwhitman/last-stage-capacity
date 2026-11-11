"""Public API stability test for last_stage_capacity/__init__.py.

Verifies:
- __all__ matches the expected public API symbols exactly (no accidental additions/removals).
- Every symbol in __all__ can be imported directly from last_stage_capacity.
- Every symbol is a class / callable subclass of torch.nn.Module.
- Every symbol matches its expected parameter signature, default values, and type annotations.
- Every block can be instantiated and invoked on dummy tensors without downloading models.
"""

import inspect
from typing import get_type_hints, Literal, Optional, List
import pytest
import torch
import torch.nn as nn
import last_stage_capacity


EXPECTED_PUBLIC_API = {
    "BottleneckBlock": {
        "params": {
            "in_channels": {"type": int, "has_default": False},
            "out_channels": {"type": int, "has_default": False},
            "bottleneck_ratio": {"type": float, "default": 0.25},
            "stride": {"type": int, "default": 1},
            "use_se": {"type": bool, "default": False},
            "se_reduction": {"type": int, "default": 16},
        },
    },
    "ProgressiveNarrowing": {
        "params": {
            "channels": {"type": (list, List[int]), "has_default": False},
            "bottleneck_ratio": {"type": float, "default": 0.25},
            "use_se": {"type": bool, "default": True},
            "se_reduction": {"type": int, "default": 16},
        },
    },
    "WidthScaler": {
        "params": {
            "module": {"type": nn.Module, "has_default": False},
            "width_scale": {"type": float, "has_default": False},
        },
    },
    "EmbeddingCompressor": {
        "params": {
            "in_features": {"type": int, "has_default": False},
            "out_features": {"type": int, "has_default": False},
            "hidden_ratio": {"type": float, "default": 0.5},
            "dropout": {"type": float, "default": 0.1},
            "norm": {"type": Literal["layernorm", "batchnorm", "none"], "default": "layernorm"},
        },
    },
    "DepthwiseFinal": {
        "params": {
            "in_channels": {"type": int, "has_default": False},
            "out_channels": {"type": int, "has_default": False},
            "kernel_size": {"type": int, "default": 3},
            "stride": {"type": int, "default": 1},
            "expansion_ratio": {"type": float, "default": 1.0},
        },
    },
    "SqueezeExcitation": {
        "params": {
            "channels": {"type": int, "has_default": False},
            "reduction": {"type": int, "default": 16},
        },
    },
    "CapacityReductionHead": {
        "params": {
            "in_features": {"type": int, "has_default": False},
            "num_classes": {"type": int, "has_default": False},
            "hidden_ratio": {"type": float, "default": 0.5},
            "dropout": {"type": float, "default": 0.2},
            "norm": {"type": Literal["layernorm", "batchnorm", "none"], "default": "layernorm"},
        },
    },
    "LinearProjectionReduction": {
        "params": {
            "in_channels": {"type": int, "has_default": False},
            "out_channels": {"type": int, "has_default": False},
            "bias": {"type": bool, "default": True},
        },
    },
    "SEReduction": {
        "params": {
            "in_channels": {"type": int, "has_default": False},
            "reduction_ratio": {"type": float, "default": 0.25},
            "bias": {"type": bool, "default": False},
            "activation": {"type": Literal["relu", "swish", "gelu"], "default": "relu"},
        },
    },
    "ConditionalCapacityBlock": {
        "params": {
            "in_channels": {"type": int, "has_default": False},
            "reduction_ratio": {"type": float, "default": 0.25},
            "min_channels": {"type": Optional[int], "default": None},
            "gate_bias_init": {"type": float, "default": -2.0},
        },
    },
    "CapacityReductionStack": {
        "params": {
            "channels": {"type": (list, list[int]), "has_default": False},
            "reduction_ratios": {"type": (float, float | list[float]), "default": 0.5},
            "block_type": {"type": Literal["linear", "se", "conditional"], "default": "se"},
        },
    },
}


def test_all_defined_and_matches_exact_set():
    """Verify __all__ is defined and contains exact expected public API set."""
    assert hasattr(last_stage_capacity, "__all__"), "__all__ is missing in last_stage_capacity"
    exported_names = set(last_stage_capacity.__all__)
    expected_names = set(EXPECTED_PUBLIC_API.keys())

    missing = expected_names - exported_names
    extra = exported_names - expected_names

    assert not missing, f"Missing public API names in last_stage_capacity.__all__: {missing}"
    assert not extra, f"Unexpected extra public API names in last_stage_capacity.__all__: {extra}"
    assert len(last_stage_capacity.__all__) == len(exported_names), "__all__ contains duplicate entries"


@pytest.mark.parametrize("name", list(EXPECTED_PUBLIC_API.keys()))
def test_symbol_importable_and_callable(name):
    """Verify each exported symbol is importable, callable, and an nn.Module class."""
    assert hasattr(last_stage_capacity, name), f"'{name}' in __all__ but not an attribute of last_stage_capacity"
    obj = getattr(last_stage_capacity, name)
    assert callable(obj), f"'{name}' must be callable"
    assert inspect.isclass(obj), f"'{name}' must be a class"
    assert issubclass(obj, nn.Module), f"'{name}' must be a subclass of torch.nn.Module"


@pytest.mark.parametrize("name,spec", list(EXPECTED_PUBLIC_API.items()))
def test_symbol_signatures_and_types(name, spec):
    """Verify constructor parameters, default values, and type annotations match specification."""
    cls = getattr(last_stage_capacity, name)
    sig = inspect.signature(cls.__init__)
    type_hints = get_type_hints(cls.__init__)

    expected_params = spec["params"]
    actual_params = [p for p in sig.parameters.keys() if p != "self"]

    assert actual_params == list(expected_params.keys()), (
        f"Parameter list mismatch for '{name}'. Expected: {list(expected_params.keys())}, got: {actual_params}"
    )

    for param_name, param_spec in expected_params.items():
        param = sig.parameters[param_name]

        # Verify defaults
        if "default" in param_spec:
            assert param.default == param_spec["default"], (
                f"Default value mismatch for {name}.{param_name}: "
                f"expected {param_spec['default']}, got {param.default}"
            )
        else:
            assert param.default is inspect.Parameter.empty, (
                f"Parameter {name}.{param_name} should not have a default value"
            )

        # Verify type hints
        assert param_name in type_hints, f"Missing type annotation for {name}.{param_name}"
        expected_type = param_spec["type"]
        actual_type = type_hints[param_name]

        if isinstance(expected_type, tuple):
            assert actual_type in expected_type, (
                f"Type annotation mismatch for {name}.{param_name}: "
                f"expected one of {expected_type}, got {actual_type}"
            )
        else:
            assert actual_type == expected_type, (
                f"Type annotation mismatch for {name}.{param_name}: "
                f"expected {expected_type}, got {actual_type}"
            )


def test_instantiation_and_execution_without_downloads():
    """Verify every public API module instantiates and executes without downloading models."""
    # 1. BottleneckBlock
    block = last_stage_capacity.BottleneckBlock(64, 32, bottleneck_ratio=0.25, stride=1, use_se=True)
    out = block(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 32, 8, 8)

    # 2. ProgressiveNarrowing
    pn = last_stage_capacity.ProgressiveNarrowing([64, 32, 16], bottleneck_ratio=0.5, use_se=True)
    out = pn(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 16, 8, 8)

    # 3. WidthScaler
    submod = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
    ws = last_stage_capacity.WidthScaler(submod, width_scale=0.5)
    out = ws(torch.randn(2, 16, 8, 8))
    assert out.shape == (2, 16, 8, 8)

    # 4. EmbeddingCompressor
    ec = last_stage_capacity.EmbeddingCompressor(128, 64, hidden_ratio=0.5, norm="layernorm")
    out = ec(torch.randn(2, 128))
    assert out.shape == (2, 64)

    # 5. DepthwiseFinal
    df = last_stage_capacity.DepthwiseFinal(64, 32, kernel_size=3, stride=1)
    out = df(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 32, 8, 8)

    # 6. SqueezeExcitation
    se = last_stage_capacity.SqueezeExcitation(64, reduction=8)
    out = se(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 64, 8, 8)

    # 7. CapacityReductionHead
    head = last_stage_capacity.CapacityReductionHead(128, 10, hidden_ratio=0.5, dropout=0.2, norm="layernorm")
    out = head(torch.randn(2, 128))
    assert out.shape == (2, 10)

    # 8. LinearProjectionReduction
    lpr = last_stage_capacity.LinearProjectionReduction(64, 32, bias=True)
    out = lpr(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 32, 8, 8)

    # 9. SEReduction
    ser = last_stage_capacity.SEReduction(64, reduction_ratio=0.5, bias=False, activation="relu")
    out = ser(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 32, 8, 8)

    # 10. ConditionalCapacityBlock
    ccb = last_stage_capacity.ConditionalCapacityBlock(64, reduction_ratio=0.5, min_channels=32)
    out = ccb(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 32, 8, 8)

    # 11. CapacityReductionStack
    crs = last_stage_capacity.CapacityReductionStack([64, 32, 16], reduction_ratios=0.5, block_type="se")
    outs = crs(torch.randn(2, 64, 8, 8))
    assert len(outs) == 2
    assert outs[0].shape == (2, 32, 8, 8)
    assert outs[1].shape == (2, 16, 8, 8)
