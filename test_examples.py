"""Tests for last_stage_capacity.examples demo entry point."""
import sys
from pathlib import Path
import socket
import pytest
import torch

_echo_root = Path(__file__).parent
if str(_echo_root) not in sys.path:
    sys.path.insert(0, str(_echo_root))

from last_stage_capacity.examples import demo, ResNetWithLastStageReduction


def test_examples_demo_cpu_end_to_end(monkeypatch, capsys):
    """
    Test that invokes the demo() entry point in last_stage_capacity/examples.py
    end-to-end on CPU, asserting it runs to completion with random init only,
    no pretrained weights and no network downloads.
    """
    # Guard against network downloads: disable socket connection
    def block_network(*args, **kwargs):
        raise RuntimeError("Network downloads are not permitted during tests")

    monkeypatch.setattr(socket, "socket", block_network)
    monkeypatch.setattr(socket, "create_connection", block_network)

    # Track module instantiations to ensure no pretrained weights are loaded
    created_modules = []
    orig_init = ResNetWithLastStageReduction.__init__

    def tracked_resnet_init(self, *args, **kwargs):
        created_modules.append(self)
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(ResNetWithLastStageReduction, "__init__", tracked_resnet_init)

    # Force torch default device to CPU if applicable and run demo()
    demo()

    # Capture stdout to assert demo executed all sections
    captured = capsys.readouterr()
    assert "Last-Stage Capacity Reduction — demo" in captured.out
    assert "ResNetWithLastStageReduction:" in captured.out
    assert "Bottleneck efficiency (same in/out channels):" in captured.out
    assert "WidthScaler (50% width reduction):" in captured.out

    # Verify ResNetWithLastStageReduction was created and ran on CPU
    assert len(created_modules) == 1
    resnet_model = created_modules[0]
    for param in resnet_model.parameters():
        assert param.device.type == "cpu"
