import pytest
import runpy
from unittest.mock import patch, MagicMock

@patch('timm.create_model')
def test_check_convnext(mock_create_model, capsys):
    """Test check_convnext.py runs without error and prints expected output."""
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model
    mock_model.named_modules.return_value = []
    mock_model.named_children.return_value = []
    
    mock_model.stages = [MagicMock() for _ in range(4)]
    
    runpy.run_path('check_convnext.py')
    
    out, err = capsys.readouterr()
    assert "=== Pool layers ===" in out
    assert "=== Top-level children ===" in out
    assert "=== stages.3 accessible? ===" in out
    assert "=== Searching for final conv stage ===" in out

@patch('timm.create_model')
def test_check_convnext_head(mock_create_model, capsys):
    """Test check_convnext_head.py runs without error and prints expected output."""
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model
    mock_model.head.fc.out_features = 1000
    
    runpy.run_path('check_convnext_head.py')
    
    out, err = capsys.readouterr()
    assert "head type:" in out
    assert "--- After patching head.fc ---" in out

def test_debug_scale(capsys):
    """Test debug_scale.py runs without error and prints expected output."""
    runpy.run_path('debug_scale.py')
    out, err = capsys.readouterr()
    assert "Before scaling:" in out
    assert "After scaling:" in out
    assert "Input:" in out
    assert "Output:" in out
