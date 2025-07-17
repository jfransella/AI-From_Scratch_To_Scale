import sys
import importlib.util
from pathlib import Path
import pytest
import torch

# Patch sys.path for Perceptron src
perceptron_src = Path(__file__).parent.parent.parent / "models" / "01_perceptron" / "src"
sys.path.insert(0, str(perceptron_src))


def import_module_from_src(module_name):
    spec = importlib.util.spec_from_file_location(module_name, perceptron_src / f"{module_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_import_and_utilities():
    """Test that evaluate module functions can be imported and basic validation works."""
    from unittest.mock import Mock, patch
    
    # Mock the problematic imports before importing evaluate
    with patch.dict('sys.modules', {
        'utils': Mock(),
        'data_utils': Mock(), 
        'engine.evaluator': Mock(),
        'plotting': Mock()
    }):
        try:
            evaluate = import_module_from_src("evaluate")
        except ImportError as e:
            # Skip test if relative import fails
            pytest.skip(f"Skipping due to ImportError: {e}")
        
        # Test load_model_from_checkpoint with a dummy config and missing file
        with patch('evaluate.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            with pytest.raises(ValueError, match="Could not load model"):
                evaluate.load_model_from_checkpoint("not_a_real_checkpoint.pth", {"input_size": 2})
        
        # Test prepare_evaluation_data with mocked load_dataset
        with patch('evaluate.load_dataset') as mock_load_dataset:
            with patch('evaluate.get_logger') as mock_logger:
                mock_logger.return_value = Mock()
                
                # Mock successful data loading
                sample_X = torch.randn(50, 4)
                sample_y = torch.randint(0, 2, (50,)).float()
                mock_load_dataset.return_value = (sample_X, sample_y)
                
                dummy_cfg = {"dataset_name": "debug_small", "dataset_params": {}}
                X_eval, y_eval = evaluate.prepare_evaluation_data(dummy_cfg, split="test")
                
                # Should return test split (last 20%)
                expected_start = int(0.8 * len(sample_X))
                expected_size = len(sample_X) - expected_start
                assert len(X_eval) == expected_size
                                 assert len(y_eval) == expected_size


def test_train_import_and_utilities():
    train = import_module_from_src("train")
    # Test create_data_split with dummy tensors
    X = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,), dtype=torch.float32)
    split = train.create_data_split(X, y, validation_split=0.2, test_split=0.2, random_state=42)
    assert hasattr(split, "x_train")
    assert hasattr(split, "y_train")
    assert split.x_train.shape[1] == 2
    # Test that main() can be imported (do not call, as it expects CLI args)
    assert hasattr(train, "main") 