"""
Integration tests for ADALINE engine framework integration.

Tests the ADALINEEngineAdapter, DataSplit support, and unified dataset loading
to ensure backward compatibility and new functionality work correctly.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add models directory to path for imports
models_dir = Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_dir / "02_adaline" / "src"))

try:
    from engine_integration import (
        ADALINEEngineAdapter, 
        create_datasplit_from_dataset,
        create_enhanced_adaline_config
    )
    from model import create_adaline
    from config import get_experiment_config
    from data_loader import load_adaline_train_data, load_adaline_eval_data
    ADALINE_AVAILABLE = True
    SKIP_REASON = "ADALINE available"
except ImportError as e:
    ADALINE_AVAILABLE = False
    SKIP_REASON = f"ADALINE modules not available: {e}"


@pytest.mark.skipif(not ADALINE_AVAILABLE, reason=SKIP_REASON)
class TestADALINEEngineIntegration:
    """Test ADALINE engine integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_datasplit_from_dataset(self):
        """Test DataSplit creation from dataset name."""
        # Test with a small synthetic dataset
        data_split = create_datasplit_from_dataset('debug_small')
        
        # Verify DataSplit structure
        assert hasattr(data_split, 'x_train')
        assert hasattr(data_split, 'y_train')
        assert hasattr(data_split, 'x_val')
        assert hasattr(data_split, 'y_val')
        assert hasattr(data_split, 'x_test')
        assert hasattr(data_split, 'y_test')
        
        # Verify data shapes
        assert len(data_split.x_train) > 0
        assert len(data_split.y_train) > 0
        assert len(data_split.x_val) > 0
        assert len(data_split.y_val) > 0
        assert len(data_split.x_test) > 0
        assert len(data_split.y_test) > 0
        
        # Verify consistent shapes
        assert len(data_split.x_train) == len(data_split.y_train)
        assert len(data_split.x_val) == len(data_split.y_val)
        assert len(data_split.x_test) == len(data_split.y_test)
    
    def test_datasplit_ratios(self):
        """Test DataSplit ratio validation."""
        # Test custom ratios
        data_split = create_datasplit_from_dataset(
            'debug_small',
            train_split=0.6,
            val_split=0.2,
            test_split=0.2
        )
        
        total_samples = len(data_split.x_train) + len(data_split.x_val) + len(data_split.x_test)
        
        # Check approximate ratios (allowing for rounding)
        train_ratio = len(data_split.x_train) / total_samples
        val_ratio = len(data_split.x_val) / total_samples
        test_ratio = len(data_split.x_test) / total_samples
        
        assert abs(train_ratio - 0.6) < 0.1
        assert abs(val_ratio - 0.2) < 0.1
        assert abs(test_ratio - 0.2) < 0.1
    
    def test_adaline_engine_adapter_creation(self):
        """Test ADALINEEngineAdapter creation."""
        # Create ADALINE model
        config = get_experiment_config('debug_small')
        adaline = create_adaline(config)
        
        # Create adapter
        adapter = ADALINEEngineAdapter(adaline)
        
        assert adapter.model == adaline
        assert hasattr(adapter, 'logger')
        assert hasattr(adapter, 'train_with_datasplit')
    
    def test_adaline_training_with_datasplit(self):
        """Test ADALINE training using DataSplit."""
        # Create ADALINE model
        config = get_experiment_config('debug_small')
        adaline = create_adaline(config)
        
        # Create DataSplit
        data_split = create_datasplit_from_dataset('debug_small')
        
        # Create adapter and train
        adapter = ADALINEEngineAdapter(adaline)
        result = adapter.train_with_datasplit(data_split)
        
        # Verify TrainingResult structure
        assert hasattr(result, 'final_loss')
        assert hasattr(result, 'final_train_accuracy')
        assert hasattr(result, 'final_val_accuracy')
        assert hasattr(result, 'final_test_accuracy')
        assert hasattr(result, 'epochs_trained')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'loss_history')
        assert hasattr(result, 'experiment_name')
        assert hasattr(result, 'model_architecture')
        assert hasattr(result, 'hyperparameters')
        
        # Verify training occurred
        assert result.epochs_trained > 0
        assert result.final_loss >= 0
        assert result.model_architecture == 'ADALINE'
        assert 'learning_rate' in result.hyperparameters
        assert 'epochs' in result.hyperparameters
    
    def test_adaline_evaluation_on_splits(self):
        """Test ADALINE evaluation on different data splits."""
        # Create ADALINE model and train briefly
        config = get_experiment_config('debug_small')
        config.epochs = 5  # Quick training
        adaline = create_adaline(config)
        
        # Create DataSplit
        data_split = create_datasplit_from_dataset('debug_small')
        
        # Train model
        adapter = ADALINEEngineAdapter(adaline)
        result = adapter.train_with_datasplit(data_split)
        
        # Verify we got evaluation results
        assert result.final_train_accuracy is not None
        assert result.final_val_accuracy is not None
        assert result.final_test_accuracy is not None
        
        # Accuracies should be reasonable (not NaN, not negative)
        assert 0 <= result.final_train_accuracy <= 1
        assert 0 <= result.final_val_accuracy <= 1
        assert 0 <= result.final_test_accuracy <= 1
    
    def test_enhanced_adaline_config(self):
        """Test enhanced ADALINE configuration."""
        base_config = get_experiment_config('debug_small')
        enhanced_config = create_enhanced_adaline_config(base_config)
        
        # Verify enhanced features were added
        assert hasattr(enhanced_config, 'track_validation')
        assert hasattr(enhanced_config, 'validation_frequency')
        assert hasattr(enhanced_config, 'early_stopping')
        assert hasattr(enhanced_config, 'patience')
        assert hasattr(enhanced_config, 'save_best_model')
        
        # Verify values
        assert enhanced_config.track_validation is True
        assert enhanced_config.validation_frequency > 0
        assert enhanced_config.early_stopping is True
        assert enhanced_config.patience > 0
    
    def test_backward_compatibility(self):
        """Test that existing ADALINE functionality still works."""
        # Test original data loading functions still work
        X_train, y_train = load_adaline_train_data('debug_small')
        X_eval, y_eval = load_adaline_eval_data('debug_small')
        
        # Verify data shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_eval.shape[0] == y_eval.shape[0]
        assert X_train.shape[1] == X_eval.shape[1]  # Same features
        
        # Test original model creation and training
        config = get_experiment_config('debug_small')
        config.epochs = 3  # Quick test
        adaline = create_adaline(config)
        
        # Original training should still work
        training_results = adaline.fit(X_train, y_train)
        
        assert 'final_mse' in training_results
        assert 'epochs_trained' in training_results
        assert 'converged' in training_results
    
    def test_data_conversion_in_adapter(self):
        """Test tensor to numpy conversion in adapter."""
        import torch
        
        # Create ADALINE model
        config = get_experiment_config('debug_small')
        adaline = create_adaline(config)
        adapter = ADALINEEngineAdapter(adaline)
        
        # Test tensor conversion
        x_tensor = torch.randn(10, 2)
        y_tensor = torch.randn(10, 1)
        
        x_np, y_np = adapter._convert_to_numpy(x_tensor, y_tensor)
        
        assert isinstance(x_np, np.ndarray)
        assert isinstance(y_np, np.ndarray)
        assert x_np.dtype == np.float32
        assert y_np.dtype == np.float32
        assert x_np.shape == (10, 2)
        assert y_np.shape == (10, 1)
    
    def test_error_handling(self):
        """Test error handling in engine integration."""
        # Test with invalid dataset name
        with pytest.raises(Exception):
            create_datasplit_from_dataset('nonexistent_dataset')
        
        # Test with invalid split ratios
        with pytest.raises(ValueError):
            create_datasplit_from_dataset(
                'debug_small',
                train_split=0.5,
                val_split=0.3,
                test_split=0.3  # Sum > 1.0
            )


@pytest.mark.skipif(not ADALINE_AVAILABLE, reason=SKIP_REASON)
class TestADALINEDataLoaderIntegration:
    """Test ADALINE data loader integration."""
    
    def test_unified_data_loading(self):
        """Test unified data loading functionality."""
        # Test all supported datasets
        datasets = ['debug_small', 'iris_setosa_versicolor', 'xor_problem']
        
        for dataset_name in datasets:
            try:
                X_train, y_train = load_adaline_train_data(dataset_name)
                X_eval, y_eval = load_adaline_eval_data(dataset_name)
                
                # Verify data was loaded
                assert X_train is not None
                assert y_train is not None
                assert X_eval is not None
                assert y_eval is not None
                
                # Verify shapes are consistent
                assert X_train.shape[0] == y_train.shape[0]
                assert X_eval.shape[0] == y_eval.shape[0]
                assert X_train.shape[1] == X_eval.shape[1]
                
            except Exception as e:
                # Skip datasets that might not be available
                if "not found" not in str(e).lower():
                    raise
    
    def test_dataset_validation(self):
        """Test dataset validation functionality."""
        # This should work without errors
        X_train, y_train = load_adaline_train_data('debug_small')
        
        # Verify basic validation passed
        assert X_train.ndim == 2
        assert y_train.ndim == 2
        assert X_train.shape[0] > 0
        assert y_train.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__]) 