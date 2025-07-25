# Model Test Template

"""
Test template for neural network models.
Supports both simple and engine-based testing patterns.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import sys
import os
from pathlib import Path

# Add model source to path
model_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(model_dir))

try:
    # Try engine-based imports
    from model import ModelTemplate
    from config import get_training_config
    from constants import MODEL_CONFIG, DATASET_SPECS
    ENGINE_AVAILABLE = True
except ImportError:
    # Fallback to simple imports
    try:
        from model import SimpleModelTemplate
        from config import SimpleExperimentConfig
        ENGINE_AVAILABLE = False
    except ImportError:
        pytest.skip("Model implementation not found", allow_module_level=True)


class TestModelImplementation:
    """Test suite for model implementation."""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sample data for testing."""
        batch_size = 32
        input_size = 2
        num_classes = 2
        
        X = torch.randn(batch_size, input_size)
        y = torch.randint(0, num_classes, (batch_size,))
        
        return X, y
    
    @pytest.fixture
    def model(self) -> nn.Module:
        """Create model instance for testing."""
        if ENGINE_AVAILABLE:
            config = get_training_config("debug")
            return ModelTemplate(config.model_config)
        else:
            config = SimpleExperimentConfig(
                input_size=2,
                output_size=2,
                hidden_size=64,
                activation="relu"
            )
            return SimpleModelTemplate(config)
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_model_forward_pass(self, model, sample_data):
        """Test forward pass produces correct output shape."""
        X, y = sample_data
        output = model(X)
        
        assert output.shape[0] == X.shape[0]  # Batch size preserved
        assert output.dim() == 2  # 2D output for classification
        assert not torch.isnan(output).any()  # No NaN values
        assert torch.isfinite(output).all()  # All finite values
    
    def test_model_backward_pass(self, model, sample_data):
        """Test that backward pass works correctly."""
        X, y = sample_data
        output = model(X)
        
        # Create loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        for param in model.parameters():
            assert param.grad is not None, "Gradient is None"
            assert torch.isfinite(param.grad).all(), "Gradient contains non-finite values"
    
    def test_model_training_mode(self, model):
        """Test training/eval mode switching."""
        # Test training mode
        model.train()
        assert model.training
        
        # Test eval mode
        model.eval()
        assert not model.training
    
    def test_model_parameters(self, model):
        """Test model has learnable parameters."""
        params = list(model.parameters())
        assert len(params) > 0, "Model has no parameters"
        
        total_params = sum(p.numel() for p in params)
        assert total_params > 0, "Model has no learnable parameters"
    
    def test_model_output_range(self, model, sample_data):
        """Test model output is in reasonable range."""
        X, _ = sample_data
        with torch.no_grad():
            output = model(X)
            
        # For classification, logits can be any range, but should be reasonable
        assert output.abs().max() < 100, "Output values too large"
    
    def test_model_deterministic(self, model, sample_data):
        """Test model produces deterministic outputs."""
        X, _ = sample_data
        
        # Set model to eval mode
        model.eval()
        
        with torch.no_grad():
            output1 = model(X)
            output2 = model(X)
            
        torch.testing.assert_close(output1, output2, msg="Model is not deterministic")
    
    def test_model_different_batch_sizes(self, model):
        """Test model works with different batch sizes."""
        input_size = 2
        
        for batch_size in [1, 4, 16, 32]:
            X = torch.randn(batch_size, input_size)
            output = model(X)
            assert output.shape[0] == batch_size


class TestModelTraining:
    """Test suite for training functionality."""
    
    @pytest.fixture
    def simple_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create simple linearly separable dataset."""
        n_samples = 100
        X = torch.randn(n_samples, 2)
        # Create linearly separable labels
        y = (X[:, 0] + X[:, 1] > 0).long()
        return X, y
    
    def test_training_loop_basic(self, simple_dataset):
        """Test basic training loop functionality."""
        X, y = simple_dataset
        
        if ENGINE_AVAILABLE:
            config = get_training_config("debug")
            model = ModelTemplate(config.model_config)
        else:
            config = SimpleExperimentConfig(
                input_size=2,
                output_size=2,
                learning_rate=0.01,
                max_epochs=10
            )
            model = SimpleModelTemplate(config)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Training should reduce loss
        assert final_loss < initial_loss, "Training did not reduce loss"
    
    def test_overfitting_capability(self, simple_dataset):
        """Test model can overfit to small dataset."""
        X, y = simple_dataset[:10]  # Very small dataset
        
        if ENGINE_AVAILABLE:
            config = get_training_config("debug")
            model = ModelTemplate(config.model_config)
        else:
            config = SimpleExperimentConfig(
                input_size=2,
                output_size=2,
                learning_rate=0.1,
                max_epochs=100
            )
            model = SimpleModelTemplate(config)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        
        # Train until convergence
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            
            if loss.item() < 0.01:  # Converged
                break
        
        # Model should achieve low loss
        assert loss.item() < 0.1, "Model cannot overfit to small dataset"


class TestModelConfiguration:
    """Test suite for configuration management."""
    
    def test_config_creation(self):
        """Test configuration can be created."""
        if ENGINE_AVAILABLE:
            config = get_training_config("debug")
            assert config is not None
            assert hasattr(config, 'model_config')
        else:
            config = SimpleExperimentConfig()
            assert config is not None
            assert hasattr(config, 'input_size')
    
    def test_config_validation(self):
        """Test configuration validation."""
        if ENGINE_AVAILABLE:
            # Test valid config
            config = get_training_config("debug")
            assert config.model_config.input_size > 0
            assert config.model_config.output_size > 0
        else:
            # Test valid config
            config = SimpleExperimentConfig(input_size=10, output_size=5)
            assert config.input_size == 10
            assert config.output_size == 5
            
            # Test invalid config
            with pytest.raises((ValueError, AssertionError)):
                config = SimpleExperimentConfig(input_size=0, output_size=5)


class TestModelUtils:
    """Test utility functions if available."""
    
    def test_constants_available(self):
        """Test that constants are available."""
        if ENGINE_AVAILABLE:
            assert MODEL_CONFIG is not None
            assert DATASET_SPECS is not None
    
    def test_device_handling(self):
        """Test device handling works correctly."""
        # Test CPU
        model_cpu = self._create_model("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = self._create_model("cuda")
            assert next(model_cuda.parameters()).device.type == "cuda"
    
    def _create_model(self, device: str) -> nn.Module:
        """Helper to create model on specific device."""
        if ENGINE_AVAILABLE:
            config = get_training_config("debug")
            model = ModelTemplate(config.model_config)
        else:
            config = SimpleExperimentConfig()
            model = SimpleModelTemplate(config)
        
        return model.to(device)


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_end_to_end_training(self):
        """Test complete training workflow."""
        # Create synthetic data
        n_samples = 200
        X = torch.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).long()
        
        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train model
        if ENGINE_AVAILABLE:
            config = get_training_config("debug")
            model = ModelTemplate(config.model_config)
            lr = config.training.learning_rate
            epochs = min(config.training.max_epochs, 50)
        else:
            config = SimpleExperimentConfig(max_epochs=50)
            model = SimpleModelTemplate(config)
            lr = config.learning_rate
            epochs = config.max_epochs
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_predictions = test_output.argmax(dim=1)
            accuracy = (test_predictions == y_test).float().mean()
        
        # Should achieve reasonable accuracy on simple dataset
        assert accuracy > 0.7, f"Model accuracy too low: {accuracy:.3f}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
