"""
Integration tests for Perceptron training pipeline.

This module tests the complete training workflow including data loading,
model creation, training, and evaluation.
"""

import os
import sys
import tempfile
from pathlib import Path
import importlib.util
import pytest
import torch
import numpy as np

# Patch sys.path for Perceptron src
perceptron_src = (
    Path(__file__).parent.parent.parent / "models" / "01_perceptron" / "src"
)
sys.path.insert(0, str(perceptron_src))

from data_utils import load_dataset  # noqa: E402
from engine.trainer import Trainer  # noqa: E402


def import_from_src(module_name, symbol):
    spec = importlib.util.spec_from_file_location(
        module_name, perceptron_src / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, symbol) if symbol else mod


Perceptron = import_from_src("model", "Perceptron")
get_model_config = import_from_src("config", "get_model_config")
get_training_config = import_from_src("config", "get_training_config")
get_dataset_config = import_from_src("config", "get_dataset_config")


class TestPerceptronTrainingPipeline:
    """Test suite for complete Perceptron training pipeline."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_experiment = "debug_small"

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_data_loading_pipeline(self):
        """Test that data loading works correctly."""
        # Get dataset configuration
        dataset_config = get_dataset_config(self.test_experiment)

        # Load dataset
        X, y = load_dataset(
            dataset_config["dataset_name"], dataset_config["dataset_params"]
        )

        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Verify data properties
        assert X.dim() == 2, "Input should be 2D"
        # Note: New data_utils returns 2D targets for consistency [batch_size, 1]
        assert y.dim() in [1, 2], "Target should be 1D or 2D"
        if y.dim() == 2:
            assert y.shape[1] == 1, "2D targets should have shape [batch_size, 1]"
        assert X.shape[0] == y.shape[0], "Number of samples should match"
        assert X.shape[1] == dataset_config["input_size"], "Input size should match"

        # Verify data types
        assert X.dtype == torch.float32, "Input should be float32"
        assert y.dtype == torch.float32, "Target should be float32"

        # Verify target values are binary
        unique_targets = torch.unique(y)
        assert len(unique_targets) <= 2, "Should be binary classification"
        assert torch.all(
            (unique_targets == 0) | (unique_targets == 1)
        ), "Targets should be 0 or 1"

    def test_model_creation_pipeline(self):
        """Test that model creation works correctly."""
        # Get model configuration
        model_config = get_model_config(self.test_experiment)

        # Create model
        model = Perceptron(**model_config)

        # Verify model properties
        assert model.input_size == model_config["input_size"]
        assert model.learning_rate == model_config["learning_rate"]
        assert model.activation == model_config["activation"]

        # Test forward pass
        test_input = torch.randn(5, model_config["input_size"])
        output = model.forward(test_input)

        assert output.shape == (5, 1), "Output shape should be (batch_size, 1)"
        assert output.dtype == torch.float32, "Output should be float32"

    def test_training_configuration_pipeline(self):
        """Test that training configuration works correctly."""
        # Get training configuration
        training_config = get_training_config(self.test_experiment)

        # Verify configuration properties
        assert hasattr(training_config, "experiment_name")
        assert hasattr(training_config, "model_name")
        assert hasattr(training_config, "learning_rate")
        assert hasattr(training_config, "max_epochs")
        assert hasattr(training_config, "device")

        # Verify configuration values
        assert training_config.experiment_name == self.test_experiment
        assert training_config.model_name == "Perceptron"
        assert training_config.learning_rate > 0
        assert training_config.max_epochs > 0

    def test_single_epoch_training(self):
        """Test single epoch training workflow."""
        # Get configurations
        training_config = get_training_config(self.test_experiment)
        model_config = get_model_config(self.test_experiment)
        dataset_config = get_dataset_config(self.test_experiment)

        # Load data
        x, y = load_dataset(
            dataset_config["dataset_name"], dataset_config["dataset_params"]
        )

        # Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Create model
        model = Perceptron(**model_config)

        # Create simple data splits for testing
        n_samples = len(x)
        train_size = int(0.8 * n_samples)

        x_train, y_train = x[:train_size], y[:train_size]
        x_val, y_val = x[train_size:], y[train_size:]

        # Test single training step
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=training_config.learning_rate
        )

        # Forward pass
        outputs = model.forward(x_train)
        loss = model.get_loss(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify training step completed
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"

        # Test evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model.forward(x_val)
            val_loss = model.get_loss(val_outputs, y_val)

            # Make predictions
            predictions = model.predict(x_val)

            # Calculate accuracy
            accuracy = (predictions == y_val).float().mean()

        # Verify evaluation results
        assert val_loss.item() >= 0, "Validation loss should be non-negative"
        assert 0 <= accuracy.item() <= 1, "Accuracy should be between 0 and 1"

    def test_model_save_load_integration(self):
        """Test model save and load integration."""
        # Create model
        model_config = get_model_config(self.test_experiment)
        model = Perceptron(**model_config)

        # Create test data
        test_input = torch.randn(10, model_config["input_size"])
        original_output = model.forward(test_input)

        # Save model
        checkpoint_path = os.path.join(self.temp_dir, "test_model.pth")
        model.save_model(checkpoint_path)

        # Verify file was created
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

        # Load model
        loaded_model = Perceptron.load_model(checkpoint_path)

        # Verify model state is preserved
        loaded_output = loaded_model.forward(test_input)
        assert torch.allclose(
            original_output, loaded_output
        ), "Outputs should be identical"

        # Verify model parameters are preserved
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(param1, param2), "Parameters should be identical"

    def test_training_with_shared_engine(self):
        """Test training using the shared engine infrastructure."""
        # Get configurations
        training_config = get_training_config(self.test_experiment)
        model_config = get_model_config(self.test_experiment)
        dataset_config = get_dataset_config(self.test_experiment)

        # Load data
        x, y = load_dataset(
            dataset_config["dataset_name"], dataset_config["dataset_params"]
        )

        # Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Create data splits
        n_samples = len(x)
        train_size = int(0.8 * n_samples)

        x_train, y_train = x[:train_size], y[:train_size]
        x_val, y_val = x[train_size:], y[train_size:]

        # Create model
        model = Perceptron(**model_config)

        # Create trainer with correct API
        trainer = Trainer(training_config)

        # Test that trainer can be created
        assert trainer is not None, "Trainer should be created successfully"


class TestPerceptronEvaluationPipeline:
    """Test suite for Perceptron evaluation pipeline."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_experiment = "debug_small"

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_evaluation_metrics_computation(self):
        """Test that evaluation metrics are computed correctly."""
        # Create a simple trained model
        model_config = get_model_config(self.test_experiment)
        model = Perceptron(**model_config)

        # Create test data
        x_test = torch.randn(100, model_config["input_size"])
        y_test = torch.randint(0, 2, (100,)).float()

        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)

        # Compute metrics manually
        accuracy = (predictions == y_test).float().mean()

        # Verify metrics
        assert 0 <= accuracy.item() <= 1, "Accuracy should be between 0 and 1"
        assert probabilities.shape == (
            100,
            2,
        ), "Probabilities should have correct shape"
        assert torch.allclose(
            probabilities.sum(dim=1), torch.ones(100)
        ), "Probabilities should sum to 1"

    def test_evaluation_with_saved_model(self):
        """Test evaluation using a saved model."""
        # Create and save a model
        model_config = get_model_config(self.test_experiment)
        model = Perceptron(**model_config)

        checkpoint_path = os.path.join(self.temp_dir, "eval_test_model.pth")
        model.save_model(checkpoint_path)

        # Load model for evaluation
        loaded_model = Perceptron.load_model(checkpoint_path)

        # Create test data
        x_test = torch.randn(50, model_config["input_size"])
        y_test = torch.randint(0, 2, (50,)).float()

        # Evaluate loaded model
        loaded_model.eval()
        with torch.no_grad():
            predictions = loaded_model.predict(x_test)
            accuracy = (predictions == y_test).float().mean()

        # Verify evaluation works
        assert 0 <= accuracy.item() <= 1, "Accuracy should be between 0 and 1"


if __name__ == "__main__":
    pytest.main([__file__])
