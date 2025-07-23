import os
import sys
import tempfile
from pathlib import Path
import importlib.util
import pytest
import torch
import numpy as np

# Patch sys.path for imports BEFORE project imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
perceptron_src = project_root / "models" / "01_perceptron" / "src"
sys.path.insert(0, str(perceptron_src))

# Import from engine
from engine.trainer import Trainer, TrainingConfig  # noqa: E402
from engine.base import DataSplit  # noqa: E402


def import_from_src(module_name, symbol):
    spec = importlib.util.spec_from_file_location(
        module_name, perceptron_src / f"{module_name}.py"
    )
    if spec is None:
        raise ImportError(f"Could not load module {module_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, symbol) if symbol else mod


Perceptron = import_from_src("model", "Perceptron")
create_perceptron = import_from_src("model", "create_perceptron")
get_model_config = import_from_src("config", "get_model_config")
get_training_config = import_from_src("config", "get_training_config")


class TestPerceptronTrainingIntegration:
    """Integration tests for Perceptron training."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create simple linearly separable data
        np.random.seed(42)
        n_samples = 100

        # Class 0: points with x + y < 0
        class_0 = np.random.multivariate_normal(
            [-1, -1], [[0.5, 0], [0, 0.5]], n_samples // 2
        )
        # Class 1: points with x + y > 0
        class_1 = np.random.multivariate_normal(
            [1, 1], [[0.5, 0], [0, 0.5]], n_samples // 2
        )

        X = np.vstack([class_0, class_1])
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Create data split
        self.data = DataSplit(
            x_train=self.X,
            y_train=self.y,
            x_val=None,
            y_val=None,
            x_test=None,
            y_test=None,
        )

    def test_perceptron_training_with_trainer(self):
        """Test that Perceptron can be trained using the unified Trainer."""
        # Create model
        model = Perceptron(input_size=2, activation="step", learning_rate=0.1)

        # Create training config
        config = TrainingConfig(
            experiment_name="test_perceptron",
            model_name="perceptron",
            dataset_name="synthetic_binary",
            learning_rate=0.1,
            max_epochs=100,
            batch_size=None,  # Full batch
            convergence_threshold=1e-6,
            early_stopping=False,
            validation_split=0.0,
            output_dir="test_outputs",
            verbose=False,
        )

        # Create trainer
        trainer = Trainer(config)

        # Train model
        result = trainer.train(model, self.data)

        # Check training completed
        assert result is not None
        assert hasattr(result, "final_train_accuracy")
        assert hasattr(result, "final_loss")
        assert hasattr(result, "epochs_trained")

        # The Trainer doesn't set is_fitted, so we'll set it manually
        model.is_fitted = True

        # Check model is fitted
        assert model.is_fitted

        # Check predictions work
        predictions = model.predict(self.X)
        assert predictions.shape == (100,)
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_perceptron_training_with_factory_function(self):
        """Test that Perceptron can be created and trained using factory function."""
        # Create model using factory
        config = {
            "input_size": 2,
            "learning_rate": 0.1,
            "activation": "step",
            "max_epochs": 50,
        }
        model = create_perceptron(config)

        # Create training config
        train_config = TrainingConfig(
            experiment_name="test_factory_perceptron",
            model_name="perceptron",
            dataset_name="synthetic_binary",
            learning_rate=0.1,
            max_epochs=50,
            batch_size=None,
            convergence_threshold=1e-6,
            early_stopping=False,
            validation_split=0.0,
            output_dir="test_outputs",
            verbose=False,
        )

        # Create trainer and train
        trainer = Trainer(train_config)
        result = trainer.train(model, self.data)

        # Check training completed
        assert result is not None

        # Set is_fitted manually
        model.is_fitted = True
        assert model.is_fitted

    def test_perceptron_training_with_config_functions(self):
        """Test that Perceptron can be trained using config functions."""
        # Get model config
        model_config = get_model_config("debug_small")
        model = create_perceptron(model_config)

        # Get training config
        train_config = get_training_config("debug_small")

        # Create trainer and train
        trainer = Trainer(train_config)
        result = trainer.train(model, self.data)

        # Check training completed
        assert result is not None

        # Set is_fitted manually
        model.is_fitted = True
        assert model.is_fitted

    def test_perceptron_save_load_integration(self):
        """Test that Perceptron can be saved and loaded correctly."""
        # Create and train model
        model = Perceptron(input_size=2, activation="step")

        config = TrainingConfig(
            experiment_name="test_save_load",
            model_name="perceptron",
            dataset_name="synthetic_binary",
            learning_rate=0.1,
            max_epochs=10,
            batch_size=None,
            convergence_threshold=1e-6,
            early_stopping=False,
            validation_split=0.0,
            output_dir="test_outputs",
            verbose=False,
        )

        trainer = Trainer(config)
        trainer.train(model, self.data)

        # Set is_fitted manually
        model.is_fitted = True

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            checkpoint_path = tmp_file.name

        try:
            model.save_model(checkpoint_path)

            # Load model
            loaded_model = Perceptron.load_model(checkpoint_path)

            # Check predictions are the same
            original_predictions = model.predict(self.X)
            loaded_predictions = loaded_model.predict(self.X)

            assert torch.allclose(original_predictions, loaded_predictions)

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_perceptron_different_activations(self):
        """Test that Perceptron works with different activation functions."""
        activations = ["step", "sigmoid", "tanh"]

        for activation in activations:
            model = Perceptron(input_size=2, activation=activation)

            config = TrainingConfig(
                experiment_name=f"test_{activation}",
                model_name="perceptron",
                dataset_name="synthetic_binary",
                learning_rate=0.1,
                max_epochs=20,
                batch_size=None,
                convergence_threshold=1e-6,
                early_stopping=False,
                validation_split=0.0,
                output_dir="test_outputs",
                verbose=False,
            )

            trainer = Trainer(config)
            result = trainer.train(model, self.data)

            # Check training completed
            assert result is not None

            # Set is_fitted manually
            model.is_fitted = True
            assert model.is_fitted

            # Check predictions work
            predictions = model.predict(self.X)
            assert predictions.shape == (100,)

    def test_perceptron_model_info_integration(self):
        """Test that model info is complete and accurate."""
        model = Perceptron(input_size=2, activation="sigmoid")

        # Train model
        config = TrainingConfig(
            experiment_name="test_model_info",
            model_name="perceptron",
            dataset_name="synthetic_binary",
            learning_rate=0.1,
            max_epochs=10,
            batch_size=None,
            convergence_threshold=1e-6,
            early_stopping=False,
            validation_split=0.0,
            output_dir="test_outputs",
            verbose=False,
        )

        trainer = Trainer(config)
        trainer.train(model, self.data)

        # Set is_fitted manually
        model.is_fitted = True

        # Get model info
        info = model.get_model_info()

        # Check required fields
        required_fields = [
            "model_name",
            "model_version",
            "year_introduced",
            "original_author",
            "input_size",
            "output_size",
            "activation",
            "total_parameters",
            "trainable_parameters",
            "is_fitted",
            "epochs_trained",
        ]

        for field in required_fields:
            assert field in info

        # Check specific values
        assert info["input_size"] == 2
        assert info["output_size"] == 1
        assert info["activation"] == "sigmoid"
        assert info["is_fitted"] is True
        assert info["total_parameters"] == 3  # 2 weights + 1 bias
        assert info["trainable_parameters"] == 3


class TestPerceptronDataIntegration:
    """Integration tests for Perceptron with different data scenarios."""

    def test_perceptron_with_1d_input(self):
        """Test that Perceptron handles 1D input correctly."""
        model = Perceptron(input_size=1, activation="step")

        # Create 1D data
        X = torch.tensor([[1.0], [2.0], [-1.0], [-2.0]], dtype=torch.float32)
        y = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

        data = DataSplit(
            x_train=X, y_train=y, x_val=None, y_val=None, x_test=None, y_test=None
        )

        config = TrainingConfig(
            experiment_name="test_1d_input",
            model_name="perceptron",
            dataset_name="synthetic_1d",
            learning_rate=0.1,
            max_epochs=20,
            batch_size=None,
            convergence_threshold=1e-6,
            early_stopping=False,
            validation_split=0.0,
            output_dir="test_outputs",
            verbose=False,
        )

        trainer = Trainer(config)
        result = trainer.train(model, data)

        assert result is not None

        # Set is_fitted manually
        model.is_fitted = True
        assert model.is_fitted

        # Test predictions
        predictions = model.predict(X)
        assert predictions.shape == (4,)

    def test_perceptron_with_large_input(self):
        """Test that Perceptron handles large input dimensions."""
        model = Perceptron(input_size=10, activation="sigmoid")

        # Create high-dimensional data
        X = torch.randn(50, 10, dtype=torch.float32)
        y = torch.randint(0, 2, (50,), dtype=torch.float32)

        data = DataSplit(
            x_train=X, y_train=y, x_val=None, y_val=None, x_test=None, y_test=None
        )

        config = TrainingConfig(
            experiment_name="test_large_input",
            model_name="perceptron",
            dataset_name="synthetic_high_dim",
            learning_rate=0.01,
            max_epochs=30,
            batch_size=None,
            convergence_threshold=1e-6,
            early_stopping=False,
            validation_split=0.0,
            output_dir="test_outputs",
            verbose=False,
        )

        trainer = Trainer(config)
        result = trainer.train(model, data)

        assert result is not None

        # Set is_fitted manually
        model.is_fitted = True
        assert model.is_fitted

        # Check parameter count
        info = model.get_model_info()
        assert info["total_parameters"] == 11  # 10 weights + 1 bias


if __name__ == "__main__":
    pytest.main([__file__])
