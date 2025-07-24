"""
Perceptron model implementation using PyTorch.

This module implements the classic Rosenblatt Perceptron (1957), the first
artificial neural network capable of learning linearly separable patterns.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Handle torch imports gracefully
try:
    import torch

    if (
        hasattr(torch, "__version__")
        and hasattr(torch, "nn")
        and hasattr(torch, "tensor")
    ):
        import torch.nn.functional as F
        from torch import nn

        _TORCH_AVAILABLE = True
        # Use actual torch classes
        BaseNNModule = nn.Module
        TorchTensor = torch.Tensor
    else:
        # torch exists but is broken
        _TORCH_AVAILABLE = False
        torch = None
        nn = None
        F = None

        # Create dummy base classes
        class BaseNNModule:
            def __init__(self):
                pass

            def parameters(self):
                return []

            def to(self, device):
                return self

            def train(self, mode=True):
                return self

            def eval(self):
                return self

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        TorchTensor = Any

        # Create dummy nn module with Linear class
        class DummyNN:
            class Linear:
                def __init__(self, in_features, out_features, bias=True):
                    self.in_features = in_features
                    self.out_features = out_features
                    self.weight = DummyParameter((out_features, in_features))
                    self.bias = DummyParameter((out_features,)) if bias else None

                def __call__(self, x):
                    return x  # Dummy forward pass

            class BCEWithLogitsLoss:
                def __call__(self, input, target):
                    return DummyTensor(0.0)  # Dummy loss

            class init:
                @staticmethod
                def zeros_(tensor):
                    pass  # Dummy weight initialization

                @staticmethod
                def xavier_normal_(tensor):
                    pass

                @staticmethod
                def xavier_uniform_(tensor):
                    pass

                @staticmethod
                def kaiming_normal_(tensor, **kwargs):
                    pass

                @staticmethod
                def kaiming_uniform_(tensor, **kwargs):
                    pass

                @staticmethod
                def normal_(tensor, mean=0, std=1):
                    pass

        class DummyParameter:
            def __init__(self, shape):
                self.data = DummyTensor(shape)

        class DummyTensor:
            def __init__(self, value):
                self.value = value

            def item(self):
                return 0.0

            def cpu(self):
                return self

            def numpy(self):
                import numpy as np

                return np.array([[0.0]])  # Dummy numpy array

            def tolist(self):
                return [[0.0]]

        class DummyTorch:
            @staticmethod
            def no_grad():
                return DummyContext()

            @staticmethod
            def device(device_str):
                return DummyDevice(device_str)

        class DummyDevice:
            def __init__(self, device_str):
                self.type = "cpu"
                self.index = None

            def __str__(self):
                return "cpu"

        class DummyContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        nn = DummyNN()
        torch = DummyTorch()

except ImportError:
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False

    # Create dummy base classes
    class BaseNNModule:
        def __init__(self):
            pass

        def parameters(self):
            return []

        def to(self, device):
            return self

        def train(self, mode=True):
            return self

        def eval(self):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    TorchTensor = Any

    # Create dummy nn module with Linear class
    class DummyNN:
        class Linear:
            def __init__(self, in_features, out_features, bias=True):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = DummyParameter((out_features, in_features))
                self.bias = DummyParameter((out_features,)) if bias else None

            def __call__(self, x):
                return x  # Dummy forward pass

        class BCEWithLogitsLoss:
            def __call__(self, input, target):
                return DummyTensor(0.0)  # Dummy loss

        class init:
            @staticmethod
            def zeros_(tensor):
                pass  # Dummy weight initialization

            @staticmethod
            def xavier_normal_(tensor):
                pass

            @staticmethod
            def xavier_uniform_(tensor):
                pass

            @staticmethod
            def kaiming_normal_(tensor, **kwargs):
                pass

            @staticmethod
            def kaiming_uniform_(tensor, **kwargs):
                pass

            @staticmethod
            def normal_(tensor, mean=0, std=1):
                pass

    class DummyParameter:
        def __init__(self, shape):
            self.data = DummyTensor(shape)

    class DummyTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return 0.0

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.array([[0.0]])  # Dummy numpy array

        def tolist(self):
            return [[0.0]]

    class DummyTorch:
        @staticmethod
        def no_grad():
            return DummyContext()

        @staticmethod
        def device(device_str):
            return DummyDevice(device_str)

    class DummyDevice:
        def __init__(self, device_str):
            self.type = "cpu"
            self.index = None

        def __str__(self):
            return "cpu"

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    nn = DummyNN()
    torch = DummyTorch()

# Import model-specific components
# Always use absolute imports for better compatibility
from constants import AUTHORS, MODEL_NAME, MODEL_VERSION, YEAR_INTRODUCED

from engine.base import BaseModel

# Import shared packages
from utils import get_logger, set_random_seed


class Perceptron(
    BaseNNModule, BaseModel
):  # pylint: disable=too-many-instance-attributes
    """
    Classic Perceptron implementation with BaseModel interface.

    The Perceptron is a linear binary classifier that learns a decision boundary
    to separate two classes. It uses the perceptron learning rule to adjust weights
    based on misclassified examples.

    Key characteristics:
    - Binary classification only
    - Linear decision boundary
    - Guaranteed to converge if data is linearly separable
    - Will not converge if data is not linearly separable

    Historical Context:
    - Introduced in 1957 by Frank Rosenblatt
    - First neural network that could learn from data
    - Foundation for all modern neural networks
    - Inspired the development of multi-layer networks

    Args:
        input_size: Number of input features
        learning_rate: Learning rate for weight updates
        max_epochs: Maximum number of training epochs
        tolerance: Convergence tolerance for loss
        activation: Activation function ('step', 'sigmoid', 'tanh')
        init_method: Weight initialization method
        random_state: Random seed for reproducibility
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_size: int,
        learning_rate: float = 0.1,  # Changed from DEFAULT_LEARNING_RATE
        max_epochs: int = 100,  # Changed from DEFAULT_MAX_EPOCHS
        tolerance: float = 1e-6,  # Changed from DEFAULT_TOLERANCE
        activation: str = "step",  # Changed from DEFAULT_ACTIVATION
        init_method: str = "zeros",  # Changed from DEFAULT_INIT_METHOD
        random_state: Optional[int] = None,
    ):
        super().__init__()

        # Validate input parameters
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")

        # Store configuration
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.activation = activation
        self.init_method = init_method
        self.random_state = random_state

        # Initialize logger
        self.logger = get_logger(__name__)

        # Set random seed if provided
        if random_state is not None:
            set_random_seed(random_state)

        # Define the linear layer (weights + bias)
        self.linear = nn.Linear(input_size, 1)

        # Initialize weights
        self._initialize_weights()

        # Ensure parameters require gradients (important for training)
        if _TORCH_AVAILABLE and torch is not None:
            for param in self.parameters():
                param.requires_grad = True

        # Training state
        self.is_fitted = False
        self.training_history = {"loss": [], "accuracy": [], "epochs_trained": 0}

        self.logger.info(
            "Perceptron initialized: %d inputs, %s activation", input_size, activation
        )

    def _initialize_weights(self):
        """Initialize weights based on the specified method."""
        with torch.no_grad():
            if self.init_method == "zeros":
                nn.init.zeros_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)
            elif self.init_method == "normal":
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.linear.bias)
            elif self.init_method == "xavier":
                nn.init.xavier_uniform_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)
            else:  # random
                nn.init.uniform_(self.linear.weight, -0.5, 0.5)
                nn.init.zeros_(self.linear.bias)

    def _apply_activation(self, x_input: TorchTensor) -> TorchTensor:
        """Apply the specified activation function."""
        if self.activation == "step":
            # Use differentiable approximation of step function for training
            if self.training:
                # Steep sigmoid approximates step function but maintains gradients
                return torch.sigmoid(10.0 * x_input)
            # True step function for inference
            return (x_input >= 0).float()
        if self.activation == "sigmoid":
            return torch.sigmoid(x_input)
        if self.activation == "tanh":
            return torch.tanh(x_input)
        # Default to step with same logic
        if self.training:
            return torch.sigmoid(10.0 * x_input)
        return (x_input >= 0).float()

    def forward(self, x: TorchTensor) -> TorchTensor:
        """
        Forward pass through the perceptron.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Linear transformation
        output = self.linear(x)

        # Apply activation function
        return self._apply_activation(output)

    def predict(self, x: TorchTensor) -> TorchTensor:
        """
        Make binary predictions.

        Args:
            x: Input tensor

        Returns:
            Binary predictions (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # For binary classification, threshold at 0.5
            predictions = (outputs >= 0.5).float().squeeze()
            return predictions

    def predict_proba(self, x: TorchTensor) -> TorchTensor:
        """
        Get prediction probabilities.

        Args:
            x: Input tensor

        Returns:
            Probability tensor of shape (batch_size, 2) for binary classification
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x).squeeze()

            # Convert to probabilities for binary classification
            if self.activation == "sigmoid":
                prob_pos = outputs
            else:
                # For step function, use distance from decision boundary
                raw_output = self.linear(x).squeeze()
                prob_pos = torch.sigmoid(raw_output)

            prob_neg = 1 - prob_pos
            # Ensure both are at least 1D
            if prob_pos.dim() == 0:
                prob_pos = prob_pos.unsqueeze(0)
                prob_neg = prob_neg.unsqueeze(0)
            probs = torch.stack([prob_neg, prob_pos], dim=1)
            return probs

    def get_loss(self, outputs: TorchTensor, targets: TorchTensor) -> TorchTensor:
        """
        Compute loss for training.

        Uses BCEWithLogitsLoss for numerical stability.

        Args:
            outputs: Raw model outputs (logits) [batch_size, 1]
            targets: Target labels [batch_size] or [batch_size, 1]

        Returns:
            Loss tensor
        """
        # Ensure outputs are [batch_size, 1]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        # Ensure targets are [batch_size, 1] to match outputs
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        elif targets.dim() > 1:
            targets = targets.squeeze()
            if targets.dim() == 0:  # Handle single sample case
                targets = targets.unsqueeze(0)
            targets = targets.unsqueeze(1)

        targets = targets.float()

        criterion = nn.BCEWithLogitsLoss()
        return criterion(outputs, targets)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information following wandb integration plan standards.

        Returns:
            Dictionary containing model metadata and current state
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            # Core identification
            "name": "Perceptron",
            "full_name": "Rosenblatt Perceptron",
            "category": "foundation",
            "module": 1,
            "pattern": "engine-based",
            # Historical context
            "year_introduced": YEAR_INTRODUCED,
            "authors": AUTHORS,
            "paper_title": "The perceptron: a probabilistic model for information storage and organization in the brain",
            "key_innovations": [
                "First neural network that could learn from data",
                "Perceptron learning rule for weight updates",
                "Foundation for all modern neural networks",
                "Demonstrated machine learning capabilities",
            ],
            # Architecture details
            "architecture_type": "single-layer",
            "input_size": self.input_size,
            "output_size": 1,
            "parameter_count": total_params,
            "trainable_parameters": trainable_params,
            "activation_function": self.activation,
            "hidden_layers": 0,
            # Training characteristics
            "learning_algorithm": "perceptron-rule",
            "loss_function": "bce-with-logits",
            "optimizer": "sgd",
            "convergence_guarantee": "linearly separable data",
            # Implementation details
            "framework": "pytorch",
            "precision": "float32",
            "device_support": ["cpu", "cuda", "mps"],
            "device": (
                str(next(iter(self.parameters())).device)
                if list(self.parameters())
                else "cpu"
            ),
            # Educational metadata
            "difficulty_level": "beginner",
            "estimated_training_time": "seconds to minutes",
            "key_learning_objectives": [
                "Understand linear decision boundaries",
                "Learn perceptron learning rule",
                "Discover limitations of linear models",
                "Foundation of neural networks",
            ],
            # Training state
            "is_fitted": self.is_fitted,
            "epochs_trained": self.training_history.get("epochs_trained", 0),
            "converged": len(self.training_history.get("loss", [])) > 0
            and self.training_history["loss"][-1] <= self.tolerance,
            # Training configuration
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "init_method": self.init_method,
            # Current weights (for analysis)
            "weights": self.linear.weight.data.cpu().numpy().tolist(),
            "bias": self.linear.bias.data.cpu().numpy().tolist(),
            # Legacy compatibility
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "original_author": AUTHORS[0] if AUTHORS else "Frank Rosenblatt",
            "total_parameters": total_params,
            "activation": self.activation,  # Legacy field name for activation_function
        }

    def save_checkpoint(
        self,
        filepath: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint.

        Args:
            filepath (str): Path to save checkpoint
            epoch (int, optional): Current epoch number
            optimizer_state (dict, optional): Optimizer state dict
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_info": self.get_model_info(),
            "epoch": epoch,
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, filepath)
        self.logger.info("Saved checkpoint to %s", filepath)

    @classmethod
    def load_from_checkpoint(cls, filepath: str, **model_kwargs):
        """
        Load model from checkpoint.

        Args:
            filepath (str): Path to checkpoint file
            **model_kwargs: Additional arguments for model initialization

        Returns:
            Perceptron: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location="cpu")

        # Extract model parameters from checkpoint
        model_info = checkpoint.get("model_info", {})
        input_size = model_info.get("input_size", model_kwargs.get("input_size", 2))
        learning_rate = model_kwargs.get("learning_rate", 0.1)
        max_epochs = model_kwargs.get("max_epochs", 100)
        tolerance = model_kwargs.get("tolerance", 0.01)
        activation = model_kwargs.get("activation", "step")
        init_method = model_kwargs.get("init_method", "zeros")
        random_state = model_kwargs.get("random_state", None)

        # Create model instance
        model = cls(
            input_size=input_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            tolerance=tolerance,
            activation=activation,
            init_method=init_method,
            random_state=random_state,
            **{
                k: v
                for k, v in model_kwargs.items()
                if k
                not in [
                    "input_size",
                    "learning_rate",
                    "max_epochs",
                    "tolerance",
                    "activation",
                    "init_method",
                    "random_state",
                ]
            },
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore training state
        model.is_fitted = True
        model.training_history["epochs_trained"] = checkpoint.get("epoch", 0)

        model.logger.info("Loaded model from %s", filepath)
        return model

    def save_model(self, filepath: str):
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save both model state and metadata
        save_dict = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "input_size": self.input_size,
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "tolerance": self.tolerance,
                "activation": self.activation,
                "init_method": self.init_method,
                "random_state": self.random_state,
            },
            "training_history": self.training_history,
            "model_info": self.get_model_info(),
        }

        torch.save(save_dict, save_path)
        self.logger.info("Model saved to %s", save_path)

    @classmethod
    def load_model(cls, filepath: str) -> "Perceptron":
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded Perceptron model
        """
        save_dict = torch.load(filepath, map_location="cpu")

        # Extract configuration
        config = save_dict["model_config"]

        # Create new model instance
        model = cls(
            input_size=config["input_size"],
            learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"],
            tolerance=config["tolerance"],
            activation=config["activation"],
            init_method=config["init_method"],
            random_state=config["random_state"],
        )

        # Load state
        model.load_state_dict(save_dict["model_state_dict"])
        model.training_history = save_dict.get("training_history", {})
        model.is_fitted = True

        model.logger.info("Model loaded from %s", filepath)
        return model

    def fit(self, x_data: TorchTensor, y_target: TorchTensor) -> Dict[str, Any]:
        """
        Fit the perceptron using the classic perceptron learning rule.

        This method provides the traditional perceptron training for
        educational purposes and compatibility.

        Args:
            X: Training data
            y: Training labels

        Returns:
            Training history
        """
        self.logger.info("Starting perceptron training")

        # Reset training state
        self.training_history = {"loss": [], "accuracy": [], "epochs_trained": 0}

        # Training loop
        for epoch in range(self.max_epochs):
            # Forward pass
            predictions = self.predict(x_data)

            # Compute accuracy
            accuracy = (predictions == y_target).float().mean().item()

            # Compute loss (number of misclassifications)
            loss = (predictions != y_target).float().sum().item()

            # Store metrics
            self.training_history["loss"].append(loss)
            self.training_history["accuracy"].append(accuracy)
            self.training_history["epochs_trained"] = epoch + 1

            # Check convergence
            if loss <= self.tolerance:
                self.logger.info("Converged at epoch %d", epoch + 1)
                break

            # Perceptron learning rule
            self.train()
            for x_sample, y_sample in zip(x_data, y_target):
                prediction = self.predict(x_sample.unsqueeze(0))
                if prediction != y_sample:
                    # Update weights using proper parameter operations
                    with torch.no_grad():
                        if y_sample == 1:  # Misclassified positive
                            self.linear.weight.data += (
                                self.learning_rate * x_sample.unsqueeze(0)
                            )
                            self.linear.bias.data += self.learning_rate
                        else:  # Misclassified negative
                            self.linear.weight.data -= (
                                self.learning_rate * x_sample.unsqueeze(0)
                            )
                            self.linear.bias.data -= self.learning_rate

        self.is_fitted = True
        self.logger.info("Training completed: %.4f accuracy", accuracy)
        return self.training_history

    def fit_historical(
        self, x_data: TorchTensor, y_target: TorchTensor, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train using the original 1957 Rosenblatt Perceptron Learning Rule.

        This method implements the historical algorithm without gradients:
        - Uses step function activation
        - Only updates weights on errors
        - Simple rule-based learning (no calculus!)

        Args:
            x_data: Input features
            y_target: Target labels (0 or 1)
            verbose: Print training progress

        Returns:
            Training results and history
        """
        self.logger.info("🏛️ Training with HISTORICAL Perceptron Learning Rule (1957)")
        self.logger.info("   • No gradients, no backpropagation")
        self.logger.info("   • Error-driven weight updates only")
        self.logger.info("   • Step function activation")

        # Ensure data is 2D
        if x_data.dim() == 1:
            x_data = x_data.unsqueeze(0)
        if y_target.dim() == 1:
            y_target = y_target.unsqueeze(1)

        n_samples = x_data.shape[0]

        # Initialize weights to zeros (historical default)
        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            self.linear.bias.fill_(0.0)

        # Training history
        history = {
            "loss": [],
            "accuracy": [],
            "errors": [],
            "weight_updates": [],
            "converged": False,
            "convergence_epoch": None,
        }

        # Historical perceptron training loop
        for epoch in range(self.max_epochs):
            epoch_errors = 0
            epoch_updates = 0

            # Process each sample individually (historical approach)
            for i in range(n_samples):
                x_i = x_data[i : i + 1]  # Keep batch dimension
                y_i = y_target[i : i + 1]

                # Forward pass with step function
                with torch.no_grad():
                    raw_output = self.linear(x_i)
                    prediction = (raw_output >= 0.0).float()  # Step function

                    # Check for error
                    error = y_i - prediction

                    if error.abs().sum() > 0:  # Only update on errors
                        epoch_errors += 1
                        epoch_updates += 1

                        # Historical perceptron learning rule
                        # w = w + η * (target - prediction) * input
                        weight_update = self.learning_rate * error * x_i
                        bias_update = self.learning_rate * error

                        # Apply updates
                        self.linear.weight.data += weight_update
                        self.linear.bias.data += bias_update

            # Compute epoch metrics
            with torch.no_grad():
                outputs = self.linear(x_data)
                predictions = (outputs >= 0.0).float()
                accuracy = (predictions == y_target).float().mean().item()

                # Use simple squared error for loss (not BCE)
                loss = torch.mean((y_target - predictions) ** 2).item()

            # Store history
            history["loss"].append(loss)
            history["accuracy"].append(accuracy)
            history["errors"].append(epoch_errors)
            history["weight_updates"].append(epoch_updates)

            # Progress logging
            if verbose or epoch % max(1, self.max_epochs // 10) == 0:
                self.logger.info(
                    f"Epoch {epoch:3d}: Accuracy={accuracy:.4f}, Errors={epoch_errors:2d}, Loss={loss:.6f}"
                )

            # Convergence check (no errors = perfect classification)
            if epoch_errors == 0:
                history["converged"] = True
                history["convergence_epoch"] = epoch
                if verbose:
                    self.logger.info(f"✅ CONVERGED at epoch {epoch}! No more errors.")
                    self.logger.info(
                        "🎉 Historical perceptron found perfect linear separator!"
                    )
                break

            # Early stopping if loss is below tolerance
            if loss < self.tolerance:
                history["converged"] = True
                history["convergence_epoch"] = epoch
                break

        # Update training state
        self.is_fitted = True
        self.training_history.update(history)

        final_accuracy = history["accuracy"][-1]

        if verbose:
            self.logger.info(f"\n🏛️ Historical Perceptron Training Complete:")
            self.logger.info(f"   Final Accuracy: {final_accuracy:.4f}")
            self.logger.info(f"   Epochs Trained: {epoch + 1}")
            self.logger.info(f"   Converged: {history['converged']}")

            if final_accuracy >= 0.99:
                self.logger.info(
                    "🎯 Perfect separation achieved! (Linearly separable data)"
                )
            elif final_accuracy < 0.6:
                self.logger.info(
                    "⚠️  Poor performance - likely non-linearly separable data"
                )
                self.logger.info(
                    "   This demonstrates the fundamental limitation Minsky & Papert identified!"
                )

        return {
            "final_accuracy": final_accuracy,
            "epochs_trained": epoch + 1,
            "converged": history["converged"],
            "convergence_epoch": history.get("convergence_epoch"),
            "final_loss": history["loss"][-1],
            "total_errors": sum(history["errors"]),
            "algorithm": "historical_perceptron_1957",
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"Perceptron(input_size={self.input_size}, "
            f"activation='{self.activation}', "
            f"lr={self.learning_rate}, "
            f"fitted={self.is_fitted})"
        )


def create_perceptron(config: Dict[str, Any]) -> Perceptron:
    """
    Factory function to create a Perceptron from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Perceptron model
    """
    return Perceptron(
        input_size=config.get("input_size", 2),
        learning_rate=config.get(
            "learning_rate", 0.1
        ),  # Changed from DEFAULT_LEARNING_RATE
        max_epochs=config.get("max_epochs", 100),  # Changed from DEFAULT_MAX_EPOCHS
        tolerance=config.get("tolerance", 0.01),  # Changed from DEFAULT_TOLERANCE
        activation=config.get("activation", "step"),  # Changed from DEFAULT_ACTIVATION
        init_method=config.get(
            "init_method", "zeros"
        ),  # Changed from DEFAULT_INIT_METHOD
        random_state=config.get("random_state", None),
    )
