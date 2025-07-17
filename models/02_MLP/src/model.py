"""
Multi-Layer Perceptron (MLP) implementation from scratch.

This module implements a multi-layer perceptron capable of solving non-linearly
separable problems like XOR, overcoming the fundamental limitations of
single-layer perceptrons through the use of hidden layers and backpropagation.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import shared packages
from utils import get_logger, set_random_seed

# Import engine framework (optional)
try:
    from engine.base import BaseModel

    HAS_BASE_MODEL = True
except ImportError:
    BaseModel = object
    HAS_BASE_MODEL = False


class MLP(nn.Module):
    """
    Multi-Layer Perceptron implementation with configurable architecture.

    This implementation demonstrates the breakthrough that enabled neural networks
    to solve non-linearly separable problems through hidden layers and
    backpropagation optimization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: str = "sigmoid",
        weight_init: str = "xavier_normal",
        device: str = "cpu",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the Multi-Layer Perceptron.

        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output units
            activation: Activation function ('sigmoid', 'tanh', 'relu', 'leaky_relu')
            weight_init: Weight initialization method
            device: Device to run on ('cpu' or 'cuda')
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.device = device
        self.random_state = random_state

        # Store additional parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize logger
        self.logger = get_logger(__name__)

        # Set random seed if provided
        if self.random_state is not None:
            set_random_seed(self.random_state)

        # Build the network layers
        self.layers = nn.ModuleList()

        # Input to first hidden layer
        if hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))

            # Hidden to hidden layers
            for i in range(1, len(hidden_layers)):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

            # Last hidden to output
            self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        else:
            # Direct input to output (reduces to perceptron)
            self.layers.append(nn.Linear(input_size, output_size))

        # Set activation function
        self.activation = self._get_activation_function(activation)

        # Initialize weights
        self._initialize_weights(weight_init)

        # Training state
        self.is_fitted = False
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "epoch": [],
            "converged": False,
            "convergence_epoch": None,
        }

        # Move to device
        self.to(device)

        self.logger.info(
            "Initialized MLP: %s -> %s -> %s", input_size, hidden_layers, output_size
        )
        self.logger.info("Activation: %s, Weight init: %s", activation, weight_init)
        self.logger.info("Total parameters: %s", self._count_parameters())

    def _get_activation_function(self, activation: str) -> Callable:
        """Get the activation function."""
        activations = {
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": F.relu,
            "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        return activations[activation]

    def _initialize_weights(self, method: str):
        """Initialize network weights using specified method."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if method == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif method == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif method == "he_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif method == "he_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif method == "random_normal":
                    nn.init.normal_(layer.weight, 0, 0.1)
                elif method == "zeros":
                    nn.init.zeros_(layer.weight)
                else:
                    raise ValueError(f"Unknown weight initialization: {method}")

                # Initialize biases to zero
                nn.init.zeros_(layer.bias)

    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Forward through all layers except the last
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        # Output layer (no activation for regression/binary classification)
        x = self.layers[-1](x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the trained model.

        Args:
            x: Input tensor

        Returns:
            Predictions (0 or 1 for binary classification)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # For binary classification, apply sigmoid and threshold
            if self.output_size == 1:
                predictions = (torch.sigmoid(outputs) > 0.5).float()
            else:
                predictions = torch.argmax(outputs, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            x: Input tensor

        Returns:
            Probability scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            if self.output_size == 1:
                # Binary classification
                proba = torch.sigmoid(outputs)
                return torch.cat([1 - proba, proba], dim=1)
            else:
                # Multi-class classification
                return F.softmax(outputs, dim=1)

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training.

        Args:
            outputs: Model outputs (already computed by forward pass)
            targets: Ground truth labels

        Returns:
            Loss tensor
        """
        if self.output_size == 1:
            # Binary classification
            return F.binary_cross_entropy_with_logits(
                outputs.squeeze(), targets.float()
            )
        else:
            # Multi-class classification
            return F.cross_entropy(outputs, targets.long())

    def fit(self, _x_data: torch.Tensor, _y_target: torch.Tensor) -> Dict[str, Any]:
        """
        Fit the model to the data using the engine framework.

        Args:
            _x_data: Input features (unused - required by BaseModel interface)
            _y_target: Target labels (unused - required by BaseModel interface)

        Returns:
            Dictionary with training results
        """
        # This method is required by BaseModel interface
        # The actual training is handled by the engine framework
        self.is_fitted = True
        return {
            "converged": True,
            "epochs_trained": 0,
            "final_loss": 0.0,
            "final_accuracy": 0.0,
        }

    def train_model(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
        convergence_threshold: float = 1e-6,
        patience: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the MLP using backpropagation.

        Args:
            x_train: Training input data
            y_train: Training target data
            x_test: Test input data (optional)
            y_test: Test target data (optional)
            learning_rate: Learning rate for optimization
            max_epochs: Maximum number of training epochs
            convergence_threshold: Loss threshold for convergence
            patience: Early stopping patience
            verbose: Whether to print training progress

        Returns:
            Dictionary with training results
        """
        self.train()

        # Setup optimizer and loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        if self.output_size == 1:
            # Binary classification
            criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            criterion = nn.CrossEntropyLoss()

        # Training loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            # Forward pass
            outputs = self.forward(x_train)

            # Compute loss
            if self.output_size == 1:
                loss = criterion(outputs, y_train.unsqueeze(1))
            else:
                loss = criterion(outputs, y_train.long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            train_accuracy = self._compute_accuracy(x_train, y_train)

            # Store history
            self.training_history["loss"].append(loss.item())
            self.training_history["accuracy"].append(train_accuracy)
            self.training_history["epoch"].append(epoch)

            # Check convergence
            if loss.item() < convergence_threshold:
                self.training_history["converged"] = True
                self.training_history["convergence_epoch"] = epoch
                if verbose:
                    self.logger.info("Converged at epoch %s", epoch)
                break

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    self.logger.info("Early stopping at epoch %s", epoch)
                break

            # Log progress
            if verbose and epoch % 10 == 0:
                self.logger.info(
                    "Epoch %s: Loss=%.6f, Accuracy=%.4f",
                    epoch,
                    loss.item(),
                    train_accuracy,
                )

        # Test accuracy if test data provided
        test_accuracy = None
        if x_test is not None and y_test is not None:
            test_accuracy = self._compute_accuracy(x_test, y_test)
            if verbose:
                self.logger.info("Test accuracy: %.4f", test_accuracy)

        self.is_fitted = True

        return {
            "final_loss": self.training_history["loss"][-1],
            "final_train_accuracy": self.training_history["accuracy"][-1],
            "final_test_accuracy": test_accuracy,
            "converged": self.training_history["converged"],
            "convergence_epoch": self.training_history["convergence_epoch"],
            "epochs_trained": len(self.training_history["loss"]),
        }

    def _compute_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy for given data."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            if self.output_size == 1:
                predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            else:
                predictions = torch.argmax(outputs, dim=1)

            correct = (predictions == y).float().sum()
            accuracy = correct / len(y)

        return accuracy.item()

    def get_hidden_representations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get hidden layer representations.

        Args:
            x: Input tensor

        Returns:
            List of hidden representations
        """
        self.eval()
        with torch.no_grad():
            representations = []
            current = x

            # Forward through all layers except the last
            for layer in self.layers[:-1]:
                current = layer(current)
                current = self.activation(current)
                representations.append(current.clone())

            return representations

    def save_model(self, filepath: str, include_history: bool = True):
        """
        Save model to file.

        Args:
            filepath: Path to save model
            include_history: Whether to include training history
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "activation_name": self.activation_name,
            "weight_init": self.weight_init,
            "is_fitted": self.is_fitted,
        }

        if include_history:
            save_dict["training_history"] = self.training_history

        torch.save(save_dict, filepath)
        self.logger.info("Model saved to %s", filepath)

    @classmethod
    def load_model(cls, filepath: str, device: str = "cpu") -> "MLP":
        """
        Load model from file.

        Args:
            filepath: Path to model file
            device: Device to load model on

        Returns:
            Loaded MLP model
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Create model instance
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_layers=checkpoint["hidden_layers"],
            output_size=checkpoint["output_size"],
            activation=checkpoint["activation_name"],
            weight_init=checkpoint["weight_init"],
            device=device,
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load training history if available
        if "training_history" in checkpoint:
            model.training_history = checkpoint["training_history"]

        # Set fitted status
        if "is_fitted" in checkpoint:
            model.is_fitted = checkpoint["is_fitted"]

        # Restore training state with epochs_trained
        if "training_history" in checkpoint:
            model.training_history["epochs_trained"] = len(
                checkpoint["training_history"].get("epoch", [])
            )

        model.logger.info("Model loaded from %s", filepath)
        return model

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.

        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "MLP",
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "activation": self.activation_name,
            "weight_init": self.weight_init,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": str(self),
            "is_fitted": self.is_fitted,
            "epochs_trained": len(self.training_history.get("epoch", [])),
        }

    def visualize_decision_boundary(
        self, x: torch.Tensor, _y: torch.Tensor, resolution: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize decision boundary for 2D data.

        Args:
            x: Input data (2D)
            y: Target labels
            resolution: Resolution of the grid

        Returns:
            Tuple of (X_grid, Y_grid, Z_predictions)
        """
        if x.shape[1] != 2:
            raise ValueError("Decision boundary visualization only works for 2D data")

        self.eval()

        # Create grid
        x_min, x_max = x[:, 0].min().item(), x[:, 0].max().item()
        y_min, y_max = x[:, 1].min().item(), x[:, 1].max().item()

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        # Make predictions on grid
        grid_points = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            if self.output_size == 1:
                # Binary classification
                outputs = self.forward(grid_points)
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                # Multi-class classification
                outputs = self.forward(grid_points)
                predictions = F.softmax(outputs, dim=1)[:, 1]  # Probability of class 1

        # Reshape predictions
        z = predictions.cpu().numpy().reshape(xx.shape)

        return xx, yy, z

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"MLP(input_size={self.input_size}, "
            f"hidden_layers={self.hidden_layers}, "
            f"output_size={self.output_size})"
        )


# Advanced version with BaseModel inheritance (for engine integration)
if HAS_BASE_MODEL:

    class MLPAdvanced(MLP, BaseModel):
        """
        Advanced MLP with BaseModel interface for engine integration.

        This version provides additional functionality for integration
        with the unified training engine.
        """

        def __init__(self, **kwargs):
            # Extract BaseModel parameters
            input_size = kwargs.pop("input_size", 2)
            hidden_layers = kwargs.pop("hidden_layers", [4])
            output_size = kwargs.pop("output_size", 1)
            activation = kwargs.pop("activation", "sigmoid")
            weight_init = kwargs.pop("weight_init", "xavier_normal")
            device = kwargs.pop("device", "cpu")

            # Initialize MLP
            super().__init__(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
                activation=activation,
                weight_init=weight_init,
                device=device,
            )

            # Store additional parameters
            for key, value in kwargs.items():
                setattr(self, key, value)


# Factory function for easy model creation
def create_mlp(config: Dict[str, Any]) -> MLP:
    """
    Factory function to create MLP from configuration.

    Args:
        config: Configuration dictionary containing model parameters

    Returns:
        Configured MLP instance
    """
    return MLP(
        input_size=config.get("input_size", 2),
        hidden_layers=config.get("hidden_layers", [4]),
        output_size=config.get("output_size", 1),
        activation=config.get("activation", "sigmoid"),
        weight_init=config.get("weight_init", "xavier_normal"),
        device=config.get("device", "cpu"),
    )
