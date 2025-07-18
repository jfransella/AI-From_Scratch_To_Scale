"""
Perceptron Model Implementation.

This module implements the classic Rosenblatt Perceptron (1957) using the unified
model infrastructure. The Perceptron is a foundational neural network that can
learn linear decision boundaries through iterative weight updates.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from engine.base import BaseModel

# Import shared packages
from utils import get_logger, set_random_seed

# Import model-specific components
try:
    from .constants import AUTHORS, MODEL_NAME, MODEL_VERSION, YEAR_INTRODUCED
except ImportError:
    # Fallback for direct imports (e.g., during testing)
    from constants import AUTHORS, MODEL_NAME, MODEL_VERSION, YEAR_INTRODUCED


class Perceptron(nn.Module, BaseModel):  # pylint: disable=too-many-instance-attributes
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

    def _apply_activation(self, x_input: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def predict(self, x: torch.Tensor) -> torch.Tensor:
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

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training.

        Args:
            outputs: Model outputs (already computed by forward pass)
            targets: Ground truth labels

        Returns:
            Loss tensor
        """
        # Use BCE loss for binary classification
        # Ensure outputs are probabilities in [0, 1]
        criterion = nn.BCELoss()
        if self.activation in ("step", "tanh"):
            # Convert outputs to probabilities for loss computation
            outputs = torch.sigmoid(outputs)
        elif self.activation == "sigmoid":
            # Already in [0, 1]
            pass
        else:
            # Default: apply sigmoid
            outputs = torch.sigmoid(outputs)
        return criterion(outputs.squeeze(), targets.float())

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model metadata and current state
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            # Model metadata
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "year_introduced": YEAR_INTRODUCED,
            "original_author": AUTHORS[0] if AUTHORS else "Frank Rosenblatt",
            # Architecture
            "input_size": self.input_size,
            "hidden_size": None,  # Perceptron has no hidden layers
            "output_size": 1,
            "activation": self.activation,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            # Training configuration
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "init_method": self.init_method,
            # Current state
            "is_fitted": self.is_fitted,
            "epochs_trained": self.training_history.get("epochs_trained", 0),
            "device": str(next(self.parameters()).device),
            # Weights (for analysis)
            "weights": self.linear.weight.data.cpu().numpy().tolist(),
            "bias": self.linear.bias.data.cpu().numpy().tolist(),
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

    def fit(self, x_data: torch.Tensor, y_target: torch.Tensor) -> Dict[str, Any]:
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
