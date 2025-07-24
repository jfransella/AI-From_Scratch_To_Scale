"""
PyTorch Wrapper for Pure NumPy Perceptron

This module provides a thin PyTorch wrapper around the pure NumPy perceptron
implementation, allowing it to work with the unified engine infrastructure
while preserving the educational value of the from-scratch implementation.

Design Philosophy:
- Core logic remains in pure NumPy (educational)
- PyTorch wrapper provides engine compatibility (practical)
- Students can see both the pure algorithm and modern integration
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from engine.base import BaseModel
from utils import get_logger

try:
    from .constants import AUTHORS, MODEL_NAME, MODEL_VERSION, YEAR_INTRODUCED
    from .pure_perceptron import PurePerceptron
except ImportError:
    # For direct execution
    from constants import AUTHORS, MODEL_NAME, MODEL_VERSION, YEAR_INTRODUCED
    from pure_perceptron import PurePerceptron


class PerceptronWrapper(nn.Module, BaseModel):
    """
    PyTorch wrapper for the pure NumPy Perceptron implementation.

    This class bridges the educational pure NumPy implementation with
    the practical PyTorch-based engine infrastructure. The core learning
    logic remains in NumPy to preserve educational value.

    Key Features:
    - Uses PurePerceptron for actual computations
    - Provides PyTorch tensor interface for engine compatibility
    - Maintains access to pure implementation for educational analysis
    - Handles data conversion between NumPy and PyTorch automatically
    """

    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.1,
        max_epochs: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Create the pure NumPy perceptron
        self.pure_perceptron = PurePerceptron(
            input_size=input_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            random_state=random_state,
        )

        # Store configuration for compatibility
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state

        # Create PyTorch parameters that mirror the NumPy weights
        # These are used for engine compatibility but the pure perceptron
        # maintains the actual training logic
        self.linear = nn.Linear(input_size, 1, bias=True)
        self._sync_pytorch_params()

        # Training state
        self.training_history = {}

        # Logger
        self.logger = get_logger(__name__)

        self.logger.info("Initialized PerceptronWrapper with Pure NumPy core")

    def _sync_pytorch_params(self):
        """Sync PyTorch parameters with pure perceptron weights."""
        if hasattr(self.pure_perceptron, "weights"):
            with torch.no_grad():
                self.linear.weight.data = torch.from_numpy(
                    self.pure_perceptron.weights.reshape(1, -1)
                ).float()
                self.linear.bias.data = torch.tensor(
                    [self.pure_perceptron.bias]
                ).float()

    def _sync_pure_params(self):
        """Sync pure perceptron weights with PyTorch parameters."""
        self.pure_perceptron.weights = self.linear.weight.data.numpy().flatten()
        self.pure_perceptron.bias = self.linear.bias.data.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the pure perceptron implementation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Convert to NumPy for pure perceptron
        x_np = x.cpu().numpy()

        # Use pure perceptron for predictions
        if x_np.ndim == 1:
            predictions = [self.pure_perceptron.predict_single(x_np)]
        else:
            predictions = []
            for sample in x_np:
                predictions.append(self.pure_perceptron.predict_single(sample))

        # Convert back to PyTorch tensor
        return torch.tensor(
            predictions, dtype=torch.float32, device=x.device
        ).unsqueeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make binary predictions using pure perceptron."""
        self.eval()
        with torch.no_grad():
            x_np = x.cpu().numpy()
            predictions = self.pure_perceptron.predict(x_np)
            return torch.from_numpy(predictions).float()

    def fit_pure(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train using the pure perceptron learning rule.

        This method exposes the educational pure implementation directly,
        allowing students to see the original 1957 algorithm in action.

        Args:
            X: Training features (NumPy array)
            y: Training labels (NumPy array)
            verbose: Show training progress

        Returns:
            Training history from pure perceptron
        """
        self.logger.info("Training with pure perceptron learning rule")

        # Train the pure perceptron
        history = self.pure_perceptron.fit(X, y, verbose=verbose)

        # Sync PyTorch parameters
        self._sync_pytorch_params()

        # Store history
        self.training_history = history

        return history

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for engine compatibility.

        Note: The pure perceptron uses the discrete learning rule,
        but for engine compatibility we provide a continuous loss.
        """
        # Handle 2D targets by squeezing if needed
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Simple MSE loss for binary classification
        return nn.functional.mse_loss(outputs.squeeze(), targets.float())

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        pure_info = self.pure_perceptron.get_model_info()

        return {
            # Model metadata
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "year_introduced": YEAR_INTRODUCED,
            "original_author": AUTHORS[0] if AUTHORS else "Frank Rosenblatt",
            "implementation": "Pure NumPy + PyTorch Wrapper",
            # Architecture from pure perceptron
            "input_size": pure_info["input_size"],
            "learning_rate": pure_info["learning_rate"],
            "max_epochs": pure_info["max_epochs"],
            "is_fitted": pure_info["is_fitted"],
            "algorithm": pure_info["algorithm"],
            # Current weights
            "weights": pure_info["weights"],
            "bias": pure_info["bias"],
            # Training history
            "training_history": pure_info["training_history"],
            # PyTorch compatibility info
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "device": str(next(self.parameters()).device),
        }

    def save_model(self, filepath: str):
        """Save both pure perceptron and PyTorch state."""
        # Save pure perceptron
        pure_path = str(Path(filepath).with_suffix(".pure.json"))
        self.pure_perceptron.save_model(pure_path)

        # Save PyTorch state
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "random_state": self.random_state,
            },
            "pure_perceptron_path": pure_path,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, filepath)

        self.logger.info(f"Saved wrapped perceptron to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "PerceptronWrapper":
        """Load wrapped perceptron from file."""
        checkpoint = torch.load(filepath, map_location="cpu")

        # Create wrapper instance
        model = cls(**checkpoint["config"])

        # Load PyTorch state
        model.load_state_dict(checkpoint["state_dict"])

        # Load pure perceptron if available
        if "pure_perceptron_path" in checkpoint:
            pure_path = checkpoint["pure_perceptron_path"]
            if Path(pure_path).exists():
                model.pure_perceptron = PurePerceptron.load_model(pure_path)

        # Restore training history
        model.training_history = checkpoint.get("training_history", {})

        return model

    def get_pure_perceptron(self) -> PurePerceptron:
        """
        Get access to the pure NumPy implementation for educational analysis.

        This allows students to:
        - Examine the actual learning rule implementation
        - See weight update history
        - Understand the convergence behavior
        - Analyze decision boundaries
        """
        return self.pure_perceptron

    def demonstrate_learning_rule(
        self, X: np.ndarray, y: np.ndarray, max_steps: int = 5
    ):
        """
        Educational method to demonstrate the perceptron learning rule step by step.

        Args:
            X: Training features
            y: Training labels
            max_steps: Maximum steps to demonstrate
        """
        print("🎓 Perceptron Learning Rule Demonstration")
        print("=" * 50)
        print("Rule: If wrong, w = w + η(target - prediction)x")
        print("      If wrong, b = b + η(target - prediction)")
        print()

        # Reset to fresh weights
        self.pure_perceptron._initialize_weights()

        step = 0
        for i, (x, target) in enumerate(zip(X, y)):
            if step >= max_steps:
                break

            # Current state
            prediction = self.pure_perceptron.predict_single(x)
            raw_output = self.pure_perceptron._compute_output(x)

            print(f"Step {step + 1}:")
            print(f"  Input: {x}")
            print(f"  Weights: {self.pure_perceptron.weights}")
            print(f"  Bias: {self.pure_perceptron.bias:.3f}")
            print(f"  Raw output: {raw_output:.3f}")
            print(f"  Prediction: {prediction}")
            print(f"  Target: {target}")

            if prediction != target:
                # Show the update
                error = target - prediction
                old_weights = self.pure_perceptron.weights.copy()
                old_bias = self.pure_perceptron.bias

                # Apply update
                self.pure_perceptron.weights += (
                    self.pure_perceptron.learning_rate * error * x
                )
                self.pure_perceptron.bias += self.pure_perceptron.learning_rate * error

                print(f"  ❌ Wrong! Applying update...")
                print(f"  Error: {error}")
                print(
                    f"  Weight update: {old_weights} + {self.pure_perceptron.learning_rate} × {error} × {x}"
                )
                print(f"                 = {self.pure_perceptron.weights}")
                print(
                    f"  Bias update: {old_bias:.3f} + {self.pure_perceptron.learning_rate} × {error}"
                )
                print(f"               = {self.pure_perceptron.bias:.3f}")
            else:
                print(f"  ✅ Correct! No update needed.")

            print()
            step += 1


def create_perceptron(input_size: int, **kwargs) -> PerceptronWrapper:
    """
    Factory function to create a perceptron with pure NumPy core.

    This is the recommended way to create perceptrons that maintain
    educational value while providing engine compatibility.
    """
    return PerceptronWrapper(input_size=input_size, **kwargs)


if __name__ == "__main__":
    # Educational demonstration
    print("🔗 Perceptron Wrapper: Pure NumPy + Engine Compatibility")
    print("=" * 65)

    # Create wrapped perceptron
    wrapper = create_perceptron(input_size=2, learning_rate=0.1, max_epochs=100)

    # AND function data
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])

    print("\n📚 Step-by-step learning demonstration:")
    wrapper.demonstrate_learning_rule(X_and, y_and, max_steps=3)

    print("\n🏃 Full training with pure implementation:")
    history = wrapper.fit_pure(X_and, y_and, verbose=False)

    accuracy = wrapper.pure_perceptron.compute_accuracy(X_and, y_and)
    print(f"Final accuracy: {accuracy:.1%}")
    print(f"Converged: {history['converged']}")
    print(f"Epochs: {history['epochs_trained']}")

    # Show PyTorch compatibility
    print(f"\n🔧 PyTorch compatibility:")
    X_torch = torch.from_numpy(X_and).float()
    y_torch = torch.from_numpy(y_and).float()

    with torch.no_grad():
        outputs = wrapper.forward(X_torch)
        loss = wrapper.get_loss(outputs, y_torch)

    print(f"PyTorch forward pass: {outputs.squeeze().numpy()}")
    print(f"PyTorch loss: {loss.item():.4f}")

    print(f"\n✨ Best of both worlds:")
    print(f"   - Pure NumPy core for education")
    print(f"   - PyTorch wrapper for engine compatibility")
    print(f"   - Students see the real 1957 algorithm!")
