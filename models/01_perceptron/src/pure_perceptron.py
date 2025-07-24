"""
Pure NumPy Perceptron Implementation - True to 1957 Original

This module implements the classic Rosenblatt Perceptron (1957) using only NumPy,
staying faithful to the original algorithm and educational goals of understanding
neural networks from first principles.

Key Features:
- Implements the original perceptron learning rule
- Uses only NumPy for core computations
- Shows explicit weight updates
- Maintains historical fidelity to 1957 algorithm
- Educational focus on understanding fundamentals
"""

<<<<<<< HEAD
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
=======
>>>>>>> 3048305baf15e05456e16ae347f669533e0d7110
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils import get_logger, set_random_seed


class PurePerceptron:
    """
    Pure NumPy implementation of the 1957 Perceptron.

    This implementation stays faithful to Rosenblatt's original algorithm,
    using only basic NumPy operations to demonstrate the fundamental
    learning mechanics without framework abstractions.

    The Classic Perceptron Learning Rule:
    - If prediction is correct: no change
    - If prediction is wrong: w = w + η(target - prediction)x
                             b = b + η(target - prediction)

    Args:
        input_size: Number of input features
        learning_rate: Learning rate η for weight updates
        max_epochs: Maximum training epochs
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.1,
        max_epochs: int = 100,
        random_state: Optional[int] = None,
    ):
        # Validate inputs
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")

        # Store configuration
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state

        # Initialize logger first
        self.logger = get_logger(__name__)

        # Set random seed
        if random_state is not None:
            set_random_seed(random_state)

        # Initialize weights and bias
        self._initialize_weights()

        # Training state
        self.is_fitted = False
        self.training_history = {
            "epochs_trained": 0,
            "training_errors": [],
            "weight_history": [],
            "bias_history": [],
            "converged": False,
        }

    def _initialize_weights(self) -> None:
        """Initialize weights and bias using simple random initialization."""
        # Small random weights (original perceptron used random initialization)
        self.weights = np.random.uniform(-0.5, 0.5, self.input_size)
        self.bias = 0.0  # Start with zero bias

        self.logger.debug(f"Initialized weights: {self.weights}")
        self.logger.debug(f"Initialized bias: {self.bias}")

    def _step_function(self, x: float) -> int:
        """Classic step function activation: 1 if x >= 0, else 0."""
        return 1 if x >= 0 else 0

    def _compute_output(self, x: np.ndarray) -> float:
        """Compute raw output before activation."""
        return np.dot(self.weights, x) + self.bias

    def predict_single(self, x: np.ndarray) -> int:
        """
        Make prediction for a single sample.

        Args:
            x: Input features (1D array)

        Returns:
            Prediction (0 or 1)
        """
        raw_output = self._compute_output(x)
        return self._step_function(raw_output)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple samples.

        Args:
            X: Input features (2D array: samples × features)

        Returns:
            Predictions array (1D: samples)
        """
        if X.ndim == 1:
            return np.array([self.predict_single(X)])

        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return np.array(predictions)

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train the perceptron using the classic learning rule.

        The Original Perceptron Learning Algorithm:
        1. Initialize weights randomly
        2. For each training example:
           - Compute prediction
           - If wrong: update weights and bias
           - If correct: no change
        3. Repeat until convergence or max_epochs

        Args:
            X: Training features (2D array: samples × features)
            y: Training labels (1D array: samples, values 0 or 1)
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        if X.shape[1] != self.input_size:
            raise ValueError(
                f"X must have {self.input_size} features, got {X.shape[1]}"
            )

        # Convert y to 0/1 if needed
        unique_labels = np.unique(y)
        if len(unique_labels) > 2:
            raise ValueError("Perceptron only supports binary classification")
        if not all(label in [0, 1] for label in unique_labels):
            # Map to 0/1
            y = (y == unique_labels[1]).astype(int)

        self.logger.info(
            "Training perceptron on %d samples with %d features", len(X), X.shape[1]
        )

        # Reset training state
        self.training_history = {
            "epochs_trained": 0,
            "training_errors": [],
            "weight_history": [self.weights.copy()],
            "bias_history": [self.bias],
            "converged": False,
            "weight_updates": 0,
        }

        # Training loop
        for epoch in range(self.max_epochs):
            errors = 0
            weight_updates = 0

            # Process each training example
            for i, (x, target) in enumerate(zip(X, y)):
                # Compute prediction
                prediction = self.predict_single(x)

                # Check if prediction is wrong
                if prediction != target:
                    errors += 1
                    weight_updates += 1

                    # Apply perceptron learning rule
                    error = target - prediction  # +1 or -1

                    # Update weights: w = w + η * error * x
                    self.weights += self.learning_rate * error * x

                    # Update bias: b = b + η * error
                    self.bias += self.learning_rate * error

                    if verbose:
                        self.logger.info(
                            f"Epoch {epoch+1}, Sample {i+1}: "
                            f"target={target}, pred={prediction}, "
                            f"updated weights and bias"
                        )

            # Record training progress
            self.training_history["training_errors"].append(errors)
            self.training_history["weight_history"].append(self.weights.copy())
            self.training_history["bias_history"].append(self.bias)
            self.training_history["weight_updates"] += weight_updates

            if verbose:
                self.logger.info("Epoch %d: %d errors", epoch + 1, errors)

            # Check for convergence
            if errors == 0:
                self.training_history["converged"] = True
                self.training_history["epochs_trained"] = epoch + 1
                self.logger.info(f"Converged after {epoch+1} epochs!")
                break

        if not self.training_history["converged"]:
            self.training_history["epochs_trained"] = self.max_epochs
            self.logger.warning(f"Did not converge after {self.max_epochs} epochs")

        self.is_fitted = True
        return self.training_history

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_decision_boundary_info(self) -> Dict[str, Any]:
        """
        Get decision boundary parameters for visualization.

        For 2D data, the decision boundary is the line:
        w1*x1 + w2*x2 + b = 0

        Returns:
            Dictionary with boundary parameters
        """
        if self.input_size != 2:
            return {"error": "Decision boundary only available for 2D data"}

        w1, w2 = self.weights
        b = self.bias

        # Line equation: w1*x1 + w2*x2 + b = 0
        # Solving for x2: x2 = -(w1*x1 + b) / w2

        return {
            "weights": [w1, w2],
            "bias": b,
            "slope": -w1 / w2 if w2 != 0 else float("inf"),
            "intercept": -b / w2 if w2 != 0 else None,
            "equation": f"{w1:.3f}*x1 + {w2:.3f}*x2 + {b:.3f} = 0",
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": "Pure Perceptron (1957)",
            "algorithm": "Rosenblatt Perceptron Learning Rule",
            "implementation": "Pure NumPy",
            "input_size": self.input_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "is_fitted": self.is_fitted,
            "weights": self.weights.tolist() if hasattr(self, "weights") else None,
            "bias": float(self.bias) if hasattr(self, "bias") else None,
            "training_history": self.training_history,
        }

    def save_model(self, filepath: str) -> None:
        """Save model to JSON file."""
        model_data = {
            "config": {
                "input_size": self.input_size,
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "random_state": self.random_state,
            },
            "weights": self.weights.tolist() if hasattr(self, "weights") else None,
            "bias": float(self.bias) if hasattr(self, "bias") else None,
            "is_fitted": self.is_fitted,
            "training_history": self.training_history,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        self.logger.info(f"Saved model to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "PurePerceptron":
        """Load model from JSON file."""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # Create model instance
        model = cls(**model_data["config"])

        # Restore weights and state
        if model_data["weights"] is not None:
            model.weights = np.array(model_data["weights"])
        if model_data["bias"] is not None:
            model.bias = model_data["bias"]

        model.is_fitted = model_data["is_fitted"]
        model.training_history = model_data["training_history"]

        return model


def demonstrate_xor_limitation():
    """
    Demonstrate why the perceptron cannot solve XOR.

    This educational function shows the fundamental limitation
    that motivated multi-layer networks.
    """
    print("🚫 XOR Problem: Perceptron's Fundamental Limitation")
    print("=" * 50)

    # XOR truth table
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR output

    print("XOR Truth Table:")
    for i, (x, y) in enumerate(zip(X_xor, y_xor)):
        print(f"  {x[0]} XOR {x[1]} = {y}")

    # Try to train perceptron
    perceptron = PurePerceptron(input_size=2, learning_rate=0.1, max_epochs=100)
    history = perceptron.fit(X_xor, y_xor, verbose=False)

    # Test predictions
    predictions = perceptron.predict(X_xor)
    accuracy = perceptron.compute_accuracy(X_xor, y_xor)

    print("\nPerceptron Results:")
    print(f"  Converged: {history['converged']}")
    print(f"  Epochs: {history['epochs_trained']}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Predictions: {predictions}")
    print(f"  Expected:    {y_xor}")

    print("\n💡 Why XOR is impossible for a perceptron:")
    print("   The perceptron can only learn linear decision boundaries.")
    print("   XOR requires a non-linear boundary - it's not linearly separable!")
    print("   This limitation led to the development of multi-layer networks.")

    return perceptron, history


if __name__ == "__main__":
    # Educational demonstration
    print("🧠 Pure Perceptron: Understanding the 1957 Algorithm")
    print("=" * 60)

    # Simple linearly separable example
    print("\n✅ AND Function: Should Work!")
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])  # AND output

    perceptron = PurePerceptron(input_size=2, learning_rate=0.1, max_epochs=100)
    history = perceptron.fit(X_and, y_and, verbose=True)

    predictions = perceptron.predict(X_and)
    accuracy = perceptron.compute_accuracy(X_and, y_and)

<<<<<<< HEAD
    print("\nAND Function Results:")
=======
    print(f"\nAND Function Results:")
>>>>>>> 3048305baf15e05456e16ae347f669533e0d7110
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Final weights: {perceptron.weights}")
    print(f"  Final bias: {perceptron.bias}")

    # Show the limitation
    print("\n" + "=" * 60)
    demonstrate_xor_limitation()
