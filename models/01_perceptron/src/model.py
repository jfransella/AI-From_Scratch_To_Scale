"""
Perceptron model implementation.

Implements the classic Perceptron algorithm by Frank Rosenblatt (1957),
the first artificial neural network capable of learning.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
import logging

from utils import get_logger, set_random_seed
from utils.exceptions import ModelError, TrainingError
from constants import (
    DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS, DEFAULT_TOLERANCE,
    DEFAULT_ACTIVATION, DEFAULT_INIT_METHOD, validate_learning_rate,
    validate_epochs, validate_activation
)


class Perceptron(nn.Module):
    """
    Classic Perceptron implementation.
    
    The Perceptron is a linear binary classifier that learns a decision boundary
    to separate two classes. It uses the perceptron learning rule to adjust weights
    based on misclassified examples.
    
    Key characteristics:
    - Binary classification only
    - Linear decision boundary
    - Guaranteed to converge if data is linearly separable
    - Will not converge if data is not linearly separable
    
    Historical note: This is the first neural network that could learn,
    introduced by Frank Rosenblatt in 1957.
    """
    
    def __init__(self, 
                 n_features: int,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 max_epochs: int = DEFAULT_MAX_EPOCHS,
                 tolerance: float = DEFAULT_TOLERANCE,
                 activation: str = DEFAULT_ACTIVATION,
                 init_method: str = DEFAULT_INIT_METHOD,
                 random_state: Optional[int] = None):
        """
        Initialize the Perceptron.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for weight updates
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance (minimum error change)
            activation: Activation function ('step' or 'sign')
            init_method: Weight initialization method
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        self.n_features = n_features
        self.learning_rate = validate_learning_rate(learning_rate)
        self.max_epochs = validate_epochs(max_epochs)
        self.tolerance = tolerance
        self.activation = validate_activation(activation)
        self.init_method = init_method
        self.random_state = random_state
        
        if random_state is not None:
            set_random_seed(random_state)
        
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.zeros(n_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        
        # Initialize weights according to specified method
        self._initialize_weights()
        
        # Training state
        self.is_fitted = False
        self.n_epochs_trained = 0
        self.training_history = {
            'epochs': [],
            'errors': [],
            'weights': [],
            'bias': [],
            'accuracy': [],
            'converged': False
        }
        
        self.logger.info(f"Initialized Perceptron: features={n_features}, lr={learning_rate}, "
                        f"activation={activation}, init={init_method}")
    
    def _initialize_weights(self):
        """Initialize weights according to the specified method."""
        with torch.no_grad():
            if self.init_method == "zeros":
                # Historical default - start with zero weights
                self.weights.fill_(0.0)
                self.bias.fill_(0.0)
                
            elif self.init_method == "random_normal":
                # Small random normal initialization
                self.weights.normal_(mean=0.0, std=0.1)
                self.bias.normal_(mean=0.0, std=0.1)
                
            elif self.init_method == "random_uniform":
                # Small random uniform initialization
                self.weights.uniform_(-0.1, 0.1)
                self.bias.uniform_(-0.1, 0.1)
                
            else:
                raise ModelError(f"Unknown initialization method: {self.init_method}")
        
        self.logger.debug(f"Initialized weights with method: {self.init_method}")
    
    def _activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function to the input."""
        if self.activation == "step":
            # Step function: 0 if x < 0, 1 if x >= 0
            return (x >= 0).float()
        elif self.activation == "sign":
            # Sign function: -1 if x < 0, 1 if x >= 0
            return torch.sign(x)
        else:
            raise ModelError(f"Unknown activation function: {self.activation}")
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the perceptron.
        
        Args:
            X: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output tensor of shape (batch_size,)
        """
        if X.dim() != 2:
            raise ModelError(f"Expected 2D input, got {X.dim()}D")
        
        if X.shape[1] != self.n_features:
            raise ModelError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Linear combination: y = X @ w + b
        linear_output = torch.matmul(X, self.weights) + self.bias
        
        # Apply activation function
        output = self._activation_function(linear_output)
        
        return output.squeeze()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            X: Input tensor
            
        Returns:
            Predicted labels
        """
        self.eval()
        with torch.no_grad():
            return self.forward(X)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """
        Train the perceptron using the perceptron learning rule.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            Dictionary with training history
            
        Raises:
            ModelError: If input validation fails
            TrainingError: If training fails
        """
        try:
            self._validate_training_data(X, y)
            
            # Convert to appropriate format for perceptron
            if self.activation == "step":
                # For step function, labels should be 0/1
                y_train = self._convert_labels_to_binary(y)
            else:  # sign function
                # For sign function, labels should be -1/+1
                y_train = self._convert_labels_to_signed(y)
            
            self.logger.info(f"Starting Perceptron training: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Reset training state
            self.training_history = {
                'epochs': [],
                'errors': [],
                'weights': [],
                'bias': [],
                'accuracy': [],
                'converged': False
            }
            
            self.train()
            
            # Training loop
            for epoch in range(self.max_epochs):
                epoch_errors = 0
                
                # Store current weights for history
                current_weights = self.weights.data.clone()
                current_bias = self.bias.data.clone()
                
                # Iterate through all samples
                for i in range(X.shape[0]):
                    x_i = X[i:i+1]  # Keep batch dimension
                    y_i = y_train[i]
                    
                    # Make prediction
                    prediction = self.forward(x_i).item()
                    
                    # Check for error
                    if prediction != y_i:
                        epoch_errors += 1
                        
                        # Perceptron learning rule: w = w + lr * (y_true - y_pred) * x
                        error = y_i - prediction
                        
                        with torch.no_grad():
                            self.weights += self.learning_rate * error * X[i]
                            self.bias += self.learning_rate * error
                        
                        if hasattr(self, 'debug_weight_updates') and self.debug_weight_updates:
                            self.logger.debug(f"Weight update at sample {i}: error={error:.3f}")
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = self.forward(X)
                    if self.activation == "step":
                        accuracy = (predictions == y_train).float().mean().item()
                    else:  # sign function
                        accuracy = (torch.sign(predictions) == y_train).float().mean().item()
                
                # Store history
                self.training_history['epochs'].append(epoch + 1)
                self.training_history['errors'].append(epoch_errors)
                self.training_history['weights'].append(current_weights.numpy())
                self.training_history['bias'].append(current_bias.numpy())
                self.training_history['accuracy'].append(accuracy)
                
                # Check for convergence
                if epoch_errors == 0:
                    self.training_history['converged'] = True
                    self.logger.info(f"Converged after {epoch + 1} epochs with perfect accuracy")
                    break
                
                # Check for minimal improvement (alternative convergence criterion)
                if epoch > 0:
                    error_change = abs(self.training_history['errors'][-1] - 
                                     self.training_history['errors'][-2])
                    if error_change < self.tolerance:
                        self.logger.info(f"Converged after {epoch + 1} epochs (minimal error change)")
                        break
                
                if epoch % 10 == 0 or epoch < 10:
                    self.logger.debug(f"Epoch {epoch + 1}: {epoch_errors} errors, "
                                    f"accuracy={accuracy:.4f}")
            
            self.n_epochs_trained = len(self.training_history['epochs'])
            self.is_fitted = True
            
            if not self.training_history['converged']:
                self.logger.warning(f"Did not converge after {self.max_epochs} epochs. "
                                  f"Final accuracy: {accuracy:.4f}")
            
            self.logger.info(f"Training completed: {self.n_epochs_trained} epochs, "
                           f"final accuracy: {accuracy:.4f}")
            
            return self.training_history
            
        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e
    
    def _validate_training_data(self, X: torch.Tensor, y: torch.Tensor):
        """Validate training data format and content."""
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ModelError("X and y must be torch tensors")
        
        if X.dim() != 2:
            raise ModelError(f"X must be 2D, got {X.dim()}D")
        
        if y.dim() != 1:
            raise ModelError(f"y must be 1D, got {y.dim()}D")
        
        if X.shape[0] != y.shape[0]:
            raise ModelError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        if X.shape[1] != self.n_features:
            raise ModelError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Check for valid labels
        unique_labels = torch.unique(y)
        if len(unique_labels) != 2:
            raise ModelError(f"Perceptron requires exactly 2 classes, got {len(unique_labels)}")
    
    def _convert_labels_to_binary(self, y: torch.Tensor) -> torch.Tensor:
        """Convert labels to 0/1 format for step activation."""
        unique_labels = torch.unique(y)
        if len(unique_labels) != 2:
            raise ModelError("Binary classification requires exactly 2 unique labels")
        
        # Map to 0/1
        y_binary = torch.zeros_like(y, dtype=torch.float32)
        y_binary[y == unique_labels[1]] = 1.0
        
        return y_binary
    
    def _convert_labels_to_signed(self, y: torch.Tensor) -> torch.Tensor:
        """Convert labels to -1/+1 format for sign activation."""
        unique_labels = torch.unique(y)
        if len(unique_labels) != 2:
            raise ModelError("Binary classification requires exactly 2 unique labels")
        
        # Map to -1/+1
        y_signed = torch.full_like(y, -1.0, dtype=torch.float32)
        y_signed[y == unique_labels[1]] = 1.0
        
        return y_signed
    
    def get_decision_boundary_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get parameters for plotting the decision boundary.
        
        Returns:
            Tuple of (weights, bias) for the decision boundary
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before getting decision boundary")
        
        return self.weights.data.clone(), self.bias.data.clone()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        info = {
            'model_name': 'Perceptron',
            'n_features': self.n_features,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'tolerance': self.tolerance,
            'activation': self.activation,
            'init_method': self.init_method,
            'is_fitted': self.is_fitted,
            'n_epochs_trained': self.n_epochs_trained,
            'converged': self.training_history.get('converged', False),
            'n_parameters': self.n_features + 1  # weights + bias
        }
        
        if self.is_fitted:
            info.update({
                'final_accuracy': self.training_history['accuracy'][-1] if self.training_history['accuracy'] else 0.0,
                'final_errors': self.training_history['errors'][-1] if self.training_history['errors'] else 0,
                'weights': self.weights.data.numpy().tolist(),
                'bias': self.bias.data.item()
            })
        
        return info
    
    def save_model(self, filepath: str):
        """Save model state to file."""
        if not self.is_fitted:
            self.logger.warning("Saving unfitted model")
        
        state = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'training_history': self.training_history
        }
        
        torch.save(state, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state from file."""
        state = torch.load(filepath, map_location='cpu')
        
        self.load_state_dict(state['model_state_dict'])
        self.training_history = state['training_history']
        self.is_fitted = state['model_info']['is_fitted']
        self.n_epochs_trained = state['model_info']['n_epochs_trained']
        
        self.logger.info(f"Model loaded from {filepath}")


def create_perceptron(config: Dict[str, Any]) -> Perceptron:
    """
    Factory function to create a Perceptron from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Perceptron instance
        
    Example:
        config = get_config("iris_binary")
        model = create_perceptron(config)
    """
    # Extract relevant parameters from config
    perceptron_params = {
        'learning_rate': config.get('learning_rate', DEFAULT_LEARNING_RATE),
        'max_epochs': config.get('max_epochs', DEFAULT_MAX_EPOCHS),
        'tolerance': config.get('tolerance', DEFAULT_TOLERANCE),
        'activation': config.get('activation', DEFAULT_ACTIVATION),
        'init_method': config.get('init_method', DEFAULT_INIT_METHOD),
        'random_state': config.get('seed', None)
    }
    
    # Note: n_features will be set when we see the data
    # This is a partial factory - complete initialization happens during training
    logger = get_logger(__name__)
    logger.info(f"Created Perceptron factory with config: {perceptron_params}")
    
    return perceptron_params


if __name__ == "__main__":
    # Test the Perceptron implementation
    print("Testing Perceptron implementation...")
    
    # Create simple test data
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])  # XOR pattern (should fail)
    
    # Test perceptron creation
    perceptron = Perceptron(n_features=2, learning_rate=0.1, max_epochs=10)
    print(f"✓ Created Perceptron: {perceptron.n_features} features")
    
    # Test training (should not converge for XOR)
    try:
        history = perceptron.fit(X, y)
        print(f"✓ Training completed: {len(history['epochs'])} epochs")
        print(f"  Converged: {history['converged']}")
        print(f"  Final accuracy: {history['accuracy'][-1]:.3f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
    
    # Test simple linearly separable data
    X_simple = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_simple = torch.tensor([0, 1, 0, 1])  # Simple separable pattern
    
    perceptron_simple = Perceptron(n_features=2, learning_rate=0.1, max_epochs=50)
    history_simple = perceptron_simple.fit(X_simple, y_simple)
    print(f"✓ Simple data training: converged={history_simple['converged']}")
    
    print("Perceptron tests completed!") 