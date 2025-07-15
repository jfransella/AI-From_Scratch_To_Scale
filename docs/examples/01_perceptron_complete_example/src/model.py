"""
Perceptron model implementation.

This file contains the complete implementation of the Perceptron model,
following the project's established patterns and best practices.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

from constants import (
    MODEL_NAME,
    ACTIVATION_FUNCTIONS,
    ERROR_MESSAGES,
    MODEL_CAPABILITIES,
    MODEL_LIMITATIONS
)

logger = logging.getLogger(__name__)


class Perceptron(nn.Module):
    """
    Perceptron model implementation.
    
    The Perceptron is a single-layer neural network that learns to classify
    linearly separable data using the Perceptron Learning Rule.
    
    Historical Context:
    - Introduced by Frank Rosenblatt in 1957
    - First artificial neural network with a learning algorithm
    - Inspired by biological neurons
    - Foundation for modern neural networks
    
    Key Innovation:
    - Learning algorithm that adjusts weights based on errors
    - Guaranteed convergence on linearly separable data
    - Binary classification capability
    
    Architecture:
    - Single layer of weights
    - Step activation function
    - Bias term (optional)
    
    Attributes:
        config: Configuration object containing model parameters
        weights: Learnable weight parameters
        bias: Learnable bias parameter (if enabled)
        activation_fn: Activation function (step, sign, or linear)
        training_history: Records of training progress
    """
    
    def __init__(self, config):
        """
        Initialize the Perceptron model.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model parameters
        self.weights = nn.Parameter(torch.randn(config.input_size, config.output_size))
        
        # Initialize bias if enabled
        if getattr(config, 'bias', True):
            self.bias = nn.Parameter(torch.zeros(config.output_size))
        else:
            self.bias = None
        
        # Set activation function
        self.activation_fn = self._get_activation_function(
            getattr(config, 'activation', 'step')
        )
        
        # Initialize training history
        self.training_history = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'weight_norms': [],
            'converged': False
        }
        
        # Initialize weights properly
        self._initialize_weights()
        
        logger.info(f"Initialized {MODEL_NAME} with {self.parameter_count()} parameters")
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        required_attrs = ['input_size', 'output_size', 'learning_rate']
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"Missing required configuration: {attr}")
        
        if self.config.input_size <= 0:
            raise ValueError(ERROR_MESSAGES['invalid_input_size'])
        
        if self.config.output_size <= 0:
            raise ValueError(ERROR_MESSAGES['invalid_output_size'])
        
        if self.config.learning_rate <= 0:
            raise ValueError(ERROR_MESSAGES['invalid_learning_rate'])
    
    def _get_activation_function(self, activation: str):
        """Get the activation function by name."""
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(ERROR_MESSAGES['invalid_activation'])
        
        if activation == 'step':
            return self._step_activation
        elif activation == 'sign':
            return self._sign_activation
        elif activation == 'linear':
            return self._linear_activation
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def _step_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Step activation function (classic Perceptron)."""
        return (x >= 0).float()
    
    def _sign_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Sign activation function (-1 or 1)."""
        return torch.sign(x)
    
    def _linear_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Linear activation function (for comparison)."""
        return x
    
    def _initialize_weights(self):
        """Initialize weights using appropriate method."""
        # Small random weights for Perceptron
        nn.init.normal_(self.weights, mean=0.0, std=0.1)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Perceptron.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Validate input shape
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        if x.size(1) != self.config.input_size:
            raise ValueError(f"Expected input size {self.config.input_size}, got {x.size(1)}")
        
        # Linear transformation
        output = torch.matmul(x, self.weights)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Apply activation function
        output = self.activation_fn(output)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions as tensor
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step using Perceptron Learning Rule.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            y: Target tensor of shape (batch_size, output_size)
            
        Returns:
            Dictionary containing training metrics
        """
        self.train()
        
        # Forward pass
        predictions = self.forward(x)
        
        # Calculate error
        error = y - predictions
        
        # Calculate accuracy
        accuracy = (predictions == y).float().mean().item()
        
        # Calculate loss (number of misclassifications)
        loss = (predictions != y).float().sum().item()
        
        # Perceptron Learning Rule: update weights only on misclassified examples
        if loss > 0:
            # Update weights: w = w + lr * (y - y_pred) * x
            weight_update = self.config.learning_rate * torch.matmul(x.t(), error)
            self.weights.data += weight_update
            
            # Update bias if present
            if self.bias is not None:
                bias_update = self.config.learning_rate * error.sum(dim=0)
                self.bias.data += bias_update
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'error_count': int(loss),
            'weight_norm': self.weights.norm().item()
        }
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            
            # Calculate metrics
            accuracy = (predictions == y).float().mean().item()
            error_count = (predictions != y).float().sum().item()
            
            # Additional metrics for binary classification
            if self.config.output_size == 1:
                # Convert to binary for metrics calculation
                y_binary = (y > 0.5).float()
                pred_binary = (predictions > 0.5).float()
                
                # True positives, false positives, etc.
                tp = ((pred_binary == 1) & (y_binary == 1)).float().sum().item()
                tn = ((pred_binary == 0) & (y_binary == 0)).float().sum().item()
                fp = ((pred_binary == 1) & (y_binary == 0)).float().sum().item()
                fn = ((pred_binary == 0) & (y_binary == 1)).float().sum().item()
                
                # Precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return {
                    'accuracy': accuracy,
                    'error_count': error_count,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'tn': tn,
                    'fp': fp,
                    'fn': fn
                }
            else:
                return {
                    'accuracy': accuracy,
                    'error_count': error_count
                }
    
    def fit(self, x: torch.Tensor, y: torch.Tensor, 
            max_epochs: Optional[int] = None,
            verbose: bool = False) -> Dict[str, Any]:
        """
        Train the Perceptron using the Perceptron Learning Rule.
        
        Args:
            x: Training input tensor
            y: Training target tensor
            max_epochs: Maximum number of epochs (from config if None)
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        max_epochs = max_epochs or self.config.epochs
        
        if verbose:
            print(f"Training {MODEL_NAME} for up to {max_epochs} epochs")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"Data shape: {x.shape}")
        
        converged = False
        
        for epoch in range(max_epochs):
            # Training step
            metrics = self.train_step(x, y)
            
            # Record history
            self.training_history['epochs'].append(epoch)
            self.training_history['losses'].append(metrics['loss'])
            self.training_history['accuracies'].append(metrics['accuracy'])
            self.training_history['weight_norms'].append(metrics['weight_norm'])
            
            # Check for convergence (no errors)
            if metrics['error_count'] == 0:
                converged = True
                if verbose:
                    print(f"Converged at epoch {epoch}!")
                break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Loss={metrics['loss']:.2f}, "
                      f"Accuracy={metrics['accuracy']:.3f}")
        
        self.training_history['converged'] = converged
        
        if not converged and verbose:
            print(f"Did not converge after {max_epochs} epochs")
            print("This may indicate the data is not linearly separable")
        
        return self.training_history
    
    def get_decision_boundary(self, x_range: Tuple[float, float] = (-2, 2),
                             y_range: Tuple[float, float] = (-2, 2),
                             resolution: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get decision boundary for 2D visualization.
        
        Args:
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution
            
        Returns:
            Tuple of (X, Y, Z) for contour plotting
        """
        if self.config.input_size != 2:
            raise ValueError("Decision boundary only available for 2D input")
        
        # Create meshgrid
        x = np.arange(x_range[0], x_range[1], resolution)
        y = np.arange(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Flatten for prediction
        grid_points = np.c_[X.ravel(), Y.ravel()]
        grid_tensor = torch.FloatTensor(grid_points)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.predict(grid_tensor)
            Z = predictions.numpy().reshape(X.shape)
        
        return X, Y, Z
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get current weights and bias as numpy arrays."""
        result = {
            'weights': self.weights.detach().numpy().copy()
        }
        
        if self.bias is not None:
            result['bias'] = self.bias.detach().numpy().copy()
        
        return result
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights and bias from numpy arrays."""
        if 'weights' in weights:
            self.weights.data = torch.FloatTensor(weights['weights'])
        
        if 'bias' in weights and self.bias is not None:
            self.bias.data = torch.FloatTensor(weights['bias'])
    
    def parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': MODEL_NAME,
            'input_size': self.config.input_size,
            'output_size': self.config.output_size,
            'parameter_count': self.parameter_count(),
            'activation': getattr(self.config, 'activation', 'step'),
            'has_bias': self.bias is not None,
            'capabilities': MODEL_CAPABILITIES,
            'limitations': MODEL_LIMITATIONS,
            'training_history': self.training_history
        }
    
    def __repr__(self):
        """String representation of the model."""
        return (f"Perceptron(input_size={self.config.input_size}, "
                f"output_size={self.config.output_size}, "
                f"activation={getattr(self.config, 'activation', 'step')}, "
                f"bias={self.bias is not None})")


# Factory function for creating Perceptron models
def create_perceptron(config) -> Perceptron:
    """
    Factory function to create a Perceptron model.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized Perceptron model
    """
    return Perceptron(config)


# Utility functions for working with Perceptron models
def is_linearly_separable(x: torch.Tensor, y: torch.Tensor,
                         max_epochs: int = 1000,
                         learning_rate: float = 0.1) -> bool:
    """
    Test if data is linearly separable using a Perceptron.
    
    Args:
        x: Input data
        y: Target data
        max_epochs: Maximum epochs to test
        learning_rate: Learning rate for test
        
    Returns:
        True if data is linearly separable, False otherwise
    """
    # Create a simple config for testing
    class TestConfig:
        def __init__(self):
            self.input_size = x.size(1)
            self.output_size = y.size(1)
            self.learning_rate = learning_rate
            self.epochs = max_epochs
            self.activation = 'step'
    
    # Create and train test model
    test_model = Perceptron(TestConfig())
    history = test_model.fit(x, y, max_epochs=max_epochs, verbose=False)
    
    return history['converged']


if __name__ == "__main__":
    # Example usage and testing
    print("Perceptron Model Implementation")
    print("=" * 40)
    
    # Create a simple test configuration
    class TestConfig:
        def __init__(self):
            self.input_size = 2
            self.output_size = 1
            self.learning_rate = 0.1
            self.epochs = 100
            self.activation = 'step'
    
    # Test with AND gate data
    config = TestConfig()
    model = Perceptron(config)
    
    # Create AND gate data
    x_and = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = torch.FloatTensor([[0], [0], [0], [1]])
    
    print(f"Model: {model}")
    print(f"Parameters: {model.parameter_count()}")
    
    # Test forward pass
    output = model.forward(x_and)
    print(f"Initial output: {output.flatten()}")
    
    # Train model
    print("\nTraining on AND gate:")
    history = model.fit(x_and, y_and, verbose=True)
    
    # Test final output
    final_output = model.predict(x_and)
    print(f"Final output: {final_output.flatten()}")
    
    # Test linear separability
    print(f"\nAND gate linearly separable: {is_linearly_separable(x_and, y_and)}")
    
    # Test with XOR gate (should not be linearly separable)
    x_xor = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = torch.FloatTensor([[0], [1], [1], [0]])
    
    print(f"XOR gate linearly separable: {is_linearly_separable(x_xor, y_xor)}")
    
    print("\nModel implementation test completed!") 