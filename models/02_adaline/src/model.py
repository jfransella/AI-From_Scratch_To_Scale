"""
ADALINE Model Implementation.

Implementation of the Adaptive Linear Neuron (ADALINE) using the Delta Rule
learning algorithm. Follows NumPy-based implementation for educational clarity.
"""

import numpy as np
from typing import Dict, Any
import logging

try:
    from .constants import YEAR_INTRODUCED, AUTHORS
    from .config import ADALINEConfig
except ImportError:
    # For direct execution
    from constants import YEAR_INTRODUCED, AUTHORS
    from config import ADALINEConfig

logger = logging.getLogger(__name__)


class ADALINE:
    """
    ADALINE (Adaptive Linear Neuron) implementation using NumPy.
    
    Key Features:
    - Linear activation (no step function)
    - Delta Rule learning algorithm
    - Continuous error-based updates
    - Mean squared error loss
    - Pure NumPy implementation for educational clarity
    
    Historical Context:
    - Introduced in 1960 by Bernard Widrow and Ted Hoff
    - First neural network with continuous activation
    - Foundation for modern gradient descent methods
    
    Args:
        config: ADALINEConfig object with model parameters
    """
    
    def __init__(self, config: ADALINEConfig):
        self.config = config
        
        # Initialize weights and bias
        self.weights = None
        self.bias = None
        self._initialize_weights()
        
        # Training state
        self.is_fitted = False
        self.training_history = {
            "loss": [],
            "mse": [],
            "epochs_trained": 0
        }
        
        logger.info(f"Initialized ADALINE with input_size={config.input_size}")
    
    def _initialize_weights(self):
        """Initialize weights with small random values."""
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.1, (self.config.input_size,))
        self.bias = 0.0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (linear output).
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Linear output (no activation applied)
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Linear transformation: y = w^T * x + b
        return np.dot(x, self.weights) + self.bias
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions (binary classification).
        
        Args:
            x: Input array
            
        Returns:
            Binary predictions (0 or 1)
        """
        linear_output = self.forward(x)
        # Convert to binary: > 0.5 -> 1, <= 0.5 -> 0
        return (linear_output > 0.5).astype(float)
    
    def fit(self, x_data: np.ndarray, y_target: np.ndarray) -> Dict[str, Any]:
        """
        Train using Delta Rule algorithm.
        
        Args:
            x_data: Input features
            y_target: Target values (continuous)
            
        Returns:
            Training results dictionary
        """
        # Ensure data is in correct format
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        if y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)
        
        # Delta Rule training loop
        for epoch in range(self.config.epochs):
            total_error = 0
            
            # Process each sample individually (Delta Rule)
            for i in range(len(x_data)):
                # Forward pass for single sample
                x_i = x_data[i:i+1]  # Keep 2D shape
                y_i = y_target[i:i+1]
                
                # Forward pass
                linear_output = self.forward(x_i)
                
                # Compute error for this sample
                error = y_i.flatten()[0] - linear_output[0]
                total_error += error ** 2
                
                # Delta Rule weight update for this sample
                # w += η * error * x
                self.weights += self.config.learning_rate * error * x_i.flatten()
                self.bias += self.config.learning_rate * error
            
            # Calculate MSE for this epoch
            mse = total_error / len(x_data)
            
            # Record training history
            self.training_history["loss"].append(mse)
            self.training_history["mse"].append(mse)
            
            # Log progress
            if epoch % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch}: MSE = {mse:.6f}")
            
            # Check convergence
            if mse < self.config.tolerance:
                logger.info(f"Converged at epoch {epoch}")
                break
        
        self.training_history["epochs_trained"] = epoch + 1
        self.is_fitted = True
        
        return {
            "converged": mse < self.config.tolerance,
            "final_mse": mse,
            "epochs_trained": epoch + 1
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata following wandb integration plan standards."""
        return {
            # Core identification
            "name": "ADALINE",
            "full_name": "Adaptive Linear Neuron",
            "category": "foundation",
            "module": 1,
            "pattern": "simple",
            
            # Historical context
            "year_introduced": YEAR_INTRODUCED,
            "authors": AUTHORS,
            "paper_title": "Adaptive switching circuits",
            "key_innovations": [
                "First neural network with continuous activation",
                "Delta Rule (LMS) learning algorithm",
                "Continuous error-based weight updates",
                "Foundation for modern gradient descent"
            ],
            
            # Architecture details
            "architecture_type": "single-layer",
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "parameter_count": len(self.weights) + 1,  # weights + bias
            "activation_function": "linear",
            
            # Training characteristics
            "learning_algorithm": "delta-rule",
            "loss_function": "mse",
            "optimizer": "custom-delta-rule",
            
            # Implementation details
            "framework": "numpy",
            "precision": "float32",
            "device_support": ["cpu"],
            
            # Educational metadata
            "difficulty_level": "beginner",
            "estimated_training_time": "seconds",
            "key_learning_objectives": [
                "Understand Delta Rule vs Perceptron Rule",
                "Learn continuous vs discrete learning",
                "Foundation of gradient descent",
                "Continuous activation functions"
            ],
            
            # Training state
            "is_fitted": self.is_fitted,
            "training_config": self.config.__dict__
        }


def create_adaline(config: ADALINEConfig) -> ADALINE:
    """Factory function to create ADALINE model."""
    return ADALINE(config) 