"""
ADALINE Model Implementation.

Implementation of the Adaptive Linear Neuron (ADALINE) using the Delta Rule
learning algorithm. Follows the Simple implementation pattern.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

from constants import MODEL_NAME, YEAR_INTRODUCED, AUTHORS
from config import ADALINEConfig

logger = logging.getLogger(__name__)


class ADALINE(nn.Module):
    """
    ADALINE (Adaptive Linear Neuron) implementation.
    
    Key Features:
    - Linear activation (no step function)
    - Delta Rule learning algorithm
    - Continuous error-based updates
    - Mean squared error loss
    
    Historical Context:
    - Introduced in 1960 by Bernard Widrow and Ted Hoff
    - First neural network with continuous activation
    - Foundation for modern gradient descent methods
    
    Args:
        config: ADALINEConfig object with model parameters
    """
    
    def __init__(self, config: ADALINEConfig):
        super().__init__()
        self.config = config
        
        # Linear layer (no activation function)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=True)
        
        # Initialize weights with small random values
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
        with torch.no_grad():
            self.linear.weight.normal_(0, 0.1)
            self.linear.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (linear output).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Linear output (no activation applied)
        """
        return self.linear(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (binary classification via sign).
        
        Args:
            x: Input tensor
            
        Returns:
            Binary predictions (0 or 1)
        """
        with torch.no_grad():
            linear_output = self.forward(x)
            # Convert to binary: positive -> 1, negative -> 0
            return (linear_output > 0).float()
    
    def fit(self, x_data: torch.Tensor, y_target: torch.Tensor) -> Dict[str, Any]:
        """
        Train using Delta Rule algorithm.
        
        Args:
            x_data: Input features
            y_target: Target values (continuous)
            
        Returns:
            Training results dictionary
        """
        self.train()
        
        # Delta Rule training loop
        for epoch in range(self.config.epochs):
            # Forward pass
            linear_output = self.forward(x_data)
            
            # Compute error (Delta Rule)
            error = y_target - linear_output
            mse = torch.mean(error ** 2)
            
            # Delta Rule weight update
            with torch.no_grad():
                # Weight update: w = w + η * error * input
                # For each sample: w += η * error[i] * x[i]
                for i in range(len(x_data)):
                    error_i = error[i]
                    x_i = x_data[i]
                    
                    # Update weights: w += η * error * x
                    self.linear.weight += self.config.learning_rate * error_i * x_i.unsqueeze(0)
                    self.linear.bias += self.config.learning_rate * error_i
            
            # Record training history
            self.training_history["loss"].append(mse.item())
            self.training_history["mse"].append(mse.item())
            
            # Log progress
            if epoch % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch}: MSE = {mse.item():.6f}")
            
            # Check convergence
            if mse.item() < self.config.tolerance:
                logger.info(f"Converged at epoch {epoch}")
                break
        
        self.training_history["epochs_trained"] = epoch + 1
        self.is_fitted = True
        
        return {
            "converged": mse.item() < self.config.tolerance,
            "final_mse": mse.item(),
            "epochs_trained": epoch + 1
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            "model_name": MODEL_NAME,
            "year_introduced": YEAR_INTRODUCED,
            "authors": AUTHORS,
            "architecture": "Single linear layer",
            "activation": "Linear (none)",
            "learning_rule": "Delta Rule (LMS)",
            "parameters": sum(p.numel() for p in self.parameters()),
            "is_fitted": self.is_fitted,
            "config": self.config
        }


def create_adaline(config: ADALINEConfig) -> ADALINE:
    """Factory function to create ADALINE model."""
    return ADALINE(config) 