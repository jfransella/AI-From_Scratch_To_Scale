# pylint: skip-file
# flake8: noqa
# type: ignore
"""
Template for model.py - Neural Network Model Implementation

This template provides the basic structure for implementing neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and clarity.

Replace MODEL_NAME with the actual model name (e.g., "Perceptron", "MLP", etc.)
Replace DESCRIPTION with a brief description of what the model does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# Import shared utilities
from utils import setup_logging, get_logger, set_random_seed
from constants import MODEL_NAME

# Optional BaseModel import (for advanced implementations)
try:
    from engine.base import BaseModel
    HAS_BASE_MODEL = True
except ImportError:
    BaseModel = object
    HAS_BASE_MODEL = False

# Set up logging
logger = setup_logging(__name__)


class ModelTemplate(nn.Module):
    """
    Template model implementation.
    
    This implementation follows the original model architecture. Key innovations include:
    - Innovation 1
    - Innovation 2
    - Innovation 3
    
    Historical Context:
    - Introduced in YEAR by AUTHOR(S)
    - Solved the problem of PROBLEM_SOLVED
    - Improved upon PREVIOUS_LIMITATIONS
    
    Args:
        input_size (int): Dimension of input features
        hidden_size (int): Number of hidden units (if applicable)
        output_size (int): Number of output classes/units
        **kwargs: Additional model-specific parameters
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: Optional[int] = None, 
        output_size: int = 1,
        **kwargs
    ):
        super(ModelTemplate, self).__init__()
        
        # Store architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Store additional parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Set random seed if provided
        if hasattr(self, 'random_state') and self.random_state is not None:
            set_random_seed(self.random_state)
        
        # Initialize model architecture
        self._build_model()
        
        # Initialize weights (if custom initialization needed)
        self._initialize_weights()
        
        # Initialize training state
        self.is_fitted = False
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "epochs_trained": 0
        }
        
        logger.info(f"Initialized ModelTemplate with input_size={input_size}, "
                   f"hidden_size={hidden_size}, output_size={output_size}")
    
    def _build_model(self):
        """
        Build the model architecture.
        
        This method should define all the layers and components of the model.
        Keep this separate from __init__ for clarity.
        """
        # TODO: Define model layers here
        # Example:
        # self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        # self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        
        raise NotImplementedError("Implement model architecture in _build_model()")
    
    def _initialize_weights(self):
        """
        Initialize model weights.
        
        Use this method if the model requires custom weight initialization
        beyond PyTorch defaults. For historical accuracy, match the original
        paper's initialization scheme when possible.
        """
        # TODO: Implement custom weight initialization if needed
        # Example:
        # for layer in self.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass
        # Example:
        # x = F.relu(self.layer1(x))
        # x = self.layer2(x)
        # return x
        
        raise NotImplementedError("Implement forward pass in forward()")
    
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
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': MODEL_NAME,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': str(self),
            'is_fitted': self.is_fitted,
            'epochs_trained': self.training_history.get('epochs_trained', 0)
        }
    
    def save_checkpoint(self, filepath: str, epoch: Optional[int] = None, optimizer_state: Optional[Dict[str, Any]] = None):
        """
        Save model checkpoint.
        
        Args:
            filepath (str): Path to save checkpoint
            epoch (int, optional): Current epoch number
            optimizer_state (dict, optional): Optimizer state dict
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'epoch': epoch,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @classmethod
    def load_from_checkpoint(cls, filepath: str, **model_kwargs):
        """
        Load model from checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            ModelTemplate: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Extract model parameters from checkpoint
        model_info = checkpoint.get('model_info', {})
        input_size = model_info.get('input_size', model_kwargs.get('input_size'))
        hidden_size = model_info.get('hidden_size', model_kwargs.get('hidden_size'))
        output_size = model_info.get('output_size', model_kwargs.get('output_size'))
        
        # Create model instance
        model = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            **model_kwargs
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        model.is_fitted = True
        model.training_history['epochs_trained'] = checkpoint.get('epoch', 0)
        
        logger.info(f"Loaded model from {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(input_size={self.input_size}, " \
               f"hidden_size={self.hidden_size}, output_size={self.output_size})"


def create_model(config: dict) -> ModelTemplate:
    """
    Create model instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ModelTemplate: Model instance
    """
    return ModelTemplate(
        input_size=config['input_size'],
        hidden_size=config.get('hidden_size', None),
        output_size=config['output_size'],
        **{k: v for k, v in config.items() 
           if k not in ['input_size', 'hidden_size', 'output_size']}
    )


# =============================================================================
# ADVANCED MODEL IMPLEMENTATION (Engine Integration)
# =============================================================================

if HAS_BASE_MODEL:
    class ModelTemplateAdvanced(ModelTemplate, BaseModel):
        """
        Advanced model implementation with engine integration.
        
        This class provides the BaseModel interface required by the engine framework
        while maintaining all the functionality of the basic model.
        """
        
        def __init__(self, **kwargs):
            # Extract BaseModel parameters
            base_model_params = {}
            model_params = {}
            
            for key, value in kwargs.items():
                if key in ['x_data', 'y_target']:
                    base_model_params[key] = value
                else:
                    model_params[key] = value
            
            # Initialize the base model
            super().__init__(**model_params)
            
            # Store BaseModel data if provided
            if base_model_params:
                self.x_data = base_model_params.get('x_data')
                self.y_target = base_model_params.get('y_target')
        
        def fit(self, x_data: torch.Tensor, y_target: torch.Tensor) -> Dict[str, Any]:
            """
            Fit the model to the data using the engine framework.
            
            Args:
                x_data: Input features
                y_target: Target labels
                
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
                "final_accuracy": 0.0
            }
        
        def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Compute loss for engine framework.
            
            Args:
                outputs: Model outputs
                targets: Target labels
                
            Returns:
                Loss tensor
            """
            return super().get_loss(outputs, targets) 