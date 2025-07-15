"""
Template for model.py - Neural Network Model Implementation

This template provides the basic structure for implementing neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and clarity.

Replace [MODEL_NAME] with the actual model name (e.g., "Perceptron", "MLP", etc.)
Replace [DESCRIPTION] with a brief description of what the model does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

# Import shared utilities
from utils import setup_logging
from constants import MODEL_NAME

# Set up logging
logger = setup_logging(__name__)


class [MODEL_NAME](nn.Module):
    """
    [DESCRIPTION]
    
    This implementation follows the original [MODEL_NAME] architecture as described
    in [ORIGINAL_PAPER_REFERENCE]. Key innovations include:
    - [KEY_INNOVATION_1]
    - [KEY_INNOVATION_2]
    - [KEY_INNOVATION_3]
    
    Historical Context:
    - Introduced in [YEAR] by [AUTHOR(S)]
    - Solved the problem of [PROBLEM_SOLVED]
    - Improved upon [PREVIOUS_LIMITATIONS]
    
    Args:
        input_size (int): Dimension of input features
        hidden_size (int): Number of hidden units (if applicable)
        output_size (int): Number of output classes/units
        [OTHER_ARCHITECTURE_PARAMS]: [DESCRIPTION]
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = None, 
        output_size: int = 1,
        **kwargs
    ):
        super([MODEL_NAME], self).__init__()
        
        # Store architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize model architecture
        self._build_model()
        
        # Initialize weights (if custom initialization needed)
        self._initialize_weights()
        
        logger.info(f"Initialized {MODEL_NAME} with input_size={input_size}, "
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
            'architecture': str(self)
        }
    
    def save_checkpoint(self, filepath: str, epoch: int = None, optimizer_state: dict = None):
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
            [MODEL_NAME]: Loaded model instance
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
        
        logger.info(f"Loaded checkpoint from {filepath}")
        return model


# Factory function for easy model creation
def create_model(config: dict) -> [MODEL_NAME]:
    """
    Factory function to create model from configuration.
    
    Args:
        config (dict): Configuration dictionary containing model parameters
        
    Returns:
        [MODEL_NAME]: Configured model instance
    """
    return [MODEL_NAME](
        input_size=config.get('input_size', 2),
        hidden_size=config.get('hidden_size', None),
        output_size=config.get('output_size', 1),
        # Add other model-specific parameters as needed
    )


if __name__ == "__main__":
    # Quick test of model instantiation
    print(f"Testing {MODEL_NAME} model...")
    
    # Create a simple test configuration
    test_config = {
        'input_size': 4,
        'hidden_size': 8,
        'output_size': 3
    }
    
    # Create model
    model = create_model(test_config)
    
    # Print model info
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, test_config['input_size'])
    try:
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except NotImplementedError:
        print("Forward pass not implemented yet - this is expected for template") 