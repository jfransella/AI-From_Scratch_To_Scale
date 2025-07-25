"""
Template for model.py - Neural Network Model Implementation

This template provides the basic structure for implementing neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and clarity.

Replace MODEL_NAME with the actual model name (e.g., "Perceptron", "MLP", etc.)
Replace DESCRIPTION with a brief description of what the model does.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Handle torch imports gracefully with fallback support
try:
    import torch

    if (
        hasattr(torch, "__version__")
        and hasattr(torch, "nn")
        and hasattr(torch, "tensor")
    ):
        import torch.nn.functional as F
        from torch import nn

        _TORCH_AVAILABLE = True
        BaseNNModule = nn.Module
        TorchTensor = torch.Tensor
    else:
        # torch exists but is broken
        _TORCH_AVAILABLE = False
        torch = None
        nn = None
        F = None

        # Create dummy base classes for compatibility
        class BaseNNModule:
            def __init__(self):
                pass

            def parameters(self):
                return []

            def to(self, device):
                return self

            def train(self, mode=True):
                return self

            def eval(self):
                return self

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        TorchTensor = Any

except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

    # Create dummy base classes for compatibility
    class BaseNNModule:
        def __init__(self):
            pass

        def parameters(self):
            return []

        def to(self, device):
            return self

        def train(self, mode=True):
            return self

        def eval(self):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    TorchTensor = Any

# Import model-specific components
from constants import AUTHORS, MODEL_NAME, MODEL_VERSION, YEAR_INTRODUCED
from engine.base import BaseModel

# Import shared utilities
from utils import get_logger, set_random_seed


class ModelTemplate(BaseNNModule, BaseModel):
    """
    Template model implementation with BaseModel interface.
    
    This implementation follows the original model architecture. Key innovations include:
    - Innovation 1: [Replace with actual innovation]
    - Innovation 2: [Replace with actual innovation]  
    - Innovation 3: [Replace with actual innovation]
    
    Historical Context:
    - Introduced in [YEAR] by [AUTHOR(S)]
    - Solved the problem of [PROBLEM_SOLVED]
    - Improved upon [PREVIOUS_LIMITATIONS]
    
    Args:
        input_size (int): Dimension of input features
        learning_rate (float): Learning rate for training
        max_epochs (int): Maximum number of training epochs
        tolerance (float): Convergence tolerance for loss
        hidden_size (int, optional): Number of hidden units (if applicable)
        output_size (int): Number of output classes/units
        activation (str): Activation function to use
        init_method (str): Weight initialization method
        random_state (int, optional): Random seed for reproducibility
        **kwargs: Additional model-specific parameters
    """
    
    def __init__(
        self, 
        input_size: int,
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        tolerance: float = 1e-6,
        hidden_size: Optional[int] = None, 
        output_size: int = 1,
        activation: str = "relu",
        init_method: str = "xavier_normal",
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        # Validate input parameters
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        # Store architecture parameters
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.init_method = init_method
        self.random_state = random_state
        
        # Store additional parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Set random seed if provided
        if random_state is not None:
            set_random_seed(random_state)
        
        # Initialize model architecture
        self._build_model()
        
        # Initialize weights (if custom initialization needed)
        self._initialize_weights()
        
        # Ensure parameters require gradients
        if _TORCH_AVAILABLE and torch is not None:
            for param in self.parameters():
                param.requires_grad = True
        
        # Initialize training state
        self.is_fitted = False
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "epochs_trained": 0
        }
        
        self.logger.info(f"Initialized {MODEL_NAME} with input_size={input_size}, "
                        f"hidden_size={hidden_size}, output_size={output_size}")
    
    def _build_model(self):
        """
        Build the model architecture.
        
        This method should define all the layers and components of the model.
        Keep this separate from __init__ for clarity.
        """
        # TODO: Define model layers here
        # Example for simple linear model:
        # self.linear = nn.Linear(self.input_size, self.output_size)
        
        # Example for model with hidden layer:
        # if self.hidden_size is not None:
        #     self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        #     self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        # else:
        #     self.linear = nn.Linear(self.input_size, self.output_size)
        
        raise NotImplementedError("Implement model architecture in _build_model()")
    
    def _initialize_weights(self):
        """
        Initialize model weights based on the specified method.
        
        Use this method if the model requires custom weight initialization
        beyond PyTorch defaults. For historical accuracy, match the original
        paper's initialization scheme when possible.
        """
        if not _TORCH_AVAILABLE:
            return
            
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if self.init_method == "zeros":
                        nn.init.zeros_(module.weight)
                        nn.init.zeros_(module.bias)
                    elif self.init_method == "xavier_normal":
                        nn.init.xavier_normal_(module.weight)
                        nn.init.zeros_(module.bias)
                    elif self.init_method == "xavier_uniform":
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
                    elif self.init_method == "kaiming_normal":
                        nn.init.kaiming_normal_(module.weight)
                        nn.init.zeros_(module.bias)
                    elif self.init_method == "normal":
                        nn.init.normal_(module.weight, mean=0.0, std=0.01)
                        nn.init.zeros_(module.bias)
                    else:  # random
                        nn.init.uniform_(module.weight, -0.5, 0.5)
                        nn.init.zeros_(module.bias)
    
    def forward(self, x: TorchTensor) -> TorchTensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # TODO: Implement forward pass
        # Example for simple linear model:
        # return self.linear(x)
        
        # Example for model with hidden layer:
        # if self.hidden_size is not None:
        #     x = F.relu(self.layer1(x))  # or use self._apply_activation(self.layer1(x))
        #     x = self.layer2(x)
        # else:
        #     x = self.linear(x)
        # return x
        
        raise NotImplementedError("Implement forward pass in forward()")
    
    def _apply_activation(self, x: TorchTensor) -> TorchTensor:
        """Apply the specified activation function."""
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "step":
            # Use differentiable approximation during training
            if self.training:
                return torch.sigmoid(10.0 * x)
            return (x >= 0).float()
        else:
            # Default to ReLU
            return F.relu(x)
    
    def predict(self, x: TorchTensor) -> TorchTensor:
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
    
    def predict_proba(self, x: TorchTensor) -> TorchTensor:
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
                # Binary classification - ensure proper probability format
                prob_pos = torch.sigmoid(outputs).squeeze()
                if prob_pos.dim() == 0:
                    prob_pos = prob_pos.unsqueeze(0)
                prob_neg = 1 - prob_pos
                return torch.stack([prob_neg, prob_pos], dim=1)
            else:
                # Multi-class classification
                return F.softmax(outputs, dim=1)
    
    def get_loss(self, outputs: TorchTensor, targets: TorchTensor) -> TorchTensor:
        """
        Compute loss for training.
        
        Args:
            outputs: Model outputs (logits from forward pass)
            targets: Ground truth labels
            
        Returns:
            Loss tensor
        """
        if self.output_size == 1:
            # Binary classification - use BCEWithLogitsLoss for numerical stability
            # Ensure outputs are [batch_size, 1] and targets are [batch_size, 1]
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            elif targets.dim() > 1:
                targets = targets.squeeze()
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                targets = targets.unsqueeze(1)
            
            targets = targets.float()
            criterion = nn.BCEWithLogitsLoss()
            return criterion(outputs, targets)
        else:
            # Multi-class classification
            return F.cross_entropy(outputs, targets.long())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information following project standards.
        
        Returns:
            Dictionary containing model metadata and current state
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters()) if _TORCH_AVAILABLE else 0
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) if _TORCH_AVAILABLE else 0
        
        return {
            # Core identification
            "name": MODEL_NAME,
            "full_name": f"Template {MODEL_NAME}",
            "category": "template",
            "module": 0,
            "pattern": "engine-based",
            # Historical context
            "year_introduced": YEAR_INTRODUCED,
            "authors": AUTHORS,
            "paper_title": "Template Model Implementation",
            "key_innovations": [
                "Template implementation for educational purposes",
                "Demonstrates best practices for model development",
                "Provides foundation for more complex models",
            ],
            # Architecture details
            "architecture_type": "template",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "parameter_count": total_params,
            "trainable_parameters": trainable_params,
            "activation_function": self.activation,
            # Training characteristics
            "learning_algorithm": "template",
            "loss_function": "bce-with-logits" if self.output_size == 1 else "cross-entropy",
            "optimizer": "template",
            "convergence_guarantee": "depends on implementation",
            # Implementation details
            "framework": "pytorch" if _TORCH_AVAILABLE else "numpy",
            "precision": "float32",
            "device_support": ["cpu", "cuda", "mps"] if _TORCH_AVAILABLE else ["cpu"],
            "device": (
                str(next(iter(self.parameters())).device)
                if _TORCH_AVAILABLE and list(self.parameters())
                else "cpu"
            ),
            # Educational metadata
            "difficulty_level": "template",
            "estimated_training_time": "template",
            "key_learning_objectives": [
                "Understand model template structure",
                "Learn implementation patterns",
                "Foundation for real models",
            ],
            # Training state
            "is_fitted": self.is_fitted,
            "epochs_trained": self.training_history.get("epochs_trained", 0),
            "converged": len(self.training_history.get("loss", [])) > 0
            and self.training_history["loss"][-1] <= self.tolerance,
            # Training configuration
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "init_method": self.init_method,
            # Legacy compatibility
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "total_parameters": total_params,
            "activation": self.activation,
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
            
        if _TORCH_AVAILABLE:
            torch.save(checkpoint, filepath)
            self.logger.info(f"Saved checkpoint to {filepath}")
        else:
            self.logger.warning("Cannot save checkpoint: PyTorch not available")
    
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
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Cannot load checkpoint: PyTorch not available")
            
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Extract model parameters from checkpoint
        model_info = checkpoint.get("model_info", {})
        input_size = model_info.get("input_size", model_kwargs.get("input_size"))
        hidden_size = model_info.get("hidden_size", model_kwargs.get("hidden_size"))
        output_size = model_info.get("output_size", model_kwargs.get("output_size", 1))
        
        # Create model instance
        model = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            **model_kwargs
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore training state
        model.is_fitted = True
        model.training_history["epochs_trained"] = checkpoint.get("epoch", 0)
        
        model.logger.info(f"Loaded model from {filepath}")
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
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "activation": self.activation,
                "init_method": self.init_method,
                "random_state": self.random_state,
            },
            "training_history": self.training_history,
            "model_info": self.get_model_info(),
        }
        
        if _TORCH_AVAILABLE:
            torch.save(save_dict, save_path)
            self.logger.info(f"Model saved to {save_path}")
        else:
            self.logger.warning("Cannot save model: PyTorch not available")
    
    @classmethod
    def load_model(cls, filepath: str) -> "ModelTemplate":
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ModelTemplate model
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Cannot load model: PyTorch not available")
            
        save_dict = torch.load(filepath, map_location="cpu")
        
        # Extract configuration
        config = save_dict["model_config"]
        
        # Create new model instance
        model = cls(
            input_size=config["input_size"],
            learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"],
            tolerance=config["tolerance"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"],
            activation=config["activation"],
            init_method=config["init_method"],
            random_state=config["random_state"],
        )
        
        # Load state
        model.load_state_dict(save_dict["model_state_dict"])
        model.training_history = save_dict.get("training_history", {})
        model.is_fitted = True
        
        model.logger.info(f"Model loaded from {filepath}")
        return model
    
    def fit(self, x_data: TorchTensor, y_target: TorchTensor) -> Dict[str, Any]:
        """
        Fit the model to the data using the engine framework.
        
        This method provides the BaseModel interface required by the engine.
        The actual training is handled by the engine framework.
        
        Args:
            x_data: Input features
            y_target: Target labels
            
        Returns:
            Dictionary with training results
        """
        # Store data for compatibility
        self.x_data = x_data
        self.y_target = y_target
        
        # This method is required by BaseModel interface
        # The actual training is handled by the engine framework
        self.is_fitted = True
        return {
            "converged": True,
            "epochs_trained": 0,
            "final_loss": 0.0,
            "final_accuracy": 0.0
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(input_size={self.input_size}, " \
               f"hidden_size={self.hidden_size}, output_size={self.output_size})"


def create_model(config: Dict[str, Any]) -> ModelTemplate:
    """
    Create model instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ModelTemplate: Model instance
    """
    return ModelTemplate(
        input_size=config['input_size'],
        learning_rate=config.get('learning_rate', 0.01),
        max_epochs=config.get('max_epochs', 100),
        tolerance=config.get('tolerance', 1e-6),
        hidden_size=config.get('hidden_size', None),
        output_size=config.get('output_size', 1),
        activation=config.get('activation', 'relu'),
        init_method=config.get('init_method', 'xavier_normal'),
        random_state=config.get('random_state', None),
        **{k: v for k, v in config.items() 
           if k not in ['input_size', 'learning_rate', 'max_epochs', 'tolerance',
                       'hidden_size', 'output_size', 'activation', 'init_method', 'random_state']}
    )


# =============================================================================
# SIMPLE IMPLEMENTATION PATTERN (Alternative to Engine-based)
# =============================================================================

class SimpleModelTemplate(BaseNNModule):
    """
    Simple model implementation without BaseModel interface.
    
    This class provides a simpler alternative for models that don't need
    the full engine framework integration. Use this for educational or
    simple implementations (like 03_MLP pattern).
    """
    
    def __init__(self, input_size: int, hidden_size: Optional[int] = None, output_size: int = 1, **kwargs):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Build model
        self._build_model()
        
        # Training state
        self.is_fitted = False
        self.training_history = []
    
    def _build_model(self):
        """Build the simple model architecture."""
        # TODO: Implement simple model architecture
        raise NotImplementedError("Implement simple model architecture")
    
    def forward(self, x: TorchTensor) -> TorchTensor:
        """Simple forward pass."""
        # TODO: Implement simple forward pass
        raise NotImplementedError("Implement simple forward pass")
    
    def fit(self, X: TorchTensor, y: TorchTensor, **kwargs) -> Dict[str, Any]:
        """Simple fit method."""
        # TODO: Implement simple training loop
        self.is_fitted = True
        return {"converged": True}
    
    def predict(self, X: TorchTensor) -> TorchTensor:
        """Simple predict method."""
        return self.forward(X)


# =============================================================================
# MODEL CREATION UTILITIES
# =============================================================================

def create_simple_model(config: Dict[str, Any]) -> SimpleModelTemplate:
    """Create simple model instance from configuration."""
    return SimpleModelTemplate(
        input_size=config['input_size'],
        hidden_size=config.get('hidden_size', None),
        output_size=config.get('output_size', 1),
        **{k: v for k, v in config.items() 
           if k not in ['input_size', 'hidden_size', 'output_size']}
    )


def get_model_class(pattern: str = "engine"):
    """
    Get the appropriate model class based on implementation pattern.
    
    Args:
        pattern: "engine" for BaseModel integration, "simple" for basic implementation
        
    Returns:
        Model class
    """
    if pattern == "engine":
        return ModelTemplate
    elif pattern == "simple":
        return SimpleModelTemplate
    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'engine' or 'simple'.") 