"""
PyTorch Wrapper for Pure NumPy ADALINE

This module provides a thin PyTorch wrapper around the pure NumPy ADALINE
implementation, allowing it to work with the unified engine infrastructure
while preserving the educational value of the Delta Rule algorithm.

Design Philosophy:
- Core logic remains in pure NumPy (educational) 
- PyTorch wrapper provides engine compatibility (practical)
- Students can see both the Delta Rule and modern integration
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np
from torch import nn

from engine.base import BaseModel
from utils import get_logger

try:
    from .model import ADALINE
    from .config import ADALINEConfig, get_experiment_config
    from .constants import AUTHORS, MODEL_NAME, YEAR_INTRODUCED
except ImportError:
    # For direct execution
    from model import ADALINE
    from config import ADALINEConfig, get_experiment_config
    from constants import AUTHORS, MODEL_NAME, YEAR_INTRODUCED


class ADALINEWrapper(nn.Module, BaseModel):
    """
    PyTorch wrapper for the pure NumPy ADALINE implementation.
    
    This class bridges the educational pure NumPy implementation with
    the practical PyTorch-based engine infrastructure. The core Delta Rule
    learning logic remains in NumPy to preserve educational value.
    
    Key Features:
    - Uses pure ADALINE for actual computations and Delta Rule learning
    - Provides PyTorch tensor interface for engine compatibility
    - Maintains access to pure implementation for educational analysis
    - Handles data conversion between NumPy and PyTorch automatically
    """
    
    def __init__(
        self,
        config: ADALINEConfig,
        **kwargs
    ):
        super().__init__()
        
        # Create the pure NumPy ADALINE
        self.pure_adaline = ADALINE(config)
        
        # Store configuration for compatibility
        self.config = config
        self.input_size = config.input_size
        self.learning_rate = config.learning_rate
        self.max_epochs = config.epochs
        
        # Create PyTorch parameters that mirror the NumPy weights
        # These are used for engine compatibility but the pure ADALINE
        # maintains the actual training logic
        self.linear = nn.Linear(config.input_size, 1, bias=True)
        self._sync_pytorch_params()
        
        # Training state
        self.training_history = {}
        
        # Logger
        self.logger = get_logger(__name__)
        
        self.logger.info("Initialized ADALINEWrapper with Pure NumPy core")
    
    def _sync_pytorch_params(self):
        """Sync PyTorch parameters with pure ADALINE weights."""
        if hasattr(self.pure_adaline, 'weights') and self.pure_adaline.weights is not None:
            with torch.no_grad():
                self.linear.weight.data = torch.from_numpy(
                    self.pure_adaline.weights.reshape(1, -1)
                ).float()
                self.linear.bias.data = torch.tensor([self.pure_adaline.bias]).float()
    
    def _sync_pure_params(self):
        """Sync pure ADALINE weights with PyTorch parameters."""
        self.pure_adaline.weights = self.linear.weight.data.numpy().flatten()
        self.pure_adaline.bias = self.linear.bias.data.item()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the pure ADALINE implementation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (continuous values, not thresholded)
        """
        # Convert to NumPy for pure ADALINE
        x_np = x.cpu().numpy()
        
        # Use pure ADALINE for predictions (linear output)
        # ADALINE forward gives continuous output, predict gives binary
        predictions = self.pure_adaline.forward(x_np).flatten()
        
        # Convert back to PyTorch tensor
        return torch.tensor(predictions, dtype=torch.float32, device=x.device).unsqueeze(-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using pure ADALINE (continuous output)."""
        self.eval()
        with torch.no_grad():
            x_np = x.cpu().numpy()
            predictions = self.pure_adaline.predict(x_np)
            return torch.from_numpy(predictions).float()
    
    def predict_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Make binary predictions by thresholding at 0.5."""
        predictions = self.predict(x)
        return (predictions >= 0.5).float()
    
    def fit_pure(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
        """
        Train using the pure ADALINE Delta Rule.
        
        This method exposes the educational pure implementation directly,
        allowing students to see the original Delta Rule algorithm in action.
        
        Args:
            X: Training features (NumPy array)
            y: Training labels (NumPy array)
            verbose: Show training progress
            
        Returns:
            Training history from pure ADALINE
        """
        self.logger.info("Training with pure ADALINE Delta Rule")
        
        # Handle 2D y arrays by squeezing if needed
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze()
        
        # Train the pure ADALINE
        history = self.pure_adaline.fit(X, y)
        
        # Sync PyTorch parameters
        self._sync_pytorch_params()
        
        # Store history
        self.training_history = history
        
        return history
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for engine compatibility.
        
        ADALINE uses Mean Squared Error (MSE) loss, which aligns with
        the Delta Rule's continuous error-based learning.
        """
        # Handle 2D targets by squeezing if needed
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # MSE loss for continuous targets
        return nn.functional.mse_loss(outputs.squeeze(), targets.float())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        pure_info = self.pure_adaline.get_model_info()
        
        return {
            # Model metadata
            "model_name": pure_info.get("model_name", MODEL_NAME),
            "year_introduced": pure_info.get("year_introduced", YEAR_INTRODUCED),
            "original_author": pure_info.get("authors", AUTHORS),
            "implementation": "Pure NumPy + PyTorch Wrapper",
            
            # Architecture from pure ADALINE
            "input_size": self.input_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "is_fitted": pure_info.get("is_fitted", False),
            "algorithm": "Delta Rule (LMS)",
            
            # Current weights
            "weights": pure_info.get("weights", []).tolist() if hasattr(pure_info.get("weights", []), 'tolist') else pure_info.get("weights", []),
            "bias": float(pure_info.get("bias", 0.0)) if pure_info.get("bias") is not None else 0.0,
            
            # Training history
            "training_history": self.training_history,
            
            # PyTorch compatibility info
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "device": str(next(self.parameters()).device),
        }
    
    def save_model(self, filepath: str):
        """Save both pure ADALINE and PyTorch state."""
        # Save pure ADALINE (assuming it has a save method, otherwise create one)
        pure_path = str(Path(filepath).with_suffix('.pure.json'))
        # Note: The pure ADALINE might not have a save method, so we'll store the config and weights
        import json
        pure_data = {
            'config': {
                'input_size': self.config.input_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'tolerance': self.config.tolerance,
            },
            'weights': self.pure_adaline.weights.tolist() if self.pure_adaline.weights is not None else None,
            'bias': float(self.pure_adaline.bias) if self.pure_adaline.bias is not None else None,
            'is_fitted': self.pure_adaline.is_fitted,
            'training_history': getattr(self.pure_adaline, 'training_history', {})
        }
        
        with open(pure_path, 'w') as f:
            json.dump(pure_data, f, indent=2)
        
        # Save PyTorch state
        checkpoint = {
            'state_dict': self.state_dict(),
            'config_dict': {
                'name': self.config.name,
                'description': self.config.description,
                'input_size': self.config.input_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'tolerance': self.config.tolerance,
            },
            'pure_adaline_path': pure_path,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        
        self.logger.info(f"Saved wrapped ADALINE to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> "ADALINEWrapper":
        """Load wrapped ADALINE from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Reconstruct config
        config_dict = checkpoint['config_dict']
        config = ADALINEConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            input_size=config_dict['input_size'],
            learning_rate=config_dict['learning_rate'],
            epochs=config_dict['epochs'],
            tolerance=config_dict['tolerance']
        )
        
        # Create wrapper instance
        model = cls(config)
        
        # Load PyTorch state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load pure ADALINE if available
        if 'pure_adaline_path' in checkpoint:
            pure_path = checkpoint['pure_adaline_path']
            if Path(pure_path).exists():
                import json
                with open(pure_path, 'r') as f:
                    pure_data = json.load(f)
                
                if pure_data['weights'] is not None:
                    model.pure_adaline.weights = np.array(pure_data['weights'])
                if pure_data['bias'] is not None:
                    model.pure_adaline.bias = pure_data['bias']
                
                model.pure_adaline.is_fitted = pure_data['is_fitted']
                model.pure_adaline.training_history = pure_data.get('training_history', {})
        
        # Restore training history
        model.training_history = checkpoint.get('training_history', {})
        
        return model
    
    def get_pure_adaline(self) -> ADALINE:
        """
        Get access to the pure NumPy implementation for educational analysis.
        
        This allows students to:
        - Examine the actual Delta Rule implementation
        - See weight update history and MSE progression
        - Understand the continuous learning behavior
        - Analyze convergence patterns
        """
        return self.pure_adaline
    
    def demonstrate_delta_rule(self, X: np.ndarray, y: np.ndarray, max_steps: int = 5):
        """
        Educational method to demonstrate the Delta Rule step by step.
        
        Args:
            X: Training features  
            y: Training labels
            max_steps: Maximum steps to demonstrate
        """
        print("🎓 ADALINE Delta Rule Demonstration")
        print("=" * 50)
        print("Rule: Δw = η × (target - prediction) × input")
        print("      Δb = η × (target - prediction)")
        print("Note: Updates based on ERROR MAGNITUDE, not just wrong/right")
        print()
        
        # Reset to fresh weights
        self.pure_adaline._initialize_weights()
        
        step = 0
        for i, (x, target) in enumerate(zip(X, y)):
            if step >= max_steps:
                break
                
            # Current state  
            prediction = self.pure_adaline.forward(x.reshape(1, -1))[0]
            error = target - prediction
            
            print(f"Step {step + 1}:")
            print(f"  Input: {x}")
            print(f"  Weights: {self.pure_adaline.weights}")
            print(f"  Bias: {self.pure_adaline.bias:.3f}")
            print(f"  Prediction: {prediction:.3f}")
            print(f"  Target: {target:.3f}")
            print(f"  Error: {error:.3f}")
            
            if abs(error) > 1e-6:  # Only update if significant error
                # Show the update
                old_weights = self.pure_adaline.weights.copy()
                old_bias = self.pure_adaline.bias
                
                # Apply Delta Rule
                weight_update = self.pure_adaline.config.learning_rate * error * x
                bias_update = self.pure_adaline.config.learning_rate * error
                
                self.pure_adaline.weights += weight_update
                self.pure_adaline.bias += bias_update
                
                print(f"  📈 Delta Rule Update:")
                print(f"  Weight change: η × error × input = {self.pure_adaline.config.learning_rate} × {error:.3f} × {x}")
                print(f"                 = {weight_update}")
                print(f"  New weights: {old_weights} + {weight_update} = {self.pure_adaline.weights}")
                print(f"  Bias change: {self.pure_adaline.config.learning_rate} × {error:.3f} = {bias_update:.3f}")
                print(f"  New bias: {old_bias:.3f} + {bias_update:.3f} = {self.pure_adaline.bias:.3f}")
            else:
                print(f"  ✅ Error small - minimal update needed")
            
            print()
            step += 1


def create_adaline_wrapper(config: ADALINEConfig) -> ADALINEWrapper:
    """
    Factory function to create an ADALINE with pure NumPy core and PyTorch wrapper.
    
    This is the recommended way to create ADALINEs that maintain
    educational value while providing engine compatibility.
    """
    return ADALINEWrapper(config)


def create_adaline_from_experiment(experiment_name: str) -> ADALINEWrapper:
    """
    Factory function to create ADALINE from experiment configuration.
    
    Args:
        experiment_name: Name of the experiment configuration
        
    Returns:
        ADALINEWrapper instance configured for the experiment
    """
    config = get_experiment_config(experiment_name)
    return create_adaline_wrapper(config)


if __name__ == "__main__":
    # Educational demonstration
    print("🔗 ADALINE Wrapper: Pure NumPy + Engine Compatibility")
    print("=" * 65)
    
    # Create wrapped ADALINE
    config = get_experiment_config('debug_small')
    wrapper = create_adaline_wrapper(config)
    
    # Simple linear data for ADALINE demonstration
    X_linear = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_linear = np.array([0.0, 0.5, 0.5, 1.0])  # Linear relationship
    
    print("\n📚 Step-by-step Delta Rule demonstration:")
    wrapper.demonstrate_delta_rule(X_linear, y_linear, max_steps=3)
    
    print("\n🏃 Full training with pure implementation:")
    history = wrapper.fit_pure(X_linear, y_linear, verbose=False)
    
    predictions = wrapper.pure_adaline.predict(X_linear)
    mse = np.mean((predictions - y_linear) ** 2)
    print(f"Final MSE: {mse:.6f}")
    print(f"Converged: {history.get('converged', False)}")
    print(f"Epochs: {history.get('epochs_trained', 0)}")
    
    # Show PyTorch compatibility
    print(f"\n🔧 PyTorch compatibility:")
    X_torch = torch.from_numpy(X_linear).float()
    y_torch = torch.from_numpy(y_linear).float()
    
    with torch.no_grad():
        outputs = wrapper.forward(X_torch)
        loss = wrapper.get_loss(outputs, y_torch)
        
    print(f"PyTorch forward pass: {outputs.squeeze().numpy()}")
    print(f"PyTorch loss: {loss.item():.6f}")
    
    print(f"\n✨ Best of both worlds:")
    print(f"   - Pure NumPy core shows Delta Rule learning")
    print(f"   - PyTorch wrapper provides engine compatibility")
    print(f"   - Students see continuous error-based updates!") 