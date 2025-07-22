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
from typing import Any, Dict

import torch
import numpy as np
from torch import nn

from engine.base import BaseModel
from utils import get_logger

try:
    from .model import ADALINE
    from .config import ADALINEConfig, get_experiment_config
except ImportError:
    # For direct execution
    from model import ADALINE
    from config import ADALINEConfig, get_experiment_config


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
    
    def fit(self, x_data: torch.Tensor, y_target: torch.Tensor) -> Dict[str, Any]:
        """
        Train the model using the BaseModel interface.
        
        This method provides the BaseModel-compatible interface required
        for engine integration while delegating to the pure ADALINE implementation.
        
        Args:
            x_data: Input features as torch tensor
            y_target: Target values as torch tensor
            
        Returns:
            Training results dictionary
        """
        # Convert torch tensors to numpy for pure ADALINE
        x_np = x_data.cpu().numpy()
        y_np = y_target.cpu().numpy()
        
        # Use the pure ADALINE implementation
        results = self.fit_pure(x_np, y_np)
        
        return results
    
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
        """
        Get comprehensive model information following wandb integration plan standards.
        
        Returns:
            Standardized model information dictionary
        """
        base_info = self.pure_adaline.get_model_info()
        
        # Enhance with wrapper-specific information
        wrapper_info = {
            # Update pattern to indicate engine-based wrapper
            "pattern": "engine-based",
            
            # Framework details (hybrid approach)
            "framework": "numpy+pytorch",
            "wrapper_framework": "pytorch",
            "core_framework": "numpy",
            
            # Architecture details
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "pytorch_parameters": dict(self.named_parameters()),
            
            # Training characteristics
            "supports_batching": True,
            "supports_gradients": True,
            "engine_compatible": True,
            
            # Implementation details  
            "device_support": ["cpu", "gpu", "mps"],
            "precision": "float32",
            
            # Wrapper capabilities
            "wandb_integration": True,
            "artifact_logging": True,
            "visualization_support": True,
            "educational_features": [
                "Access to pure NumPy implementation",
                "Delta Rule step-by-step demonstration", 
                "PyTorch/NumPy comparison",
                "Engine framework integration"
            ]
        }
        
        # Merge base info with wrapper enhancements
        base_info.update(wrapper_info)
        return base_info
    
    def save_model(self, filepath: str):
        """
        Save the wrapped ADALINE model.
        
        Saves both the PyTorch wrapper state and the pure NumPy implementation
        to ensure complete model persistence.
        """
        self.logger.info(f"Saving ADALINEWrapper to {filepath}")
        
        # Create save directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pure ADALINE state
        pure_state = {
            'weights': self.pure_adaline.weights,
            'bias': self.pure_adaline.bias,
            'config': self.pure_adaline.config.__dict__,
            'training_history': self.pure_adaline.training_history,
            'is_fitted': self.pure_adaline.is_fitted
        }
        
        # Save combined state
        save_state = {
            'model_type': 'ADALINEWrapper',
            'pure_adaline_state': pure_state,
            'pytorch_state_dict': self.state_dict(),
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        torch.save(save_state, filepath)
        self.logger.info("Model saved successfully")
    
    @classmethod
    def load_model(cls, filepath: str) -> "ADALINEWrapper":
        """
        Load a saved ADALINEWrapper model.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded ADALINEWrapper instance
        """
        # Load the saved state
        saved_state = torch.load(filepath, map_location='cpu')
        
        if saved_state.get('model_type') != 'ADALINEWrapper':
            raise ValueError("File is not a saved ADALINEWrapper model")
        
        # Reconstruct config
        from config import ADALINEConfig
        config_dict = saved_state['config']
        config = ADALINEConfig(**config_dict)
        
        # Create new instance
        wrapper = cls(config)
        
        # Restore pure ADALINE state
        pure_state = saved_state['pure_adaline_state']
        wrapper.pure_adaline.weights = pure_state['weights']
        wrapper.pure_adaline.bias = pure_state['bias']
        wrapper.pure_adaline.training_history = pure_state['training_history']
        wrapper.pure_adaline.is_fitted = pure_state['is_fitted']
        
        # Restore PyTorch state
        wrapper.load_state_dict(saved_state['pytorch_state_dict'])
        
        # Restore training history
        wrapper.training_history = saved_state.get('training_history', {})
        
        return wrapper
    
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

    def fit_with_wandb(self, X: np.ndarray, y: np.ndarray, config: 'ADALINEConfig') -> Dict[str, Any]:
        """
        Train ADALINE with wandb tracking integration.
        
        This method demonstrates how to use the BaseModel wandb methods
        with the Simple Pattern implementation.
        
        Args:
            X: Training features
            y: Training labels  
            config: ADALINEConfig with wandb settings
            
        Returns:
            Training results with wandb tracking
        """
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb_tags = config.wandb_tags or ["adaline", config.dataset, "delta-rule"]
            success = self.init_wandb(
                project=config.wandb_project,
                name=f"adaline-{config.name}",
                tags=wandb_tags,
                config=config.__dict__,
                notes=config.wandb_notes or f"ADALINE training on {config.dataset}",
                mode=config.wandb_mode
            )
            
            if success:
                self.logger.info("✅ Wandb tracking enabled for ADALINE")
                # Log model info as initial metrics
                model_info = self.get_model_info()
                self.log_metrics({
                    "model/parameters": model_info.get("total_parameters", 0),
                    "model/input_size": self.input_size,
                    "config/learning_rate": config.learning_rate,
                    "config/max_epochs": config.epochs
                }, step=0)
            else:
                self.logger.warning("⚠️ Wandb setup failed, training without tracking")
        
        # Train using pure ADALINE with epoch-by-epoch logging
        self.logger.info("Training ADALINE with Delta Rule and wandb tracking")
        
        # Handle 2D y arrays by squeezing if needed
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze()
        
        # Modified training loop to log metrics to wandb
        for epoch in range(config.epochs):
            
            # Train for one epoch (simplified single epoch training)
            total_error = 0
            for i in range(len(X)):
                x_i = X[i:i+1]
                y_i = y[i:i+1] if y.ndim > 0 else np.array([y[i]])
                
                # Forward pass
                linear_output = self.pure_adaline.forward(x_i)
                error = y_i[0] - linear_output[0]
                total_error += error ** 2
                
                # Delta Rule update
                self.pure_adaline.weights += config.learning_rate * error * x_i.flatten()
                self.pure_adaline.bias += config.learning_rate * error
            
            # Calculate metrics
            mse = total_error / len(X)
            predictions = self.pure_adaline.predict(X)
            accuracy = np.mean((predictions > 0.5) == (y > 0.5)) if y.dtype == bool or np.all(np.isin(y, [0, 1])) else None
            
            # Log to wandb if enabled
            if config.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
                metrics = {
                    "train/mse": mse,
                    "train/total_error": total_error,
                    "weights/mean": float(np.mean(self.pure_adaline.weights)),
                    "weights/std": float(np.std(self.pure_adaline.weights)),
                    "bias": float(self.pure_adaline.bias),
                    "epoch": epoch
                }
                
                if accuracy is not None:
                    metrics["train/accuracy"] = float(accuracy)
                
                self.log_metrics(metrics, step=epoch)
            
            # Console logging  
            if epoch % config.log_interval == 0:
                self.logger.info(f"Epoch {epoch}: MSE = {mse:.6f}")
            
            # Check convergence
            if mse < config.tolerance:
                self.logger.info(f"Converged at epoch {epoch}")
                break
        
        # Sync PyTorch parameters
        self._sync_pytorch_params()
        
        # Final results
        self.pure_adaline.training_history["epochs_trained"] = epoch + 1
        self.pure_adaline.is_fitted = True
        
        # Log final metrics to wandb
        if config.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
            final_metrics = {
                "final/mse": mse,
                "final/converged": mse < config.tolerance,
                "final/epochs_trained": epoch + 1
            }
            if accuracy is not None:
                final_metrics["final/accuracy"] = float(accuracy)
            
            self.log_metrics(final_metrics, step=epoch + 1)
            
            # Finish wandb run
            self.finish_wandb()
            self.logger.info("📊 Wandb tracking completed")
        
        results = {
            "converged": mse < config.tolerance,
            "final_mse": mse,
            "epochs_trained": epoch + 1,
            "final_accuracy": float(accuracy) if accuracy is not None else None
        }
        
        self.training_history = results
        return results


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