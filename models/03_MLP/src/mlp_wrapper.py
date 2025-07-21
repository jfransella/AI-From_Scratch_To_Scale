"""
PyTorch Wrapper for Pure NumPy MLP with Backpropagation.

This module provides a PyTorch wrapper around the pure NumPy MLP implementation,
enabling integration with the unified engine infrastructure while preserving
the educational value of visible backpropagation.

Design Philosophy:
- Pure NumPy core shows actual backpropagation algorithm (educational)
- PyTorch wrapper provides engine compatibility (practical) 
- Students see the breakthrough: XOR problem solved with hidden layers
- Visible gradient flow and weight updates for deep learning understanding

Historical Significance:
The backpropagation algorithm (Rumelhart, Hinton, Williams 1986) was the key
breakthrough that made multi-layer networks practical by providing an efficient
method to compute gradients for all layers through the chain rule.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from torch import nn

from engine.base import BaseModel
from utils import get_logger

try:
    from .pure_mlp import PureMLP, create_pure_mlp
    from .config import MLPExperimentConfig
    from .constants import AUTHORS, MODEL_NAME, YEAR_INTRODUCED
except ImportError:
    # For direct execution
    from pure_mlp import PureMLP, create_pure_mlp
    from config import MLPExperimentConfig
    from constants import AUTHORS, MODEL_NAME, YEAR_INTRODUCED


class MLPWrapper(nn.Module, BaseModel):
    """
    PyTorch wrapper for Pure NumPy MLP with visible backpropagation.
    
    This class bridges the educational pure NumPy MLP implementation with
    the practical PyTorch-based engine infrastructure. The core backpropagation
    algorithm remains in NumPy to preserve educational transparency.
    
    Key Educational Features:
    - Uses pure MLP for backpropagation computations and learning
    - Shows gradient flow through hidden layers step-by-step
    - Demonstrates XOR problem solution (impossible for single layer)
    - Maintains access to pure implementation for algorithm analysis
    - Provides PyTorch interface for engine integration
    
    The Breakthrough:
    MLPs with hidden layers can solve non-linearly separable problems
    like XOR, demonstrating the power of representation learning.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int = 1,
        activation: str = "sigmoid",
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
        tolerance: float = 1e-6,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        # Create the pure NumPy MLP - this is where the magic happens!
        self.pure_mlp = PureMLP(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            activation=activation,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            tolerance=tolerance,
            random_state=random_state
        )
        
        # Store configuration for compatibility
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        
        # Create PyTorch layers that mirror the NumPy architecture
        # These are used for engine compatibility but pure MLP does the learning
        self.torch_layers = nn.ModuleList()
        
        # Build PyTorch architecture matching pure MLP
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.torch_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # Sync parameters initially
        self._sync_pytorch_params()
        
        # Training state
        self.training_history = {}
        
        # Logger
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Initialized MLPWrapper with Pure NumPy Backpropagation core")
        self.logger.info(f"Architecture: {input_size} -> {hidden_layers} -> {output_size}")
        self.logger.info(f"Total parameters: {self.pure_mlp._count_parameters()}")
    
    def _sync_pytorch_params(self):
        """Sync PyTorch parameters with pure MLP weights."""
        if self.pure_mlp.weights:
            with torch.no_grad():
                for i, (W, b) in enumerate(zip(self.pure_mlp.weights, self.pure_mlp.biases)):
                    self.torch_layers[i].weight.data = torch.from_numpy(W.T).float()
                    self.torch_layers[i].bias.data = torch.from_numpy(b.flatten()).float()
    
    def _sync_pure_params(self):
        """Sync pure MLP weights with PyTorch parameters."""
        if self.torch_layers:
            for i, layer in enumerate(self.torch_layers):
                self.pure_mlp.weights[i] = layer.weight.data.T.numpy()
                self.pure_mlp.biases[i] = layer.bias.data.numpy().reshape(1, -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using pure MLP implementation.
        
        This routes through the pure NumPy implementation to ensure
        students see the actual forward propagation algorithm.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor from pure MLP
        """
        # Convert to NumPy for pure MLP
        x_np = x.cpu().numpy()
        
        # Use pure MLP forward pass
        output, _, _ = self.pure_mlp.forward(x_np)
        
        # Convert back to PyTorch tensor
        return torch.from_numpy(output).float().to(x.device)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using pure MLP."""
        self.eval()
        with torch.no_grad():
            x_np = x.cpu().numpy()
            predictions = self.pure_mlp.predict(x_np)
            return torch.from_numpy(predictions).float()
    
    def predict_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Make binary predictions by thresholding at 0.5."""
        predictions = self.predict(x)
        return (predictions >= 0.5).float()
    
    def fit_pure(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
        """
        Train using pure MLP backpropagation algorithm.
        
        This method exposes the educational pure implementation directly,
        allowing students to see the complete backpropagation algorithm
        working through hidden layers to solve non-linear problems.
        
        Args:
            X: Training features (NumPy array)
            y: Training labels (NumPy array)
            verbose: Show detailed backpropagation steps
            
        Returns:
            Training history from pure MLP
        """
        self.logger.info("Training with pure MLP backpropagation")
        
        # Handle 2D y arrays by keeping them 2D for MLP
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Train the pure MLP using backpropagation
        history = self.pure_mlp.fit(X, y, verbose=verbose)
        
        # Sync PyTorch parameters after pure training
        self._sync_pytorch_params()
        
        # Store history
        self.training_history = history
        
        return history
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for engine compatibility.
        
        MLP typically uses MSE loss for regression or BCELoss for binary classification.
        """
        # Handle target dimensions
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze()
        if outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze()
        
        # Use MSE loss (consistent with pure MLP)
        return nn.functional.mse_loss(outputs, targets.float())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        pure_info = self.pure_mlp.get_model_info()
        
        return {
            # Model metadata
            "model_name": pure_info.get("model_name", MODEL_NAME),
            "year_introduced": pure_info.get("year_introduced", YEAR_INTRODUCED),
            "original_author": pure_info.get("authors", AUTHORS),
            "implementation": "Pure NumPy Backpropagation + PyTorch Wrapper",
            "algorithm": "Backpropagation (Chain Rule)",
            
            # Architecture from pure MLP
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "activation": self.activation,
            "total_parameters": pure_info.get("total_parameters", 0),
            
            # Training configuration
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "is_fitted": pure_info.get("is_fitted", False),
            
            # Current weights (from pure implementation)
            "weights": pure_info.get("weights", []),
            "biases": pure_info.get("biases", []),
            
            # Training history
            "training_history": self.training_history,
            
            # Capabilities
            "can_solve_xor": len(self.hidden_layers) > 0 and all(h > 0 for h in self.hidden_layers),
            "is_nonlinear": True,
            
            # PyTorch compatibility info
            "torch_parameters": sum(p.numel() for p in self.parameters()),
            "device": str(next(self.parameters()).device),
        }
    
    def save_model(self, filepath: str):
        """Save both pure MLP and PyTorch state."""
        # Save pure MLP
        pure_path = str(Path(filepath).with_suffix('.pure.json'))
        self.pure_mlp.save_model(pure_path)
        
        # Save PyTorch state
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'activation': self.activation,
                'learning_rate': self.learning_rate,
                'max_epochs': self.max_epochs,
                'tolerance': self.tolerance,
            },
            'pure_mlp_path': pure_path,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        
        self.logger.info(f"Saved wrapped MLP to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> "MLPWrapper":
        """Load wrapped MLP from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Reconstruct wrapper
        config = checkpoint['config']
        model = cls(**config)
        
        # Load PyTorch state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load pure MLP if available
        if 'pure_mlp_path' in checkpoint:
            pure_path = checkpoint['pure_mlp_path']
            if Path(pure_path).exists():
                model.pure_mlp = PureMLP.load_model(pure_path)
                model._sync_pytorch_params()
        
        # Restore training history
        model.training_history = checkpoint.get('training_history', {})
        
        return model
    
    def get_pure_mlp(self) -> PureMLP:
        """
        Get access to the pure NumPy implementation for educational analysis.
        
        This allows students to:
        - Examine the actual backpropagation implementation
        - See gradient flow through hidden layers
        - Understand weight update mechanics for each layer
        - Analyze how hidden layers create internal representations
        - Demonstrate XOR problem solution step-by-step
        """
        return self.pure_mlp
    
    def demonstrate_backpropagation(self, X: np.ndarray, y: np.ndarray, max_steps: int = 3):
        """
        Educational demonstration of backpropagation algorithm.
        
        This shows students exactly how gradients flow backwards through
        the network and how weights are updated using the chain rule.
        
        Args:
            X: Training features
            y: Training labels
            max_steps: Maximum training steps to demonstrate
        """
        print("🧠 BACKPROPAGATION ALGORITHM DEMONSTRATION")
        print("=" * 55)
        print("The Chain Rule in Action: How Neural Networks Learn")
        print()
        print(f"Network: {self.input_size} -> {self.hidden_layers} -> {self.output_size}")
        print(f"Algorithm: Backpropagation (Rumelhart, Hinton, Williams 1986)")
        print()
        
        # Ensure proper shapes
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Reset to fresh weights for clean demo
        self.pure_mlp._initialize_weights()
        
        print("📚 Forward → Backward → Update Pattern:")
        print("-" * 40)
        
        for step in range(min(max_steps, len(X))):
            x_i = X[step:step+1]
            y_i = y[step:step+1]
            
            print(f"\nStep {step + 1}: Input {x_i.flatten()} → Target {y_i.flatten()}")
            
            # Forward pass with detailed output
            output, activations, z_values = self.pure_mlp.forward(x_i)
            
            print(f"  📈 Forward Pass:")
            for layer_idx, activation in enumerate(activations):
                if layer_idx == 0:
                    print(f"    Input: {activation.flatten()}")
                elif layer_idx < len(activations) - 1:
                    print(f"    Hidden {layer_idx}: {activation.flatten()}")
                else:
                    print(f"    Output: {activation.flatten()}")
            
            # Show loss
            loss = self.pure_mlp._compute_loss(y_i, output)
            print(f"  💥 Loss: {loss:.6f}")
            
            # Backpropagation with verbose output
            print(f"  🔄 Backpropagation:")
            self.pure_mlp._backpropagate(activations, z_values, y_i, verbose=True)
            
            print("  ⬆️  Weights updated via chain rule!")
        
        print(f"\n✨ Key Insights:")
        print(f"   • Forward pass: Data flows through layers")
        print(f"   • Backward pass: Gradients flow backwards via chain rule")
        print(f"   • Hidden layers: Create internal representations")
        print(f"   • XOR solution: Non-linear problems become solvable!")
    
    def solve_xor_problem(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Educational demonstration: Solving XOR with backpropagation.
        
        This shows the breakthrough moment when MLPs overcome the
        fundamental limitation of single-layer perceptrons.
        
        Returns:
            Training results and XOR solution analysis
        """
        if verbose:
            print("🎯 THE XOR BREAKTHROUGH")
            print("=" * 30)
            print("Solving the problem that stumped single-layer networks!")
            print()
        
        # XOR dataset - the classic non-linearly separable problem
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_xor = np.array([[0], [1], [1], [0]], dtype=float)
        
        if verbose:
            print("XOR Truth Table (impossible for single layer):")
            for x, y in zip(X_xor, y_xor):
                print(f"  {x} → {y[0]}")
            print()
        
        # Train the MLP
        if verbose:
            print("🏃 Training MLP with backpropagation...")
        
        history = self.fit_pure(X_xor, y_xor, verbose=False)
        
        # Test the solution
        predictions = self.pure_mlp.predict(X_xor)
        binary_preds = self.pure_mlp.predict_binary(X_xor)
        accuracy = self.pure_mlp._compute_accuracy(y_xor, predictions)
        
        if verbose:
            print(f"\n📊 Results after {len(history['epoch'])} epochs:")
            print("-" * 30)
            for i, (x, y_true, y_pred, y_bin) in enumerate(zip(X_xor, y_xor, predictions, binary_preds)):
                print(f"  {x} → True: {y_true[0]}, Pred: {y_pred[0]:.3f}, Binary: {y_bin[0]}")
            
            print(f"\nAccuracy: {accuracy:.1%}")
            
            if accuracy > 0.9:
                print("🎉 SUCCESS! MLP solved XOR with hidden layers!")
                print("🧠 The power of representation learning unlocked!")
            else:
                print("😞 Needs more training - but the architecture is capable!")
            
            print(f"\n🔍 Decision boundary: Non-linear (unlike Perceptron)")
            print(f"🔑 Key: Hidden layers transform input space!")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'binary_predictions': binary_preds.tolist(),
            'training_history': history,
            'xor_solved': accuracy > 0.9
        }


def create_mlp_wrapper(
    input_size: int,
    hidden_layers: List[int],
    **kwargs
) -> MLPWrapper:
    """
    Factory function to create an MLP with pure NumPy backpropagation core
    and PyTorch wrapper for engine compatibility.
    
    This is the recommended way to create MLPs that provide both
    educational value (visible backpropagation) and practical utility
    (engine integration).
    
    Args:
        input_size: Number of input features
        hidden_layers: List of hidden layer sizes
        **kwargs: Additional configuration parameters
        
    Returns:
        MLPWrapper instance with pure NumPy core and PyTorch compatibility
    """
    return MLPWrapper(input_size=input_size, hidden_layers=hidden_layers, **kwargs)


def create_xor_solver(learning_rate: float = 0.5, max_epochs: int = 500) -> MLPWrapper:
    """
    Create an MLP specifically configured to solve the XOR problem.
    
    This is a pre-configured MLP that demonstrates the breakthrough
    capability of multi-layer networks to solve non-linearly separable problems.
    
    Args:
        learning_rate: Learning rate for training
        max_epochs: Maximum training epochs
        
    Returns:
        MLPWrapper configured for XOR problem
    """
    return create_mlp_wrapper(
        input_size=2,
        hidden_layers=[4],  # Small hidden layer sufficient for XOR
        output_size=1,
        activation='sigmoid',
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        random_state=42
    )


if __name__ == "__main__":
    # Educational demonstration of MLP hybrid approach
    print("🔗 MLP Wrapper: Pure NumPy Backpropagation + Engine Compatibility")
    print("=" * 75)
    
    # Create XOR solver
    mlp = create_xor_solver(learning_rate=1.0, max_epochs=200)
    
    print("\n📚 Step-by-step backpropagation demonstration:")
    X_demo = np.array([[0, 1], [1, 0]], dtype=float)
    y_demo = np.array([[1], [1]], dtype=float)
    mlp.demonstrate_backpropagation(X_demo, y_demo, max_steps=2)
    
    print("\n🎯 XOR Problem Solution:")
    results = mlp.solve_xor_problem(verbose=True)
    
    # Show PyTorch compatibility
    print(f"\n🔧 PyTorch Engine Compatibility:")
    X_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    y_torch = torch.tensor([[1], [1]], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = mlp.forward(X_torch)
        loss = mlp.get_loss(outputs, y_torch)
    
    print(f"PyTorch forward pass: {outputs.detach().numpy().flatten()}")
    print(f"PyTorch loss: {loss.item():.6f}")
    
    print(f"\n✨ The Hybrid Advantage:")
    print(f"   🎓 Pure NumPy: Students see backpropagation algorithm")
    print(f"   🔧 PyTorch: Engine compatibility and practical deployment")
    print(f"   🧠 Educational: XOR breakthrough moment visible!")
    print(f"   🚀 Practical: Ready for complex training pipelines!") 