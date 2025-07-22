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
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from torch import nn

from engine.base import BaseModel
from utils import get_logger

try:
    from .pure_mlp import PureMLP
    from .config import MLPExperimentConfig
    from .constants import AUTHORS, YEAR_INTRODUCED
except ImportError:
    # For direct execution
    from pure_mlp import PureMLP
    from config import MLPExperimentConfig
    from constants import AUTHORS, YEAR_INTRODUCED


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
        
        self.logger.info("Initialized MLPWrapper with Pure NumPy Backpropagation core")
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
    
    def fit(self, x_data: torch.Tensor, y_target: torch.Tensor) -> Dict[str, Any]:
        """
        Fit the model to the data using the engine framework.
        
        This abstract method is required by BaseModel interface.
        Converts PyTorch tensors to NumPy and delegates to pure implementation.
        
        Args:
            x_data: Input features as PyTorch tensor
            y_target: Target labels as PyTorch tensor
            
        Returns:
            Dictionary with training results
        """
        # Convert to NumPy for pure MLP
        X_np = x_data.cpu().numpy()
        y_np = y_target.cpu().numpy()
        
        # Train using pure implementation
        history = self.fit_pure(X_np, y_np)
        
        # Sync PyTorch parameters after training
        self._sync_pytorch_params()
        
        return history
    
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
        """
        Get comprehensive model information for wandb integration.
        
        Returns:
            Dictionary containing detailed model metadata and current state,
            following the standardized structure across all models.
        """
        pure_info = self.pure_mlp.get_model_info()
        
        return {
            # Core identification
            "name": "MLP",
            "full_name": "Multi-Layer Perceptron",
            "category": "foundation",
            "module": 1,
            "pattern": "engine-based",
            
            # Historical context
            "year_introduced": YEAR_INTRODUCED,
            "authors": AUTHORS,
            "paper_title": "Perceptrons: An Introduction to Computational Geometry",
            "key_innovations": [
                "First neural network capable of universal function approximation",
                "Solved XOR problem that single-layer perceptrons cannot handle",
                "Enabled learning of hierarchical feature representations",
                "Backpropagation learning algorithm",
                "Multi-layer gradient-based optimization"
            ],
            
            # Architecture details
            "architecture_type": "multi-layer-feedforward",
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "total_layers": len(self.hidden_layers) + 1,  # +1 for output layer
            "parameter_count": pure_info.get("total_parameters", self._count_total_parameters()),
            "trainable_parameters": pure_info.get("total_parameters", self._count_total_parameters()),
            "activation_function": self.activation,
            "weight_initialization": "xavier_normal",
            "has_bias": True,
            
            # Training characteristics
            "learning_algorithm": "backpropagation",
            "loss_function": "binary-cross-entropy" if self.output_size == 1 else "cross-entropy",
            "optimizer": "sgd",
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "convergence_tolerance": self.tolerance,
            "early_stopping": True,
            "batch_processing": False,  # Full batch for educational clarity
            
            # Implementation details
            "framework": "numpy+pytorch",
            "core_implementation": "pure-numpy-backpropagation",
            "wrapper_framework": "pytorch",
            "precision": "float32",
            "device": str(next(self.parameters()).device),
            "engine_compatible": True,
            "wandb_integration": True,
            
            # Capabilities and characteristics
            "can_solve_xor": len(self.hidden_layers) > 0 and all(h > 0 for h in self.hidden_layers),
            "is_nonlinear": True,
            "universal_approximator": True,
            "handles_non_linearly_separable": True,
            "gradient_based": True,
            "supports_backpropagation": True,
            
            # Educational metadata
            "difficulty_level": "intermediate",
            "estimated_training_time": "1-5 minutes",
            "key_learning_objectives": [
                "Understand backpropagation algorithm",
                "See gradient flow through hidden layers",
                "Witness XOR breakthrough moment",
                "Learn non-linear pattern recognition",
                "Grasp universal approximation theorem"
            ],
            "breakthrough_significance": "Solved AI Winter problem - first network to handle non-linear patterns",
            
            # Current state
            "is_fitted": pure_info.get("is_fitted", False),
            "training_history": self.training_history,
            "weights_info": {
                "layer_count": len(self.hidden_layers) + 1,
                "weight_matrices": len(pure_info.get("weights", [])),
                "bias_vectors": len(pure_info.get("biases", []))
            },
            
            # Configuration used
            "training_config": {
                "input_size": self.input_size,
                "hidden_layers": self.hidden_layers,
                "output_size": self.output_size,
                "activation": self.activation,
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "tolerance": self.tolerance
            }
        }
    
    def _count_total_parameters(self) -> int:
        """Count total parameters in the model."""
        if hasattr(self.pure_mlp, '_count_parameters'):
            return self.pure_mlp._count_parameters()
        return sum(p.numel() for p in self.parameters())
    
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
        print("Algorithm: Backpropagation (Rumelhart, Hinton, Williams 1986)")
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

    def fit_with_wandb(self, X: np.ndarray, y: np.ndarray, config: 'MLPExperimentConfig', 
                       verbose: bool = False) -> Dict[str, Any]:
        """
        Train MLP with comprehensive wandb tracking.
        
        This method demonstrates the educational power of MLP wandb integration,
        including backpropagation visualization, XOR breakthrough tracking,
        and gradient flow analysis.
        
        Args:
            X: Training features
            y: Training labels
            config: MLPExperimentConfig with wandb settings
            verbose: Show detailed training progress
            
        Returns:
            Training results with wandb tracking
        """
        # Initialize wandb with MLP-specific configuration
        if config.use_wandb:
            success = self.init_wandb(
                project=config.wandb_project,
                name=config.wandb_name,
                tags=config.wandb_tags,
                config=config.__dict__,
                notes=config.wandb_notes,
                mode=config.wandb_mode
            )
            
            if success:
                self.logger.info("✅ MLP wandb tracking enabled")
                
                # Log initial model architecture
                model_info = self.get_model_info()
                architecture_metrics = {
                    "model/input_size": self.input_size,
                    "model/hidden_layers": len(self.hidden_layers),
                    "model/first_hidden_size": self.hidden_layers[0] if self.hidden_layers else 0,
                    "model/output_size": self.output_size,
                    "model/total_parameters": model_info.get("total_parameters", 0),
                    "model/can_solve_xor": int(model_info.get("can_solve_xor", False)),  # Convert bool to int
                    "config/learning_rate": config.learning_rate,
                    "config/max_epochs": config.max_epochs
                }
                self.log_metrics(architecture_metrics, step=0)
                
                # Enable model watching for gradient visualization
                if config.wandb_watch_model:
                    self.watch_model(
                        log=config.wandb_watch_log,
                        log_freq=config.wandb_watch_freq
                    )
                    self.logger.info("📊 MLP gradient watching enabled")
            else:
                self.logger.warning("⚠️ Wandb setup failed, training without tracking")
        
        # XOR breakthrough detection
        is_xor_problem = "xor" in config.dataset_type.lower()
        if is_xor_problem:
            self.logger.info("🎯 XOR Problem Detected - Tracking breakthrough moment!")
        
        # Enhanced training with epoch-by-epoch wandb logging
        self.logger.info("Training MLP with enhanced wandb tracking")
        
        # Initialize training history
        self.pure_mlp.training_history = {
            'loss': [],
            'accuracy': [],
            'epoch': [],
            'converged': False,
            'convergence_epoch': None,
            'final_weights': [],
            'weight_updates': []
        }
        
        # Prepare data
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Training loop with detailed wandb logging
        for epoch in range(config.max_epochs):
            # Store weights for change analysis
            old_weights = [W.copy() for W in self.pure_mlp.weights]
            
            # Training step
            epoch_loss = 0
            epoch_accuracy = 0
            n_samples = X.shape[0]
            
            for i in range(n_samples):
                x_i = X[i:i+1]
                y_i = y[i:i+1]
                
                # Forward pass with activations
                output, activations, z_values = self.pure_mlp.forward(x_i)
                
                # Compute sample loss
                sample_loss = self.pure_mlp._compute_loss(y_i, output)
                sample_accuracy = self.pure_mlp._compute_accuracy(y_i, output)
                epoch_loss += sample_loss
                epoch_accuracy += sample_accuracy
                
                # Backpropagation step
                self.pure_mlp._backpropagate(activations, z_values, y_i, verbose=False)
            
            # Average metrics
            epoch_loss /= n_samples
            epoch_accuracy /= n_samples
            
            # Calculate weight changes for educational analysis
            weight_changes = []
            for old_W, new_W in zip(old_weights, self.pure_mlp.weights):
                weight_change = np.mean(np.abs(new_W - old_W))
                weight_changes.append(weight_change)
            
            # Log comprehensive metrics to wandb
            if config.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
                metrics = {
                    "train/loss": float(epoch_loss),
                    "train/accuracy": float(epoch_accuracy),
                    "train/error_rate": float(1.0 - epoch_accuracy),
                    "epoch": epoch
                }
                
                # Layer-specific metrics for educational value
                if config.wandb_log_layer_activations and epoch % 10 == 0:
                    # Sample forward pass for activation analysis
                    _, sample_activations, _ = self.pure_mlp.forward(X[:1])
                    for layer_idx, activation in enumerate(sample_activations[1:]):  # Skip input
                        metrics[f"activations/layer_{layer_idx}_mean"] = float(np.mean(activation))
                        metrics[f"activations/layer_{layer_idx}_std"] = float(np.std(activation))
                
                # Weight change analysis
                if config.wandb_log_weight_histograms:
                    for layer_idx, change in enumerate(weight_changes):
                        metrics[f"weights/layer_{layer_idx}_change"] = float(change)
                        metrics[f"weights/layer_{layer_idx}_mean"] = float(np.mean(self.pure_mlp.weights[layer_idx]))
                        metrics[f"weights/layer_{layer_idx}_std"] = float(np.std(self.pure_mlp.weights[layer_idx]))
                
                # XOR breakthrough tracking
                if is_xor_problem and config.wandb_log_xor_breakthrough:
                    if epoch_accuracy > 0.9:  # Close to solving XOR
                        metrics["xor/near_breakthrough"] = 1
                    if epoch_accuracy >= 0.99:  # XOR solved!
                        metrics["xor/breakthrough_achieved"] = 1
                        if epoch < 50:
                            metrics["xor/fast_breakthrough"] = 1
                
                # Learning dynamics
                if epoch > 0:
                    prev_loss = self.pure_mlp.training_history.get('loss', [float('inf')])[-1] if hasattr(self.pure_mlp, 'training_history') else float('inf')
                    metrics["learning/loss_change"] = float(epoch_loss - prev_loss)
                    metrics["learning/is_improving"] = float(epoch_loss < prev_loss)
                
                self.log_metrics(metrics, step=epoch + 1)
            
            # Console logging
            if verbose or (epoch % max(1, config.max_epochs // 20) == 0):
                self.logger.info(f"Epoch {epoch:4d}: Loss = {epoch_loss:.6f}, Accuracy = {epoch_accuracy:.4f}")
                
                # XOR-specific progress messages
                if is_xor_problem:
                    if epoch_accuracy > 0.9:
                        self.logger.info("🔥 Approaching XOR breakthrough!")
                    if epoch_accuracy >= 0.99:
                        self.logger.info("🎉 XOR BREAKTHROUGH ACHIEVED!")
            
            # Store in pure MLP history
            self.pure_mlp.training_history['loss'].append(epoch_loss)
            self.pure_mlp.training_history['accuracy'].append(epoch_accuracy)
            self.pure_mlp.training_history['epoch'].append(epoch)
            
            # Convergence check
            if epoch_loss < config.convergence_threshold:
                self.logger.info(f"✅ Converged at epoch {epoch}")
                break
        
        # Sync PyTorch parameters
        self._sync_pytorch_params()
        
        # Final results and wandb logging
        self.pure_mlp.is_fitted = True
        final_results = {
            "converged": epoch_loss < config.convergence_threshold,
            "final_loss": float(epoch_loss),
            "final_train_accuracy": float(epoch_accuracy),
            "epochs_trained": epoch + 1,
            "xor_solved": is_xor_problem and epoch_accuracy >= 0.99,
            "breakthrough_epoch": epoch + 1 if is_xor_problem and epoch_accuracy >= 0.99 else None,
            "convergence_epoch": epoch + 1 if epoch_loss < config.convergence_threshold else None
        }
        
        # Log final metrics
        if config.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
            final_metrics = {
                "final/loss": final_results["final_loss"],
                "final/accuracy": final_results["final_train_accuracy"],
                "final/converged": int(final_results["converged"]),  # Convert bool to int
                "final/epochs_trained": final_results["epochs_trained"]
            }
            
            if is_xor_problem:
                final_metrics["final/xor_solved"] = int(final_results["xor_solved"])  # Convert bool to int
                if final_results["breakthrough_epoch"]:
                    final_metrics["final/breakthrough_epoch"] = final_results["breakthrough_epoch"]
            
            self.log_metrics(final_metrics)  # Final summary metrics - no step needed
            
            # Finish wandb run
            self.finish_wandb()
            self.logger.info("📊 MLP wandb tracking completed")
        
        self.training_history = final_results
        return final_results


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