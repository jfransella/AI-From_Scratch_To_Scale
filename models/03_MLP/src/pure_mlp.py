"""
Pure NumPy Multi-Layer Perceptron with Backpropagation.

This module implements the classic backpropagation algorithm from scratch using
only NumPy, allowing students to see the exact mathematical operations that
enable neural networks to learn complex, non-linear patterns.

Educational Focus:
- Visible forward propagation through hidden layers
- Step-by-step backpropagation with chain rule applications
- Weight update mechanics for each layer
- The XOR breakthrough moment when hidden layers solve non-linear problems

Historical Context:
The backpropagation algorithm, popularized by Rumelhart, Hinton, and Williams (1986),
was the key breakthrough that made multi-layer neural networks practical by
providing an efficient way to compute gradients for all layers.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import json
from pathlib import Path

from utils import get_logger, set_random_seed

try:
    from .constants import MODEL_NAME, YEAR_INTRODUCED, AUTHORS
except ImportError:
    from constants import MODEL_NAME, YEAR_INTRODUCED, AUTHORS


class PureMLP:
    """
    Pure NumPy Multi-Layer Perceptron with visible backpropagation.
    
    This implementation prioritizes educational clarity over efficiency,
    making every step of the backpropagation algorithm visible to students.
    
    Key Features:
    - Forward propagation through arbitrary hidden layers
    - Backpropagation with visible gradient computations
    - Support for sigmoid, tanh, and ReLU activations
    - Step-by-step training demonstrations
    - XOR problem solving capability
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
        """
        Initialize Pure NumPy MLP.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes [h1, h2, ...]
            output_size: Number of output units
            activation: Activation function ('sigmoid', 'tanh', 'relu')
            learning_rate: Learning rate for weight updates
            max_epochs: Maximum training epochs
            tolerance: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Training state
        self.is_fitted = False
        self.training_history = {}
        
        # Logger
        self.logger = get_logger(__name__)
        
        # Set random seed
        if random_state is not None:
            set_random_seed(random_state)
            np.random.seed(random_state)
        
        # Build network architecture
        self.layers = []
        self.weights = []
        self.biases = []
        
        # Create layer structure: input -> hidden -> ... -> output
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layer_info = {
                'input_size': layer_sizes[i],
                'output_size': layer_sizes[i + 1],
                'is_output': i == len(layer_sizes) - 2
            }
            self.layers.append(layer_info)
        
        # Initialize weights and biases
        self._initialize_weights()
        
        self.logger.info(f"Initialized Pure MLP: {input_size} -> {hidden_layers} -> {output_size}")
        self.logger.info(f"Activation: {activation}, Learning rate: {learning_rate}")
        self.logger.info(f"Total parameters: {self._count_parameters()}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        self.weights = []
        self.biases = []
        
        for layer in self.layers:
            # Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
            fan_in = layer['input_size']
            fan_out = layer['output_size']
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            
            # Initialize weights
            W = np.random.normal(0, scale, (fan_in, fan_out))
            self.weights.append(W)
            
            # Initialize biases to small positive values
            b = np.zeros((1, fan_out))
            self.biases.append(b)
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "sigmoid":
            return self._sigmoid(x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute activation function derivative."""
        if self.activation == "sigmoid":
            s = self._sigmoid(x)
            return s * (1 - s)
        elif self.activation == "tanh":
            t = np.tanh(x)
            return 1 - t**2
        elif self.activation == "relu":
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network.
        
        This method returns intermediate activations and pre-activations
        for use in backpropagation, making the forward pass visible.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            output: Final output
            activations: List of activations for each layer (including input)
            z_values: List of pre-activation values for backprop
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Store activations and pre-activations for backprop
        activations = [X]  # Start with input
        z_values = []      # Pre-activation values
        
        current_input = X
        
        # Forward through each layer
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation: z = X @ W + b
            z = current_input @ W + b
            z_values.append(z)
            
            # Apply activation (except for output layer which might be linear)
            if i < len(self.weights) - 1:  # Hidden layers
                a = self._activation_function(z)
            else:  # Output layer - keep linear for regression, sigmoid for classification
                if self.output_size == 1:
                    a = self._sigmoid(z)  # Binary classification
                else:
                    a = z  # Multi-class or regression
            
            activations.append(a)
            current_input = a
        
        return activations[-1], activations, z_values
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        output, _, _ = self.forward(X)
        return output
    
    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions by thresholding at 0.5."""
        predictions = self.predict(X)
        return (predictions >= 0.5).astype(int)
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary classification accuracy."""
        binary_pred = (y_pred >= 0.5).astype(int)
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.squeeze()
        if binary_pred.ndim == 2 and binary_pred.shape[1] == 1:
            binary_pred = binary_pred.squeeze()
        return np.mean(y_true == binary_pred)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
        """
        Train the MLP using backpropagation.
        
        This implementation shows the complete backpropagation algorithm,
        making gradient computation and weight updates visible to students.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Show training progress
            
        Returns:
            Training history and results
        """
        # Prepare data
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples = X.shape[0]
        
        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'epoch': [],
            'converged': False,
            'convergence_epoch': None,
            'final_weights': [],
            'weight_updates': []
        }
        
        if verbose:
            print("🧠 MLP Training with Backpropagation")
            print("=" * 45)
            print(f"Architecture: {self.input_size} -> {self.hidden_layers} -> {self.output_size}")
            print(f"Training samples: {n_samples}")
            print()
        
        # Training loop
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            
            # For each training sample (stochastic gradient descent)
            for i in range(n_samples):
                x_i = X[i:i+1]  # Keep batch dimension
                y_i = y[i:i+1]
                
                # FORWARD PASS
                output, activations, z_values = self.forward(x_i)
                
                # Compute loss and accuracy
                sample_loss = self._compute_loss(y_i, output)
                sample_accuracy = self._compute_accuracy(y_i, output)
                epoch_loss += sample_loss
                epoch_accuracy += sample_accuracy
                
                # BACKWARD PASS (The Heart of Backpropagation!)
                self._backpropagate(activations, z_values, y_i, verbose and epoch < 3)
            
            # Average metrics over epoch
            epoch_loss /= n_samples
            epoch_accuracy /= n_samples
            
            # Store history
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            history['epoch'].append(epoch)
            
            # Check convergence
            if epoch_loss < self.tolerance:
                history['converged'] = True
                history['convergence_epoch'] = epoch
                if verbose:
                    print(f"✅ Converged at epoch {epoch}")
                break
            
            # Progress reporting
            if verbose and (epoch % max(1, self.max_epochs // 10) == 0 or epoch < 5):
                print(f"Epoch {epoch:4d}: Loss = {epoch_loss:.6f}, Accuracy = {epoch_accuracy:.4f}")
        
        # Store final state
        history['final_weights'] = [W.copy() for W in self.weights]
        history['final_loss'] = epoch_loss
        history['final_train_accuracy'] = epoch_accuracy
        history['epochs_trained'] = epoch + 1
        self.is_fitted = True
        self.training_history = history
        
        if verbose:
            print(f"\n✨ Training Complete!")
            print(f"Final Loss: {epoch_loss:.6f}")
            print(f"Final Accuracy: {epoch_accuracy:.4f}")
            print(f"Epochs trained: {epoch + 1}")
        
        return history
    
    def _backpropagate(self, activations: List[np.ndarray], z_values: List[np.ndarray], 
                      y_true: np.ndarray, verbose: bool = False):
        """
        The Backpropagation Algorithm - Made Visible!
        
        This method implements the chain rule to compute gradients for all layers,
        working backwards from output to input.
        
        Args:
            activations: Forward pass activations
            z_values: Pre-activation values
            y_true: True target values
            verbose: Show detailed gradient computations
        """
        if verbose:
            print("\n🔄 BACKPROPAGATION STEP:")
            print("-" * 30)
        
        # Start with output layer error
        output = activations[-1]
        
        # Output layer gradient (dL/da for output layer)
        # For MSE loss: dL/da = 2 * (a - y) / n
        output_error = output - y_true  # Shape: (1, output_size)
        
        if verbose:
            print(f"Output Error: {output_error.flatten()}")
        
        # Backpropagate through layers (from last to first)
        deltas = []
        
        for layer_idx in range(len(self.weights) - 1, -1, -1):
            if layer_idx == len(self.weights) - 1:
                # Output layer
                if self.output_size == 1:
                    # Binary classification with sigmoid
                    # dL/dz = dL/da * da/dz = error * sigmoid'(z)
                    z = z_values[layer_idx]
                    sigmoid_deriv = self._activation_derivative(z)
                    delta = output_error * sigmoid_deriv
                else:
                    # Multi-class or regression
                    delta = output_error
            else:
                # Hidden layers
                # dL/dz = (dL/dz_next @ W_next.T) * activation'(z)
                z = z_values[layer_idx]
                activation_deriv = self._activation_derivative(z)
                delta = (deltas[0] @ self.weights[layer_idx + 1].T) * activation_deriv
            
            deltas.insert(0, delta)  # Insert at beginning
            
            if verbose:
                print(f"Layer {layer_idx} delta: {delta.flatten()}")
        
        # Update weights and biases using gradients
        for layer_idx in range(len(self.weights)):
            # dL/dW = activation_input.T @ delta
            # dL/db = delta
            
            activation_input = activations[layer_idx]  # Input to this layer
            delta = deltas[layer_idx]
            
            # Compute gradients
            W_gradient = activation_input.T @ delta
            b_gradient = delta.mean(axis=0, keepdims=True)  # Average over batch
            
            # Update weights (Gradient Descent)
            self.weights[layer_idx] -= self.learning_rate * W_gradient
            self.biases[layer_idx] -= self.learning_rate * b_gradient
            
            if verbose:
                print(f"Layer {layer_idx} weight update: {np.mean(np.abs(W_gradient)):.6f}")
    
    def demonstrate_xor_learning(self, max_epochs: int = 100):
        """
        Educational demonstration of MLP solving the XOR problem.
        
        This shows the key breakthrough moment when hidden layers
        enable neural networks to solve non-linearly separable problems.
        """
        print("🎯 MLP vs XOR Problem - The Breakthrough Moment!")
        print("=" * 55)
        print("This is what single-layer perceptrons CANNOT solve...")
        print()
        
        # XOR dataset
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_xor = np.array([[0], [1], [1], [0]], dtype=float)
        
        print("XOR Truth Table:")
        for i, (x, y) in enumerate(zip(X_xor, y_xor)):
            print(f"  {x} -> {y[0]}")
        print()
        
        # Train on XOR
        print("🏃 Training MLP to solve XOR...")
        self.max_epochs = max_epochs
        history = self.fit(X_xor, y_xor, verbose=False)
        
        # Test predictions
        predictions = self.predict(X_xor)
        binary_preds = self.predict_binary(X_xor)
        
        print("\n📊 Results:")
        print("-" * 20)
        for i, (x, y_true, y_pred, y_bin) in enumerate(zip(X_xor, y_xor, predictions, binary_preds)):
            print(f"Input: {x} | True: {y_true[0]} | Pred: {y_pred[0]:.3f} | Binary: {y_bin[0]}")
        
        final_accuracy = self._compute_accuracy(y_xor, predictions)
        print(f"\nFinal Accuracy: {final_accuracy:.2%}")
        
        if final_accuracy > 0.9:
            print("🎉 SUCCESS! MLP solved XOR - hidden layers work!")
        else:
            print("😞 Not quite there - try more epochs or different architecture")
        
        print(f"\n🧠 Key Insight:")
        print("   Hidden layers create internal representations that")
        print("   transform the input space to make problems linearly separable!")
        
        return history
    
    def get_decision_boundary_info(self) -> Dict[str, Any]:
        """Get information about the decision boundary for visualization."""
        return {
            'model_type': 'MLP',
            'architecture': f"{self.input_size}-{'-'.join(map(str, self.hidden_layers))}-{self.output_size}",
            'activation': self.activation,
            'is_nonlinear': True,
            'can_solve_xor': len(self.hidden_layers) > 0 and all(h > 0 for h in self.hidden_layers),
            'weights': [W.tolist() for W in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': MODEL_NAME,
            'year_introduced': YEAR_INTRODUCED,
            'authors': AUTHORS,
            'algorithm': 'Backpropagation',
            'implementation': 'Pure NumPy',
            
            # Architecture
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'activation': self.activation,
            'total_parameters': self._count_parameters(),
            
            # Training config
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'tolerance': self.tolerance,
            'is_fitted': self.is_fitted,
            
            # Current state
            'weights': [W.tolist() for W in self.weights] if self.weights else [],
            'biases': [b.tolist() for b in self.biases] if self.biases else [],
            'training_history': self.training_history
        }
    
    def save_model(self, filepath: str):
        """Save model to JSON file."""
        model_data = {
            'config': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'activation': self.activation,
                'learning_rate': self.learning_rate,
                'max_epochs': self.max_epochs,
                'tolerance': self.tolerance,
                'random_state': self.random_state
            },
            'state': {
                'weights': [W.tolist() for W in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'is_fitted': self.is_fitted,
                'training_history': self.training_history
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"Saved Pure MLP to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> "PureMLP":
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        config = model_data['config']
        model = cls(**config)
        
        # Restore state
        state = model_data['state']
        model.weights = [np.array(W) for W in state['weights']]
        model.biases = [np.array(b) for b in state['biases']]
        model.is_fitted = state['is_fitted']
        model.training_history = state['training_history']
        
        return model


def create_pure_mlp(input_size: int, hidden_layers: List[int], **kwargs) -> PureMLP:
    """
    Factory function to create a Pure NumPy MLP.
    
    This is the recommended way to create MLPs that show
    the backpropagation algorithm clearly to students.
    """
    return PureMLP(input_size=input_size, hidden_layers=hidden_layers, **kwargs)


def demonstrate_backpropagation():
    """
    Educational demonstration of backpropagation on XOR problem.
    
    This function provides a complete walkthrough of how MLPs
    overcome the limitations of single-layer perceptrons.
    """
    print("🎓 BACKPROPAGATION ALGORITHM DEMONSTRATION")
    print("=" * 50)
    print("Solving XOR: The Problem Perceptrons Cannot Handle")
    print()
    
    # Create a small MLP for clear visualization
    mlp = create_pure_mlp(
        input_size=2,
        hidden_layers=[3],  # Small hidden layer for clarity
        output_size=1,
        activation='sigmoid',
        learning_rate=1.0,  # Higher LR for faster demo
        max_epochs=50
    )
    
    # Demonstrate the learning process
    history = mlp.demonstrate_xor_learning(max_epochs=50)
    
    # Show architecture details
    print(f"\n🏗️  Network Architecture:")
    print(f"   Input Layer: {mlp.input_size} units")
    for i, h in enumerate(mlp.hidden_layers):
        print(f"   Hidden Layer {i+1}: {h} units ({mlp.activation})")
    print(f"   Output Layer: {mlp.output_size} unit (sigmoid)")
    print(f"   Total Parameters: {mlp._count_parameters()}")
    
    # Show final decision boundary info
    boundary_info = mlp.get_decision_boundary_info()
    print(f"\n✨ Decision Boundary: {boundary_info['architecture']}")
    print(f"   Can solve XOR: {boundary_info['can_solve_xor']}")
    
    return mlp, history


if __name__ == "__main__":
    # Run the educational demonstration
    demonstrate_backpropagation() 