"""
Multi-Layer Perceptron (MLP) implementation from scratch.

This module implements a multi-layer perceptron capable of solving non-linearly
separable problems like XOR, overcoming the fundamental limitations of
single-layer perceptrons through the use of hidden layers and backpropagation.
"""

import os
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layer Perceptron implementation with configurable architecture.
    
    This implementation demonstrates the breakthrough that enabled neural networks
    to solve non-linearly separable problems through hidden layers and
    backpropagation optimization.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_layers: List[int],
                 output_size: int,
                 activation: str = "sigmoid",
                 weight_init: str = "xavier_normal",
                 device: str = "cpu"):
        """
        Initialize the Multi-Layer Perceptron.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output units
            activation: Activation function ('sigmoid', 'tanh', 'relu', 'leaky_relu')
            weight_init: Weight initialization method
            device: Device to run on ('cpu' or 'cuda')
        """
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.device = device
        
        # Build the network layers
        self.layers = nn.ModuleList()
        
        # Input to first hidden layer
        if hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            
            # Hidden to hidden layers
            for i in range(1, len(hidden_layers)):
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            
            # Last hidden to output
            self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        else:
            # Direct input to output (reduces to perceptron)
            self.layers.append(nn.Linear(input_size, output_size))
        
        # Set activation function
        self.activation = self._get_activation_function(activation)
        
        # Initialize weights
        self._initialize_weights(weight_init)
        
        # Training history
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "epoch": [],
            "converged": False,
            "convergence_epoch": None
        }
        
        # Move to device
        self.to(device)
        
        logging.info(f"Initialized MLP: {input_size} -> {hidden_layers} -> {output_size}")
        logging.info(f"Activation: {activation}, Weight init: {weight_init}")
        logging.info(f"Total parameters: {self._count_parameters()}")
    
    def _get_activation_function(self, activation: str) -> Callable:
        """Get the activation function."""
        activations = {
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": F.relu,
            "leaky_relu": lambda x: F.leaky_relu(x, 0.01)
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self, method: str):
        """Initialize network weights using specified method."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if method == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif method == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif method == "he_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif method == "he_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif method == "random_normal":
                    nn.init.normal_(layer.weight, 0, 0.1)
                elif method == "zeros":
                    nn.init.zeros_(layer.weight)
                else:
                    raise ValueError(f"Unknown weight initialization: {method}")
                
                # Initialize biases to zero
                nn.init.zeros_(layer.bias)
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Forward through all layers except the last
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation for regression/binary classification)
        x = self.layers[-1](x)
        
        return x
    
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
    
    def train_model(self, 
                    x_train: torch.Tensor,
                    y_train: torch.Tensor,
                    x_test: Optional[torch.Tensor] = None,
                    y_test: Optional[torch.Tensor] = None,
                    learning_rate: float = 0.1,
                    max_epochs: int = 1000,
                    convergence_threshold: float = 1e-6,
                    patience: int = 50,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Train the MLP using backpropagation.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            x_test: Test input data (optional)
            y_test: Test target data (optional)
            learning_rate: Learning rate for optimization
            max_epochs: Maximum number of training epochs
            convergence_threshold: Loss threshold for convergence
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training results
        """
        self.train()
        
        # Setup optimizer and loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
        if self.output_size == 1:
            # Binary classification
            criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Forward pass
            outputs = self.forward(x_train)
            
            # Compute loss
            if self.output_size == 1:
                loss = criterion(outputs, y_train.unsqueeze(1))
            else:
                loss = criterion(outputs, y_train.long())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            train_acc = self._compute_accuracy(x_train, y_train)
            
            # Store history
            self.training_history["loss"].append(loss.item())
            self.training_history["accuracy"].append(train_acc)
            self.training_history["epoch"].append(epoch)
            
            # Check convergence
            if loss.item() < convergence_threshold:
                self.training_history["converged"] = True
                self.training_history["convergence_epoch"] = epoch
                if verbose:
                    print(f"Converged at epoch {epoch} with loss {loss.item():.6f}")
                break
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress logging
            if verbose and (epoch % 100 == 0 or epoch < 10):
                test_info = ""
                if x_test is not None and y_test is not None:
                    test_acc = self._compute_accuracy(x_test, y_test)
                    test_info = f", Test Acc: {test_acc:.4f}"
                
                print(f"Epoch {epoch:4d}: Loss: {loss.item():.6f}, "
                      f"Train Acc: {train_acc:.4f}{test_info}")
        
        # Final evaluation
        final_train_acc = self._compute_accuracy(x_train, y_train)
        results = {
            "final_loss": loss.item(),
            "final_train_accuracy": final_train_acc,
            "epochs_trained": epoch + 1,
            "converged": self.training_history["converged"],
            "convergence_epoch": self.training_history["convergence_epoch"]
        }
        
        if x_test is not None and y_test is not None:
            final_test_acc = self._compute_accuracy(x_test, y_test)
            results["final_test_accuracy"] = final_test_acc
        
        if verbose:
            print("\nTraining completed!")
            print(f"Final loss: {results['final_loss']:.6f}")
            print(f"Final train accuracy: {results['final_train_accuracy']:.4f}")
            if "final_test_accuracy" in results:
                print(f"Final test accuracy: {results['final_test_accuracy']:.4f}")
        
        return results
    
    def _compute_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy."""
        self.eval()
        with torch.no_grad():
            predictions = self.predict(x)
            if self.output_size == 1:
                accuracy = (predictions.squeeze() == y).float().mean().item()
            else:
                accuracy = (predictions == y).float().mean().item()
        self.train()
        return accuracy
    
    def get_hidden_representations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get hidden layer representations for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of hidden layer activations
        """
        self.eval()
        activations = []
        
        with torch.no_grad():
            current = x
            for layer in self.layers[:-1]:  # All layers except output
                current = layer(current)
                current = self.activation(current)
                activations.append(current.clone())
        
        return activations
    
    def save_model(self, filepath: str, include_history: bool = True):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            include_history: Whether to include training history
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "architecture": {
                "input_size": self.input_size,
                "hidden_layers": self.hidden_layers,
                "output_size": self.output_size,
                "activation": self.activation_name,
                "weight_init": self.weight_init
            },
            "device": self.device
        }
        
        if include_history:
            save_dict["training_history"] = self.training_history
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(save_dict, filepath)
        logging.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = "cpu") -> "MLP":
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded MLP model
        """
        save_dict = torch.load(filepath, map_location=device)
        
        # Recreate model
        arch = save_dict["architecture"]
        model = cls(
            input_size=arch["input_size"],
            hidden_layers=arch["hidden_layers"],
            output_size=arch["output_size"],
            activation=arch["activation"],
            weight_init=arch["weight_init"],
            device=device
        )
        
        # Load weights
        model.load_state_dict(save_dict["model_state_dict"])
        
        # Load training history if available
        if "training_history" in save_dict:
            model.training_history = save_dict["training_history"]
        
        logging.info(f"Model loaded from {filepath}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture and training."""
        return {
            "architecture": {
                "input_size": self.input_size,
                "hidden_layers": self.hidden_layers,
                "output_size": self.output_size,
                "total_parameters": self._count_parameters()
            },
            "configuration": {
                "activation": self.activation_name,
                "weight_init": self.weight_init,
                "device": self.device
            },
            "training_status": {
                "epochs_trained": len(self.training_history["epoch"]),
                "converged": self.training_history["converged"],
                "convergence_epoch": self.training_history["convergence_epoch"],
                "final_loss": self.training_history["loss"][-1] if self.training_history["loss"] else None,
                "final_accuracy": self.training_history["accuracy"][-1] if self.training_history["accuracy"] else None
            }
        }
    
    def visualize_decision_boundary(self, x: torch.Tensor, y: torch.Tensor, 
                                   resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data for visualizing the decision boundary (for 2D input).
        
        Args:
            x: Input data (should be 2D)
            y: Target labels
            resolution: Grid resolution for boundary visualization
            
        Returns:
            Tuple of (X_grid, Y_grid, Z_predictions) for plotting
        """
        if x.shape[1] != 2:
            raise ValueError("Decision boundary visualization only supports 2D input")
        
        self.eval()
        
        # Create grid
        x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
        y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Make predictions on grid
        grid_points = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], 
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            if self.output_size == 1:
                predictions = torch.sigmoid(self.forward(grid_points))
                Z = predictions.cpu().numpy().reshape(xx.shape)
            else:
                outputs = self.forward(grid_points)
                predictions = F.softmax(outputs, dim=1)
                Z = predictions[:, 1].cpu().numpy().reshape(xx.shape)
        
        return xx, yy, Z
    
    def __repr__(self) -> str:
        """String representation of the model."""
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        architecture = " -> ".join(map(str, layer_sizes))
        return f"MLP({architecture}, activation={self.activation_name})" 