"""
Base classes and interfaces for the AI From Scratch to Scale training engine.

This module defines abstract base classes and data structures that provide
a unified interface for training and evaluating different model architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn


@dataclass
class TrainingResult:
    """Results from a training session."""
    
    # Training metrics
    final_loss: float
    final_train_accuracy: float
    final_val_accuracy: Optional[float] = None
    final_test_accuracy: Optional[float] = None
    
    # Training progress
    epochs_trained: int = 0
    total_training_time: float = 0.0
    converged: bool = False
    convergence_epoch: Optional[int] = None
    
    # History tracking
    loss_history: List[float] = field(default_factory=list)
    train_accuracy_history: List[float] = field(default_factory=list)
    val_accuracy_history: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # Experiment metadata
    experiment_name: str = "unnamed"
    model_architecture: str = "unknown"
    dataset_name: str = "unknown"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Model artifacts
    best_model_path: Optional[str] = None
    final_model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_loss": self.final_loss,
            "final_train_accuracy": self.final_train_accuracy,
            "final_val_accuracy": self.final_val_accuracy,
            "final_test_accuracy": self.final_test_accuracy,
            "epochs_trained": self.epochs_trained,
            "total_training_time": self.total_training_time,
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
            "loss_history": self.loss_history,
            "train_accuracy_history": self.train_accuracy_history,
            "val_accuracy_history": self.val_accuracy_history,
            "epoch_times": self.epoch_times,
            "experiment_name": self.experiment_name,
            "model_architecture": self.model_architecture,
            "dataset_name": self.dataset_name,
            "hyperparameters": self.hyperparameters,
            "best_model_path": self.best_model_path,
            "final_model_path": self.final_model_path
        }


@dataclass  
class EvaluationResult:
    """Results from model evaluation."""
    
    # Core metrics
    accuracy: float
    loss: float
    
    # Additional metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Per-class metrics (for multi-class)
    per_class_accuracy: Optional[Dict[str, float]] = None
    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None
    
    # Prediction details
    predictions: Optional[List[Any]] = None
    probabilities: Optional[List[List[float]]] = None
    ground_truth: Optional[List[Any]] = None
    
    # Evaluation metadata
    num_samples: int = 0
    evaluation_time: float = 0.0
    model_name: str = "unknown"
    dataset_name: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "loss": self.loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            # Note: predictions, probabilities, ground_truth not included 
            # in dict by default to avoid large serialization
        }


class BaseModel(ABC):
    """
    Abstract base class for all models in the AI From Scratch to Scale project.
    
    This interface ensures that all models can be used with the unified
    training and evaluation engine, regardless of their specific implementation.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture and status."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save the model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load_model(cls, filepath: str) -> "BaseModel":
        """Load a model from disk."""
        pass
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities (optional, default implementation).
        
        Models can override this if they support probability outputs.
        """
        # Default: return predictions as one-hot probabilities
        predictions = self.predict(x)
        if predictions.dim() == 1:
            # Binary classification
            probs = torch.zeros(predictions.shape[0], 2)
            probs[range(len(predictions)), predictions.long()] = 1.0
            return probs
        else:
            # Multi-class: assume predictions are already probabilities
            return predictions
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss (optional, default implementation).
        
        Models can override this for custom loss functions.
        """
        if outputs.shape[1] == 1:
            # Binary classification
            criterion = nn.BCEWithLogitsLoss()
            return criterion(outputs, targets.unsqueeze(1))
        else:
            # Multi-class classification
            criterion = nn.CrossEntropyLoss()
            return criterion(outputs, targets.long())
    
    def to(self, device: str):
        """Move model to device (default implementation for PyTorch models)."""
        if hasattr(self, 'device'):
            self.device = device
        if isinstance(self, nn.Module):
            return super().to(device)
        return self
    
    def train(self):
        """Set model to training mode (default implementation)."""
        if isinstance(self, nn.Module):
            return super().train()
        return self
    
    def eval(self):
        """Set model to evaluation mode (default implementation)."""
        if isinstance(self, nn.Module):
            return super().eval()
        return self


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: Optional[torch.Tensor] = None
    y_val: Optional[torch.Tensor] = None
    x_test: Optional[torch.Tensor] = None
    y_test: Optional[torch.Tensor] = None
    
    def to_device(self, device: str) -> "DataSplit":
        """Move all tensors to specified device."""
        return DataSplit(
            x_train=self.x_train.to(device),
            y_train=self.y_train.to(device),
            x_val=self.x_val.to(device) if self.x_val is not None else None,
            y_val=self.y_val.to(device) if self.y_val is not None else None,
            x_test=self.x_test.to(device) if self.x_test is not None else None,
            y_test=self.y_test.to(device) if self.y_test is not None else None
        )
    
    def get_split_info(self) -> Dict[str, int]:
        """Get information about data split sizes."""
        info = {
            "train_size": len(self.x_train),
        }
        if self.x_val is not None:
            info["val_size"] = len(self.x_val)
        if self.x_test is not None:
            info["test_size"] = len(self.x_test)
        return info


class ModelAdapter:
    """
    Adapter to make existing models compatible with the BaseModel interface.
    
    This allows models that don't inherit from BaseModel to work with
    the training engine.
    """
    
    def __init__(self, model: Any, model_type: str = "unknown"):
        """
        Initialize adapter.
        
        Args:
            model: The model to adapt
            model_type: String identifier for model type
        """
        self.model = model
        self.model_type = model_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapted model."""
        if hasattr(self.model, 'forward'):
            return self.model.forward(x)
        elif callable(self.model):
            return self.model(x)
        else:
            raise NotImplementedError(f"Model {type(self.model)} doesn't support forward pass")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the adapted model."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            # Default: use forward pass and apply thresholding
            outputs = self.forward(x)
            if outputs.shape[1] == 1:
                return (torch.sigmoid(outputs) > 0.5).float()
            else:
                return torch.argmax(outputs, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the adapted model."""
        info = {"model_type": self.model_type}
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        return info
    
    def save_model(self, filepath: str):
        """Save the adapted model."""
        if hasattr(self.model, 'save_model'):
            self.model.save_model(filepath)
        else:
            torch.save(self.model.state_dict(), filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> "ModelAdapter":
        """Load an adapted model (basic implementation)."""
        # This is a basic implementation - specific adapters should override
        # Would need model creation logic here
        raise NotImplementedError("ModelAdapter.load_model needs specific implementation")
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        return getattr(self.model, name) 