"""
Base classes and interfaces for the unified training engine.

Provides abstract base classes and data structures that define the interface
for models, training results, and other core components of the training system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Handle torch imports gracefully
try:
    import torch

    if (
        hasattr(torch, "__version__")
        and hasattr(torch, "nn")
        and hasattr(torch, "tensor")
    ):
        import torch.nn as nn

        _TORCH_AVAILABLE = True
        TorchTensor = torch.Tensor
    else:
        # torch exists but is broken
        _TORCH_AVAILABLE = False
        torch = None
        nn = None
        TorchTensor = Any
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False
    TorchTensor = Any


from pathlib import Path

# Optional wandb integration
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


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
            "final_model_path": self.final_model_path,
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
    Includes comprehensive wandb integration for experiment tracking.
    """

    def __init__(self):
        """Initialize base model with wandb tracking capabilities."""
        self.wandb_run = None
        self.wandb_config = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def forward(self, x: TorchTensor) -> TorchTensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def fit(self, x_data: TorchTensor, y_target: TorchTensor) -> Dict[str, Any]:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, x: TorchTensor) -> TorchTensor:
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def get_loss(self, outputs: TorchTensor, targets: TorchTensor) -> TorchTensor:
        """Compute loss for the given outputs and targets."""
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

    # New wandb integration methods
    def init_wandb(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
    ) -> bool:
        """
        Initialize Weights & Biases tracking for this model.

        Args:
            project: Wandb project name (defaults to model-specific)
            name: Run name (defaults to experiment name)
            tags: List of tags for categorization
            config: Configuration dictionary to log
            notes: Optional notes about the experiment
            mode: Wandb mode ("online", "offline", "disabled")

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not _WANDB_AVAILABLE:
            self.logger.warning("wandb not available, skipping initialization")
            return False

        try:
            # Get model-specific defaults
            model_info = self.get_model_info()

            # Set default project name based on model
            if project is None:
                model_name = model_info.get("name", "unknown").lower()
                project = f"ai-from-scratch-{model_name}"

            # Set default tags based on model info
            if tags is None:
                tags = []

            # Add model-specific tags
            tags.extend(
                [
                    model_info.get("name", "unknown").lower(),
                    model_info.get("category", "unknown"),
                    f"module-{model_info.get('module', 'unknown')}",
                ]
            )

            # Merge configurations
            wandb_config = {}
            if config:
                wandb_config.update(config)
            wandb_config.update(model_info)

            # Initialize wandb with proper mode handling
            init_kwargs = {
                "project": project,
                "name": name,
                "tags": list(set(tags)),  # Remove duplicates
                "config": wandb_config,
                "notes": notes,
                "reinit": True,  # Allow multiple inits
            }

            # Only add mode if it's a valid wandb mode
            if mode in ["online", "offline", "disabled"]:
                init_kwargs["mode"] = mode

            self.wandb_run = wandb.init(**init_kwargs)

            self.wandb_config = wandb_config
            self.logger.info(f"Wandb initialized for project: {project}")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            return False

    def log_metrics(
        self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True
    ):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch number
            commit: Whether to commit the log entry
        """
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step, commit=commit)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to wandb: {e}")

    def log_artifact(
        self,
        filepath: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Log an artifact (model checkpoint, visualization, etc.) to wandb.

        Args:
            filepath: Path to the file to log
            artifact_type: Type of artifact ("model", "dataset", "visualization")
            name: Artifact name (defaults to filename)
            description: Artifact description
        """
        if self.wandb_run is not None:
            try:
                if name is None:
                    name = Path(filepath).stem

                artifact = wandb.Artifact(
                    name=name, type=artifact_type, description=description
                )
                artifact.add_file(filepath)
                self.wandb_run.log_artifact(artifact)

                self.logger.info(f"Logged artifact: {name} ({artifact_type})")

            except Exception as e:
                self.logger.warning(f"Failed to log artifact to wandb: {e}")

    def log_image(
        self, image_path: str, caption: Optional[str] = None, step: Optional[int] = None
    ):
        """
        Log an image to wandb.

        Args:
            image_path: Path to the image file
            caption: Image caption
            step: Training step/epoch
        """
        if self.wandb_run is not None:
            try:
                image = wandb.Image(image_path, caption=caption)
                self.wandb_run.log({"visualization": image}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log image to wandb: {e}")

    def watch_model(self, log: str = "gradients", log_freq: int = 100) -> None:
        """
        Watch model gradients and parameters with wandb.

        Args:
            log: What to log ("gradients", "parameters", "all")
            log_freq: Frequency of logging
        """
        if not _TORCH_AVAILABLE:
            self.logger.info("⚠️ Torch not available - skipping model watching")
            return

        if (
            self.wandb_run is not None
            and nn is not None
            and isinstance(self, nn.Module)
        ):
            try:
                import wandb

                wandb.watch(self, log=log, log_freq=log_freq)
                self.logger.info(
                    f"✅ Wandb watching model - log: {log}, freq: {log_freq}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to setup wandb model watching: {e}")
        else:
            self.logger.info(
                "⚠️ Skipping wandb model watching - no active run or not a torch Module"
            )

    def finish_wandb(self):
        """Finish the wandb run and cleanup."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                self.wandb_run = None
                self.logger.info("Wandb run finished")
            except Exception as e:
                self.logger.warning(f"Failed to finish wandb run: {e}")

    def get_wandb_url(self) -> Optional[str]:
        """Get the URL for the current wandb run."""
        if self.wandb_run is not None:
            return self.wandb_run.get_url()
        return None

    def get_wandb_id(self) -> Optional[str]:
        """Get the ID for the current wandb run."""
        if self.wandb_run is not None:
            return self.wandb_run.id
        return None

    def predict_proba(self, x: TorchTensor) -> TorchTensor:
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

    def get_loss(self, outputs: TorchTensor, targets: TorchTensor) -> TorchTensor:
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
        if hasattr(self, "device"):
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

    x_train: TorchTensor
    y_train: TorchTensor
    x_val: Optional[TorchTensor] = None
    y_val: Optional[TorchTensor] = None
    x_test: Optional[TorchTensor] = None
    y_test: Optional[TorchTensor] = None

    def to_device(self, device: str) -> "DataSplit":
        """Move all tensors to specified device."""
        return DataSplit(
            x_train=self.x_train.to(device),
            y_train=self.y_train.to(device),
            x_val=self.x_val.to(device) if self.x_val is not None else None,
            y_val=self.y_val.to(device) if self.y_val is not None else None,
            x_test=self.x_test.to(device) if self.x_test is not None else None,
            y_test=self.y_test.to(device) if self.y_test is not None else None,
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

    def forward(self, x: TorchTensor) -> TorchTensor:
        """Forward pass through the adapted model."""
        if hasattr(self.model, "forward"):
            return self.model.forward(x)
        elif callable(self.model):
            return self.model(x)
        else:
            raise NotImplementedError(
                f"Model {type(self.model)} doesn't support forward pass"
            )

    def predict(self, x: TorchTensor) -> TorchTensor:
        """Make predictions with the adapted model."""
        if hasattr(self.model, "predict"):
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
        if hasattr(self.model, "get_model_info"):
            info.update(self.model.get_model_info())
        return info

    def save_model(self, filepath: str):
        """Save the adapted model."""
        if hasattr(self.model, "save_model"):
            self.model.save_model(filepath)
        else:
            torch.save(self.model.state_dict(), filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "ModelAdapter":
        """Load an adapted model (basic implementation)."""
        # This is a basic implementation - specific adapters should override
        # Would need model creation logic here
        raise NotImplementedError(
            "ModelAdapter.load_model needs specific implementation"
        )

    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        return getattr(self.model, name)
