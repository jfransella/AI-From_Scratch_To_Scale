"""
Unified training engine for AI From Scratch to Scale project.

Provides a consistent training interface across all models while supporting
both simple implementations and advanced features like learning rate scheduling,
early stopping, and comprehensive logging.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

# Handle torch imports gracefully
try:
    import torch
    if hasattr(torch, '__version__') and hasattr(torch, 'optim') and hasattr(torch, 'nn'):
        import torch.optim as optim
        import torch.nn as nn
        _TORCH_AVAILABLE = True
        TorchTensor = torch.Tensor
    else:
        # torch exists but is broken - create dummy torch
        _TORCH_AVAILABLE = False
        
        class DummyDevice:
            def __init__(self, device_str):
                self.type = "cpu"
            def __str__(self):
                return "cpu"
                
        class DummyTorch:
            @staticmethod
            def device(device_str):
                return DummyDevice(device_str)
        
        class DummyOptim:
            class SGD:
                def __init__(self, params, lr=0.01, **kwargs):
                    self.param_groups = [{'lr': lr, 'params': list(params)}]
                def step(self):
                    pass
                def zero_grad(self):
                    pass
            
            class Adam:
                def __init__(self, params, lr=0.001, **kwargs):
                    self.param_groups = [{'lr': lr, 'params': list(params)}]
                def step(self):
                    pass
                def zero_grad(self):
                    pass
                    
            class AdamW:
                def __init__(self, params, lr=0.001, **kwargs):
                    self.param_groups = [{'lr': lr, 'params': list(params)}]
                def step(self):
                    pass
                def zero_grad(self):
                    pass
        
        torch = DummyTorch()
        optim = DummyOptim()
        nn = None
        TorchTensor = Any
except ImportError:
    torch = None
    optim = None  
    nn = None
    _TORCH_AVAILABLE = False
    TorchTensor = Any
    
    # Create dummy torch for ImportError case too
    class DummyDevice:
        def __init__(self, device_str):
            self.type = "cpu"
        def __str__(self):
            return "cpu"
            
    class DummyTorch:
        @staticmethod
        def device(device_str):
            return DummyDevice(device_str)
    
    class DummyOptim:
        class SGD:
            def __init__(self, params, lr=0.01, **kwargs):
                self.param_groups = [{'lr': lr, 'params': list(params)}]
            def step(self):
                pass
            def zero_grad(self):
                pass
        
        class Adam:
            def __init__(self, params, lr=0.001, **kwargs):
                self.param_groups = [{'lr': lr, 'params': list(params)}]
            def step(self):
                pass
            def zero_grad(self):
                pass
                
        class AdamW:
            def __init__(self, params, lr=0.001, **kwargs):
                self.param_groups = [{'lr': lr, 'params': list(params)}]
            def step(self):
                pass
            def zero_grad(self):
                pass
    
    torch = DummyTorch()
    optim = DummyOptim()

# Optional wandb integration
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from utils import set_random_seed, get_logger
from .base import BaseModel, TrainingResult, DataSplit, ModelAdapter


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""

    # Experiment metadata
    experiment_name: str = "unnamed_experiment"
    model_name: str = "unknown_model"
    dataset_name: str = "unknown_dataset"

    # Training hyperparameters
    learning_rate: float = 0.01
    max_epochs: int = 1000
    batch_size: Optional[int] = None  # None for full batch

    # Optimization
    optimizer_type: str = "sgd"  # "sgd", "adam", "rmsprop"
    momentum: float = 0.0
    weight_decay: float = 0.0

    # Learning rate scheduling
    lr_scheduler: Optional[str] = None  # "step", "exponential", "cosine"
    lr_step_size: int = 100
    lr_gamma: float = 0.1

    # Convergence and early stopping
    convergence_threshold: float = 1e-6
    patience: int = 50
    early_stopping: bool = True

    # Validation
    validation_split: float = 0.0  # 0.0 means no validation split
    validation_freq: int = 1  # Validate every N epochs

    # Checkpointing
    save_best_model: bool = True
    save_final_model: bool = True
    checkpoint_freq: int = 0  # 0 means no intermediate checkpoints
    output_dir: str = "outputs"

    # Logging and tracking
    log_freq: int = 10  # Log every N epochs
    verbose: bool = True
    
    # Enhanced wandb configuration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    
    # Advanced wandb features
    wandb_watch_model: bool = False
    wandb_watch_log: str = "gradients"  # "gradients", "parameters", "all"
    wandb_watch_freq: int = 100
    
    # Artifact configuration
    wandb_log_checkpoints: bool = True
    wandb_log_visualizations: bool = True
    wandb_log_datasets: bool = False
    
    # Group and sweep support
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_sweep_id: Optional[str] = None

    # Reproducibility
    random_seed: Optional[int] = None

    # Device
    device: str = "auto"

    def __post_init__(self):
        """Validate and setup configuration."""
        if self.use_wandb and not _WANDB_AVAILABLE:
            logging.warning("wandb requested but not available, disabling")
            self.use_wandb = False

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class Trainer:
    """
    Unified training engine for all model types.

    This trainer provides a consistent interface for training different models
    while handling experiment tracking, checkpointing, and monitoring.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = get_logger("ai_from_scratch")

        # Setup random seed if specified
        if config.random_seed is not None:
            set_random_seed(config.random_seed)

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Initialize wandb if requested
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        self.logger.info(f"Trainer initialized - Device: {self.device}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        if not _WANDB_AVAILABLE:
            self.logger.warning("wandb not available, skipping initialization")
            return

        # Skip trainer-level wandb if model will handle it
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
            self.logger.info("Skipping trainer wandb - model will handle wandb integration")
            return

        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project or "ai-from-scratch-to-scale",
                name=self.config.experiment_name,
                tags=self.config.wandb_tags,
                config=self.config.__dict__,
            )
            self.logger.info("wandb tracking initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False
    
    def _init_trainer_wandb(self):
        """
        Initialize wandb at trainer level with enhanced configuration.
        
        This method sets up trainer-level wandb configuration and prepares
        for model-level wandb integration.
        """
        if not self.config.use_wandb or not _WANDB_AVAILABLE:
            return
        
        try:
            # Generate default project name if not provided
            if not self.config.wandb_project:
                model_name = self.config.model_name.lower().replace(" ", "-")
                self.config.wandb_project = f"ai-from-scratch-{model_name}"
            
            # Generate default run name if not provided
            if not self.config.wandb_name:
                self.config.wandb_name = f"{self.config.experiment_name}-{self.config.model_name}"
            
            # Add automatic tags
            auto_tags = [
                self.config.model_name.lower(),
                self.config.experiment_name,
                "trainer-managed"
            ]
            
            # Merge with user-provided tags
            all_tags = list(set(self.config.wandb_tags + auto_tags))
            self.config.wandb_tags = all_tags
            
            self.logger.info(f"Trainer wandb configured: project={self.config.wandb_project}, "
                             f"name={self.config.wandb_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure trainer wandb: {e}")
            self.config.use_wandb = False
    
    def _setup_model_wandb(self, model: BaseModel) -> bool:
        """
        Set up wandb integration for the model.
        
        Args:
            model: The model to set up wandb for
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        if not self.config.use_wandb or not hasattr(model, 'init_wandb'):
            return False
        
        try:
            # Initialize model wandb
            success = model.init_wandb(
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                tags=self.config.wandb_tags,
                config=self.config.__dict__,
                notes=self.config.wandb_notes,
                mode=self.config.wandb_mode
            )
            
            if success and self.config.wandb_watch_model:
                # Set up model watching
                model.watch_model(
                    log=self.config.wandb_watch_log,
                    log_freq=self.config.wandb_watch_freq
                )
                self.logger.info("Model watching enabled")
            
            return success
            
        except Exception as e:
            self.logger.warning(f"Failed to setup model wandb: {e}")
            return False
    
    def _log_training_metrics(self, model: BaseModel, epoch: int, metrics: Dict[str, Any]):
        """
        Log training metrics to wandb through the model.
        
        Args:
            model: The model to log metrics through
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        if (self.config.use_wandb and 
            hasattr(model, 'log_metrics') and
            hasattr(model, 'wandb_run') and 
            model.wandb_run is not None):
            
            try:
                # Add epoch to metrics
                metrics_with_epoch = {"epoch": epoch, **metrics}
                model.log_metrics(metrics_with_epoch, step=epoch)
                
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def _log_checkpoint_artifact(self, model: BaseModel, checkpoint_path: str, 
                                 epoch: int, description: str = None):
        """
        Log model checkpoint as wandb artifact.
        
        Args:
            model: The model to log artifact through
            checkpoint_path: Path to the checkpoint file
            epoch: Current epoch number
            description: Optional description for the artifact
        """
        if (self.config.use_wandb and 
            self.config.wandb_log_checkpoints and
            hasattr(model, 'log_artifact') and
            hasattr(model, 'wandb_run') and 
            model.wandb_run is not None):
            
            try:
                if not description:
                    description = f"Model checkpoint at epoch {epoch}"
                
                model.log_artifact(
                    checkpoint_path,
                    artifact_type="model",
                    description=description
                )
                self.logger.info(f"Logged checkpoint artifact: {checkpoint_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to log checkpoint artifact: {e}")
    
    def _finish_model_wandb(self, model: BaseModel):
        """
        Finish wandb run for the model.
        
        Args:
            model: The model to finish wandb for
        """
        if (self.config.use_wandb and 
            hasattr(model, 'finish_wandb')):
            
            try:
                model.finish_wandb()
                self.logger.info("Model wandb run finished")
                
            except Exception as e:
                self.logger.warning(f"Failed to finish model wandb: {e}")

    def _setup_optimizer(self, model: BaseModel) -> Any:
        """Setup optimizer based on configuration."""
        if self.config.optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:  # Default to SGD
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        return optimizer

    def _setup_scheduler(
        self, optimizer: Any
    ) -> Optional[Any]:
        """Setup learning rate scheduler if specified."""
        if self.config.lr_scheduler is None:
            return None

        if self.config.lr_scheduler.lower() == "step":
            return Any.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler.lower() == "exponential":
            return Any.ExponentialLR(
                optimizer, gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler.lower() == "cosine":
            return Any.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
        else:
            self.logger.warning(f"Unknown scheduler: {self.config.lr_scheduler}")
            return None

    def _split_data(self, data: DataSplit) -> DataSplit:
        """Split training data into train/validation if requested."""
        if self.config.validation_split <= 0:
            return data

        # Split training data
        train_size = len(data.x_train)
        val_size = int(train_size * self.config.validation_split)

        if val_size == 0:
            return data

        # Random split
        indices = torch.randperm(train_size)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        return DataSplit(
            x_train=data.x_train[train_indices],
            y_train=data.y_train[train_indices],
            x_val=data.x_train[val_indices],
            y_val=data.y_train[val_indices],
            x_test=data.x_test,
            y_test=data.y_test,
        )

    def _compute_accuracy(
        self, model: BaseModel, x: TorchTensor, y: TorchTensor
    ) -> float:
        """Compute accuracy for given data."""
        # Store original training mode
        was_training = model.training
        
        model.eval()
        with torch.no_grad():
            outputs = model.forward(x)
            if hasattr(model, 'predict'):
                predictions = model.predict(x)
            else:
                # Default binary classification
                predictions = (outputs >= 0.5).float().squeeze()
            
            correct = (predictions == y).float().sum()
            accuracy = correct / len(y)
            
        # Restore original training mode
        model.train(was_training)
        
        return accuracy.item()

    def _create_data_loaders(self, data_split: DataSplit):
        """Create data loaders for training and validation."""
        # For simple models like Perceptron, we use full batch training
        # So we return the data directly without DataLoader wrappers
        return data_split.x_train, data_split.x_val if data_split.x_val is not None else None

    def _log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics to wandb and console."""
        if self.config.verbose and epoch % self.config.log_freq == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Epoch {epoch:4d}: {metrics_str}")

        # Log to wandb if available - use model's wandb run if it exists
        if (hasattr(self, 'current_model') and 
            hasattr(self.current_model, 'wandb_run') and 
            self.current_model.wandb_run is not None):
            try:
                self.current_model.log_metrics(metrics, step=epoch)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics via model wandb: {e}")
        elif self.config.use_wandb and self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=epoch)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to wandb: {e}")

    def _save_checkpoint(self, model: BaseModel, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not (self.config.save_best_model or self.config.save_final_model):
            return

        output_dir = Path(self.config.output_dir)

        # Save checkpoint
        if self.config.checkpoint_freq > 0 and epoch % self.config.checkpoint_freq == 0:
            checkpoint_path = (
                output_dir / f"{self.config.experiment_name}_epoch_{epoch}.pth"
            )
            model.save_model(str(checkpoint_path))
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self, model: BaseModel, data_split: DataSplit) -> Dict[str, Any]:
        """
        Train a model using the configured training loop.

        Args:
            model: Model to train (must inherit from BaseModel)
            data_split: DataSplit containing train/val/test data

        Returns:
            Training results dictionary
        """
        # Store reference to current model for wandb integration
        self.current_model = model
        
        start_time = time.time()
        self.logger.info(f"Starting training: {self.config.experiment_name}")

        # Setup model and training
        model = model.to(self.device)
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)

        # Get data loaders
        train_loader, val_loader = self._create_data_loaders(data_split)

        # Initialize result tracking
        result = TrainingResult(
            experiment_name=self.config.experiment_name,
            model_architecture=self.config.model_name,
            dataset_name=self.config.dataset_name,
            hyperparameters=self.config.__dict__.copy(),
            final_loss=float("inf"),
            final_train_accuracy=0.0,
        )

        # Training state
        best_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Data splits: {data_split.get_split_info()}")

        # Training loop
        model.train()
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()

            # Forward pass - inputs don't need gradients in classification
            outputs = model.forward(data_split.x_train)
            loss = model.get_loss(outputs, data_split.y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Learning rate scheduling
            if scheduler:
                scheduler.step()

            # Compute metrics
            train_acc = self._compute_accuracy(model, data_split.x_train, data_split.y_train)

            metrics = {
                "epoch": epoch,
                "loss": loss.item(),
                "train_accuracy": train_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            # Validation metrics
            if data_split.x_val is not None and epoch % self.config.validation_freq == 0:
                val_acc = self._compute_accuracy(model, data_split.x_val, data_split.y_val)
                metrics["val_accuracy"] = val_acc
                result.val_accuracy_history.append(val_acc)

            # Update training history
            result.loss_history.append(loss.item())
            result.train_accuracy_history.append(train_acc)
            result.epoch_times.append(time.time() - epoch_start)

            # Logging
            self._log_metrics(epoch, metrics)

            # Checkpointing
            self._save_checkpoint(model, epoch, metrics)

            # Convergence check
            if loss.item() < self.config.convergence_threshold:
                result.converged = True
                result.convergence_epoch = epoch
                if self.config.verbose:
                    self.logger.info(
                        f"Converged at epoch {epoch} with loss {loss.item():.6f}"
                    )
                break

            # Early stopping
            if self.config.early_stopping:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0

                    # Save best model
                    if self.config.save_best_model:
                        best_path = (
                            Path(self.config.output_dir)
                            / f"{self.config.experiment_name}_best.pth"
                        )
                        model.save_model(str(best_path))
                        result.best_model_path = str(best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        if self.config.verbose:
                            self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        # Final evaluation
        result.epochs_trained = epoch + 1
        result.total_training_time = time.time() - start_time
        result.final_loss = loss.item()
        result.final_train_accuracy = self._compute_accuracy(
            model, data_split.x_train, data_split.y_train
        )

        if data_split.x_val is not None:
            result.final_val_accuracy = self._compute_accuracy(
                model, data_split.x_val, data_split.y_val
            )

        if data_split.x_test is not None:
            result.final_test_accuracy = self._compute_accuracy(
                model, data_split.x_test, data_split.y_test
            )

        # Save final model
        if self.config.save_final_model:
            final_path = (
                Path(self.config.output_dir)
                / f"{self.config.experiment_name}_final.pth"
            )
            model.save_model(str(final_path))
            result.final_model_path = str(final_path)

        # Final logging
        if self.config.verbose:
            self.logger.info("\nTraining completed!")
            self.logger.info(f"Total time: {result.total_training_time:.2f} seconds")
            self.logger.info(f"Final loss: {result.final_loss:.6f}")
            self.logger.info(f"Final train accuracy: {result.final_train_accuracy:.4f}")
            if result.final_val_accuracy:
                self.logger.info(
                    f"Final validation accuracy: {result.final_val_accuracy:.4f}"
                )
            if result.final_test_accuracy:
                self.logger.info(
                    f"Final test accuracy: {result.final_test_accuracy:.4f}"
                )

        # Cleanup wandb
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        self.logger.info(f"Training completed: {self.config.experiment_name}")
        return result

    def train_multiple(self, experiments: List[tuple]) -> Dict[str, TrainingResult]:
        """
        Train multiple experiments in sequence.

        Args:
            experiments: List of (model, data, config_overrides) tuples

        Returns:
            Dictionary mapping experiment names to results
        """
        results = {}

        for i, (model, data, config_overrides) in enumerate(experiments):
            # Update config for this experiment
            experiment_config = TrainingConfig(**self.config.__dict__)
            if config_overrides:
                for key, value in config_overrides.items():
                    setattr(experiment_config, key, value)

            # Create trainer for this experiment
            trainer = Trainer(experiment_config)

            # Train
            result = trainer.train(model, data)
            results[experiment_config.experiment_name] = result

            self.logger.info(
                f"Completed experiment {i+1}/{len(experiments)}: {experiment_config.experiment_name}"
            )

        return results
