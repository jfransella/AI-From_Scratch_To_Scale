"""
Unified training engine for AI From Scratch to Scale project.

This module provides a model-agnostic training framework that works with
any model implementing the BaseModel interface, featuring experiment tracking,
checkpointing, and comprehensive training loop management.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.optim as optim

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
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
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
        
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project or "ai-from-scratch-to-scale",
                name=self.config.experiment_name,
                tags=self.config.wandb_tags,
                config=self.config.__dict__
            )
            self.logger.info("wandb tracking initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False
    
    def _setup_optimizer(self, model: BaseModel) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        if self.config.optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:  # Default to SGD
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        
        return optimizer
    
    def _setup_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler if specified."""
        if self.config.lr_scheduler is None:
            return None
        
        if self.config.lr_scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs
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
            y_test=data.y_test
        )
    
    def _compute_accuracy(self, model: BaseModel, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy."""
        model.eval()
        with torch.no_grad():
            predictions = model.predict(x)
            if predictions.dim() == 2 and predictions.shape[1] == 1:
                predictions = predictions.squeeze()
            accuracy = (predictions == y).float().mean().item()
        model.train()
        return accuracy
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        if self.config.verbose and epoch % self.config.log_freq == 0:
            metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.logger.info(f"Epoch {epoch:4d}: {metric_str}")
        
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.log(metrics, step=epoch)
    
    def _save_checkpoint(self, model: BaseModel, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not (self.config.save_best_model or self.config.save_final_model):
            return
        
        output_dir = Path(self.config.output_dir)
        
        # Save checkpoint
        if self.config.checkpoint_freq > 0 and epoch % self.config.checkpoint_freq == 0:
            checkpoint_path = output_dir / f"{self.config.experiment_name}_epoch_{epoch}.pth"
            model.save_model(str(checkpoint_path))
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, model: BaseModel, data: DataSplit) -> TrainingResult:
        """
        Train a model using the configured training loop.
        
        Args:
            model: Model to train (must implement BaseModel interface)
            data: Training/validation/test data
            
        Returns:
            TrainingResult with comprehensive training information
        """
        self.logger.info(f"Starting training: {self.config.experiment_name}")
        
        # Setup data splits
        data = self._split_data(data)
        data = data.to_device(str(self.device))
        
        # Move model to device
        model = model.to(str(self.device))
        
        # Setup training components
        if not isinstance(model, BaseModel):
            model = ModelAdapter(model, self.config.model_name)
        
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)
        
        # Initialize result tracking
        result = TrainingResult(
            experiment_name=self.config.experiment_name,
            model_architecture=self.config.model_name,
            dataset_name=self.config.dataset_name,
            hyperparameters=self.config.__dict__.copy(),
            final_loss=float('inf'),
            final_train_accuracy=0.0
        )
        
        # Training state
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Data splits: {data.get_split_info()}")
        
        # Training loop
        model.train()
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            
            # Forward pass
            outputs = model.forward(data.x_train)
            loss = model.get_loss(outputs, data.y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Compute metrics
            train_acc = self._compute_accuracy(model, data.x_train, data.y_train)
            
            metrics = {
                "epoch": epoch,
                "loss": loss.item(),
                "train_accuracy": train_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # Validation metrics
            if data.x_val is not None and epoch % self.config.validation_freq == 0:
                val_acc = self._compute_accuracy(model, data.x_val, data.y_val)
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
                    self.logger.info(f"Converged at epoch {epoch} with loss {loss.item():.6f}")
                break
            
            # Early stopping
            if self.config.early_stopping:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    
                    # Save best model
                    if self.config.save_best_model:
                        best_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_best.pth"
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
        result.final_train_accuracy = self._compute_accuracy(model, data.x_train, data.y_train)
        
        if data.x_val is not None:
            result.final_val_accuracy = self._compute_accuracy(model, data.x_val, data.y_val)
        
        if data.x_test is not None:
            result.final_test_accuracy = self._compute_accuracy(model, data.x_test, data.y_test)
        
        # Save final model
        if self.config.save_final_model:
            final_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_final.pth"
            model.save_model(str(final_path))
            result.final_model_path = str(final_path)
        
        # Final logging
        if self.config.verbose:
            self.logger.info("\nTraining completed!")
            self.logger.info(f"Total time: {result.total_training_time:.2f} seconds")
            self.logger.info(f"Final loss: {result.final_loss:.6f}")
            self.logger.info(f"Final train accuracy: {result.final_train_accuracy:.4f}")
            if result.final_val_accuracy:
                self.logger.info(f"Final validation accuracy: {result.final_val_accuracy:.4f}")
            if result.final_test_accuracy:
                self.logger.info(f"Final test accuracy: {result.final_test_accuracy:.4f}")
        
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
            
            self.logger.info(f"Completed experiment {i+1}/{len(experiments)}: {experiment_config.experiment_name}")
        
        return results 