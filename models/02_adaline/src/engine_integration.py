"""
Engine framework integration for ADALINE model.

This module provides enhanced engine integration for ADALINE while maintaining
the educational simplicity of the basic implementation. It bridges the gap between
the simple pattern and the engine framework for better dataset strategy compliance.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

from utils import get_logger
try:
    from .data_loader import load_adaline_train_data, load_adaline_eval_data
except ImportError:
    from data_loader import load_adaline_train_data, load_adaline_eval_data

# Import engine components if available
try:
    from engine.base import DataSplit, TrainingResult
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    # Create minimal fallback classes for compatibility
    class DataSplit:
        def __init__(self, x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None):
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val
            self.x_test = x_test
            self.y_test = y_test
    
    class TrainingResult:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class ADALINEEngineAdapter:
    """
    Adapter class to provide engine compatibility for ADALINE.
    
    This class wraps the simple ADALINE implementation to work with
    the engine framework's DataSplit structure while maintaining
    educational clarity.
    """
    
    def __init__(self, adaline_model):
        """
        Initialize adapter with ADALINE model.
        
        Args:
            adaline_model: Instance of ADALINE model
        """
        self.model = adaline_model
        self.logger = get_logger(__name__)
    
    def train_with_datasplit(self, data_split: DataSplit) -> TrainingResult:
        """
        Train ADALINE using engine DataSplit structure.
        
        Args:
            data_split: Engine DataSplit containing train/val/test data
            
        Returns:
            TrainingResult with comprehensive metrics
        """
        self.logger.info("Training ADALINE with engine DataSplit")
        
        # Convert DataSplit to numpy arrays for ADALINE
        x_train, y_train = self._convert_to_numpy(data_split.x_train, data_split.y_train)
        
        # Train the model
        training_results = self.model.fit(x_train, y_train)
        
        # Evaluate on validation set if available
        val_results = {}
        if data_split.x_val is not None and data_split.y_val is not None:
            x_val, y_val = self._convert_to_numpy(data_split.x_val, data_split.y_val)
            val_results = self._evaluate_on_split(x_val, y_val, "validation")
        
        # Evaluate on test set if available
        test_results = {}
        if data_split.x_test is not None and data_split.y_test is not None:
            x_test, y_test = self._convert_to_numpy(data_split.x_test, data_split.y_test)
            test_results = self._evaluate_on_split(x_test, y_test, "test")
        
        # Create comprehensive TrainingResult
        result = TrainingResult(
            # Core training metrics
            final_loss=training_results['final_mse'],
            final_train_accuracy=training_results.get('final_accuracy', 0.0),
            final_val_accuracy=val_results.get('accuracy'),
            final_test_accuracy=test_results.get('accuracy'),
            
            # Training progress
            epochs_trained=training_results['epochs_trained'],
            total_training_time=0.0,  # ADALINE doesn't track time by default
            converged=training_results['converged'],
            convergence_epoch=training_results['epochs_trained'] if training_results['converged'] else None,
            
            # History tracking
            loss_history=training_results.get('loss_history', []),
            train_accuracy_history=training_results.get('accuracy_history', []),
            val_accuracy_history=[],  # Could be enhanced to track during training
            
            # Experiment metadata
            experiment_name=getattr(self.model.config, 'name', 'adaline_experiment'),
            model_architecture='ADALINE',
            dataset_name=getattr(self.model.config, 'dataset', 'unknown'),
            hyperparameters={
                'learning_rate': self.model.config.learning_rate,
                'epochs': self.model.config.epochs,
                'tolerance': self.model.config.tolerance
            }
        )
        
        self.logger.info(f"Training completed - Converged: {result.converged}, "
                        f"Final loss: {result.final_loss:.6f}")
        
        return result
    
    def _convert_to_numpy(self, x_data, y_data) -> Tuple[np.ndarray, np.ndarray]:
        """Convert tensor data to numpy arrays for ADALINE."""
        if torch.is_tensor(x_data):
            x_np = x_data.detach().cpu().numpy()
        else:
            x_np = np.array(x_data)
        
        if torch.is_tensor(y_data):
            y_np = y_data.detach().cpu().numpy()
        else:
            y_np = np.array(y_data)
        
        # Ensure correct shapes
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        
        return x_np.astype(np.float32), y_np.astype(np.float32)
    
    def _evaluate_on_split(self, x_data: np.ndarray, y_data: np.ndarray, 
                          split_name: str) -> Dict[str, float]:
        """Evaluate model on a data split."""
        try:
            # Forward pass
            predictions = self.model.predict(x_data)
            
            # Calculate metrics
            mse = np.mean((y_data.flatten() - predictions.flatten()) ** 2)
            
            # Binary accuracy (assuming binary classification)
            pred_binary = (predictions > 0.5).astype(int)
            y_binary = (y_data > 0.5).astype(int)
            accuracy = np.mean(pred_binary.flatten() == y_binary.flatten())
            
            results = {
                'mse': float(mse),
                'accuracy': float(accuracy)
            }
            
            self.logger.debug(f"{split_name} evaluation: MSE={mse:.6f}, Accuracy={accuracy:.4f}")
            return results
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate on {split_name}: {e}")
            return {'mse': float('inf'), 'accuracy': 0.0}


def create_datasplit_from_dataset(dataset_name: str, 
                                 train_split: float = 0.7,
                                 val_split: float = 0.15,
                                 test_split: float = 0.15,
                                 random_state: int = 42) -> DataSplit:
    """
    Create DataSplit from dataset name using unified loading system.
    
    Args:
        dataset_name: Name of dataset to load
        train_split: Fraction for training
        val_split: Fraction for validation  
        test_split: Fraction for testing
        random_state: Random seed for reproducible splits
        
    Returns:
        DataSplit with train/val/test splits
        
    Example:
        data_split = create_datasplit_from_dataset('iris_setosa_versicolor')
        adapter = ADALINEEngineAdapter(adaline_model)
        result = adapter.train_with_datasplit(data_split)
    """
    logger = get_logger(__name__)
    
    # Validate split ratios
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_split + val_split + test_split}")
    
    logger.info(f"Creating DataSplit for {dataset_name} with ratios {train_split}/{val_split}/{test_split}")
    
    # Load full dataset
    x_data, y_data = load_adaline_train_data(dataset_name)
    
    # Convert to tensors for splitting
    x_tensor = torch.from_numpy(x_data).float()
    y_tensor = torch.from_numpy(y_data.flatten()).float()  # Flatten for easier handling
    
    # Create splits using sklearn
    from sklearn.model_selection import train_test_split
    
    # First split: separate out test set
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_tensor, y_tensor, 
        test_size=test_split, 
        random_state=random_state,
        stratify=y_tensor.long() if len(torch.unique(y_tensor)) > 1 else None
    )
    
    # Second split: separate train and validation from remaining data
    train_ratio = train_split / (train_split + val_split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        train_size=train_ratio,
        random_state=random_state + 1,
        stratify=y_temp.long() if len(torch.unique(y_temp)) > 1 else None
    )
    
    # Reshape y tensors to match ADALINE expectations
    y_train = y_train.unsqueeze(1)
    y_val = y_val.unsqueeze(1) 
    y_test = y_test.unsqueeze(1)
    
    # Create DataSplit
    data_split = DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test
    )
    
    logger.info(f"Created DataSplit - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    return data_split


def create_enhanced_adaline_config(base_config, enable_engine_features: bool = True):
    """
    Enhance ADALINE config for better engine integration.
    
    Args:
        base_config: Basic ADALINE configuration
        enable_engine_features: Whether to enable engine-specific features
        
    Returns:
        Enhanced configuration
    """
    if not enable_engine_features:
        return base_config
    
    # Add engine-compatible features
    enhanced_config = base_config
    
    # Add validation tracking
    enhanced_config.track_validation = True
    enhanced_config.validation_frequency = 10  # Every 10 epochs
    
    # Add early stopping
    enhanced_config.early_stopping = True
    enhanced_config.patience = 50
    
    # Add model persistence
    enhanced_config.save_best_model = True
    enhanced_config.model_checkpoint_frequency = 25
    
    return enhanced_config


def demonstrate_adaline_engine_integration():
    """
    Demonstration function showing ADALINE engine integration.
    
    This function shows how to use ADALINE with the engine framework
    while maintaining the educational simplicity.
    """
    logger = get_logger(__name__)
    logger.info("Demonstrating ADALINE engine integration")
    
    try:
        # Import ADALINE components
        from model import create_adaline
        from config import get_experiment_config
        
        # Create configuration
        config = get_experiment_config('debug_small')
        config = create_enhanced_adaline_config(config)
        
        # Create model
        adaline = create_adaline(config)
        
        # Create DataSplit
        data_split = create_datasplit_from_dataset(config.dataset)
        
        # Create adapter and train
        adapter = ADALINEEngineAdapter(adaline)
        result = adapter.train_with_datasplit(data_split)
        
        # Report results
        logger.info("Demo completed successfully!")
        logger.info(f"Final accuracy: {result.final_train_accuracy:.4f}")
        logger.info(f"Validation accuracy: {result.final_val_accuracy:.4f}")
        logger.info(f"Test accuracy: {result.final_test_accuracy:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run demonstration
    demonstrate_adaline_engine_integration() 