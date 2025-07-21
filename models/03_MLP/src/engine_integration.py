"""
Engine framework integration for MLP model.

This module provides enhanced engine integration for MLP to support DataSplit
and improve compatibility with the engine framework for better dataset strategy compliance.
"""

from typing import Dict, Any, Tuple
import torch

from utils import get_logger
from data_utils import load_dataset

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


class MLPEngineAdapter:
    """
    Adapter class to provide enhanced engine compatibility for MLP.
    
    This class enhances the MLP implementation to work better with
    the engine framework's DataSplit structure while maintaining
    MLP's existing functionality.
    """
    
    def __init__(self, mlp_model):
        """
        Initialize adapter with MLP model.
        
        Args:
            mlp_model: Instance of MLP model
        """
        self.model = mlp_model
        self.logger = get_logger(__name__)
    
    def train_with_datasplit(self, data_split: DataSplit, **train_kwargs) -> TrainingResult:
        """
        Train MLP using engine DataSplit structure.
        
        Args:
            data_split: Engine DataSplit containing train/val/test data
            **train_kwargs: Additional training parameters
            
        Returns:
            TrainingResult with comprehensive metrics
        """
        self.logger.info("Training MLP with engine DataSplit")
        
        # Extract training parameters
        learning_rate = train_kwargs.get('learning_rate', 0.1)
        max_epochs = train_kwargs.get('max_epochs', 1000)
        convergence_threshold = train_kwargs.get('convergence_threshold', 1e-6)
        patience = train_kwargs.get('patience', 50)
        verbose = train_kwargs.get('verbose', True)
        
        # Train the model using DataSplit
        training_results = self.model.train_model(
            x_train=data_split.x_train,
            y_train=data_split.y_train,
            x_test=data_split.x_val if data_split.x_val is not None else data_split.x_test,
            y_test=data_split.y_val if data_split.y_val is not None else data_split.y_test,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold,
            patience=patience,
            verbose=verbose
        )
        
        # Evaluate on test set if separate from validation
        test_results = {}
        if (data_split.x_test is not None and data_split.y_test is not None and 
            data_split.x_val is not None):
            test_results = self._evaluate_on_split(data_split.x_test, data_split.y_test, "test")
        
        # Create comprehensive TrainingResult
        result = TrainingResult(
            # Core training metrics
            final_loss=training_results.get('final_loss', 0.0),
            final_train_accuracy=training_results.get('final_train_accuracy', 0.0),
            final_val_accuracy=training_results.get('final_test_accuracy', test_results.get('accuracy')),
            final_test_accuracy=test_results.get('accuracy', training_results.get('final_test_accuracy')),
            
            # Training progress
            epochs_trained=training_results.get('epochs_trained', 0),
            total_training_time=training_results.get('training_time', 0.0),
            converged=training_results.get('converged', False),
            convergence_epoch=training_results.get('convergence_epoch'),
            
            # History tracking
            loss_history=training_results.get('loss_history', []),
            train_accuracy_history=training_results.get('train_accuracy_history', []),
            val_accuracy_history=training_results.get('test_accuracy_history', []),
            
            # Experiment metadata
            experiment_name=getattr(self.model, 'experiment_name', 'mlp_experiment'),
            model_architecture='MLP',
            dataset_name='unknown',
            hyperparameters={
                'learning_rate': learning_rate,
                'max_epochs': max_epochs,
                'convergence_threshold': convergence_threshold,
                'hidden_layers': getattr(self.model, 'hidden_layers', []),
                'activation': getattr(self.model, 'activation_name', 'unknown')
            }
        )
        
        self.logger.info(f"Training completed - Converged: {result.converged}, "
                        f"Final loss: {result.final_loss:.6f}")
        
        return result
    
    def _evaluate_on_split(self, x_data: torch.Tensor, y_data: torch.Tensor, 
                          split_name: str) -> Dict[str, float]:
        """Evaluate model on a data split."""
        try:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.forward(x_data)
                
                # Calculate loss
                if hasattr(self.model, 'get_loss'):
                    loss = self.model.get_loss(outputs, y_data).item()
                else:
                    criterion = torch.nn.BCEWithLogitsLoss()
                    loss = criterion(outputs, y_data).item()
                
                # Calculate accuracy
                predictions = (outputs >= 0.5).float()
                if y_data.dim() > 1 and y_data.shape[1] == 1:
                    y_data = y_data.squeeze(1)
                if predictions.dim() > 1 and predictions.shape[1] == 1:
                    predictions = predictions.squeeze(1)
                
                accuracy = (predictions == y_data).float().mean().item()
            
            results = {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }
            
            self.logger.debug(f"{split_name} evaluation: Loss={loss:.6f}, Accuracy={accuracy:.4f}")
            return results
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate on {split_name}: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0}


def create_mlp_datasplit_from_config(config) -> DataSplit:
    """
    Create DataSplit from MLP configuration using unified loading system.
    
    Args:
        config: MLP experiment configuration
        
    Returns:
        DataSplit with train/val/test splits
        
    Example:
        config = get_experiment_config('xor_breakthrough')
        data_split = create_mlp_datasplit_from_config(config)
        adapter = MLPEngineAdapter(mlp_model)
        result = adapter.train_with_datasplit(data_split)
    """
    logger = get_logger(__name__)
    
    # Map MLP dataset names to unified system names
    dataset_name_mapping = {
        "xor": "xor_problem",
        "circles": "circles_dataset"
    }
    
    dataset_name = dataset_name_mapping.get(config.dataset_type, config.dataset_type)
    
    logger.info(f"Creating MLP DataSplit for {dataset_name}")
    
    # Load dataset using unified system
    try:
        features, labels = load_dataset(dataset_name, config.dataset_params)
    except Exception as e:
        # Fall back to XOR if dataset loading fails
        logger.warning(f"Failed to load {dataset_name}, falling back to XOR: {e}")
        features, labels = load_dataset("xor_problem")
    
    # Convert to tensors
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Handle special case for XOR (only 4 samples)
    if config.dataset_type == "xor" or dataset_name == "xor_problem":
        # For XOR, use all data for training (only 4 samples)
        data_split = DataSplit(
            x_train=x_tensor,
            y_train=y_tensor,
            x_val=x_tensor,  # Use same data for validation
            y_val=y_tensor,
            x_test=x_tensor,  # Use same data for test
            y_test=y_tensor
        )
        logger.info(f"Created XOR DataSplit - Train/Val/Test: {len(x_tensor)} samples each")
    else:
        # For other datasets, split into train/val/test
        from sklearn.model_selection import train_test_split
        
        # First split: separate out test set (20%)
        x_temp, x_test, y_temp, y_test = train_test_split(
            x_tensor, y_tensor, 
            test_size=0.2, 
            random_state=42,
            stratify=y_tensor.long() if len(torch.unique(y_tensor)) > 1 else None
        )
        
        # Second split: separate train (75% of remaining) and validation (25% of remaining)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp,
            test_size=0.25,  # 0.25 * 0.8 = 0.2 of total for validation
            random_state=43,
            stratify=y_temp.long() if len(torch.unique(y_temp)) > 1 else None
        )
        
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


def create_enhanced_mlp_config(base_config, enable_engine_features: bool = True):
    """
    Enhance MLP config for better engine integration.
    
    Args:
        base_config: Basic MLP configuration
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
    
    # Add model persistence
    enhanced_config.save_best_model = True
    enhanced_config.model_checkpoint_frequency = 50
    
    # Enhanced logging
    enhanced_config.detailed_logging = True
    
    return enhanced_config


def demonstrate_mlp_engine_integration():
    """
    Demonstration function showing MLP engine integration.
    
    This function shows how to use MLP with the engine framework
    DataSplit structure for better strategy compliance.
    """
    logger = get_logger(__name__)
    logger.info("Demonstrating MLP engine integration")
    
    try:
        # Import MLP components
        from .model import MLP
        from .config import get_experiment_config
        
        # Create configuration
        config = get_experiment_config('quick_test')  # Use quick_test for fast demo
        config = create_enhanced_mlp_config(config)
        
        # Create model
        mlp = MLP(
            input_size=config.input_size,
            hidden_layers=config.hidden_layers,
            output_size=config.output_size,
            activation=config.activation,
            weight_init=config.weight_init,
            device='cpu'
        )
        
        # Create DataSplit
        data_split = create_mlp_datasplit_from_config(config)
        
        # Create adapter and train
        adapter = MLPEngineAdapter(mlp)
        result = adapter.train_with_datasplit(
            data_split,
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs,
            convergence_threshold=config.convergence_threshold,
            patience=config.patience,
            verbose=config.verbose
        )
        
        # Report results
        logger.info("Demo completed successfully!")
        logger.info(f"Final train accuracy: {result.final_train_accuracy:.4f}")
        logger.info(f"Final val accuracy: {result.final_val_accuracy:.4f}")
        logger.info(f"Final test accuracy: {result.final_test_accuracy:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run demonstration
    demonstrate_mlp_engine_integration() 