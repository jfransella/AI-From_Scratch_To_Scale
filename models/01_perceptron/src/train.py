"""
Training script for Perceptron model.

This script demonstrates training the Perceptron on various datasets,
including both successes (linearly separable) and failures (XOR).
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import setup_logging, set_random_seed, get_logger
from data_utils import generate_xor_dataset, generate_linear_dataset, generate_circles_dataset
from config import get_config
from model import Perceptron
from constants import MODEL_NAME


def load_dataset_from_config(config):
    """Load dataset based on configuration."""
    dataset_name = config["dataset"]
    dataset_params = config.get("dataset_params", {})
    
    logger = get_logger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "xor":
        X, y = generate_xor_dataset(**dataset_params)
    elif dataset_name == "linear":
        X, y = generate_linear_dataset(**dataset_params)
    elif dataset_name == "circles":
        X, y = generate_circles_dataset(**dataset_params)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    logger.info(f"Dataset loaded: {X.shape}, classes: {torch.unique(y).tolist()}")
    
    return X, y


def create_simple_train_test_split(X, y, train_split=0.8, random_state=None):
    """Create a simple train/test split."""
    if random_state is not None:
        set_random_seed(random_state)
    
    n_samples = X.shape[0]
    n_train = int(n_samples * train_split)
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Simple evaluation function."""
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X_test)
        
        # Convert predictions to match target format
        if model.activation == "step":
            # predictions are already 0/1
            pred_labels = predictions.long()
        else:  # sign function
            # Convert sign output to 0/1
            pred_labels = (predictions > 0).long()
        
        # Handle label mapping
        unique_labels = torch.unique(y_test)
        if len(unique_labels) == 2:
            accuracy = (pred_labels == y_test).float().mean().item()
        else:
            accuracy = 0.0
    
    return accuracy, pred_labels


def save_results(config, model, training_history, test_accuracy):
    """Save training results."""
    results = {
        'experiment': config['experiment'],
        'model_info': model.get_model_info(),
        'training_history': training_history,
        'test_accuracy': test_accuracy,
        'converged': training_history.get('converged', False)
    }
    
    # Save to outputs directory
    output_file = Path(config['model_dir']) / f"{config['experiment']}_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger = get_logger(__name__)
    logger.info(f"Results saved to {output_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Perceptron model')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., quick_test, xor_failure)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations (placeholder)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = get_config(args.experiment)
        
        # Apply command-line overrides
        if args.epochs is not None:
            config['max_epochs'] = args.epochs
        if args.learning_rate is not None:
            config['learning_rate'] = args.learning_rate
        if args.debug:
            config['log_level'] = 'DEBUG'
            config['max_epochs'] = min(config['max_epochs'], 10)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Setup logging
    try:
        logger = setup_logging(
            level=config.get('log_level', 'INFO'),
            log_dir=config.get('log_dir')
        )
        
        # Set random seed for reproducibility
        if 'seed' in config:
            set_random_seed(config['seed'])
        
        logger.info(f"Starting {MODEL_NAME} training")
        logger.info(f"Experiment: {config['experiment']}")
        logger.info(f"Configuration: lr={config['learning_rate']}, "
                   f"epochs={config['max_epochs']}, "
                   f"dataset={config['dataset']}")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return 1
    
    try:
        # Load dataset
        X, y = load_dataset_from_config(config)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = create_simple_train_test_split(
            X, y, train_split=config.get('train_split', 0.8), 
            random_state=config.get('seed', 42)
        )
        
        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # Create model
        model = Perceptron(
            n_features=X_train.shape[1],
            learning_rate=config['learning_rate'],
            max_epochs=config['max_epochs'],
            tolerance=config.get('tolerance', 1e-6),
            activation=config.get('activation', 'step'),
            random_state=config.get('seed')
        )
        
        logger.info(f"Created model: {model.get_model_info()}")
        
        # Train model
        logger.info("Starting training...")
        training_history = model.fit(X_train, y_train)
        
        # Evaluate model
        test_accuracy, predictions = evaluate_model(model, X_test, y_test)
        
        # Log results
        logger.info("Training completed!")
        logger.info(f"Converged: {training_history.get('converged', False)}")
        logger.info(f"Training epochs: {len(training_history['epochs'])}")
        logger.info(f"Final training accuracy: {training_history['accuracy'][-1]:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save results
        save_results(config, model, training_history, test_accuracy)
        
        # Save model checkpoint
        if config.get('save_model', True):
            model_path = Path(config['model_dir']) / f"{config['experiment']}_model.pth"
            model.save_model(str(model_path))
        
        # Print summary
        print("\n" + "="*60)
        print(f"EXPERIMENT: {config['experiment']}")
        print("="*60)
        print(f"Dataset: {config['dataset']}")
        print(f"Samples: {X.shape[0]} ({X_train.shape[0]} train, {X_test.shape[0]} test)")
        print(f"Features: {X.shape[1]}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Max Epochs: {config['max_epochs']}")
        print(f"Activation: {config.get('activation', 'step')}")
        print("-"*60)
        print(f"Training Epochs: {len(training_history['epochs'])}")
        print(f"Converged: {'✓' if training_history.get('converged', False) else '✗'}")
        print(f"Final Training Accuracy: {training_history['accuracy'][-1]:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("="*60)
        
        if args.visualize:
            logger.info("Visualization requested but not implemented yet")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 