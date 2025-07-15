"""
Template for config.py - Configuration Management

This template provides the basic structure for managing configuration parameters
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistent configuration handling.

Replace [MODEL_NAME] with the actual model name (e.g., "Perceptron", "MLP", etc.)
Add model-specific experiments and parameters as needed.
"""

import torch
from typing import Dict, Any


def get_config(experiment_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific experiment.
    
    This function returns a configuration dictionary containing all parameters
    needed for training and evaluation. It combines base configuration with
    experiment-specific overrides.
    
    Args:
        experiment_name (str): Name of the experiment to configure
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ValueError: If experiment_name is not recognized
    """
    
    # Base configuration - common settings for all experiments
    base_config = {
        # Model architecture parameters
        'model_name': '[MODEL_NAME]',
        'input_size': 2,  # Will be overridden by specific experiments
        'hidden_size': None,  # Not all models have hidden layers
        'output_size': 1,  # Will be overridden by specific experiments
        
        # Training parameters
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'optimizer': 'adam',  # Options: 'adam', 'sgd'
        'loss_function': 'crossentropy',  # Options: 'crossentropy', 'mse', 'bce'
        'weight_decay': 0.0,
        'momentum': 0.9,  # For SGD
        
        # Data parameters
        'dataset': 'synthetic',  # Will be overridden by specific experiments
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'shuffle': True,
        'num_workers': 0,  # For DataLoader
        
        # Reproducibility
        'seed': 42,
        'deterministic': True,
        
        # Device configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Logging and monitoring
        'log_interval': 10,  # Log every N epochs
        'save_interval': 50,  # Save checkpoint every N epochs
        'early_stopping_patience': 20,  # Stop if no improvement for N epochs
        
        # Visualization
        'plot_types': ['loss_curve'],  # Default plots to generate
        
        # Experiment metadata
        'experiment': experiment_name,
        'description': f'[MODEL_NAME] experiment: {experiment_name}',
    }
    
    # Experiment-specific configurations
    experiments = {
        
        # Example: Simple XOR problem
        'xor': {
            'description': '[MODEL_NAME] on XOR problem - classic non-linear test',
            'dataset': 'xor',
            'dataset_params': {
                'n_samples': 1000,
                'noise': 0.1,
            },
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.1,
            'epochs': 200,
            'batch_size': 16,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
        },
        
        # Example: Iris dataset (easy case)
        'iris-easy': {
            'description': '[MODEL_NAME] on Iris dataset - Setosa vs others',
            'dataset': 'iris',
            'dataset_params': {
                'classes': ['setosa', 'versicolor', 'virginica'],
                'binary_target': 'setosa',  # Setosa vs others
            },
            'input_size': 4,
            'output_size': 1,
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 32,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'feature_importance'],
        },
        
        # Example: Iris dataset (hard case)
        'iris-hard': {
            'description': '[MODEL_NAME] on Iris dataset - Versicolor vs Virginica',
            'dataset': 'iris',
            'dataset_params': {
                'classes': ['versicolor', 'virginica'],
            },
            'input_size': 4,
            'output_size': 1,
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 16,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'feature_importance'],
        },
        
        # Example: MNIST (binary classification)
        'mnist-binary': {
            'description': '[MODEL_NAME] on MNIST - 0s vs 1s',
            'dataset': 'mnist',
            'dataset_params': {
                'classes': [0, 1],
                'flatten': True,
            },
            'input_size': 784,  # 28x28 flattened
            'output_size': 1,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'sample_predictions'],
        },
        
        # Example: MNIST (multi-class)
        'mnist-multi': {
            'description': '[MODEL_NAME] on MNIST - all 10 digits',
            'dataset': 'mnist',
            'dataset_params': {
                'classes': list(range(10)),
                'flatten': True,
            },
            'input_size': 784,  # 28x28 flattened
            'output_size': 10,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'confusion_matrix', 'sample_predictions'],
        },
        
        # Example: Synthetic circles dataset
        'circles': {
            'description': '[MODEL_NAME] on concentric circles - non-linear test',
            'dataset': 'circles',
            'dataset_params': {
                'n_samples': 1000,
                'noise': 0.1,
                'factor': 0.5,
            },
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.01,
            'epochs': 300,
            'batch_size': 32,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
        },
        
        # Example: Synthetic moons dataset
        'moons': {
            'description': '[MODEL_NAME] on moons dataset - non-linear test',
            'dataset': 'moons',
            'dataset_params': {
                'n_samples': 1000,
                'noise': 0.1,
            },
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.01,
            'epochs': 200,
            'batch_size': 32,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
        },
        
        # TODO: Add more experiments as needed for this model
        # Common patterns:
        # - Strength datasets: Show where model excels
        # - Weakness datasets: Expose model limitations
        # - Progressive complexity: Simple -> Real-world -> Complex
        
    }
    
    # Get experiment configuration
    if experiment_name not in experiments:
        available_experiments = list(experiments.keys())
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {available_experiments}"
        )
    
    # Merge base config with experiment-specific config
    config = base_config.copy()
    config.update(experiments[experiment_name])
    
    # Validate configuration
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = [
        'model_name', 'input_size', 'output_size', 'learning_rate',
        'batch_size', 'epochs', 'dataset', 'experiment'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Check value ranges
    if config['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")
    
    if config['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")
    
    if config['epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if config['input_size'] <= 0:
        raise ValueError("Input size must be positive")
    
    if config['output_size'] <= 0:
        raise ValueError("Output size must be positive")
    
    # Check split ratios
    splits = [config['train_split'], config['val_split'], config['test_split']]
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test splits must sum to 1.0")
    
    # Check optimizer
    valid_optimizers = ['adam', 'sgd']
    if config['optimizer'] not in valid_optimizers:
        raise ValueError(f"Optimizer must be one of: {valid_optimizers}")
    
    # Check loss function
    valid_losses = ['crossentropy', 'mse', 'bce']
    if config['loss_function'] not in valid_losses:
        raise ValueError(f"Loss function must be one of: {valid_losses}")


def get_available_experiments() -> list:
    """
    Get list of available experiments for this model.
    
    Returns:
        list: List of available experiment names
    """
    # This is a bit of a hack to get the experiment names without duplicating code
    dummy_config = {
        'model_name': '[MODEL_NAME]',
        'input_size': 2,
        'hidden_size': None,
        'output_size': 1,
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'optimizer': 'adam',
        'loss_function': 'crossentropy',
        'weight_decay': 0.0,
        'momentum': 0.9,
        'dataset': 'synthetic',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'shuffle': True,
        'num_workers': 0,
        'seed': 42,
        'deterministic': True,
        'device': 'cpu',
        'log_interval': 10,
        'save_interval': 50,
        'early_stopping_patience': 20,
        'plot_types': ['loss_curve'],
        'experiment': 'dummy',
        'description': 'dummy',
    }
    
    experiments = {
        'xor': dummy_config,
        'iris-easy': dummy_config,
        'iris-hard': dummy_config,
        'mnist-binary': dummy_config,
        'mnist-multi': dummy_config,
        'circles': dummy_config,
        'moons': dummy_config,
    }
    
    return list(experiments.keys())


def print_config(config: Dict[str, Any]) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to print
    """
    print(f"\n{'='*50}")
    print(f"Configuration for {config['model_name']}")
    print(f"Experiment: {config['experiment']}")
    print(f"{'='*50}")
    
    # Group parameters by category
    categories = {
        'Model Architecture': ['model_name', 'input_size', 'hidden_size', 'output_size'],
        'Training': ['learning_rate', 'batch_size', 'epochs', 'optimizer', 'loss_function'],
        'Data': ['dataset', 'train_split', 'val_split', 'test_split'],
        'Other': ['seed', 'device', 'description']
    }
    
    for category, params in categories.items():
        print(f"\n{category}:")
        for param in params:
            if param in config:
                value = config[param]
                print(f"  {param}: {value}")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    # Example usage
    print("Available experiments:")
    experiments = get_available_experiments()
    for exp in experiments:
        print(f"  - {exp}")
    
    print("\nExample configuration:")
    config = get_config('xor')
    print_config(config) 