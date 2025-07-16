"""
Configuration management for Perceptron model using unified infrastructure.

Provides experiment-specific configurations using the shared TrainingConfig 
and EvaluationConfig classes from the engine package.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from engine.trainer import TrainingConfig
from engine.evaluator import EvaluationConfig
from constants import (
    MODEL_NAME, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS, DEFAULT_TOLERANCE,
    DEFAULT_ACTIVATION, DEFAULT_INIT_METHOD, ALL_EXPERIMENTS, DATASET_SPECS,
    validate_experiment, get_experiment_info, LOGS_DIR, MODELS_DIR, PLOTS_DIR
)


def get_training_config(experiment_name: str, **overrides) -> TrainingConfig:
    """
    Get TrainingConfig for a specific Perceptron experiment.
    
    Args:
        experiment_name: Name of the experiment to run
        **overrides: Any parameter overrides
        
    Returns:
        TrainingConfig instance configured for the experiment
        
    Raises:
        ValueError: If experiment_name is not supported
    """
    # Validate experiment name
    validate_experiment(experiment_name)
    
    # Get experiment info and dataset specs
    exp_info = get_experiment_info(experiment_name)
    dataset_spec = DATASET_SPECS[experiment_name]
    
    # Base configuration for all Perceptron experiments
    base_config = {
        # Experiment metadata
        "experiment_name": experiment_name,
        "model_name": MODEL_NAME,
        "dataset_name": dataset_spec["dataset_name"],
        
        # Training hyperparameters
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "batch_size": None,  # Full batch for Perceptron
        
        # Optimization (basic SGD for Perceptron)
        "optimizer_type": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
        
        # No learning rate scheduling for basic Perceptron
        "lr_scheduler": None,
        
        # Convergence and early stopping
        "convergence_threshold": DEFAULT_TOLERANCE,
        "patience": 50,
        "early_stopping": True,
        
        # Validation
        "validation_split": 0.2,
        "validation_freq": 1,
        
        # Checkpointing
        "save_best_model": True,
        "save_final_model": True,
        "checkpoint_freq": 0,  # No intermediate checkpoints for simple model
        "output_dir": str(MODELS_DIR.parent),  # outputs/
        
        # Logging and tracking
        "log_freq": 10,
        "verbose": True,
        "use_wandb": False,
        "wandb_project": "ai-from-scratch-perceptron",
        "wandb_tags": [MODEL_NAME.lower(), experiment_name],
        
        # Reproducibility
        "random_seed": 42,
        
        # Device
        "device": "cpu",  # Perceptron is simple enough for CPU
    }
    
    # Experiment-specific overrides
    experiment_configs = {
        # Debug experiments
        "debug_small": {
            "max_epochs": 20,
            "learning_rate": 0.1,
            "log_freq": 1,
            "verbose": True,
            "patience": 10
        },
        
        "debug_linear": {
            "max_epochs": 50,
            "learning_rate": 0.1,
            "log_freq": 5,
            "patience": 20
        },
        
        # Strength experiments
        "iris_binary": {
            "max_epochs": 100,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-6,
            "patience": 30,
            "wandb_tags": [MODEL_NAME.lower(), "strength", "iris"]
        },
        
        "linear_separable": {
            "max_epochs": 100,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-6,
            "patience": 30,
            "wandb_tags": [MODEL_NAME.lower(), "strength", "synthetic"]
        },
        
        "breast_cancer_binary": {
            "max_epochs": 200,
            "learning_rate": 0.01,  # Lower learning rate for more complex data
            "convergence_threshold": 1e-4,
            "patience": 50,
            "wandb_tags": [MODEL_NAME.lower(), "strength", "medical"]
        },
        
        # Weakness experiments
        "xor_problem": {
            "max_epochs": 1000,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-6,
            "patience": 200,  # Give it more time to demonstrate failure
            "early_stopping": False,  # Let it run to show non-convergence
            "wandb_tags": [MODEL_NAME.lower(), "weakness", "xor"]
        },
        
        "circles_dataset": {
            "max_epochs": 500,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-4,
            "patience": 100,
            "wandb_tags": [MODEL_NAME.lower(), "weakness", "circles"]
        },
        
        "mnist_subset": {
            "max_epochs": 300,
            "learning_rate": 0.01,  # Lower learning rate for high-dimensional data
            "convergence_threshold": 1e-3,
            "patience": 75,
            "wandb_tags": [MODEL_NAME.lower(), "weakness", "mnist"]
        }
    }
    
    # Apply experiment-specific overrides
    if experiment_name in experiment_configs:
        base_config.update(experiment_configs[experiment_name])
    
    # Apply user overrides
    base_config.update(overrides)
    
    # Create and return TrainingConfig
    return TrainingConfig(**base_config)


def get_evaluation_config(experiment_name: str, **overrides) -> EvaluationConfig:
    """
    Get EvaluationConfig for a specific Perceptron experiment.
    
    Args:
        experiment_name: Name of the experiment
        **overrides: Any parameter overrides
        
    Returns:
        EvaluationConfig instance configured for the experiment
    """
    # Validate experiment name
    validate_experiment(experiment_name)
    
    # Base evaluation configuration
    base_config = {
        # Metrics to compute
        "compute_accuracy": True,
        "compute_precision": True,
        "compute_recall": True,
        "compute_f1": True,
        "compute_confusion_matrix": True,
        
        # Per-class metrics (useful for multi-class if extended)
        "compute_per_class": True,
        
        # Prediction storage
        "store_predictions": True,
        "store_probabilities": True,
        "store_ground_truth": True,
        
        # Output configuration
        "verbose": True,
        "save_results": True,
        "output_path": str(PLOTS_DIR / f"{experiment_name}_evaluation.json"),
        
        # Device
        "device": "cpu"
    }
    
    # Apply user overrides
    base_config.update(overrides)
    
    # Create and return EvaluationConfig
    return EvaluationConfig(**base_config)


def get_model_config(experiment_name: str, **overrides) -> Dict[str, Any]:
    """
    Get model-specific configuration for Perceptron.
    
    Args:
        experiment_name: Name of the experiment
        **overrides: Any parameter overrides
        
    Returns:
        Dictionary with model configuration parameters
    """
    # Validate experiment name
    validate_experiment(experiment_name)
    
    # Get dataset specifications
    dataset_spec = DATASET_SPECS[experiment_name]
    
    # Base model configuration
    model_config = {
        "input_size": dataset_spec["input_size"],
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "tolerance": DEFAULT_TOLERANCE,
        "activation": DEFAULT_ACTIVATION,
        "init_method": DEFAULT_INIT_METHOD,
        "random_state": 42
    }
    
    # Experiment-specific model configurations
    experiment_model_configs = {
        "debug_small": {
            "learning_rate": 0.1,
            "max_epochs": 20
        },
        "debug_linear": {
            "learning_rate": 0.1,
            "max_epochs": 50
        },
        "breast_cancer_binary": {
            "learning_rate": 0.01,  # Lower learning rate for complex data
            "init_method": "xavier"  # Better initialization for high-dimensional data
        },
        "mnist_subset": {
            "learning_rate": 0.01,
            "init_method": "xavier",
            "max_epochs": 300
        }
    }
    
    # Apply experiment-specific model overrides
    if experiment_name in experiment_model_configs:
        model_config.update(experiment_model_configs[experiment_name])
    
    # Apply user overrides
    model_config.update(overrides)
    
    return model_config


def get_dataset_config(experiment_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with dataset configuration
    """
    # Validate experiment name
    validate_experiment(experiment_name)
    
    # Get dataset specifications
    dataset_spec = DATASET_SPECS[experiment_name]
    
    return {
        "dataset_name": dataset_spec["dataset_name"],
        "dataset_params": dataset_spec["dataset_params"],
        "input_size": dataset_spec["input_size"],
        "output_size": dataset_spec["output_size"],
        "expected_accuracy": dataset_spec["expected_accuracy"],
        "difficulty": dataset_spec["difficulty"],
        "description": dataset_spec["description"]
    }


def get_complete_config(experiment_name: str, **overrides) -> Dict[str, Any]:
    """
    Get complete configuration for an experiment including training, evaluation, and model configs.
    
    Args:
        experiment_name: Name of the experiment
        **overrides: Any parameter overrides
        
    Returns:
        Dictionary with all configuration components
    """
    return {
        "training_config": get_training_config(experiment_name, **overrides),
        "evaluation_config": get_evaluation_config(experiment_name, **overrides),
        "model_config": get_model_config(experiment_name, **overrides),
        "dataset_config": get_dataset_config(experiment_name),
        "experiment_info": get_experiment_info(experiment_name)
    }


def print_config_summary(experiment_name: str):
    """
    Print a summary of the configuration for an experiment.
    
    Args:
        experiment_name: Name of the experiment
    """
    try:
        config = get_complete_config(experiment_name)
        
        print(f"\n{'='*60}")
        print(f"Configuration Summary: {experiment_name}")
        print(f"{'='*60}")
        
        # Experiment info
        exp_info = config["experiment_info"]
        print(f"Model: {exp_info['model_name']}")
        print(f"Dataset: {config['dataset_config']['dataset_name']}")
        print(f"Expected Accuracy: {config['dataset_config']['expected_accuracy']:.3f}")
        print(f"Difficulty: {config['dataset_config']['difficulty']}")
        print(f"Type: {'Strength' if exp_info['is_strength'] else 'Weakness' if exp_info['is_weakness'] else 'Debug'}")
        
        # Model config
        model_config = config["model_config"]
        print(f"\nModel Configuration:")
        print(f"  Input Size: {model_config['input_size']}")
        print(f"  Learning Rate: {model_config['learning_rate']}")
        print(f"  Max Epochs: {model_config['max_epochs']}")
        print(f"  Activation: {model_config['activation']}")
        print(f"  Init Method: {model_config['init_method']}")
        
        # Training config
        training_config = config["training_config"]
        print(f"\nTraining Configuration:")
        print(f"  Optimizer: {training_config.optimizer_type}")
        print(f"  Convergence Threshold: {training_config.convergence_threshold}")
        print(f"  Patience: {training_config.patience}")
        print(f"  Validation Split: {training_config.validation_split}")
        print(f"  Device: {training_config.device}")
        
        print(f"\nDescription: {config['dataset_config']['description']}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error printing config summary: {e}")


# Legacy compatibility function for old code
def get_config(experiment_name: str, env: str = "default") -> Dict[str, Any]:
    """
    Legacy compatibility function that returns a dictionary config.
    
    This function provides backward compatibility with older code that expects
    a dictionary configuration instead of the new TrainingConfig objects.
    
    Args:
        experiment_name: Name of the experiment to run
        env: Environment (ignored in new implementation)
        
    Returns:
        Dictionary with complete configuration (flattened)
    """
    try:
        # Get all configs
        training_config = get_training_config(experiment_name)
        evaluation_config = get_evaluation_config(experiment_name)  
        model_config = get_model_config(experiment_name)
        dataset_config = get_dataset_config(experiment_name)
        
        # Flatten into single dictionary for backward compatibility
        legacy_config = {}
        
        # Add training config fields
        for field_name, field_value in training_config.__dict__.items():
            legacy_config[field_name] = field_value
        
        # Add model config fields
        legacy_config.update(model_config)
        
        # Add dataset config fields
        legacy_config.update(dataset_config)
        
        # Add some computed fields for compatibility
        legacy_config["model_name"] = MODEL_NAME
        legacy_config["experiment"] = experiment_name
        legacy_config["environment"] = env
        
        return legacy_config
        
    except Exception as e:
        raise ValueError(f"Failed to get configuration for experiment '{experiment_name}': {e}")


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Perceptron configuration system...")
    
    for experiment in ["debug_small", "iris_binary", "xor_problem"]:
        try:
            print_config_summary(experiment)
        except Exception as e:
            print(f"Error testing {experiment}: {e}")
    
    print("Configuration system test completed!") 