"""
Configuration management for Perceptron model.

Provides experiment-specific configurations with base defaults
and parameter validation following the project's configuration patterns.
"""

from typing import Dict, Any, Optional
import logging
from constants import (
    DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS, DEFAULT_TOLERANCE,
    DEFAULT_ACTIVATION, DEFAULT_INIT_METHOD, STANDARD_EXPERIMENTS,
    validate_learning_rate, validate_epochs, validate_activation,
    MODEL_NAME, LOGS_DIR, MODELS_DIR, PLOTS_DIR
)


def get_config(experiment_name: str, env: str = "default") -> Dict[str, Any]:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment to run
        env: Environment (default, debug, production)
        
    Returns:
        Dictionary with complete configuration
        
    Raises:
        ValueError: If experiment_name is not supported
        
    Example:
        config = get_config("iris_binary")
        config = get_config("xor_failure", env="debug")
    """
    # Base configuration - common to all experiments
    base_config = {
        # Model information
        "model_name": MODEL_NAME,
        "experiment": experiment_name,
        "environment": env,
        
        # Model architecture
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "tolerance": DEFAULT_TOLERANCE,
        "activation": DEFAULT_ACTIVATION,
        "init_method": DEFAULT_INIT_METHOD,
        
        # Training settings
        "shuffle_data": True,
        "validate_every": 10,
        "early_stopping": True,
        "patience": 20,
        
        # Logging and output
        "log_level": "INFO",
        "save_model": True,
        "save_plots": True,
        "log_dir": str(LOGS_DIR),
        "model_dir": str(MODELS_DIR),
        "plot_dir": str(PLOTS_DIR),
        
        # Reproducibility
        "seed": 42,
        "deterministic": True,
        
        # Device
        "device": "cpu",  # Perceptron is simple enough for CPU
        
        # Visualization
        "plot_decision_boundary": True,
        "plot_loss_curve": True,
        "plot_weight_evolution": True,
        
        # Dataset defaults
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
    }
    
    # Experiment-specific configurations
    experiments = {
        # Quick test for development
        "quick_test": {
            "dataset": "linear",
            "dataset_params": {
                "n_samples": 100,
                "n_features": 2,
                "n_classes": 2,
                "noise": 0.1,
                "random_state": 42
            },
            "max_epochs": 10,
            "learning_rate": 0.1,
            "log_level": "DEBUG"
        },
        
        # Iris binary classification - should work well
        "iris_binary": {
            "dataset": "iris_binary",
            "dataset_params": {
                "target_class": "setosa",  # setosa vs. others
                "random_state": 42
            },
            "max_epochs": 100,
            "learning_rate": 0.1,
            "tolerance": 1e-6,
            "early_stopping": True,
            "patience": 10
        },
        
        # Simple linear 2D problem - pedagogical example
        "linear_simple": {
            "dataset": "linear",
            "dataset_params": {
                "n_samples": 200,
                "n_features": 2,
                "n_classes": 2,
                "noise": 0.05,
                "random_state": 42
            },
            "max_epochs": 50,
            "learning_rate": 0.1,
            "tolerance": 1e-6
        },
        
        # XOR problem - should fail, demonstrates limitation
        "xor_failure": {
            "dataset": "xor",
            "dataset_params": {
                "n_samples": 1000,
                "noise": 0.1,
                "random_state": 42
            },
            "max_epochs": 1000,
            "learning_rate": 0.1,
            "tolerance": 1e-6,
            "early_stopping": False,  # Let it run to show it won't converge
            "note": "Expected to fail - demonstrates XOR limitation"
        },
        
        # Circles problem - also should fail
        "circles_failure": {
            "dataset": "circles",
            "dataset_params": {
                "n_samples": 500,
                "noise": 0.1,
                "factor": 0.5,
                "random_state": 42
            },
            "max_epochs": 500,
            "learning_rate": 0.1,
            "tolerance": 1e-6,
            "early_stopping": False,
            "note": "Expected to fail - non-linearly separable"
        },
        
        # Learning rate exploration
        "lr_exploration": {
            "dataset": "linear",
            "dataset_params": {
                "n_samples": 300,
                "n_features": 2,
                "n_classes": 2,
                "noise": 0.1,
                "random_state": 42
            },
            "max_epochs": 100,
            "learning_rate": 0.01,  # Different LR for comparison
            "tolerance": 1e-6
        },
        
        # Debug experiment
        "debug": {
            "dataset": "linear",
            "dataset_params": {
                "n_samples": 50,
                "n_features": 2,
                "n_classes": 2,
                "noise": 0.05,
                "random_state": 42
            },
            "max_epochs": 5,
            "learning_rate": 0.1,
            "log_level": "DEBUG",
            "validate_every": 1
        }
    }
    
    # Check if experiment exists
    if experiment_name not in experiments:
        available = list(experiments.keys())
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available experiments: {available}")
    
    # Merge base config with experiment-specific config
    config = {**base_config, **experiments[experiment_name]}
    
    # Apply environment-specific modifications
    if env == "debug":
        config.update({
            "log_level": "DEBUG",
            "max_epochs": min(config["max_epochs"], 10),
            "validate_every": 1,
            "plot_decision_boundary": True,
            "plot_loss_curve": True
        })
    elif env == "production":
        config.update({
            "log_level": "WARNING",
            "save_plots": False,
            "plot_decision_boundary": False
        })
    
    # Validate configuration
    config = _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Validate numeric parameters
    original_lr = config["learning_rate"]
    config["learning_rate"] = validate_learning_rate(config["learning_rate"])
    if config["learning_rate"] != original_lr:
        logger.warning(f"Learning rate adjusted from {original_lr} to {config['learning_rate']}")
    
    original_epochs = config["max_epochs"]
    config["max_epochs"] = validate_epochs(config["max_epochs"])
    if config["max_epochs"] != original_epochs:
        logger.warning(f"Max epochs adjusted from {original_epochs} to {config['max_epochs']}")
    
    # Validate activation function
    config["activation"] = validate_activation(config["activation"])
    
    # Validate splits sum to 1.0
    total_split = config["train_split"] + config["val_split"] + config["test_split"]
    if abs(total_split - 1.0) > 1e-6:
        logger.warning(f"Data splits sum to {total_split}, should sum to 1.0")
        # Normalize splits
        config["train_split"] /= total_split
        config["val_split"] /= total_split
        config["test_split"] /= total_split
    
    # Ensure patience is less than max_epochs
    if config.get("patience", 0) >= config["max_epochs"]:
        config["patience"] = config["max_epochs"] // 2
        logger.warning(f"Patience adjusted to {config['patience']}")
    
    return config


def get_experiment_info(experiment_name: str) -> Dict[str, Any]:
    """
    Get information about an experiment without full configuration.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with experiment metadata
        
    Example:
        info = get_experiment_info("xor_failure")
        print(info["description"])
    """
    experiment_descriptions = {
        "quick_test": {
            "description": "Quick test with small linear dataset",
            "purpose": "Development and testing",
            "expected_outcome": "Should converge quickly",
            "difficulty": "Easy"
        },
        "iris_binary": {
            "description": "Binary classification on Iris dataset (setosa vs others)",
            "purpose": "Demonstrate successful learning on real data",
            "expected_outcome": "Should achieve perfect accuracy",
            "difficulty": "Easy"
        },
        "linear_simple": {
            "description": "Simple 2D linearly separable problem",
            "purpose": "Educational demonstration of decision boundary",
            "expected_outcome": "Should converge with clear boundary",
            "difficulty": "Easy"
        },
        "xor_failure": {
            "description": "XOR problem - classic Perceptron limitation",
            "purpose": "Demonstrate fundamental limitation",
            "expected_outcome": "Should fail to converge",
            "difficulty": "Impossible"
        },
        "circles_failure": {
            "description": "Concentric circles - non-linearly separable",
            "purpose": "Another demonstration of limitation",
            "expected_outcome": "Should fail to converge",
            "difficulty": "Impossible"
        },
        "lr_exploration": {
            "description": "Test different learning rate",
            "purpose": "Explore hyperparameter sensitivity",
            "expected_outcome": "Should converge but with different dynamics",
            "difficulty": "Easy"
        },
        "debug": {
            "description": "Minimal debug configuration",
            "purpose": "Quick testing and debugging",
            "expected_outcome": "Should work with detailed logging",
            "difficulty": "Easy"
        }
    }
    
    if experiment_name not in experiment_descriptions:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return experiment_descriptions[experiment_name]


def list_experiments() -> Dict[str, str]:
    """
    List all available experiments with brief descriptions.
    
    Returns:
        Dictionary mapping experiment names to descriptions
        
    Example:
        experiments = list_experiments()
        for name, desc in experiments.items():
            print(f"{name}: {desc}")
    """
    try:
        config = get_config("quick_test")  # Get any config to extract experiment list
        # Extract experiment names from get_config function
        experiments = {}
        
        # We'll reconstruct this by trying to get info for known experiments
        known_experiments = [
            "quick_test", "iris_binary", "linear_simple", "xor_failure",
            "circles_failure", "lr_exploration", "debug"
        ]
        
        for exp_name in known_experiments:
            try:
                info = get_experiment_info(exp_name)
                experiments[exp_name] = info["description"]
            except ValueError:
                continue
                
        return experiments
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to list experiments: {e}")
        return {}


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Perceptron configuration...")
    
    # Test basic config loading
    config = get_config("iris_binary")
    print(f"✓ Loaded config for iris_binary: {config['dataset']}")
    
    # Test experiment info
    info = get_experiment_info("xor_failure")
    print(f"✓ XOR experiment info: {info['expected_outcome']}")
    
    # Test experiment listing
    experiments = list_experiments()
    print(f"✓ Available experiments: {list(experiments.keys())}")
    
    print("Configuration tests passed!") 