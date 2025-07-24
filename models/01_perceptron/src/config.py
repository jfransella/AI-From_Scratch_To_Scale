"""
Configuration management for Perceptron model using unified infrastructure.

Provides experiment-specific configurations using the shared TrainingConfig
and EvaluationConfig classes from the engine package.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

# Setup path for shared packages
sys.path.insert(0, str(Path(__file__).parent.parent))
import setup_path  # pylint: disable=unused-import,wrong-import-position

# Import from engine package
<<<<<<< HEAD
from engine import (  # noqa: E402
    TrainingConfig,
    EvaluationConfig,
=======
from engine import (  # noqa: E402  # pylint: disable=wrong-import-position
    EvaluationConfig,
    TrainingConfig,
>>>>>>> 3048305baf15e05456e16ae347f669533e0d7110
)

# Import constants - handle both direct and relative imports
try:
    from constants import (
        DATASET_SPECS,
        DEFAULT_ACTIVATION,
        DEFAULT_INIT_METHOD,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_EPOCHS,
        DEFAULT_TOLERANCE,
        MODEL_NAME,
        MODELS_DIR,
        PLOTS_DIR,
        get_experiment_info,
        validate_experiment,
    )
except ImportError:
    # Fallback for validation system - import from same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from constants import (
        DATASET_SPECS,
        DEFAULT_ACTIVATION,
        DEFAULT_INIT_METHOD,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_EPOCHS,
        DEFAULT_TOLERANCE,
        MODEL_NAME,
        MODELS_DIR,
        PLOTS_DIR,
        get_experiment_info,
        validate_experiment,
    )


def apply_wandb_defaults(config_dict: dict, experiment_name: str) -> dict:
    """Apply comprehensive wandb defaults according to integration plan standards."""

    # Auto-generate project name if not set
    if not config_dict.get("wandb_project"):
        config_dict["wandb_project"] = "ai-from-scratch-perceptron"

    # Auto-generate run name if not set
    if not config_dict.get("wandb_name"):
        config_dict["wandb_name"] = f"perceptron-{experiment_name}"

    # Enhanced auto-tagging system
    base_tags = [
        "perceptron",
        "module-1",
        "foundation",
        "engine-based",  # Perceptron uses engine framework
        experiment_name,
        "perceptron-rule",
    ]

    # Add experiment-specific tags
    if "strength" in experiment_name.lower():
        base_tags.append("strength")
    elif "weakness" in experiment_name.lower():
        base_tags.append("weakness")
    elif "debug" in experiment_name.lower():
        base_tags.append("debug")
    elif "comparison" in experiment_name.lower():
        base_tags.append("comparison")
    elif "xor" in experiment_name.lower():
        base_tags.append("limitation")

    # Add dataset-specific tags
    dataset_name = config_dict.get("dataset_name", "")
    if dataset_name:
        base_tags.append(dataset_name.lower().replace("_", "-"))

    config_dict["wandb_tags"] = base_tags

    # Auto-generate notes if not set
    if not config_dict.get("wandb_notes"):
        config_dict["wandb_notes"] = (
            f"Perceptron training on {experiment_name} dataset using "
            f"classic learning rule. Demonstrates fundamental neural "
            f"network concepts from 1957."
        )

    # Set group for organization
    if not config_dict.get("wandb_group"):
        config_dict["wandb_group"] = "module-1-foundations"

    # Set job type
    if not config_dict.get("wandb_job_type"):
        config_dict["wandb_job_type"] = "train"

    return config_dict


def create_wandb_config_dict(experiment_name: str, config_dict: dict) -> dict:
    """Create comprehensive wandb config dictionary with all relevant parameters."""
    return {
        # Experiment info
        "experiment_name": experiment_name,
        "dataset_name": config_dict.get("dataset_name"),
        # Model architecture
        "model_name": "Perceptron",
        "activation": "step",
        "learning_algorithm": "perceptron-rule",
        # Training parameters
        "learning_rate": config_dict.get("learning_rate"),
        "max_epochs": config_dict.get("max_epochs"),
        "convergence_threshold": config_dict.get("convergence_threshold"),
        "batch_size": config_dict.get("batch_size") or "full-batch",
        # Training characteristics
        "optimizer_type": config_dict.get("optimizer_type"),
        "early_stopping": config_dict.get("early_stopping"),
        "patience": config_dict.get("patience"),
        # Data parameters
        "validation_split": config_dict.get("validation_split"),
        "random_seed": config_dict.get("random_seed"),
        # Device and framework
        "device": config_dict.get("device"),
        "framework": "pytorch",
        # Logging parameters
        "log_freq": config_dict.get("log_freq"),
        "verbose": config_dict.get("verbose"),
    }


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

    # Get dataset specs
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
        # Enhanced wandb configuration following integration plan
        "use_wandb": False,  # Will be enabled via command line or overrides
        "wandb_project": None,  # Will be auto-generated
        "wandb_name": None,  # Will be auto-generated
        "wandb_tags": [],  # Will be auto-generated
        "wandb_notes": None,  # Will be auto-generated
        "wandb_mode": "online",  # "online", "offline", "disabled"
        # Advanced wandb features
        "wandb_watch_model": True,
        "wandb_watch_log": "gradients",  # "gradients", "parameters", "all"
        "wandb_watch_freq": 50,
        # Artifact configuration
        "wandb_log_checkpoints": True,
        "wandb_log_visualizations": True,
        "wandb_log_datasets": False,
        # Group and sweep support
        "wandb_group": None,  # Will be auto-generated
        "wandb_job_type": None,  # Will be auto-generated
        "wandb_sweep_id": None,
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
            "patience": 10,
        },
        "debug_linear": {
            "max_epochs": 50,
            "learning_rate": 0.1,
            "log_freq": 5,
            "patience": 20,
        },
        # Strength experiments
        "iris_binary": {
            "max_epochs": 100,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-6,
            "patience": 30,
            "wandb_tags": [MODEL_NAME.lower(), "strength", "iris"],
        },
        "linear_separable": {
            "max_epochs": 100,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-6,
            "patience": 30,
            "wandb_tags": [MODEL_NAME.lower(), "strength", "synthetic"],
        },
        "breast_cancer_binary": {
            "max_epochs": 200,
            "learning_rate": 0.01,  # Lower learning rate for more complex data
            "convergence_threshold": 1e-4,
            "patience": 50,
            "wandb_tags": [MODEL_NAME.lower(), "strength", "medical"],
        },
        # Weakness experiments
        "xor_problem": {
            "max_epochs": 1000,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-6,
            "patience": 200,  # Give it more time to demonstrate failure
            "early_stopping": False,  # Let it run to show non-convergence
            "wandb_tags": [MODEL_NAME.lower(), "weakness", "xor"],
        },
        "circles_dataset": {
            "max_epochs": 500,
            "learning_rate": 0.1,
            "convergence_threshold": 1e-4,
            "patience": 100,
            "wandb_tags": [MODEL_NAME.lower(), "weakness", "circles"],
        },
        "mnist_subset": {
            "max_epochs": 300,
            "learning_rate": 0.01,  # Lower learning rate for high-dimensional data
            "convergence_threshold": 1e-3,
            "patience": 75,
            "wandb_tags": [MODEL_NAME.lower(), "weakness", "mnist"],
        },
    }

    # Apply experiment-specific overrides
    if experiment_name in experiment_configs:
        base_config.update(experiment_configs[experiment_name])

    # Apply user overrides
    base_config.update(overrides)

    # Apply comprehensive wandb defaults following integration plan
    base_config = apply_wandb_defaults(base_config, experiment_name)

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
        "device": "cpu",
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
        "random_state": 42,
    }

    # Experiment-specific model configurations
    experiment_model_configs = {
        "debug_small": {"learning_rate": 0.1, "max_epochs": 20},
        "debug_linear": {"learning_rate": 0.1, "max_epochs": 50},
        "breast_cancer_binary": {
            "learning_rate": 0.01,  # Lower learning rate for complex data
            "init_method": "xavier",  # Better initialization for high-dimensional data
        },
        "mnist_subset": {
            "learning_rate": 0.01,
            "init_method": "xavier",
            "max_epochs": 300,
        },
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
        "description": dataset_spec["description"],
    }


def get_complete_config(experiment_name: str, **overrides) -> Dict[str, Any]:
    """
    Get complete configuration for an experiment including training,
    evaluation, and model configs.

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
        "experiment_info": get_experiment_info(experiment_name),
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
        type_str = (
            "Strength"
            if exp_info["is_strength"]
            else "Weakness" if exp_info["is_weakness"] else "Debug"
        )
        print(f"Type: {type_str}")

        # Model config
        model_config = config["model_config"]
        print("\nModel Configuration:")
        print(f"  Input Size: {model_config['input_size']}")
        print(f"  Learning Rate: {model_config['learning_rate']}")
        print(f"  Max Epochs: {model_config['max_epochs']}")
        print(f"  Activation: {model_config['activation']}")
        print(f"  Init Method: {model_config['init_method']}")

        # Training config
        training_config = config["training_config"]
        print("\nTraining Configuration:")
        print(f"  Optimizer: {training_config.optimizer_type}")
        print(f"  Convergence Threshold: {training_config.convergence_threshold}")
        print(f"  Patience: {training_config.patience}")
        print(f"  Validation Split: {training_config.validation_split}")
        print(f"  Device: {training_config.device}")

        print(f"\nDescription: {config['dataset_config']['description']}")
        print(f"{'='*60}\n")

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error printing config summary: {e}")  # pylint: disable=broad-except


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
        raise ValueError(
            f"Failed to get configuration for experiment '{experiment_name}': {e}"
        ) from e


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Perceptron configuration system...")

    for experiment in ["debug_small", "iris_binary", "xor_problem"]:
        try:
            print_config_summary(experiment)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error testing {experiment}: {e}")

    print("Configuration system test completed!")
