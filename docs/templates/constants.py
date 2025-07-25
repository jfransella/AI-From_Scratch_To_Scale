"""
Template for constants.py - Model Constants and Metadata

This template provides constants and metadata for neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and historical accuracy.

Replace MODEL_NAME with the actual model name (e.g., "Perceptron", "MLP", etc.)
Replace YEAR_INTRODUCED with the year the model was introduced
Replace AUTHORS with the original authors
"""

from pathlib import Path
from typing import Dict, Any, List

# =============================================================================
# MODEL METADATA
# =============================================================================

# Basic model information
MODEL_NAME = "ModelTemplate"
MODEL_VERSION = "1.0.0"
MODEL_DESCRIPTION = "Template model for AI From Scratch to Scale project"

# Historical information
YEAR_INTRODUCED = 2024
AUTHORS = ["AI From Scratch Team"]
ORIGINAL_PAPER_TITLE = "Template Model Implementation"
ORIGINAL_PAPER_URL = "https://github.com/ai-from-scratch-to-scale"

# Key innovations and contributions
KEY_INNOVATIONS = [
    "Template implementation for educational purposes",
    "Demonstrates best practices for model development",
    "Provides foundation for more complex models",
]

# Problems solved by this model
PROBLEMS_SOLVED = [
    "Template for model implementation",
    "Educational demonstration",
    "Best practices showcase",
]

# Limitations of this model
LIMITATIONS = [
    "Template only - not a real model",
    "For demonstration purposes only",
    "Not intended for production use",
]

# =============================================================================
# MODEL ARCHITECTURE CONSTANTS
# =============================================================================

# Default architecture parameters
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MAX_EPOCHS = 100
DEFAULT_TOLERANCE = 1e-6
DEFAULT_BATCH_SIZE = 32

# Activation function options
SUPPORTED_ACTIVATIONS = ["relu", "sigmoid", "tanh", "leaky_relu", "step"]
DEFAULT_ACTIVATION = "relu"

# Weight initialization options
SUPPORTED_INIT_METHODS = ["xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform", "zeros", "normal", "uniform"]
DEFAULT_INIT_METHOD = "xavier_normal"

# Optimizer options
SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw", "rmsprop"]
DEFAULT_OPTIMIZER = "adam"

# Training parameters
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 10.0
MIN_EPOCHS = 1
MAX_EPOCHS = 10000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1024

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# All experiments supported by this model
ALL_EXPERIMENTS = [
    "debug",
    "quick_test", 
    "standard",
    "production",
]

# Debug experiments - for quick testing and development
DEBUG_EXPERIMENTS = ["debug", "quick_test"]

# Standard experiments - for normal training
STANDARD_EXPERIMENTS = ["standard", "production"]

# Experiment difficulty levels
EXPERIMENT_DIFFICULTY = {
    "debug": "trivial",
    "quick_test": "easy", 
    "standard": "medium",
    "production": "hard",
}

# =============================================================================
# DATASET SPECIFICATIONS
# =============================================================================

DATASET_SPECS = {
    "debug": {
        "dataset_name": "synthetic_debug",
        "dataset_params": {
            "n_samples": 20,
            "n_features": 2,
            "noise": 0.0,
            "random_state": 42,
        },
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 1.0,
        "difficulty": "trivial",
        "description": "Small synthetic dataset for quick testing",
        "convergence_epochs": 10,
    },
    "quick_test": {
        "dataset_name": "synthetic_quick",
        "dataset_params": {
            "n_samples": 50,
            "n_features": 2,
            "noise": 0.05,
            "random_state": 42,
        },
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.95,
        "difficulty": "easy",
        "description": "Small dataset with minimal noise for testing",
        "convergence_epochs": 25,
    },
    "standard": {
        "dataset_name": "synthetic_standard",
        "dataset_params": {
            "n_samples": 200,
            "n_features": 2,
            "noise": 0.1,
            "random_state": 42,
        },
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.85,
        "difficulty": "medium",
        "description": "Standard synthetic dataset for training",
        "convergence_epochs": 50,
    },
    "production": {
        "dataset_name": "synthetic_production",
        "dataset_params": {
            "n_samples": 1000,
            "n_features": 2,
            "noise": 0.1,
            "random_state": 42,
        },
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.90,
        "difficulty": "hard",
        "description": "Large synthetic dataset for production testing",
        "convergence_epochs": 100,
    },
}

# =============================================================================
# WANDB INTEGRATION CONSTANTS
# =============================================================================

# W&B project configuration
WANDB_PROJECT_NAME = f"ai-from-scratch-{MODEL_NAME.lower()}"
WANDB_ENTITY = None  # Set to your W&B entity/team name
WANDB_TAGS = [MODEL_NAME.lower(), "template", "educational"]

# W&B experiment tracking
WANDB_LOG_FREQUENCY = 10  # Log every N epochs
WANDB_WATCH_MODEL = True  # Track gradients and parameters
WANDB_SAVE_CODE = True    # Save source code with runs

# Metrics to track
TRACKED_METRICS = [
    "loss",
    "accuracy", 
    "precision",
    "recall",
    "f1_score",
    "learning_rate",
    "epoch_time",
]

# =============================================================================
# FILE PATH CONSTANTS
# =============================================================================

# Base directories
MODEL_DIR = Path(__file__).parent
ROOT_DIR = MODEL_DIR.parent
OUTPUTS_DIR = MODEL_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "visualizations"
NOTEBOOKS_DIR = MODEL_DIR / "notebooks"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

# Ensure output directories exist
for directory in [OUTPUTS_DIR, LOGS_DIR, MODELS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File naming patterns
MODEL_CHECKPOINT_PATTERN = "{experiment}_epoch_{epoch:03d}.pth"
MODEL_FINAL_PATTERN = "{experiment}_final_model.pth"
PLOT_FILENAME_PATTERN = "{experiment}_{plot_type}.png"
LOG_FILENAME_PATTERN = "training_{timestamp}.log"
RESULTS_FILENAME_PATTERN = "{experiment}_results.json"

# =============================================================================
# DEVICE AND ENVIRONMENT CONSTANTS
# =============================================================================

# Supported compute devices
SUPPORTED_DEVICES = ["cpu", "cuda", "mps"]
DEFAULT_DEVICE = "cpu"

# Performance settings
DEFAULT_NUM_WORKERS = 4  # For data loading
DEFAULT_PIN_MEMORY = True  # For GPU training
DEFAULT_PERSISTENT_WORKERS = True  # For data loading efficiency

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_learning_rate(lr: float) -> float:
    """Validate and clip learning rate to acceptable range."""
    if lr < MIN_LEARNING_RATE:
        return MIN_LEARNING_RATE
    elif lr > MAX_LEARNING_RATE:
        return MAX_LEARNING_RATE
    return lr


def validate_epochs(epochs: int) -> int:
    """Validate and clip epochs to acceptable range."""
    if epochs < MIN_EPOCHS:
        return MIN_EPOCHS
    elif epochs > MAX_EPOCHS:
        return MAX_EPOCHS
    return epochs


def validate_activation(activation: str) -> str:
    """Validate activation function name."""
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation: {activation}. "
            f"Supported: {SUPPORTED_ACTIVATIONS}"
        )
    return activation


def validate_init_method(init_method: str) -> str:
    """Validate weight initialization method."""
    if init_method not in SUPPORTED_INIT_METHODS:
        raise ValueError(
            f"Unsupported init method: {init_method}. "
            f"Supported: {SUPPORTED_INIT_METHODS}"
        )
    return init_method


def validate_experiment(experiment_name: str) -> str:
    """Validate experiment name."""
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available: {ALL_EXPERIMENTS}"
        )
    return experiment_name


def get_experiment_info(experiment_name: str) -> Dict[str, Any]:
    """Get detailed information about an experiment."""
    validate_experiment(experiment_name)
    
    dataset_spec = DATASET_SPECS.get(experiment_name, {})
    
    return {
        "name": experiment_name,
        "description": dataset_spec.get("description", "No description"),
        "dataset_name": dataset_spec.get("dataset_name", "unknown"),
        "input_size": dataset_spec.get("input_size", 2),
        "output_size": dataset_spec.get("output_size", 1),
        "expected_accuracy": dataset_spec.get("expected_accuracy", 0.8),
        "difficulty": dataset_spec.get("difficulty", "medium"),
        "dataset_params": dataset_spec.get("dataset_params", {}),
    }


def validate_parameter(param_name: str, value: Any) -> Any:
    """Validate a specific parameter value."""
    if param_name == "learning_rate":
        return validate_learning_rate(value)
    elif param_name == "max_epochs":
        return validate_epochs(value)
    elif param_name == "activation":
        return validate_activation(value)
    elif param_name == "init_method":
        return validate_init_method(value)
    elif param_name == "experiment":
        return validate_experiment(value)
    else:
        return value


def get_config_for_experiment(experiment_name: str) -> Dict[str, Any]:
    """
    Get complete configuration for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Complete configuration dictionary
    """
    validate_experiment(experiment_name)
    
    dataset_spec = DATASET_SPECS[experiment_name]
    
    return {
        # Experiment metadata
        "experiment_name": experiment_name,
        "model_name": MODEL_NAME,
        "description": dataset_spec["description"],
        
        # Model architecture
        "input_size": dataset_spec["input_size"],
        "output_size": dataset_spec["output_size"],
        "activation": DEFAULT_ACTIVATION,
        "init_method": DEFAULT_INIT_METHOD,
        
        # Training configuration
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "tolerance": DEFAULT_TOLERANCE,
        "batch_size": DEFAULT_BATCH_SIZE,
        
        # Dataset configuration
        "dataset_name": dataset_spec["dataset_name"],
        "dataset_params": dataset_spec["dataset_params"],
        
        # Expected performance
        "expected_accuracy": dataset_spec["expected_accuracy"],
        "difficulty": dataset_spec["difficulty"],
        "convergence_epochs": dataset_spec.get("convergence_epochs", DEFAULT_MAX_EPOCHS),
        
        # Output configuration
        "output_dir": str(OUTPUTS_DIR),
        "save_model": True,
        "save_plots": True,
        "verbose": True,
    }


def get_wandb_config(experiment_name: str) -> Dict[str, Any]:
    """Get W&B configuration for an experiment."""
    return {
        "project": WANDB_PROJECT_NAME,
        "entity": WANDB_ENTITY,
        "tags": WANDB_TAGS + [experiment_name],
        "name": f"{MODEL_NAME}_{experiment_name}",
        "notes": f"{MODEL_NAME} training on {experiment_name} experiment",
        "config": get_config_for_experiment(experiment_name),
    }


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    device_info = {
        "available_devices": SUPPORTED_DEVICES,
        "default_device": DEFAULT_DEVICE,
        "recommended_device": DEFAULT_DEVICE,
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            device_info["cuda_available"] = True
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_info["recommended_device"] = "cuda"
        else:
            device_info["cuda_available"] = False
            
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info["mps_available"] = True
            device_info["recommended_device"] = "mps"
        else:
            device_info["mps_available"] = False
            
    except ImportError:
        device_info["pytorch_available"] = False
        
    return device_info


# =============================================================================
# MODEL-SPECIFIC CONSTANTS
# =============================================================================

# Add model-specific constants here
# These should be customized for each model implementation

MODEL_SPECIFIC_CONSTANTS = {
    "example_param": 42,
    "example_flag": True,
    "example_list": [1, 2, 3, 4, 5],
} 