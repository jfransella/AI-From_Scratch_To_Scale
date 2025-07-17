"""
Constants for Perceptron model implementation.

This file contains all fixed values, metadata, and configuration ranges for the
Perceptron model, following the project's established patterns.
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# Model Metadata
# =============================================================================

MODEL_NAME = "Perceptron"
MODEL_VERSION = "1.0.0"
YEAR_INTRODUCED = 1957
PAPER_TITLE = "The Perceptron: A Perceiving and Recognizing Automaton"
AUTHORS = ["Frank Rosenblatt"]
INSTITUTION = "Cornell Aeronautical Laboratory"
HISTORICAL_CONTEXT = "First artificial neural network, inspired by biological neurons"

# Key innovation
KEY_INNOVATION = "First learning algorithm for artificial neurons"
LEARNING_ALGORITHM = "Perceptron Learning Rule"
MATHEMATICAL_FOUNDATION = "Linear threshold function with weight updates"

# =============================================================================
# Model Architecture Constants
# =============================================================================

# Default architecture parameters
DEFAULT_INPUT_SIZE = 2
DEFAULT_OUTPUT_SIZE = 1
DEFAULT_ACTIVATION = "step"  # Classic perceptron uses step function
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_BIAS = True

# Available activation functions
ACTIVATION_FUNCTIONS = {
    "step": "Classic step function (0 or 1)",
    "sign": "Sign function (-1 or 1)",
    "linear": "Linear activation (for comparison)"
}

# =============================================================================
# Training Constants
# =============================================================================

# Default training parameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 1  # Classic perceptron uses online learning
DEFAULT_CONVERGENCE_THRESHOLD = 0.0  # Stop when error is zero
DEFAULT_MAX_ITERATIONS = 1000

# Learning rate ranges
MIN_LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 1.0
RECOMMENDED_LEARNING_RATE = 0.1

# =============================================================================
# File Paths (Windows-style)
# =============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "visualizations"

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GENERATED_DATA_DIR = DATA_DIR / "generated"

# Template paths
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_DIR = BASE_DIR / "docs"

# =============================================================================
# Experiment Configurations
# =============================================================================

# Strength experiments (where Perceptron excels)
STRENGTH_EXPERIMENTS = [
    "and_gate",
    "or_gate",
    "iris_easy",
    "mnist_binary"
]

# Weakness experiments (where Perceptron fails)
WEAKNESS_EXPERIMENTS = [
    "xor_gate",
    "iris_hard",
    "circles",
    "moons"
]

# Debug experiments
DEBUG_EXPERIMENTS = [
    "debug_small",
    "debug_overfit"
]

# All available experiments
ALL_EXPERIMENTS = STRENGTH_EXPERIMENTS + WEAKNESS_EXPERIMENTS + DEBUG_EXPERIMENTS

# =============================================================================
# Dataset Specifications
# =============================================================================

DATASET_SPECS = {
    "and_gate": {
        "description": "AND logic gate - linearly separable",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 1000,
        "expected_accuracy": 1.0,
        "difficulty": "easy"
    },
    "or_gate": {
        "description": "OR logic gate - linearly separable",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 1000,
        "expected_accuracy": 1.0,
        "difficulty": "easy"
    },
    "xor_gate": {
        "description": "XOR logic gate - NOT linearly separable",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 1000,
        "expected_accuracy": 0.5,  # Random performance
        "difficulty": "impossible"
    },
    "iris_easy": {
        "description": "Iris dataset - Setosa vs others (linearly separable)",
        "input_size": 4,
        "output_size": 1,
        "num_samples": 150,
        "expected_accuracy": 1.0,
        "difficulty": "easy"
    },
    "iris_hard": {
        "description": "Iris dataset - Versicolor vs Virginica (harder)",
        "input_size": 4,
        "output_size": 1,
        "num_samples": 100,
        "expected_accuracy": 0.7,
        "difficulty": "medium"
    },
    "mnist_binary": {
        "description": "MNIST digits - 0s vs 1s (linearly separable)",
        "input_size": 784,
        "output_size": 1,
        "num_samples": 12665,
        "expected_accuracy": 0.95,
        "difficulty": "medium"
    }
}

# =============================================================================
# Visualization Settings
# =============================================================================

# Plot types supported for this model
SUPPORTED_PLOT_TYPES = [
    "loss_curve",
    "decision_boundary",
    "weight_evolution",
    "convergence_analysis"
]

# Plot configuration
PLOT_CONFIG = {
    "figsize": (10, 6),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl",
    "save_format": "png"
}

# Decision boundary plot settings (for 2D data)
DECISION_BOUNDARY_CONFIG = {
    "resolution": 0.01,
    "alpha": 0.8,
    "contour_levels": 20,
    "point_size": 50
}

# =============================================================================
# Validation Ranges
# =============================================================================

# Parameter validation ranges
VALIDATION_RANGES = {
    "learning_rate": (MIN_LEARNING_RATE, MAX_LEARNING_RATE),
    "epochs": (1, 10000),
    "batch_size": (1, 1000),
    "input_size": (1, 100000),
    "output_size": (1, 1000)
}

# =============================================================================
# Error Messages
# =============================================================================

ERROR_MESSAGES = {
    "invalid_learning_rate": (
        f"Learning rate must be between {MIN_LEARNING_RATE} and {MAX_LEARNING_RATE}"
    ),
    "invalid_epochs": "Number of epochs must be positive",
    "invalid_batch_size": "Batch size must be positive",
    "invalid_input_size": "Input size must be positive",
    "invalid_output_size": "Output size must be positive",
    "invalid_activation": f"Activation must be one of {list(ACTIVATION_FUNCTIONS.keys())}",
    "convergence_failed": "Model failed to converge within maximum iterations",
    "data_not_linearly_separable": "Data may not be linearly separable - Perceptron cannot learn"
}

# =============================================================================
# Model Capabilities and Limitations
# =============================================================================

MODEL_CAPABILITIES = [
    "Learn linearly separable patterns",
    "Binary classification",
    "Online learning (single sample updates)",
    "Guaranteed convergence on linearly separable data",
    "Simple and interpretable weights"
]

MODEL_LIMITATIONS = [
    "Cannot learn non-linearly separable patterns (e.g., XOR)",
    "Binary classification only",
    "No hidden layers",
    "Limited expressiveness",
    "Sensitive to learning rate"
]

# =============================================================================
# Utility Functions
# =============================================================================


def get_experiment_info(experiment_name: str) -> Dict[str, Any]:
    """
    Get information about a specific experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Dictionary containing experiment information

    Raises:
        ValueError: If experiment name is not recognized
    """
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {ALL_EXPERIMENTS}"
        )

    if experiment_name in DATASET_SPECS:
        return DATASET_SPECS[experiment_name]

    # Return default info for debug experiments
    return {
        "description": f"Debug experiment: {experiment_name}",
        "input_size": DEFAULT_INPUT_SIZE,
        "output_size": DEFAULT_OUTPUT_SIZE,
        "difficulty": "debug"
    }


def validate_parameter(param_name: str, value: Any) -> bool:
    """
    Validate a parameter value against defined ranges.

    Args:
        param_name: Name of the parameter
        value: Value to validate

    Returns:
        True if valid, False otherwise
    """
    if param_name not in VALIDATION_RANGES:
        return True  # Unknown parameters are considered valid

    min_val, max_val = VALIDATION_RANGES[param_name]
    return min_val <= value <= max_val


def get_expected_performance(experiment_name: str) -> Dict[str, Any]:
    """
    Get expected performance metrics for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Dictionary containing expected performance metrics
    """
    if experiment_name in DATASET_SPECS:
        spec = DATASET_SPECS[experiment_name]
        return {
            "expected_accuracy": spec.get("expected_accuracy", 0.8),
            "difficulty": spec.get("difficulty", "medium"),
            "should_converge": spec.get("difficulty") != "impossible"
        }

    return {
        "expected_accuracy": 0.8,
        "difficulty": "medium",
        "should_converge": True
    }


def create_output_directories():
    """Create necessary output directories if they don't exist."""
    directories = [OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Module Initialization
# =============================================================================

# Create output directories when module is imported
create_output_directories()

# Export important constants
__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "YEAR_INTRODUCED",
    "PAPER_TITLE",
    "AUTHORS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_EPOCHS",
    "STRENGTH_EXPERIMENTS",
    "WEAKNESS_EXPERIMENTS",
    "ALL_EXPERIMENTS",
    "DATASET_SPECS",
    "VALIDATION_RANGES",
    "ERROR_MESSAGES",
    "MODEL_CAPABILITIES",
    "MODEL_LIMITATIONS",
    "get_experiment_info",
    "validate_parameter",
    "get_expected_performance"
]
