"""
Constants for Perceptron model implementation.

Defines model metadata, file paths, experiment configurations, and fixed values
for the Perceptron, the first artificial neural network model.
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# MODEL METADATA
# =============================================================================

# Basic model information
MODEL_NAME = "Perceptron"
MODEL_VERSION = "1.0.0"
MODEL_DESCRIPTION = (
    "The first artificial neural network capable of learning "
    "linearly separable patterns"
)

# Historical information
YEAR_INTRODUCED = 1957
AUTHORS = ["Frank Rosenblatt"]
ORIGINAL_PAPER_TITLE = "The Perceptron: A Perceiving and Recognizing Automaton"
ORIGINAL_PAPER_URL = (
    "https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf"
)

# Key innovations and contributions
KEY_INNOVATIONS = [
    "First learning algorithm for artificial neurons",
    "Demonstrated that machines could learn from experience",
    "Introduced the concept of adjustable weights",
    "Established the foundation for all future neural networks",
]

# Problems solved by this model
PROBLEMS_SOLVED = [
    "Binary classification of linearly separable data",
    "Pattern recognition with learned decision boundaries",
    "Automated feature weight adjustment",
]

# Limitations of this model
LIMITATIONS = [
    "Cannot solve non-linearly separable problems (e.g., XOR)",
    "Limited to binary classification only",
    "Convergence not guaranteed for non-separable data",
    "Single layer limits representational capacity",
]

# =============================================================================
# MODEL ARCHITECTURE CONSTANTS
# =============================================================================

# Default architecture parameters
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_EPOCHS = 100
DEFAULT_TOLERANCE = 1e-6

# Activation function options
SUPPORTED_ACTIVATIONS = ["step", "sigmoid", "tanh"]
DEFAULT_ACTIVATION = "step"

# Weight initialization options
SUPPORTED_INIT_METHODS = ["zeros", "normal", "xavier", "random"]
DEFAULT_INIT_METHOD = "zeros"

# Training parameters
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 10.0
MIN_EPOCHS = 1
MAX_EPOCHS = 10000

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# All experiments supported by this model
ALL_EXPERIMENTS = [
    # Strength experiments (should succeed)
    "iris_binary",
    "linear_separable",
    "breast_cancer_binary",
    # Weakness experiments (should struggle/fail)
    "xor_problem",
    "circles_dataset",
    "mnist_subset",
    # Debug experiments (for quick testing)
    "debug_small",
    "debug_linear",
]

# Strength experiments - datasets where Perceptron excels
STRENGTH_EXPERIMENTS = ["iris_binary", "linear_separable", "breast_cancer_binary"]

# Weakness experiments - datasets that expose Perceptron limitations
WEAKNESS_EXPERIMENTS = ["xor_problem", "circles_dataset", "mnist_subset"]

# Debug experiments - for quick testing and development
DEBUG_EXPERIMENTS = ["debug_small", "debug_linear"]

# =============================================================================
# DATASET SPECIFICATIONS
# =============================================================================

DATASET_SPECS = {
    # Strength datasets
    "iris_binary": {
        "dataset_name": "iris_binary",
        "dataset_params": {
            "classes": ["setosa", "versicolor"],
            "n_samples": None,  # Use all available
            "noise": 0.0,
        },
        "input_size": 4,
        "output_size": 1,
        "expected_accuracy": 0.98,
        "difficulty": "easy",
        "description": (
            "Binary classification of Iris setosa vs versicolor - "
            "linearly separable"
        ),
    },
    "linear_separable": {
        "dataset_name": "linear_separable",
        "dataset_params": {
            "n_samples": 200,
            "n_features": 2,
            "n_clusters_per_class": 1,
            "noise": 0.1,
        },
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.95,
        "difficulty": "easy",
        "description": "Simple 2D linearly separable synthetic data",
    },
    "breast_cancer_binary": {
        "dataset_name": "breast_cancer",
        "dataset_params": {"binary_classification": True, "normalize": True},
        "input_size": 30,
        "output_size": 1,
        "expected_accuracy": 0.85,
        "difficulty": "medium",
        "description": "Wisconsin breast cancer dataset - binary classification",
    },
    # Weakness datasets
    "xor_problem": {
        "dataset_name": "xor",
        "dataset_params": {"n_samples": 1000, "noise": 0.0},
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.50,  # Random chance - should fail
        "difficulty": "impossible",
        "description": "XOR problem - classic non-linearly separable case",
    },
    "circles_dataset": {
        "dataset_name": "circles",
        "dataset_params": {"n_samples": 300, "noise": 0.1, "factor": 0.8},
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.60,
        "difficulty": "hard",
        "description": "Concentric circles - non-linearly separable",
    },
    "mnist_subset": {
        "dataset_name": "mnist_binary",
        "dataset_params": {"digits": [0, 1], "n_samples": 1000, "flatten": True},
        "input_size": 784,
        "output_size": 1,
        "expected_accuracy": 0.70,
        "difficulty": "hard",
        "description": "MNIST digits 0 vs 1 - high dimensional, some non-linearity",
    },
    # Debug datasets
    "debug_small": {
        "dataset_name": "linear_separable",
        "dataset_params": {"n_samples": 20, "n_features": 2, "noise": 0.0},
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 1.0,
        "difficulty": "trivial",
        "description": "Small linearly separable dataset for quick testing",
    },
    "debug_linear": {
        "dataset_name": "linear_separable",
        "dataset_params": {"n_samples": 50, "n_features": 2, "noise": 0.05},
        "input_size": 2,
        "output_size": 1,
        "expected_accuracy": 0.95,
        "difficulty": "easy",
        "description": "Small dataset with minimal noise for debugging",
    },
}

# =============================================================================
# FILE PATH CONSTANTS
# =============================================================================

# Base directories
MODEL_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = MODEL_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "visualizations"
NOTEBOOKS_DIR = MODEL_DIR / "notebooks"

# Ensure output directories exist
for directory in [OUTPUTS_DIR, LOGS_DIR, MODELS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File naming patterns
MODEL_CHECKPOINT_PATTERN = "{experiment}_epoch_{epoch:03d}.pth"
PLOT_FILENAME_PATTERN = "{experiment}_{plot_type}.png"
LOG_FILENAME_PATTERN = "training_{timestamp}.log"

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_learning_rate(lr: float) -> float:
    """Validate and clip learning rate to acceptable range."""
    if lr < MIN_LEARNING_RATE:
        return MIN_LEARNING_RATE
    if lr > MAX_LEARNING_RATE:
        return MAX_LEARNING_RATE
    return lr


def validate_epochs(epochs: int) -> int:
    """Validate and clip epochs to acceptable range."""
    if epochs < MIN_EPOCHS:
        return MIN_EPOCHS
    if epochs > MAX_EPOCHS:
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
            f"Unsupported initialization: {init_method}. "
            f"Supported: {SUPPORTED_INIT_METHODS}"
        )
    return init_method


def validate_experiment(experiment_name: str) -> str:
    """Validate experiment name."""
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {ALL_EXPERIMENTS}"
        )
    return experiment_name


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_experiment_info(experiment_name: str) -> Dict[str, Any]:
    """
    Get comprehensive information about an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Dictionary with experiment metadata
    """
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    dataset_spec = DATASET_SPECS[experiment_name]

    return {
        "experiment_name": experiment_name,
        "model_name": MODEL_NAME,
        "dataset_info": dataset_spec,
        "is_strength": experiment_name in STRENGTH_EXPERIMENTS,
        "is_weakness": experiment_name in WEAKNESS_EXPERIMENTS,
        "is_debug": experiment_name in DEBUG_EXPERIMENTS,
        "expected_outcome": (
            "success"
            if experiment_name in STRENGTH_EXPERIMENTS
            else "failure" if experiment_name in WEAKNESS_EXPERIMENTS else "testing"
        ),
    }


def validate_parameter(param_name: str, value: Any) -> Any:
    """
    Validate and potentially adjust parameter values.

    Args:
        param_name: Name of the parameter
        value: Parameter value to validate

    Returns:
        Validated (and potentially adjusted) parameter value
    """
    if param_name == "learning_rate":
        return validate_learning_rate(float(value))
    if param_name == "max_epochs":
        return validate_epochs(int(value))
    if param_name == "activation":
        return validate_activation(str(value))
    if param_name == "init_method":
        return validate_init_method(str(value))
    return value


def get_expected_performance(experiment_name: str) -> Dict[str, Any]:
    """
    Get expected performance metrics for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Dictionary with expected performance metrics
    """
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    dataset_spec = DATASET_SPECS[experiment_name]

    return {
        "expected_accuracy": dataset_spec["expected_accuracy"],
        "difficulty": dataset_spec["difficulty"],
        "should_converge": experiment_name in STRENGTH_EXPERIMENTS
        or experiment_name in DEBUG_EXPERIMENTS,
        "max_reasonable_epochs": (
            50
            if dataset_spec["difficulty"] == "easy"
            else 200 if dataset_spec["difficulty"] == "medium" else 1000
        ),
        "description": dataset_spec["description"],
    }


# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

# Educational context for each experiment type
EXPERIMENT_CONTEXTS = {
    "strength": {
        "purpose": "Demonstrate Perceptron capabilities on linearly separable data",
        "expected_outcome": "High accuracy and convergence",
        "learning_objective": "Understanding when Perceptrons work well",
    },
    "weakness": {
        "purpose": "Expose Perceptron limitations on non-linearly separable data",
        "expected_outcome": "Poor accuracy and/or non-convergence",
        "learning_objective": (
            "Understanding Perceptron limitations and motivation for MLPs"
        ),
    },
    "debug": {
        "purpose": "Quick validation of implementation and setup",
        "expected_outcome": "Fast convergence on simple data",
        "learning_objective": "Verification that code works correctly",
    },
}

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Logging configuration
LOG_LEVEL_MAPPING = {
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARNING",
    "error": "ERROR",
}

DEFAULT_LOG_LEVEL = "INFO"

# Debug flags
DEBUG_WEIGHT_UPDATES = False
DEBUG_PREDICTION_DETAILS = False
DEBUG_CONVERGENCE_CRITERIA = False
