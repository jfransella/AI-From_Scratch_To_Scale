"""
Constants for Perceptron model implementation.

Defines model metadata, file paths, and fixed values for the Perceptron,
the first artificial neural network model.
"""

import os
from pathlib import Path

# =============================================================================
# MODEL METADATA
# =============================================================================

# Basic model information
MODEL_NAME = "Perceptron"
MODEL_VERSION = "1.0.0"
MODEL_DESCRIPTION = "The first artificial neural network capable of learning linearly separable patterns"

# Historical information
YEAR_INTRODUCED = 1957
ORIGINAL_AUTHOR = "Frank Rosenblatt"
ORIGINAL_PAPER_TITLE = "The Perceptron: A Perceiving and Recognizing Automaton"
ORIGINAL_PAPER_URL = "https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf"

# Key innovations and contributions
KEY_INNOVATIONS = [
    "First learning algorithm for artificial neurons",
    "Demonstrated that machines could learn from experience",
    "Introduced the concept of adjustable weights",
    "Established the foundation for all future neural networks"
]

# Problems solved by this model
PROBLEMS_SOLVED = [
    "Binary classification of linearly separable data",
    "Pattern recognition with learned decision boundaries",
    "Automated feature weight adjustment"
]

# Limitations of this model
LIMITATIONS = [
    "Cannot solve non-linearly separable problems (e.g., XOR)",
    "Limited to binary classification only",
    "Convergence not guaranteed for non-separable data",
    "Single layer limits representational capacity"
]

# =============================================================================
# MODEL ARCHITECTURE CONSTANTS
# =============================================================================

# Default architecture parameters
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_EPOCHS = 100
DEFAULT_TOLERANCE = 1e-6

# Activation function options
SUPPORTED_ACTIVATIONS = ["step", "sign"]
DEFAULT_ACTIVATION = "step"

# Weight initialization options
SUPPORTED_INIT_METHODS = ["zeros", "random_normal", "random_uniform"]
DEFAULT_INIT_METHOD = "zeros"

# Training parameters
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 10.0
MIN_EPOCHS = 1
MAX_EPOCHS = 10000

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

# Strength datasets (should perform well)
STRENGTH_DATASETS = [
    "iris_binary",      # Linearly separable subset of Iris
    "linear_2d",        # Simple 2D linearly separable data
    "breast_cancer",    # Wisconsin breast cancer dataset
]

# Weakness datasets (should struggle/fail)
WEAKNESS_DATASETS = [
    "xor",              # Classic non-linearly separable problem
    "circles",          # Concentric circles
    "iris_multiclass"   # Multi-class problem (Perceptron is binary)
]

# Default experiments
DEFAULT_EXPERIMENTS = [
    "iris_setosa_vs_others",  # Binary classification from Iris
    "linear_simple",          # Simple synthetic linear data
    "xor_failure_demo"        # Demonstrate XOR limitation
]

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
# EXPERIMENT VALIDATION
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
        raise ValueError(f"Unsupported activation: {activation}. "
                        f"Supported: {SUPPORTED_ACTIVATIONS}")
    return activation


def validate_init_method(init_method: str) -> str:
    """Validate weight initialization method."""
    if init_method not in SUPPORTED_INIT_METHODS:
        raise ValueError(f"Unsupported initialization: {init_method}. "
                        f"Supported: {SUPPORTED_INIT_METHODS}")
    return init_method


# =============================================================================
# EXPERIMENTAL SETUPS
# =============================================================================

# Standard experimental configurations
STANDARD_EXPERIMENTS = {
    "quick_test": {
        "max_epochs": 10,
        "learning_rate": 0.1,
        "dataset": "linear_2d",
        "n_samples": 100
    },
    "iris_binary": {
        "max_epochs": 100,
        "learning_rate": 0.1,
        "dataset": "iris_setosa_vs_others",
        "tolerance": 1e-6
    },
    "xor_failure": {
        "max_epochs": 1000,
        "learning_rate": 0.1,
        "dataset": "xor",
        "tolerance": 1e-6,
        "note": "Expected to fail - demonstrates limitation"
    }
}

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Logging configuration
LOG_LEVEL_MAPPING = {
    "debug": "DEBUG",
    "info": "INFO", 
    "warning": "WARNING",
    "error": "ERROR"
}

DEFAULT_LOG_LEVEL = "INFO"

# Debug flags
DEBUG_WEIGHT_UPDATES = False
DEBUG_PREDICTION_DETAILS = False
DEBUG_CONVERGENCE_CRITERIA = False

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Plot types available for Perceptron
AVAILABLE_PLOTS = [
    "loss_curve",           # Training loss over epochs
    "decision_boundary",    # 2D decision boundary visualization
    "weight_evolution",     # How weights change during training
    "convergence_analysis", # Convergence behavior analysis
    "error_analysis"        # Error distribution analysis
]

# Default plot settings
DEFAULT_PLOT_SETTINGS = {
    "figsize": (8, 6),
    "dpi": 150,
    "style": "seaborn-v0_8-whitegrid",
    "color_palette": "Set1"
}

# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

# Expected performance on standard datasets
PERFORMANCE_BENCHMARKS = {
    "iris_setosa_vs_others": {
        "expected_accuracy": 1.0,  # Should achieve perfect separation
        "max_epochs_expected": 50,
        "convergence_tolerance": 1e-6
    },
    "linear_2d": {
        "expected_accuracy": 0.95,
        "max_epochs_expected": 100,
        "convergence_tolerance": 1e-6
    },
    "xor": {
        "expected_accuracy": 0.5,  # Random performance expected
        "max_epochs_expected": float('inf'),  # Will not converge
        "note": "Demonstrates fundamental limitation"
    }
}

# =============================================================================
# HISTORICAL CONTEXT
# =============================================================================

HISTORICAL_CONTEXT = f"""
The Perceptron ({YEAR_INTRODUCED}) by {ORIGINAL_AUTHOR} was a groundbreaking moment in AI history.
It demonstrated for the first time that a machine could learn to classify patterns, 
laying the foundation for all modern neural networks.

Key Historical Significance:
- First trainable artificial neural network
- Introduced the concept of learning through weight adjustment
- Sparked the first wave of neural network research
- Led to both great excitement and the "AI Winter" when limitations were discovered

The famous XOR problem, unsolvable by a single Perceptron, led to the development 
of multi-layer networks and eventually modern deep learning.
""" 