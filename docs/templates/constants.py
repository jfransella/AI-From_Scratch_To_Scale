"""
Template for constants.py - Model Constants and Metadata

This template provides the basic structure for defining model-specific constants
and metadata in the "AI From Scratch to Scale" project. Each model should follow
this pattern for consistency.

Replace [MODEL_NAME] with the actual model name (e.g., "Perceptron", "MLP", etc.)
Replace [YEAR] with the year the model was introduced
Replace [AUTHOR] with the original author(s)
Replace [PAPER_TITLE] with the original paper title
"""

import os
from pathlib import Path

# =============================================================================
# MODEL METADATA
# =============================================================================

# Basic model information
MODEL_NAME = "[MODEL_NAME]"
MODEL_VERSION = "1.0.0"
MODEL_DESCRIPTION = "[Brief description of what this model does]"

# Historical information
YEAR_INTRODUCED = [YEAR]
ORIGINAL_AUTHOR = "[AUTHOR(S)]"
ORIGINAL_PAPER_TITLE = "[PAPER_TITLE]"
ORIGINAL_PAPER_URL = "[URL_TO_PAPER]"

# Key innovations and contributions
KEY_INNOVATIONS = [
    "[KEY_INNOVATION_1]",
    "[KEY_INNOVATION_2]",
    "[KEY_INNOVATION_3]",
]

# Problems solved by this model
PROBLEMS_SOLVED = [
    "[PROBLEM_1]",
    "[PROBLEM_2]",
]

# Limitations of this model
LIMITATIONS = [
    "[LIMITATION_1]",
    "[LIMITATION_2]",
]

# =============================================================================
# ARCHITECTURE CONSTANTS
# =============================================================================

# Default architecture parameters
DEFAULT_INPUT_SIZE = 2
DEFAULT_HIDDEN_SIZE = None  # Not all models have hidden layers
DEFAULT_OUTPUT_SIZE = 1

# Activation functions
DEFAULT_ACTIVATION = "sigmoid"  # Options: "sigmoid", "tanh", "relu", "step", etc.
AVAILABLE_ACTIVATIONS = ["sigmoid", "tanh", "relu", "step", "linear"]

# Weight initialization
DEFAULT_WEIGHT_INIT = "random"  # Options: "random", "zeros", "xavier", "he"
WEIGHT_INIT_SCALE = 0.1

# Learning parameters
DEFAULT_LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0

# =============================================================================
# TRAINING CONSTANTS
# =============================================================================

# Default training parameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = "adam"

# Early stopping
DEFAULT_PATIENCE = 20
MIN_DELTA = 1e-6

# Convergence criteria
CONVERGENCE_THRESHOLD = 1e-6
MAX_GRADIENT_NORM = 1.0

# =============================================================================
# DATA CONSTANTS
# =============================================================================

# Data splits
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15

# Data preprocessing
DEFAULT_NORMALIZE = True
DEFAULT_STANDARDIZE = False

# =============================================================================
# FILE PATHS
# =============================================================================

# Get the model directory (assuming this constants.py is in src/)
MODEL_DIR = Path(__file__).parent.parent
PROJECT_ROOT = MODEL_DIR.parent.parent

# Output directories
OUTPUT_DIR = MODEL_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Data directory
DATA_DIR = MODEL_DIR / "data"

# Notebooks directory
NOTEBOOKS_DIR = MODEL_DIR / "notebooks"

# Ensure output directories exist
for directory in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR, VISUALIZATIONS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Log levels
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file names
TRAIN_LOG_FILE = LOGS_DIR / f"{MODEL_NAME.lower()}_train.log"
EVAL_LOG_FILE = LOGS_DIR / f"{MODEL_NAME.lower()}_eval.log"

# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

# Plot settings
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 300
DEFAULT_FORMAT = "png"

# Colors for plots
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2ca02c"
ERROR_COLOR = "#d62728"

# Plot types supported by this model
SUPPORTED_PLOT_TYPES = [
    "loss_curve",
    "accuracy_curve",
    "decision_boundary",  # For 2D problems
    "feature_importance",
    "confusion_matrix",
    "sample_predictions",
]

# =============================================================================
# DEVICE CONSTANTS
# =============================================================================

# Device preferences
PREFER_GPU = True
FALLBACK_TO_CPU = True

# Memory limits
MAX_BATCH_SIZE_GPU = 1024
MAX_BATCH_SIZE_CPU = 256

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Parameter validation ranges
VALID_INPUT_SIZE_RANGE = (1, 10000)
VALID_OUTPUT_SIZE_RANGE = (1, 1000)
VALID_EPOCHS_RANGE = (1, 10000)
VALID_BATCH_SIZE_RANGE = (1, 1024)
VALID_LEARNING_RATE_RANGE = (1e-6, 1.0)

# =============================================================================
# EXPERIMENTAL CONSTANTS
# =============================================================================

# Common experiment names for this model
STRENGTH_EXPERIMENTS = [
    # TODO: Add experiments where this model excels
    # Example: "linear_classification", "simple_regression"
]

WEAKNESS_EXPERIMENTS = [
    # TODO: Add experiments where this model struggles
    # Example: "xor_problem", "non_linear_classification"
]

# Dataset-specific constants
DATASET_CONFIGS = {
    "iris": {
        "input_size": 4,
        "output_size": 3,
        "task_type": "classification",
    },
    "mnist": {
        "input_size": 784,
        "output_size": 10,
        "task_type": "classification",
    },
    "xor": {
        "input_size": 2,
        "output_size": 1,
        "task_type": "classification",
    },
    # Add more datasets as needed
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info():
    """
    Get comprehensive model information.
    
    Returns:
        dict: Dictionary containing model metadata
    """
    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "description": MODEL_DESCRIPTION,
        "year_introduced": YEAR_INTRODUCED,
        "original_author": ORIGINAL_AUTHOR,
        "original_paper": ORIGINAL_PAPER_TITLE,
        "paper_url": ORIGINAL_PAPER_URL,
        "key_innovations": KEY_INNOVATIONS,
        "problems_solved": PROBLEMS_SOLVED,
        "limitations": LIMITATIONS,
        "default_architecture": {
            "input_size": DEFAULT_INPUT_SIZE,
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "output_size": DEFAULT_OUTPUT_SIZE,
            "activation": DEFAULT_ACTIVATION,
        },
        "supported_plots": SUPPORTED_PLOT_TYPES,
        "strength_experiments": STRENGTH_EXPERIMENTS,
        "weakness_experiments": WEAKNESS_EXPERIMENTS,
    }


def validate_architecture_params(input_size, hidden_size, output_size):
    """
    Validate architecture parameters.
    
    Args:
        input_size (int): Input layer size
        hidden_size (int or None): Hidden layer size
        output_size (int): Output layer size
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not (VALID_INPUT_SIZE_RANGE[0] <= input_size <= VALID_INPUT_SIZE_RANGE[1]):
        raise ValueError(f"Input size must be in range {VALID_INPUT_SIZE_RANGE}")
    
    if not (VALID_OUTPUT_SIZE_RANGE[0] <= output_size <= VALID_OUTPUT_SIZE_RANGE[1]):
        raise ValueError(f"Output size must be in range {VALID_OUTPUT_SIZE_RANGE}")
    
    if hidden_size is not None and hidden_size <= 0:
        raise ValueError("Hidden size must be positive if specified")


def validate_training_params(epochs, batch_size, learning_rate):
    """
    Validate training parameters.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not (VALID_EPOCHS_RANGE[0] <= epochs <= VALID_EPOCHS_RANGE[1]):
        raise ValueError(f"Epochs must be in range {VALID_EPOCHS_RANGE}")
    
    if not (VALID_BATCH_SIZE_RANGE[0] <= batch_size <= VALID_BATCH_SIZE_RANGE[1]):
        raise ValueError(f"Batch size must be in range {VALID_BATCH_SIZE_RANGE}")
    
    if not (VALID_LEARNING_RATE_RANGE[0] <= learning_rate <= VALID_LEARNING_RATE_RANGE[1]):
        raise ValueError(f"Learning rate must be in range {VALID_LEARNING_RATE_RANGE}")


if __name__ == "__main__":
    # Print model information
    print("Model Information:")
    print("=" * 50)
    
    info = get_model_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    
    print("\nFile Paths:")
    print("=" * 50)
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"Visualizations Directory: {VISUALIZATIONS_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Notebooks Directory: {NOTEBOOKS_DIR}") 