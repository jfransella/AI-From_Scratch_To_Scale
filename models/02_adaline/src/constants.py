"""
ADALINE Constants and Metadata.

Historical constants and configurations for the ADALINE (Adaptive Linear Neuron)
implementation, focusing on the Delta Rule learning algorithm.
"""

from typing import Dict, Any

# =============================================================================
# HISTORICAL METADATA
# =============================================================================

MODEL_NAME = "ADALINE"
FULL_NAME = "Adaptive Linear Neuron"
YEAR_INTRODUCED = 1960
AUTHORS = ["Bernard Widrow", "Ted Hoff"]
INSTITUTION = "Stanford University"

KEY_INNOVATIONS = [
    "First neural network with continuous activation",
    "Delta Rule (Least Mean Squares) learning algorithm", 
    "Continuous error-based weight updates",
    "Foundation for modern gradient descent methods"
]

PROBLEMS_SOLVED = [
    "Continuous learning from error magnitude",
    "Smoother convergence than discrete Perceptron",
    "Better noise tolerance than step function",
    "Foundation for multi-layer training"
]

LIMITATIONS = [
    "Still limited to linear decision boundaries",
    "Cannot solve XOR problem (like Perceptron)",
    "Requires linearly separable data for classification",
    "No non-linear transformations"
]

# =============================================================================
# ALGORITHM SPECIFICATIONS
# =============================================================================

# Learning parameters
DEFAULT_LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_EPOCHS = 1000

# Activation and initialization
ACTIVATION_FUNCTION = "linear"  # No activation (continuous output)
WEIGHT_INIT_METHOD = "small_random"  # Small random weights
BIAS_INIT_VALUE = 0.0

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    # =============================================================================
    # STRENGTH DATASETS (Linearly Separable)
    # =============================================================================
    "debug_small": {
        "description": "Quick debug with minimal linearly separable data",
        "dataset": "debug_small",
        "epochs": 50,
        "learning_rate": 0.01
    },
    "linear_2d_demo": {
        "description": "Demonstrate ADALINE on 2D linearly separable points",
        "dataset": "linear_separable",
        "epochs": 200,
        "learning_rate": 0.01
    },
    "noisy_linear_demo": {
        "description": "Demonstrate ADALINE's robustness on noisy linear data",
        "dataset": "noisy_linear",
        "epochs": 300,
        "learning_rate": 0.005
    },
    "iris_strength_demo": {
        "description": "Demonstrate ADALINE on linearly separable Iris (setosa vs versicolor)",
        "dataset": "iris_setosa_versicolor",
        "epochs": 300,
        "learning_rate": 0.001
    },
    "mnist_strength_demo": {
        "description": "Demonstrate ADALINE on MNIST subset (0 vs 1) - high-dimensional linear",
        "dataset": "mnist_subset",
        "epochs": 100,
        "learning_rate": 0.0001
    },
    
    # =============================================================================
    # WEAKNESS DATASETS (Non-Linearly Separable)
    # =============================================================================
    "xor_limitation": {
        "description": "Demonstrate ADALINE's linear limitation on XOR problem",
        "dataset": "xor_problem",
        "epochs": 500,
        "learning_rate": 0.01
    },
    "iris_weakness_demo": {
        "description": "Demonstrate ADALINE's limitation on non-linearly separable Iris (versicolor vs virginica)",
        "dataset": "iris_versicolor_virginica",
        "epochs": 500,
        "learning_rate": 0.01
    },
    
    # =============================================================================
    # EDUCATIONAL COMPARISONS
    # =============================================================================
    "delta_rule_demo": {
        "description": "Demonstrate Delta Rule learning on simple linear data",
        "dataset": "simple_linear",
        "epochs": 100,
        "learning_rate": 0.01
    },
    "perceptron_comparison": {
        "description": "Direct comparison with Perceptron on linear data",
        "dataset": "linear_separable", 
        "epochs": 200,
        "learning_rate": 0.01
    },
    "convergence_study": {
        "description": "Study convergence behavior on noisy data",
        "dataset": "noisy_linear",
        "epochs": 500,
        "learning_rate": 0.005
    }
}

# Expected performance benchmarks
EXPECTED_PERFORMANCE = {
    # =============================================================================
    # STRENGTH DATASETS (Linearly Separable)
    # =============================================================================
    "debug_small": {"mse": "<0.1", "accuracy": ">95%"},
    "linear_separable": {"mse": "<0.1", "accuracy": ">90%"},
    "noisy_linear": {"mse": "<0.2", "accuracy": ">80%"},
    "simple_linear": {"mse": "<0.1", "accuracy": ">90%"},
    "iris_setosa_versicolor": {"mse": "<0.1", "accuracy": ">95%"},
    "mnist_subset": {"mse": "<0.3", "accuracy": ">85%"},
    
    # =============================================================================
    # WEAKNESS DATASETS (Non-Linearly Separable)
    # =============================================================================
    "iris_versicolor_virginica": {"mse": "high", "accuracy": "~70%"},
    "xor_problem": {"mse": "high", "accuracy": "~50%"}  # Should fail like Perceptron
}

# =============================================================================
# COMPARISON WITH PERCEPTRON
# =============================================================================

PERCEPTRON_COMPARISON = {
    "learning_rule": {
        "perceptron": "Weight update on misclassification only",
        "adaline": "Weight update based on error magnitude"
    },
    "activation": {
        "perceptron": "Step function (binary output)",
        "adaline": "Linear (continuous output)"
    },
    "error_function": {
        "perceptron": "Classification error",
        "adaline": "Mean squared error"
    },
    "convergence": {
        "perceptron": "Guaranteed if linearly separable",
        "adaline": "Converges to minimum error"
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_experiment_info(experiment_name: str) -> Dict[str, Any]:
    """Get information about a specific experiment."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return {
        "name": experiment_name,
        **EXPERIMENTS[experiment_name],
        "expected_performance": EXPECTED_PERFORMANCE.get(
            EXPERIMENTS[experiment_name]["dataset"], {}
        )
    }


def list_experiments() -> Dict[str, str]:
    """List all available experiments with descriptions."""
    return {name: str(config["description"]) for name, config in EXPERIMENTS.items()}


def validate_learning_rate(lr: float) -> float:
    """Validate and clamp learning rate to valid range."""
    if lr < MIN_LEARNING_RATE:
        return MIN_LEARNING_RATE
    elif lr > MAX_LEARNING_RATE:
        return MAX_LEARNING_RATE
    return lr


def get_expected_performance(dataset: str) -> Dict[str, str]:
    """Get expected performance metrics for a dataset."""
    return EXPECTED_PERFORMANCE.get(dataset, {}) 