"""
Constants for the Multi-Layer Perceptron (MLP) implementation.

This module defines constants and metadata for the MLP model, representing
the breakthrough that enabled neural networks to solve non-linearly separable
problems like XOR, overcoming the limitations demonstrated by the single-layer
perceptron.
"""

from typing import Dict, Any
import logging

# =============================================================================
# HISTORICAL METADATA
# =============================================================================

MODEL_METADATA = {
    "name": "Multi-Layer Perceptron (MLP)",
    "year_invented": 1969,
    "inventors": ["Marvin Minsky", "Seymour Papert", "Paul Werbos"],
    "key_breakthrough": "Backpropagation algorithm enabling training of multi-layer networks",
    "historical_context": {
        "problem_solved": "Non-linearly separable problems (XOR, complex patterns)",
        "key_innovation": "Hidden layers with non-linear activation functions",
        "ai_winter_context": "Ironically, while Minsky/Papert's 1969 book 'Perceptrons' showed single-layer limitations, it contributed to the first AI winter by discouraging neural network research",
        "renaissance": "MLPs gained prominence in the 1980s with backpropagation popularization",
        "modern_relevance": "Foundation architecture for all modern deep learning"
    },
    "significance": [
        "First neural network capable of universal function approximation",
        "Solved the XOR problem that single-layer perceptrons couldn't handle",
        "Enabled learning of hierarchical feature representations",
        "Laid groundwork for deep learning revolution",
        "Demonstrated power of gradient-based optimization"
    ]
}

# =============================================================================
# ARCHITECTURE CONSTANTS
# =============================================================================

# Standard MLP configurations
ARCHITECTURE_CONFIGS = {
    "minimal": {
        "input_size": 2,
        "hidden_layers": [2],
        "output_size": 1,
        "description": "Minimal MLP for XOR problem"
    },
    "small": {
        "input_size": 2,
        "hidden_layers": [4],
        "output_size": 1,
        "description": "Small MLP with single hidden layer"
    },
    "medium": {
        "input_size": 2,
        "hidden_layers": [8, 4],
        "output_size": 1,
        "description": "Medium MLP with two hidden layers"
    },
    "deep": {
        "input_size": 2,
        "hidden_layers": [16, 8, 4],
        "output_size": 1,
        "description": "Deeper MLP for complex pattern learning"
    }
}

# Activation functions
ACTIVATION_FUNCTIONS = {
    "sigmoid": "Classic activation enabling gradient flow",
    "tanh": "Zero-centered sigmoid variant",
    "relu": "Modern standard, addresses vanishing gradients",
    "leaky_relu": "ReLU variant preventing dead neurons"
}

# Weight initialization strategies
WEIGHT_INIT_METHODS = {
    "xavier_normal": "Xavier/Glorot normal initialization for sigmoid/tanh",
    "xavier_uniform": "Xavier/Glorot uniform initialization",
    "he_normal": "He normal initialization for ReLU activations",
    "he_uniform": "He uniform initialization for ReLU",
    "random_normal": "Simple random normal initialization",
    "zeros": "Zero initialization (poor choice for MLPs)"
}

# =============================================================================
# TRAINING CONSTANTS
# =============================================================================

# Learning rates optimized for different scenarios
LEARNING_RATES = {
    "very_slow": 0.001,
    "slow": 0.01,
    "moderate": 0.1,
    "fast": 0.5,
    "very_fast": 1.0
}

# Training hyperparameters
DEFAULT_TRAINING_CONFIG = {
    "learning_rate": 0.1,
    "max_epochs": 1000,
    "convergence_threshold": 1e-6,
    "patience": 50,  # Early stopping patience
    "batch_size": None,  # None for full batch
    "weight_decay": 0.0,
    "momentum": 0.0
}

# Loss functions
LOSS_FUNCTIONS = {
    "mse": "Mean Squared Error - regression and binary classification",
    "cross_entropy": "Cross-entropy loss for classification",
    "binary_cross_entropy": "Binary cross-entropy for binary classification"
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    "xor": {
        "name": "XOR Dataset",
        "description": "The classic non-linearly separable problem",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 4,
        "difficulty": "fundamental",
        "expected_accuracy": 1.0,
        "historical_significance": "Problem that single-layer perceptrons cannot solve"
    },
    "circles": {
        "name": "Concentric Circles",
        "description": "Non-linearly separable concentric circles",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 1000,
        "difficulty": "moderate",
        "expected_accuracy": 0.95
    },
    "moons": {
        "name": "Two Moons",
        "description": "Two interleaving half-circles",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 1000,
        "difficulty": "moderate",
        "expected_accuracy": 0.90
    },
    "spirals": {
        "name": "Two Spirals",
        "description": "Two intertwined spirals - challenging non-linear problem",
        "input_size": 2,
        "output_size": 1,
        "num_samples": 1000,
        "difficulty": "hard",
        "expected_accuracy": 0.85
    }
}

# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

PERFORMANCE_BENCHMARKS = {
    "xor_solution": {
        "description": "Successful XOR problem solution",
        "target_accuracy": 1.0,
        "max_epochs": 1000,
        "architecture": "minimal",
        "significance": "Demonstrates overcoming single-layer perceptron limitation"
    },
    "generalization_test": {
        "description": "Performance on various non-linear datasets",
        "circles_accuracy": 0.95,
        "moons_accuracy": 0.90,
        "spirals_accuracy": 0.85,
        "significance": "Validates universal function approximation capability"
    }
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_architecture_config(config: Dict[str, Any]) -> bool:
    """
    Validate that an architecture configuration is properly formatted.
    
    Args:
        config: Architecture configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ["input_size", "hidden_layers", "output_size"]
    
    if not all(key in config for key in required_keys):
        return False
    
    if not isinstance(config["hidden_layers"], list):
        return False
    
    if len(config["hidden_layers"]) == 0:
        logging.warning("No hidden layers specified - this would be a perceptron, not an MLP")
        return False
    
    if any(size <= 0 for size in config["hidden_layers"]):
        return False
        
    return True

def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration parameters.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    if config.get("learning_rate", 0) <= 0:
        return False
    
    if config.get("max_epochs", 0) <= 0:
        return False
    
    if config.get("convergence_threshold", 0) <= 0:
        return False
        
    return True


def get_recommended_config_for_problem(problem_type: str) -> Dict[str, Any]:
    """
    Get recommended architecture and training configuration for a specific problem.
    
    Args:
        problem_type: Type of problem ("xor", "circles", "moons", "spirals")
        
    Returns:
        Dict containing recommended architecture and training settings
    """
    recommendations = {
        "xor": {
            "architecture": "minimal",
            "activation": "sigmoid",
            "learning_rate": 0.5,
            "max_epochs": 1000,
            "weight_init": "xavier_normal"
        },
        "circles": {
            "architecture": "small",
            "activation": "relu",
            "learning_rate": 0.01,
            "max_epochs": 2000,
            "weight_init": "he_normal"
        },
        "moons": {
            "architecture": "medium",
            "activation": "relu",
            "learning_rate": 0.01,
            "max_epochs": 2000,
            "weight_init": "he_normal"
        },
        "spirals": {
            "architecture": "deep",
            "activation": "relu",
            "learning_rate": 0.001,
            "max_epochs": 5000,
            "weight_init": "he_normal"
        }
    }
    
    return recommendations.get(problem_type, recommendations["xor"])

# =============================================================================
# EDUCATIONAL CONTEXT
# =============================================================================

EDUCATIONAL_OBJECTIVES = [
    "Understand how hidden layers enable non-linear function approximation",
    "Experience the breakthrough moment of solving the XOR problem",
    "Learn the importance of activation functions in neural networks",
    "Observe how depth affects learning capability and complexity",
    "Appreciate the historical significance of overcoming perceptron limitations",
    "Understand gradient-based optimization in multi-layer networks"
]

XOR_EDUCATIONAL_VALUE = {
    "problem_statement": "XOR returns 1 if inputs differ, 0 if they're the same",
    "why_perceptron_fails": "XOR is not linearly separable - no single line can separate the classes",
    "mlp_solution": "Hidden layer creates new feature space where XOR becomes linearly separable",
    "broader_implications": "Demonstrates that MLPs can approximate any continuous function"
}