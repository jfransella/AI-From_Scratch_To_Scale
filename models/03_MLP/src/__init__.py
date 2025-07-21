"""
Multi-Layer Perceptron (MLP) implementation package.

This package contains the MLP implementation that solves the XOR problem
and demonstrates non-linear learning capabilities.
"""

from .model import MLP
from .pure_mlp import PureMLP, create_pure_mlp, demonstrate_backpropagation
from .mlp_wrapper import MLPWrapper, create_mlp_wrapper, create_xor_solver
from .config import MLPExperimentConfig, get_experiment_config, get_training_config
from .constants import MODEL_NAME, YEAR_INTRODUCED, AUTHORS

__version__ = "1.0.0"
__all__ = [
    "MLP",
    "PureMLP", 
    "MLPWrapper",
    "create_pure_mlp",
    "create_mlp_wrapper",
    "create_xor_solver",
    "demonstrate_backpropagation",
    "MLPExperimentConfig",
    "get_experiment_config",
    "get_training_config", 
    "MODEL_NAME",
    "YEAR_INTRODUCED",
    "AUTHORS"
]
