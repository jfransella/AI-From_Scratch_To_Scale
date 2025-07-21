"""
ADALINE (Adaptive Linear Neuron) implementation package.

This package contains the ADALINE implementation with pure NumPy core
and engine integration for the AI From Scratch to Scale project.
"""

from .model import ADALINE, create_adaline
from .adaline_wrapper import ADALINEWrapper, create_adaline_wrapper, create_adaline_from_experiment
from .config import ADALINEConfig, get_experiment_config
from .constants import MODEL_NAME, YEAR_INTRODUCED, AUTHORS

__version__ = "1.0.0"
__all__ = [
    "ADALINE",
    "ADALINEWrapper",
    "create_adaline", 
    "create_adaline_wrapper",
    "create_adaline_from_experiment",
    "ADALINEConfig",
    "get_experiment_config",
    "MODEL_NAME",
    "YEAR_INTRODUCED", 
    "AUTHORS"
] 