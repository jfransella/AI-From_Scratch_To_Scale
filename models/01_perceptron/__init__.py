"""
01_perceptron model package.

Classic Rosenblatt perceptron implementation for binary classification.
"""

from .src.model import PerceptronModel, create_perceptron
from .src.config import get_training_config, get_evaluation_config
from .src.constants import MODEL_NAME

__version__ = "1.0.0"
__all__ = [
    "PerceptronModel",
    "create_perceptron", 
    "get_training_config",
    "get_evaluation_config",
    "MODEL_NAME"
] 