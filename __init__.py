"""
AI From Scratch to Scale - Educational Neural Network Implementations

A comprehensive educational project implementing neural network architectures
from basic perceptrons to modern transformers, designed for learning and
understanding the evolution of deep learning.

This package provides:
- Shared utilities for logging, device management, and reproducibility
- Data loading and preprocessing utilities
- Training and evaluation engines
- Visualization and plotting capabilities
- Progressive model implementations with educational focus

Modules:
- utils: Core utilities (logging, devices, seeds, exceptions)
- data_utils: Dataset loading and preprocessing
- engine: Training and evaluation infrastructure  
- plotting: Visualization and analysis tools
- models: Progressive neural network implementations
"""

__version__ = "1.0.0"
__author__ = "AI From Scratch to Scale Team"
__email__ = "contact@ai-from-scratch.dev"
__description__ = "Educational neural network implementations from basic perceptrons to modern architectures"

# Core package imports
from . import data_utils, engine, plotting, utils

# Version information
VERSION = __version__

# Metadata
METADATA = {
    "name": "ai-from-scratch-to-scale",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": __description__,
    "url": "https://github.com/ai-from-scratch/ai-from-scratch-to-scale",
}

__all__ = ["utils", "data_utils", "engine", "plotting", "VERSION", "METADATA"]
