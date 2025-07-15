"""
Training and evaluation engine for AI From Scratch to Scale project.

This package provides unified training and evaluation infrastructure that works
across all model implementations in the project, from simple perceptrons to
complex deep learning architectures.

Key components:
- Trainer: Unified training loop with experiment tracking
- Evaluator: Comprehensive model evaluation and metrics
- Experiment tracking with Weights & Biases integration
- Model-agnostic interfaces for maximum flexibility
"""

from .trainer import Trainer, TrainingConfig
from .evaluator import Evaluator, EvaluationConfig, ModelMetrics
from .base import BaseModel, TrainingResult, EvaluationResult

__version__ = "1.0.0"
__all__ = [
    # Core training
    "Trainer",
    "TrainingConfig",
    # Core evaluation  
    "Evaluator",
    "EvaluationConfig",
    "ModelMetrics",
    # Base interfaces
    "BaseModel",
    "TrainingResult", 
    "EvaluationResult"
] 