"""
Visualization and plotting utilities for AI From Scratch to Scale project.

This package provides comprehensive plotting and visualization capabilities
for training metrics, model performance, decision boundaries, and comparative
analysis across different models and experiments.

Key components:
- Training plots: Loss curves, accuracy progression, learning rate schedules
- Model analysis: Decision boundaries, feature visualizations, confusion matrices
- Comparative analysis: Model comparison charts, performance heatmaps
- Publication-ready figures with consistent styling
"""

from .training_plots import TrainingPlotter, plot_training_history, plot_learning_curves
from .model_analysis import ModelPlotter, plot_decision_boundary, plot_confusion_matrix
from .comparison_plots import ComparisonPlotter, plot_model_comparison, plot_performance_heatmap
from .utils import setup_plotting_style, save_figure, create_subplots

__version__ = "1.0.0"
__all__ = [
    # Training visualization
    "TrainingPlotter",
    "plot_training_history",
    "plot_learning_curves",
    # Model analysis
    "ModelPlotter",
    "plot_decision_boundary",
    "plot_confusion_matrix",
    # Comparison plots
    "ComparisonPlotter",
    "plot_model_comparison",
    "plot_performance_heatmap",
    # Utilities
    "setup_plotting_style",
    "save_figure",
    "create_subplots"
]
