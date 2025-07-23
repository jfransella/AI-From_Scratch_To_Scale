"""
Training visualization plots for AI From Scratch to Scale project.

Provides functions for plotting training metrics, loss curves, and learning
progression to monitor and analyze model training performance.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from .utils import create_subplots, save_figure, set_axis_labels, PROJECT_COLORS


class TrainingPlotter:
    """Class for creating training-related plots."""

    def __init__(self):
        pass

    def plot_metrics(self, metrics: Dict[str, List[float]],
                     title: str = "Training Metrics",
                     save_path: Optional[str] = None) -> plt.Figure:
        """Plot training metrics over time."""
        fig, ax = create_subplots(figsize=(12, 6))

        for metric_name, values in metrics.items():
            epochs = range(len(values))
            color = PROJECT_COLORS['primary'] if 'loss' in metric_name else PROJECT_COLORS['success']
            ax.plot(epochs, values, label=metric_name, color=color, linewidth=2)

        set_axis_labels(ax, xlabel="Epoch", ylabel="Metric Value", title=title)
        ax.legend()

        if save_path:
            save_figure(fig, save_path)

        return fig


def plot_training_history(loss_history: List[float],
                          accuracy_history: Optional[List[float]] = None,
                          title: str = "Training History",
                          save_path: Optional[str] = None) -> plt.Figure:
    """Simple function to plot training history."""
    if accuracy_history:
        fig, (ax1, ax2) = create_subplots(1, 2, figsize=(15, 6))

        # Loss plot
        epochs = range(len(loss_history))
        ax1.plot(epochs, loss_history, color=PROJECT_COLORS['error'], linewidth=2)
        set_axis_labels(ax1, xlabel="Epoch", ylabel="Loss", title="Training Loss")

        # Accuracy plot
        ax2.plot(epochs, accuracy_history, color=PROJECT_COLORS['success'], linewidth=2)
        set_axis_labels(ax2, xlabel="Epoch", ylabel="Accuracy", title="Training Accuracy")
    else:
        fig, ax = create_subplots(figsize=(10, 6))
        epochs = range(len(loss_history))
        ax.plot(epochs, loss_history, color=PROJECT_COLORS['error'], linewidth=2)
        set_axis_labels(ax, xlabel="Epoch", ylabel="Loss", title=title)

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_learning_curves(train_metrics: Dict[str, List[float]],
                         val_metrics: Optional[Dict[str, List[float]]] = None,
                         title: str = "Learning Curves",
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot learning curves comparing training and validation metrics."""
    fig, ax = create_subplots(figsize=(12, 8))

    for metric_name, values in train_metrics.items():
        epochs = range(len(values))
        color = PROJECT_COLORS['primary'] if 'loss' in metric_name else PROJECT_COLORS['success']
        ax.plot(epochs, values, label=f"Train {metric_name}", color=color, linewidth=2)

        if val_metrics and metric_name in val_metrics:
            ax.plot(epochs, val_metrics[metric_name],
                    label=f"Val {metric_name}", color=color,
                    linewidth=2, linestyle='--')

    set_axis_labels(ax, xlabel="Epoch", ylabel="Metric Value", title=title)
    ax.legend()

    if save_path:
        save_figure(fig, save_path)

    return fig
