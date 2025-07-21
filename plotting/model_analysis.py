"""Model analysis and visualization plots."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from typing import Optional, Any, Sequence
from .utils import save_figure, set_axis_labels

from sklearn.metrics import confusion_matrix


class ModelPlotter:
    def __init__(self):
        pass


def plot_decision_boundary(model: Any, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary", save_path: Optional[str] = None, grid_resolution: int = 200, random_state: Optional[int] = None, cmap: str = "coolwarm") -> Figure:
    """
    Plot the decision boundary of a classifier for 2D data.

    Args:
        model: Trained model with a predict or predict_proba method.
        X: Input features, shape (n_samples, 2).
        y: True labels, shape (n_samples,).
        title: Plot title.
        save_path: If provided, save the figure to this path.
        grid_resolution: Number of points along each axis for meshgrid.
        random_state: Random seed for reproducibility (if sampling is used).
        cmap: Colormap for background.

    Returns:
        Matplotlib Figure object.
    """
    assert X.shape[1] == 2, "plot_decision_boundary only supports 2D input features."
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Ensure grid is a torch.Tensor for model prediction
    import torch
    if not isinstance(grid, torch.Tensor):
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
    else:
        grid_tensor = grid

    # Predict class for each point in the grid
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid_tensor)
        if hasattr(Z, 'detach'):
            Z = Z.detach().cpu().numpy()
        # Handle both 1D and 2D probability outputs
        if len(Z.shape) == 2 and Z.shape[1] == 2:
            Z = Z[:, 1]  # Probability of class 1
        elif len(Z.shape) == 2:
            Z = np.argmax(Z, axis=1)
        # If Z is already 1D, use as is
    else:
        Z = model.predict(grid_tensor)
        if hasattr(Z, 'detach'):
            Z = Z.detach().cpu().numpy()
    # Ensure Z has the correct shape for reshaping
    if Z.size != xx.size:
        # If Z has wrong size, take the first xx.size elements or repeat
        if Z.size > xx.size:
            Z = Z[:xx.size]
        else:
            # Repeat Z to match the required size
            Z = np.tile(Z, int(np.ceil(xx.size / Z.size)))[:xx.size]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=40)
    set_axis_labels(ax, xlabel="Feature 1", ylabel="Feature 2", title=title)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(y_true: Sequence, y_pred: Sequence, class_names: Optional[Sequence[str]] = None, title: str = "Confusion Matrix", save_path: Optional[str] = None, cmap: str = "Blues") -> Figure:
    """
    Plot a confusion matrix as a heatmap.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        class_names: List of class names (optional).
        title: Plot title.
        save_path: If provided, save the figure to this path.
        cmap: Colormap for heatmap.

    Returns:
        Matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True, ax=ax,
                xticklabels=class_names if class_names is not None else "auto",
                yticklabels=class_names if class_names is not None else "auto")
    set_axis_labels(ax, xlabel="Predicted Label", ylabel="True Label", title=title)
    if save_path:
        save_figure(fig, save_path)
    return fig 