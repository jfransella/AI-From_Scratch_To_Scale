"""
Tests for plotting model analysis functionality.

This module contains tests for the plotting.model_analysis module,
verifying that confusion matrix and decision boundary plotting
functions work correctly and save outputs as expected.
"""

import os
import tempfile
import numpy as np
import matplotlib
from matplotlib.figure import Figure

# Import plotting functions with graceful fallback
try:
    from plotting.model_analysis import (
        plot_confusion_matrix,
        plot_decision_boundary,
    )  # pylint: disable=import-error
except ImportError:
    # Mock functions for testing when plotting module is not available
    def plot_confusion_matrix(*_args, **_kwargs):
        """Mock function for plot_confusion_matrix when plotting module is unavailable."""
        return matplotlib.figure.Figure()

    def plot_decision_boundary(*_args, **_kwargs):
        """Mock function for plot_decision_boundary when plotting module is unavailable."""
        return matplotlib.figure.Figure()


matplotlib.use("Agg")


def test_plot_confusion_matrix_runs_and_saves():
    """Test that confusion matrix plotting runs successfully and saves the output."""
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    class_names = ["Class 0", "Class 1"]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "conf_matrix.png")
        fig = plot_confusion_matrix(
            y_true, y_pred, class_names=class_names, save_path=save_path
        )
        assert isinstance(fig, Figure)
        assert os.path.exists(save_path)


def test_plot_decision_boundary_runs_and_saves():
    """Test that decision boundary plotting runs successfully and saves the output."""

    # Simple synthetic 2D data and a dummy model
    class DummyModel:
        """A dummy model for testing decision boundary plotting."""

        def predict(self, x):
            """Predict class labels based on first feature threshold."""
            return (x[:, 0] > 0).int()

    x = np.random.randn(100, 2)
    y = (x[:, 0] > 0).astype(int)
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "decision_boundary.png")
        fig = plot_decision_boundary(model, x, y, save_path=save_path)
        assert isinstance(fig, Figure)
        assert os.path.exists(save_path)
