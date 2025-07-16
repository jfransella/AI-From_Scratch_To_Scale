import matplotlib
matplotlib.use("Agg")
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import tempfile
from matplotlib.figure import Figure
from plotting.model_analysis import plot_confusion_matrix, plot_decision_boundary


def test_plot_confusion_matrix_runs_and_saves():
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    class_names = ["Class 0", "Class 1"]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "conf_matrix.png")
        fig = plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)
        assert isinstance(fig, Figure)
        assert os.path.exists(save_path)


def test_plot_decision_boundary_runs_and_saves():
    # Simple synthetic 2D data and a dummy model
    class DummyModel:
        def predict(self, X):
            return (X[:, 0] > 0).astype(int)

    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "decision_boundary.png")
        fig = plot_decision_boundary(model, X, y, save_path=save_path)
        assert isinstance(fig, Figure)
        assert os.path.exists(save_path) 