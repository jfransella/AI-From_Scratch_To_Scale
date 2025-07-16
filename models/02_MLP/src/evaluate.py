#!/usr/bin/env python3
"""
Evaluation script for Multi-Layer Perceptron (MLP) experiments.

Loads a trained MLP model, evaluates on test data, prints metrics, and generates visualizations (confusion matrix, decision boundary if 2D).
"""
import sys
import argparse
from pathlib import Path
import torch

# Add project root to path for shared package imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import setup_logging
from data_utils import generate_xor_dataset, generate_circles_dataset
from config import get_experiment_config, apply_environment_overrides
from model import MLP

# Optional plotting imports
try:
    from plotting import plot_confusion_matrix, plot_decision_boundary
except ImportError:
    plot_confusion_matrix = None
    plot_decision_boundary = None

def create_dataset(config):
    if config.dataset_type == "xor":
        features, labels = generate_xor_dataset(n_samples=4, noise=0.0, random_state=42)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y, x, y
    elif config.dataset_type == "circles":
        params = config.dataset_params
        num_samples = params.get("num_samples", 1000)
        noise = params.get("noise", 0.1)
        features, labels = generate_circles_dataset(num_samples=num_samples, noise=noise)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        n_train = int(0.8 * len(x))
        indices = torch.randperm(len(x))
        x_train = x[indices[:n_train]]
        y_train = y[indices[:n_train]]
        x_test = x[indices[n_train:]]
        y_test = y[indices[n_train:]]
        return x_train, y_train, x_test, y_test
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP on classic non-linear problems")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after evaluation")
    args = parser.parse_args()

    # Load experiment config
    config = get_experiment_config(args.experiment)
    config = apply_environment_overrides(config, "default")
    device = torch.device(args.device)

    # Load test data
    _, _, x_test, y_test = create_dataset(config)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Load model
    model = MLP(
        input_size=config.input_size,
        hidden_layers=config.hidden_layers,
        output_size=config.output_size,
        activation=config.activation,
        weight_init=config.weight_init,
        device=device
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    # Evaluate
    with torch.no_grad():
        outputs = model.forward(x_test)
        preds = (outputs >= 0.5).float().squeeze()
        accuracy = (preds == y_test).float().mean().item()
        print(f"Test accuracy: {accuracy:.4f}")

    # Visualization
    if args.visualize:
        plots_dir = Path(config.output_dir) / "visualizations"
        plots_dir.mkdir(exist_ok=True)
        # Confusion matrix
        if plot_confusion_matrix is not None:
            cm_path = plots_dir / f"{config.name}_confusion_matrix.png"
            plot_confusion_matrix(
                y_true=y_test.cpu().numpy(),
                y_pred=preds.cpu().numpy(),
                class_names=["Class 0", "Class 1"],
                title="MLP Confusion Matrix",
                save_path=str(cm_path),
            )
            print(f"Confusion matrix plot saved: {cm_path}")
        else:
            print("plot_confusion_matrix not available")
        # Decision boundary (if 2D)
        if plot_decision_boundary is not None and config.input_size == 2:
            x_train, y_train, _, _ = create_dataset(config)
            boundary_path = plots_dir / f"{config.name}_decision_boundary_eval.png"
            plot_decision_boundary(
                model,
                x_train.numpy() if hasattr(x_train, 'numpy') else x_train,
                y_train.numpy() if hasattr(y_train, 'numpy') else y_train,
                title="MLP Decision Boundary (Eval)",
                save_path=str(boundary_path),
            )
            print(f"Decision boundary plot saved: {boundary_path}")
        elif config.input_size == 2:
            print("plot_decision_boundary not available")

if __name__ == "__main__":
    main() 