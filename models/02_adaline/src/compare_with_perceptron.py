"""
ADALINE vs Perceptron Comparison Script.

Side-by-side comparison of ADALINE (Delta Rule) and Perceptron (Step Rule)
to demonstrate the differences in learning behavior.
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import logging

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from config import get_experiment_config, validate_config
from model import create_adaline
from constants import get_experiment_info


def create_simple_perceptron(input_size=2):
    """Create a simple Perceptron for comparison."""

    class SimplePerceptron:
        def __init__(self, input_size=input_size, learning_rate=0.01):
            self.learning_rate = learning_rate
            self.weights = torch.randn(input_size) * 0.1
            self.bias = torch.zeros(1)
            self.training_history = {"loss": [], "errors": []}
            self.is_fitted = False

        def forward(self, x):
            """Forward pass with step function."""
            linear_output = torch.matmul(x, self.weights) + self.bias
            return torch.where(linear_output > 0, 1.0, 0.0)

        def fit(self, x_data, y_target, epochs=100):
            """Train using Perceptron learning rule."""
            y_target = y_target.flatten()

            for epoch in range(epochs):
                total_errors = 0

                for i in range(len(x_data)):
                    x_i = x_data[i]
                    y_i = y_target[i]

                    # Forward pass
                    prediction = self.forward(x_i.unsqueeze(0)).item()

                    # Perceptron learning rule (update only on error)
                    if prediction != y_i:
                        error = y_i - prediction
                        self.weights += self.learning_rate * error * x_i
                        self.bias += self.learning_rate * error
                        total_errors += 1

                # Record history
                error_rate = total_errors / len(x_data)
                self.training_history["errors"].append(total_errors)
                self.training_history["loss"].append(error_rate)

                # Early stopping if converged
                if total_errors == 0:
                    break

            self.is_fitted = True
            return {
                "converged": total_errors == 0,
                "final_errors": total_errors,
                "epochs_trained": epoch + 1,
            }

        def predict(self, x):
            """Make predictions."""
            return self.forward(x)

        def get_model_info(self):
            """Get model information."""
            return {
                "model_name": "Perceptron",
                "year_introduced": 1957,
                "learning_rule": "Perceptron Learning Rule",
                "activation": "Step function",
                "is_fitted": self.is_fitted,
            }

    return SimplePerceptron()


def load_comparison_data(dataset_name: str, n_samples: int = 100):
    """Load dataset using unified data_utils for comparison."""
    try:
        from data_utils.datasets import load_dataset
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        
        # Convert to torch tensors
        x_data = torch.tensor(X, dtype=torch.float32)
        y_data = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        return x_data, y_data
        
    except ImportError:
        print("Warning: data_utils not available, using fallback dataset generation")
        return generate_fallback_comparison_data(dataset_name, n_samples)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to simple data generation")
        return generate_fallback_comparison_data(dataset_name, n_samples)


def generate_fallback_comparison_data(dataset_type: str, n_samples: int = 100):
    """Generate fallback data for comparison."""
    torch.manual_seed(42)  # For reproducible results

    if dataset_type == "simple_linear":
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        return x, y

    elif dataset_type == "linearly_separable":
        x = torch.randn(n_samples, 2)
        y = (2 * x[:, 0] + x[:, 1] > 1).float().unsqueeze(1)
        return x, y

    elif dataset_type == "noisy_linear":
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        # Add noise
        noise = 0.1 * torch.randn_like(y)
        y = torch.clamp(y + noise, 0, 1)
        return x, y
    
    elif dataset_type == "iris_binary":
        # Simple 2D data as fallback for Iris
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        return x, y
    
    elif dataset_type == "mnist_subset":
        # Simple 2D data as fallback for MNIST
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        return x, y
    
    elif dataset_type == "xor_problem":
        # XOR-like data
        x = torch.randn(n_samples, 2)
        y = ((x[:, 0] > 0) != (x[:, 1] > 0)).float().unsqueeze(1)
        return x, y

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def run_comparison(experiment_name: str, epochs: int = 100, visualize: bool = False):
    """Run side-by-side comparison of ADALINE vs Perceptron."""

    print("=" * 60)
    print("ADALINE vs PERCEPTRON COMPARISON")
    print("=" * 60)

    # Get configuration
    config = get_experiment_config(experiment_name)
    config.epochs = epochs
    config = validate_config(config)

    print(f"Experiment: {config.name}")
    print(f"Dataset: {config.dataset}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print("-" * 60)

    # Load data
    x_data, y_data = load_comparison_data(config.dataset)
    print(f"Loaded {len(x_data)} samples")

    # Create models
    adaline = create_adaline(config)
    perceptron = create_simple_perceptron(input_size=config.input_size)
    perceptron.learning_rate = config.learning_rate

    print("\n" + "=" * 30 + " TRAINING " + "=" * 30)

    # Train ADALINE
    print("\n🔵 Training ADALINE (Delta Rule)...")
    start_time = time.time()
    adaline_results = adaline.fit(x_data, y_data)
    adaline_time = time.time() - start_time

    print(f"ADALINE Results:")
    print(f"  - Converged: {adaline_results['converged']}")
    print(f"  - Final MSE: {adaline_results['final_mse']:.6f}")
    print(f"  - Epochs trained: {adaline_results['epochs_trained']}")
    print(f"  - Training time: {adaline_time:.3f}s")

    # Train Perceptron
    print("\n🔴 Training Perceptron (Step Rule)...")
    start_time = time.time()
    perceptron_results = perceptron.fit(x_data, y_data, epochs=config.epochs)
    perceptron_time = time.time() - start_time

    print(f"Perceptron Results:")
    print(f"  - Converged: {perceptron_results['converged']}")
    print(f"  - Final errors: {perceptron_results['final_errors']}")
    print(f"  - Epochs trained: {perceptron_results['epochs_trained']}")
    print(f"  - Training time: {perceptron_time:.3f}s")

    # Evaluate both models
    print("\n" + "=" * 30 + " EVALUATION " + "=" * 30)

    # Load test data
    x_test, y_test = load_comparison_data(config.dataset, n_samples=200)

    # ADALINE evaluation
    adaline.eval()
    with torch.no_grad():
        adaline_pred = adaline.predict(x_test)
        adaline_accuracy = torch.mean((adaline_pred == y_test).float()).item()
        adaline_mse = torch.mean((y_test - adaline.forward(x_test)) ** 2).item()

    # Perceptron evaluation
    perceptron_pred = perceptron.predict(x_test)
    perceptron_accuracy = torch.mean(
        (perceptron_pred == y_test.flatten()).float()
    ).item()

    print(f"\n📊 Evaluation Results (200 test samples):")
    print(f"ADALINE:")
    print(f"  - Accuracy: {adaline_accuracy:.4f} ({adaline_accuracy*100:.2f}%)")
    print(f"  - MSE: {adaline_mse:.6f}")

    print(f"Perceptron:")
    print(f"  - Accuracy: {perceptron_accuracy:.4f} ({perceptron_accuracy*100:.2f}%)")
    print(f"  - MSE: N/A (discrete output)")

    # Learning behavior comparison
    print("\n" + "=" * 25 + " LEARNING ANALYSIS " + "=" * 25)

    adaline_epochs = len(adaline.training_history["loss"])
    perceptron_epochs = len(perceptron.training_history["loss"])

    print(f"\n📈 Learning Behavior:")
    print(f"ADALINE (Continuous Learning):")
    print(f"  - Updates: Every sample based on error magnitude")
    print(f"  - Metric: Mean Squared Error")
    print(f"  - Final MSE: {adaline.training_history['loss'][-1]:.6f}")

    print(f"Perceptron (Discrete Learning):")
    print(f"  - Updates: Only on misclassification")
    print(f"  - Metric: Classification errors")
    print(f"  - Final errors: {perceptron.training_history['errors'][-1]}")

    # Generate visualizations if requested
    if visualize:
        try:
            from visualize import plot_adaline_vs_perceptron_learning

            # Create comparison plot
            fig = plot_adaline_vs_perceptron_learning(
                adaline.training_history["loss"],
                perceptron.training_history["loss"],
                save_path="outputs/visualizations/adaline_vs_perceptron_comparison.png",
            )
            print(f"\n📊 Comparison visualization saved to outputs/visualizations/")

        except Exception as e:
            print(f"Error generating comparison plot: {e}")

    print("\n" + "=" * 60)
    print("🎓 EDUCATIONAL INSIGHTS")
    print("=" * 60)
    print(
        """
Key Differences Observed:

1. LEARNING APPROACH:
   • ADALINE: Learns from error MAGNITUDE (continuous)
   • Perceptron: Learns from error OCCURRENCE (discrete)

2. CONVERGENCE:
   • ADALINE: Smooth, gradual improvement in MSE
   • Perceptron: Stepwise reduction in classification errors

3. ROBUSTNESS:
   • ADALINE: Better noise tolerance due to continuous updates
   • Perceptron: More sensitive to outliers and noise

4. MATHEMATICAL FOUNDATION:
   • ADALINE: Delta Rule → Foundation for gradient descent
   • Perceptron: Step Rule → Foundation for discrete learning

5. HISTORICAL SIGNIFICANCE:
   • Both limited to linear decision boundaries
   • ADALINE's continuous approach enabled modern deep learning
    """
    )

    return {
        "adaline": adaline_results,
        "perceptron": perceptron_results,
        "adaline_accuracy": adaline_accuracy,
        "perceptron_accuracy": perceptron_accuracy,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare ADALINE vs Perceptron")

    parser.add_argument(
        "--experiment",
        type=str,
        default="perceptron_comparison",
        help="Experiment name to run",
    )

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    parser.add_argument(
        "--visualize", action="store_true", help="Generate comparison visualizations"
    )

    return parser.parse_args()


def main():
    """Main comparison function."""
    args = parse_args()

    try:
        results = run_comparison(
            experiment_name=args.experiment,
            epochs=args.epochs,
            visualize=args.visualize,
        )

        print("\n✅ Comparison completed successfully!")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()
