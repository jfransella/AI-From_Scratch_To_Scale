"""
ADALINE Evaluation Script.

Evaluation script for ADALINE model with comparison features.
"""

import argparse
import sys
from pathlib import Path
import torch
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_experiment_config
from model import create_adaline
from constants import get_experiment_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ADALINE model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="debug_small",
        help="Experiment name for data generation"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def load_dataset_data(dataset_name: str) -> tuple:
    """Load dataset using unified data_utils."""
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
        return generate_fallback_data(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to simple data generation")
        return generate_fallback_data(dataset_name)


def generate_fallback_data(dataset_type: str, n_samples: int = 100) -> tuple:
    """Generate fallback synthetic data for evaluation."""
    if dataset_type == "simple_linear":
        # Simple linearly separable data
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        return x, y
    
    elif dataset_type == "linearly_separable":
        # More complex linearly separable data
        x = torch.randn(n_samples, 2)
        y = (2*x[:, 0] + x[:, 1] > 1).float().unsqueeze(1)
        return x, y
    
    elif dataset_type == "noisy_linear":
        # Linearly separable data with noise
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        # Add some noise
        y = y + 0.1 * torch.randn_like(y)
        y = torch.clamp(y, 0, 1)  # Keep in [0,1] range
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


def evaluate_model(model, x_data, y_data):
    """Evaluate ADALINE model."""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        linear_output = model.forward(x_data)
        predictions = model.predict(x_data)
        
        # Calculate metrics
        mse = torch.mean((y_data - linear_output) ** 2)
        accuracy = torch.mean((predictions == y_data).float())
        
        # Calculate confusion matrix
        tp = torch.sum((predictions == 1) & (y_data == 1))
        tn = torch.sum((predictions == 0) & (y_data == 0))
        fp = torch.sum((predictions == 1) & (y_data == 0))
        fn = torch.sum((predictions == 0) & (y_data == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "mse": mse.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "confusion_matrix": {
            "tp": tp.item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item()
        }
    }


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*50)
    print("ADALINE EVALUATION RESULTS")
    print("="*50)
    
    print(f"Mean Squared Error: {results['mse']:.6f}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  True Positives: {cm['tp']}")
    print(f"  True Negatives: {cm['tn']}")
    print(f"  False Positives: {cm['fp']}")
    print(f"  False Negatives: {cm['fn']}")
    
    print("="*50)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    try:
        config = get_experiment_config(args.experiment)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create model
    model = create_adaline(config)
    
    # Load checkpoint
    try:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return
        
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Load evaluation data
    try:
        x_data, y_data = load_dataset_data(config.dataset)
        print(f"Loaded {len(x_data)} evaluation samples for {config.dataset}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    # Evaluate model
    try:
        results = evaluate_model(model, x_data, y_data)
        print_results(results)
        
        # Generate visualizations if requested
        if args.visualize:
            print("Visualization not implemented yet")
            # TODO: Implement visualization using shared plotting package
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.debug:
            raise


if __name__ == "__main__":
    main() 