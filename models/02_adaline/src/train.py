"""
ADALINE Training Script.

Training script for ADALINE model with Delta Rule learning algorithm.
Follows the Simple implementation pattern.
"""

import argparse
import sys
from pathlib import Path
import torch
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_experiment_config, validate_config, list_experiments
from model import create_adaline
from constants import get_experiment_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ADALINE model")
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="debug_small",
        help="Experiment name to run"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override number of epochs"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--list-experiments", 
        action="store_true",
        help="List available experiments"
    )
    
    parser.add_argument(
        "--experiment-info", 
        type=str,
        help="Get detailed info about specific experiment"
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
    """Generate fallback synthetic data for training."""
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


def train_adaline(config, x_data, y_data):
    """Train ADALINE model."""
    print(f"Training ADALINE on {config.dataset} dataset")
    print(f"Configuration: {config}")
    
    # Create model
    model = create_adaline(config)
    
    # Train model
    results = model.fit(x_data, y_data)
    
    # Print results
    print(f"Training completed:")
    print(f"  - Converged: {results['converged']}")
    print(f"  - Final MSE: {results['final_mse']:.6f}")
    print(f"  - Epochs trained: {results['epochs_trained']}")
    
    return model, results


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # List experiments if requested
    if args.list_experiments:
        experiments = list_experiments()
        print("Available experiments:")
        for name, desc in experiments.items():
            print(f"  {name}: {desc}")
        return
    
    # Get experiment info if requested
    if args.experiment_info:
        try:
            info = get_experiment_info(args.experiment_info)
            print(f"Experiment: {args.experiment_info}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        except ValueError as e:
            print(f"Error: {e}")
        return
    
    # Get configuration
    try:
        config = get_experiment_config(args.experiment)
        
        # Override parameters if provided
        if args.epochs:
            config.epochs = args.epochs
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        
        # Validate configuration
        config = validate_config(config)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-experiments to see available experiments")
        return
    
    # Load data
    try:
        x_data, y_data = load_dataset_data(config.dataset)
        print(f"Loaded {len(x_data)} samples for {config.dataset}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    # Train model
    try:
        model, results = train_adaline(config, x_data, y_data)
        
        # Save model if requested
        if config.save_model:
            output_dir = Path("outputs/models")
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"{config.name}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        # Generate visualizations if requested
        if args.visualize:
            try:
                from visualize import save_all_adaline_plots
                plots = save_all_adaline_plots(model, x_data, y_data)
                print(f"Generated {len(plots)} visualization plots")
            except ImportError:
                print("Visualization module not available")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if args.debug:
            raise


if __name__ == "__main__":
    main() 