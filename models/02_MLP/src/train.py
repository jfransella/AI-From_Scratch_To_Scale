#!/usr/bin/env python3
"""
Training script for Multi-Layer Perceptron (MLP) experiments.

This script demonstrates the breakthrough capability of MLPs to solve
non-linearly separable problems, particularly the famous XOR problem
that single-layer perceptrons cannot handle.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path for shared package imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import setup_logging, setup_device, set_random_seed
from data_utils import generate_xor_dataset, generate_circles_dataset

from config import (
    get_experiment_config, 
    list_available_experiments,
    get_experiment_info,
    apply_environment_overrides
)
from model import MLP


def create_dataset(config):
    """
    Create dataset based on configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    if config.dataset_type == "xor":
        # Create XOR dataset - classic 4-sample problem
        features, labels = generate_xor_dataset(n_samples=4, noise=0.0, random_state=42)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        
        # For XOR, use all data for training (only 4 samples)
        return x, y, x, y
    
    elif config.dataset_type == "circles":
        # Create circles dataset
        params = config.dataset_params
        num_samples = params.get("num_samples", 1000)
        noise = params.get("noise", 0.1)
        
        features, labels = generate_circles_dataset(num_samples=num_samples, noise=noise)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        
        # Split into train/test
        n_train = int(0.8 * len(x))
        indices = torch.randperm(len(x))
        
        x_train = x[indices[:n_train]]
        y_train = y[indices[:n_train]]
        x_test = x[indices[n_train:]]
        y_test = y[indices[n_train:]]
        
        return x_train, y_train, x_test, y_test
    
    elif config.dataset_type in ["moons", "spirals"]:
        # For now, fall back to XOR for unsupported datasets
        print(f"Warning: {config.dataset_type} dataset not yet implemented, using XOR")
        features, labels = generate_xor_dataset()
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y, x, y
    
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")


def train_experiment(config, device="cpu"):
    """
    Train a single MLP experiment.
    
    Args:
        config: Experiment configuration
        device: Device to run on
        
    Returns:
        Tuple of (model, results)
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    set_random_seed(config.random_seed)
    
    # Create dataset
    print(f"Creating {config.dataset_type} dataset...")
    x_train, y_train, x_test, y_test = create_dataset(config)
    
    print(f"Train set: {x_train.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    
    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Create model
    print(f"\nCreating MLP: {config.input_size} -> {config.hidden_layers} -> {config.output_size}")
    model = MLP(
        input_size=config.input_size,
        hidden_layers=config.hidden_layers,
        output_size=config.output_size,
        activation=config.activation,
        weight_init=config.weight_init,
        device=device
    )
    
    # Train model
    print("\nStarting training...")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Convergence threshold: {config.convergence_threshold}")
    
    start_time = time.time()
    
    results = model.train_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test if x_test is not x_train else None,
        y_test=y_test if y_test is not y_train else None,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        convergence_threshold=config.convergence_threshold,
        patience=config.patience,
        verbose=config.verbose
    )
    
    training_time = time.time() - start_time
    results["training_time"] = training_time
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Experiment completed: {config.name}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Final train accuracy: {results['final_train_accuracy']:.4f}")
    if "final_test_accuracy" in results:
        print(f"Final test accuracy: {results['final_test_accuracy']:.4f}")
    
    if results["converged"]:
        print(f"✅ Converged at epoch {results['convergence_epoch']}")
    else:
        print(f"⚠️  Did not converge within {config.max_epochs} epochs")
    
    # XOR-specific success message
    if config.dataset_type == "xor" and results["final_train_accuracy"] >= 0.99:
        print("🎉 Successfully solved the XOR problem!")
        print("   This demonstrates MLP's ability to learn non-linear patterns")
        print("   that single-layer perceptrons cannot handle.")
    
    print(f"{'='*60}")
    
    # Save model and results if requested
    if config.save_model:
        save_experiment_results(config, model, results)
    
    return model, results


def save_experiment_results(config, model, results):
    """Save experiment results to disk."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{config.name}_model.pth"
    model.save_model(str(model_path), include_history=True)
    
    # Save results summary
    results_path = output_dir / f"{config.name}_results.json"
    import json
    
    # Add config info to results
    full_results = {
        "experiment": {
            "name": config.name,
            "description": config.description,
            "dataset": config.dataset_type,
            "architecture": config.hidden_layers,
            "activation": config.activation,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs
        },
        "results": results,
        "model_info": model.get_model_info()
    }
    
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"Results saved to: {output_dir}")


def run_educational_sequence():
    """Run a sequence of experiments for educational purposes."""
    from config import get_educational_sequence
    
    print("🎓 Running educational experiment sequence...")
    print("This demonstrates the progression from basic to advanced MLP capabilities.\n")
    
    sequence = get_educational_sequence()
    results = {}
    
    device = setup_device()
    
    for exp_name in sequence:
        try:
            config = get_experiment_config(exp_name)
            config = apply_environment_overrides(config, "default")
            
            model, exp_results = train_experiment(config, device)
            results[exp_name] = exp_results
            
            # Brief pause between experiments
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error in experiment {exp_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 EDUCATIONAL SEQUENCE SUMMARY")
    print(f"{'='*80}")
    
    for exp_name, exp_results in results.items():
        status = "✅ Solved" if exp_results.get("final_train_accuracy", 0) >= 0.99 else "⚠️  Partial"
        accuracy = exp_results.get("final_train_accuracy", 0)
        epochs = exp_results.get("epochs_trained", 0)
        
        print(f"{exp_name:20} | {status} | Accuracy: {accuracy:.3f} | Epochs: {epochs:4d}")
    
    print(f"{'='*80}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Layer Perceptron (MLP) models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --experiment xor_breakthrough    # Solve the XOR problem
  python train.py --experiment quick_test          # Quick test run
  python train.py --experiment circles             # Non-linear dataset
  python train.py --list-experiments               # Show available experiments
  python train.py --educational-sequence           # Run full educational sequence
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Name of the experiment to run"
    )
    
    parser.add_argument(
        "--list-experiments", "-l",
        action="store_true",
        help="List available experiments"
    )
    
    parser.add_argument(
        "--experiment-info", "-i",
        type=str,
        help="Show information about a specific experiment"
    )
    
    parser.add_argument(
        "--educational-sequence",
        action="store_true",
        help="Run the full educational experiment sequence"
    )
    
    parser.add_argument(
        "--environment",
        choices=["default", "debug", "production"],
        default="default",
        help="Environment configuration"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Get device
    if args.device == "auto":
        device = setup_device()
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Handle different commands
    if args.list_experiments:
        print("Available experiments:")
        for exp_name in list_available_experiments():
            info = get_experiment_info(exp_name)
            print(f"  {exp_name:20} - {info['description']}")
        return
    
    if args.experiment_info:
        try:
            info = get_experiment_info(args.experiment_info)
            print(f"Experiment: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Architecture: {info['architecture']}")
            print(f"Dataset: {info['dataset']}")
            print(f"Max epochs: {info['max_epochs']}")
            print(f"Learning rate: {info['learning_rate']}")
        except KeyError:
            print(f"Unknown experiment: {args.experiment_info}")
            print("Use --list-experiments to see available options")
        return
    
    if args.educational_sequence:
        run_educational_sequence()
        return
    
    if args.experiment:
        try:
            # Load and modify config
            config = get_experiment_config(args.experiment)
            config = apply_environment_overrides(config, args.environment)
            
            # Run experiment
            model, results = train_experiment(config, device)
            
            # Special message for XOR breakthrough
            if args.experiment == "xor_breakthrough" and results["final_train_accuracy"] >= 0.99:
                print("\n🎉 BREAKTHROUGH ACHIEVED! 🎉")
                print("You have successfully demonstrated that MLPs can solve")
                print("the XOR problem that stumped single-layer perceptrons!")
                print("This is a historic moment in neural network development.")
            
        except KeyError:
            print(f"Unknown experiment: {args.experiment}")
            print("Use --list-experiments to see available options")
            return 1
        except Exception as e:
            print(f"Error running experiment: {e}")
            return 1
    else:
        print("Please specify an experiment with --experiment or use --help for options")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 