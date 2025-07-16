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

# Optional plotting imports
try:
    from plotting import plot_training_history, plot_decision_boundary
except ImportError:
    plot_training_history = None
    plot_decision_boundary = None

from utils import setup_logging, setup_device, set_random_seed, get_logger
from data_utils import generate_xor_dataset, generate_circles_dataset

from config import (
    get_experiment_config,
    list_available_experiments,
    get_experiment_info,
    apply_environment_overrides,
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

        features, labels = generate_circles_dataset(
            num_samples=num_samples, noise=noise
        )
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
    logger = get_logger("ai_from_scratch")

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"{'='*60}")

    # Set random seed for reproducibility
    set_random_seed(config.random_seed)

    # Create dataset
    logger.info(f"Creating {config.dataset_type} dataset...")
    x_train, y_train, x_test, y_test = create_dataset(config)

    logger.info(f"Train set: {x_train.shape[0]} samples")
    logger.info(f"Test set: {x_test.shape[0]} samples")

    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Create model
    logger.info(
        f"\nCreating MLP: {config.input_size} -> {config.hidden_layers} -> {config.output_size}"
    )
    model = MLP(
        input_size=config.input_size,
        hidden_layers=config.hidden_layers,
        output_size=config.output_size,
        activation=config.activation,
        weight_init=config.weight_init,
        device=device,
    )

    # Train model
    logger.info("\nStarting training...")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Max epochs: {config.max_epochs}")
    logger.info(f"Convergence threshold: {config.convergence_threshold}")

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
        verbose=config.verbose,
    )

    training_time = time.time() - start_time
    results["training_time"] = training_time

    # Log final results
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment completed: {config.name}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Final loss: {results['final_loss']:.6f}")
    logger.info(f"Final train accuracy: {results['final_train_accuracy']:.4f}")
    if "final_test_accuracy" in results:
        logger.info(f"Final test accuracy: {results['final_test_accuracy']:.4f}")

    if results["converged"]:
        logger.info(f"✅ Converged at epoch {results['convergence_epoch']}")
    else:
        logger.info(f"⚠️  Did not converge within {config.max_epochs} epochs")

    # XOR-specific success message
    if config.dataset_type == "xor" and results["final_train_accuracy"] >= 0.99:
        logger.info("🎉 Successfully solved the XOR problem!")
        logger.info("   This demonstrates MLP's ability to learn non-linear patterns")
        logger.info("   that single-layer perceptrons cannot handle.")

    logger.info(f"{'='*60}")

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
            "max_epochs": config.max_epochs,
        },
        "results": results,
        "model_info": model.get_model_info(),
    }

    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)

    logger = get_logger("ai_from_scratch")
    logger.info(f"Results saved to: {output_dir}")


def run_educational_sequence():
    """Run a sequence of experiments for educational purposes."""
    from config import get_educational_sequence

    logger = get_logger("ai_from_scratch")

    logger.info("🎓 Running educational experiment sequence...")
    logger.info(
        "This demonstrates the progression from basic to advanced MLP capabilities.\n"
    )

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
            logger.error(f"❌ Error in experiment {exp_name}: {e}")
            continue

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 EDUCATIONAL SEQUENCE SUMMARY")
    logger.info(f"{'='*80}")

    for exp_name, exp_results in results.items():
        status = (
            "✅ Solved"
            if exp_results.get("final_train_accuracy", 0) >= 0.99
            else "⚠️  Partial"
        )
        accuracy = exp_results.get("final_train_accuracy", 0)
        epochs = exp_results.get("epochs_trained", 0)

        logger.info(
            f"{exp_name:20} | {status} | Accuracy: {accuracy:.3f} | Epochs: {epochs:4d}"
        )

    logger.info(f"{'='*80}")


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
        """,
    )

    parser.add_argument(
        "--experiment", "-e", type=str, required=True, help="Name of the experiment to run"
    )

    parser.add_argument(
        "--list-experiments",
        "-l",
        action="store_true",
        help="List available experiments",
    )

    parser.add_argument(
        "--experiment-info",
        "-i",
        type=str,
        help="Show information about a specific experiment",
    )

    parser.add_argument(
        "--educational-sequence",
        action="store_true",
        help="Run the full educational experiment sequence",
    )

    parser.add_argument(
        "--environment",
        choices=["default", "debug", "production"],
        default="default",
        help="Environment configuration",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    
    # Debugging
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with reduced epochs"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="ai-from-scratch",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--tags", type=str, nargs="+", default=[],
        help="Tags to attach to the run"
    )
    
    # Model and data options
    parser.add_argument(
        "--load-checkpoint", type=str, default=None,
        help="Path to checkpoint to load before training"
    )
    parser.add_argument(
        "--no-save-checkpoint", action="store_true",
        help="Skip saving final model checkpoint"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after training",
    )

    args = parser.parse_args()

    # Setup logging with file output
    setup_logging(
        level=args.log_level,
        log_dir="outputs/logs",
        file_output=True,
        console_output=True,
    )
    logger = get_logger("ai_from_scratch")

    # Get device
    if args.device == "auto":
        device = setup_device()
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Handle different commands
    if args.list_experiments:
        logger.info("Available experiments:")
        for exp_name in list_available_experiments():
            info = get_experiment_info(exp_name)
            logger.info(f"  {exp_name:20} - {info['description']}")
        return

    if args.experiment_info:
        try:
            info = get_experiment_info(args.experiment_info)
            logger.info(f"Experiment: {info['name']}")
            logger.info(f"Description: {info['description']}")
            logger.info(f"Architecture: {info['architecture']}")
            logger.info(f"Dataset: {info['dataset']}")
            logger.info(f"Max epochs: {info['max_epochs']}")
            logger.info(f"Learning rate: {info['learning_rate']}")
        except KeyError:
            logger.error(f"Unknown experiment: {args.experiment_info}")
            logger.error("Use --list-experiments to see available options")
        return

    if args.educational_sequence:
        run_educational_sequence()
        return

    if args.experiment:
        try:
            # Build configuration overrides from command line
            overrides = {}
            if args.epochs is not None:
                overrides["max_epochs"] = args.epochs
            if args.learning_rate is not None:
                overrides["learning_rate"] = args.learning_rate
            if args.batch_size is not None:
                overrides["batch_size"] = args.batch_size
            if args.seed is not None:
                overrides["random_seed"] = args.seed
            if args.device != "auto":
                overrides["device"] = args.device
            if args.wandb:
                overrides["use_wandb"] = True
            if args.debug:
                overrides["max_epochs"] = min(overrides.get("max_epochs", 50), 20)
                overrides["log_freq"] = 1
                overrides["verbose"] = True
            
            # Load and modify config
            config = get_experiment_config(args.experiment)
            config = apply_environment_overrides(config, args.environment)
            
            # Apply command line overrides
            for key, value in overrides.items():
                setattr(config, key, value)

            # Train experiment
            x_train, y_train, x_test, y_test = create_dataset(config)
            model, results = train_experiment(config, device)

            # Special message for XOR breakthrough
            if (
                args.experiment == "xor_breakthrough"
                and results["final_train_accuracy"] >= 0.99
            ):
                logger.info("\n🎉 BREAKTHROUGH ACHIEVED! 🎉")
                logger.info("You have successfully demonstrated that MLPs can solve")
                logger.info("the XOR problem that stumped single-layer perceptrons!")
                logger.info("This is a historic moment in neural network development.")

            # Visualization integration
            if args.visualize:
                logger.info("\nGenerating visualizations...")
                plots_dir = Path(config.output_dir) / "visualizations"
                plots_dir.mkdir(exist_ok=True)
                if plot_training_history is not None:
                    plot_path = plots_dir / f"{config.name}_training_history.png"
                    plot_training_history(
                        loss_history=model.training_history["loss"],
                        accuracy_history=model.training_history["accuracy"],
                        title="MLP Training History",
                        save_path=str(plot_path),
                    )
                    logger.info(f"Training history plot saved: {plot_path}")
                else:
                    logger.warning("plot_training_history not available")
                # Plot decision boundary if input is 2D
                if plot_decision_boundary is not None and config.input_size == 2:
                    boundary_path = plots_dir / f"{config.name}_decision_boundary.png"
                    plot_decision_boundary(
                        model,
                        x_train.numpy() if hasattr(x_train, "numpy") else x_train,
                        y_train.numpy() if hasattr(y_train, "numpy") else y_train,
                        title="MLP Decision Boundary",
                        save_path=str(boundary_path),
                    )
                    logger.info(f"Decision boundary plot saved: {boundary_path}")
                elif config.input_size == 2:
                    logger.warning("plot_decision_boundary not available")
                # Clean exit after visualization
                import sys

                sys.exit(0)

        except KeyError:
            logger.error(f"Unknown experiment: {args.experiment}")
            logger.error("Use --list-experiments to see available options")
            return 1
        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            return 1
    else:
        logger.error(
            "Please specify an experiment with --experiment or use --help for options"
        )
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
