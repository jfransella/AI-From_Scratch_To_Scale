#!/usr/bin/env python3
"""
Training script for the 01_perceptron model.

This script trains the classic Rosenblatt perceptron using the unified
engine framework. Supports multiple datasets including both linearly
separable (strengths) and non-separable (limitations) examples.
"""

import sys
import argparse
import torch
from pathlib import Path
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import setup_logging, set_random_seed, get_logger
from data_utils import load_dataset
from engine.trainer import Trainer
from engine.base import DataSplit
from config import (
    get_training_config,
    get_model_config,
    get_dataset_config,
    print_config_summary,
)
from model import create_perceptron
from constants import MODEL_NAME, ALL_EXPERIMENTS


def create_data_split(
    X: torch.Tensor,
    y: torch.Tensor,
    validation_split: float = 0.2,
    test_split: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """
    Create train/validation/test data splits.

    Args:
        X: Input features
        y: Target labels
        validation_split: Fraction for validation set
        test_split: Fraction for test set
        random_state: Random seed

    Returns:
        DataSplit object with train/val/test splits
    """
    # Set random seed
    torch.manual_seed(random_state)

    n_samples = len(X)
    indices = torch.randperm(n_samples)

    # Calculate split sizes
    n_test = int(test_split * n_samples)
    n_val = int(validation_split * n_samples)
    n_train = n_samples - n_test - n_val

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val] if n_val > 0 else None
    test_idx = indices[n_train + n_val :] if n_test > 0 else None

    # Create data splits
    x_train, y_train = X[train_idx], y[train_idx]
    x_val, y_val = (X[val_idx], y[val_idx]) if val_idx is not None else (None, None)
    x_test, y_test = (
        (X[test_idx], y[test_idx]) if test_idx is not None else (None, None)
    )

    return DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def main():
    """Main training function using unified infrastructure."""
    parser = argparse.ArgumentParser(description="Train Perceptron model")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=f"Experiment name. Available: {ALL_EXPERIMENTS}",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (Perceptron uses full batch by default)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after training",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with reduced epochs"
    )
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Print configuration summary and exit",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    # Handle special arguments
    if args.list_experiments:
        print("\nAvailable Perceptron experiments:")
        print("-" * 50)
        for exp in ALL_EXPERIMENTS:
            try:
                dataset_config = get_dataset_config(exp)
                print(f"{exp:20} - {dataset_config['description']}")
            except Exception as e:
                print(f"{exp:20} - Error: {e}")
        return 0

    if args.config_summary:
        try:
            print_config_summary(args.experiment)
        except Exception as e:
            print(f"Error printing config summary: {e}")
            return 1
        return 0

    # Validate experiment
    if args.experiment not in ALL_EXPERIMENTS:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print(f"Available experiments: {ALL_EXPERIMENTS}")
        return 1

    try:
        # Load configurations
        print(f"Loading configuration for experiment: {args.experiment}")

        # Build configuration overrides from command line
        overrides = {}
        if args.epochs is not None:
            overrides["max_epochs"] = args.epochs
        if args.learning_rate is not None:
            overrides["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if args.device is not None:
            overrides["device"] = args.device
        if args.wandb:
            overrides["use_wandb"] = True
        if args.debug:
            overrides["max_epochs"] = min(overrides.get("max_epochs", 50), 20)
            overrides["log_freq"] = 1
            overrides["verbose"] = True

        # Get configurations
        training_config = get_training_config(args.experiment, **overrides)
        model_config = get_model_config(args.experiment, **overrides)
        dataset_config = get_dataset_config(args.experiment)

        # Setup logging
        setup_logging(level="DEBUG" if args.debug else "INFO")
        logger = get_logger(__name__)

        # Set random seed
        if training_config.random_seed is not None:
            set_random_seed(training_config.random_seed)

        logger.info(f"Starting {MODEL_NAME} training")
        logger.info(f"Experiment: {args.experiment}")
        logger.info(f"Dataset: {dataset_config['dataset_name']}")
        logger.info(f"Expected accuracy: {dataset_config['expected_accuracy']:.3f}")
        logger.info(f"Difficulty: {dataset_config['difficulty']}")

        # Load dataset
        logger.info("Loading dataset...")
        X, y = load_dataset(
            dataset_config["dataset_name"], dataset_config["dataset_params"]
        )

        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        logger.info(
            f"Dataset loaded: {X.shape} features, {len(torch.unique(y))} classes"
        )

        # Create data splits
        logger.info("Creating data splits...")
        data_split = create_data_split(
            X,
            y,
            validation_split=training_config.validation_split,
            test_split=0.2,  # Fixed test split
            random_state=training_config.random_seed,
        )

        split_info = data_split.get_split_info()
        logger.info(f"Data splits: {split_info}")

        # Create model
        logger.info("Creating Perceptron model...")
        model = create_perceptron(model_config)

        model_info = model.get_model_info()
        logger.info(f"Model created: {model_info['total_parameters']} parameters")
        logger.info(
            f"Architecture: {model_info['input_size']} -> {model_info['output_size']}"
        )
        logger.info(f"Activation: {model_info['activation']}")

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(training_config)

        # Train model
        logger.info("Starting training...")
        print(f"\nTraining {MODEL_NAME} on {args.experiment}")
        print(
            f"Dataset: {dataset_config['dataset_name']} ({dataset_config['difficulty']})"
        )
        print(
            f"Samples: {split_info['train_size']} train, {split_info.get('val_size', 0)} val"
        )
        print(f"Learning rate: {training_config.learning_rate}")
        print(f"Max epochs: {training_config.max_epochs}")
        print(f"Device: {training_config.device}")
        print("-" * 60)

        # Train the model
        training_result = trainer.train(model, data_split)

        # Print results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Experiment: {args.experiment}")
        print(f"Model: {MODEL_NAME}")
        print(f"Dataset: {dataset_config['dataset_name']}")
        print("-" * 60)
        print(f"Epochs trained: {training_result.epochs_trained}")
        print(f"Training time: {training_result.total_training_time:.2f} seconds")
        print(f"Converged: {'✓' if training_result.converged else '✗'}")
        print(f"Final train accuracy: {training_result.final_train_accuracy:.4f}")

        if training_result.final_val_accuracy is not None:
            print(
                f"Final validation accuracy: {training_result.final_val_accuracy:.4f}"
            )

        if training_result.final_test_accuracy is not None:
            print(f"Final test accuracy: {training_result.final_test_accuracy:.4f}")

        # Performance vs expectation
        expected_acc = dataset_config["expected_accuracy"]
        actual_acc = training_result.final_train_accuracy

        if actual_acc >= expected_acc * 0.9:
            performance = "✓ MEETS EXPECTATIONS"
        elif actual_acc >= expected_acc * 0.7:
            performance = "~ BELOW EXPECTATIONS"
        else:
            performance = "✗ POOR PERFORMANCE"

        print(f"Expected accuracy: {expected_acc:.3f}")
        print(f"Performance: {performance}")
        print("=" * 60)

        # Save additional information
        if training_result.best_model_path:
            logger.info(f"Best model saved: {training_result.best_model_path}")
        if training_result.final_model_path:
            logger.info(f"Final model saved: {training_result.final_model_path}")

        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            try:
                from plotting import generate_training_plots

                plots_dir = Path(training_config.output_dir) / "visualizations"
                plots_dir.mkdir(exist_ok=True)

                # Generate training plots
                plot_path = plots_dir / f"{args.experiment}_training.png"
                generate_training_plots(training_result, str(plot_path))
                logger.info(f"Training plots saved: {plot_path}")

                # Generate decision boundary if 2D data
                if model_config["input_size"] == 2:
                    from plotting import plot_decision_boundary

                    boundary_path = (
                        plots_dir / f"{args.experiment}_decision_boundary.png"
                    )
                    plot_decision_boundary(
                        model,
                        data_split.x_train,
                        data_split.y_train,
                        str(boundary_path),
                    )
                    logger.info(f"Decision boundary plot saved: {boundary_path}")

            except ImportError:
                logger.warning("Plotting functions not available")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")

        # Educational summary
        print(f"\nEducational Summary:")
        print(f"Description: {dataset_config['description']}")

        if args.experiment in [
            "iris_binary",
            "linear_separable",
            "debug_small",
            "debug_linear",
        ]:
            print(
                "✓ This experiment demonstrates Perceptron strengths on linearly separable data"
            )
        elif args.experiment in ["xor_problem", "circles_dataset", "mnist_subset"]:
            print(
                "⚠ This experiment exposes Perceptron limitations on non-linearly separable data"
            )
            print(
                "💡 These limitations motivated the development of multi-layer perceptrons (MLPs)"
            )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            import traceback

            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
