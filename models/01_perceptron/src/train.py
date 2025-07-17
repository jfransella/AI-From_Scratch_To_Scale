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

# Add project root to path for imports (must be first)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import setup_logging, set_random_seed, get_logger
from data_utils import load_dataset
from engine.trainer import Trainer
from engine.base import DataSplit
from .config import (
    get_training_config,
    get_model_config,
    get_dataset_config,
    print_config_summary,
)
from .model import create_perceptron
from .constants import MODEL_NAME, ALL_EXPERIMENTS

# Optional plotting imports (handled gracefully if not installed)
try:
    from plotting import plot_training_history, plot_decision_boundary
except ImportError:
    plot_training_history = None
    plot_decision_boundary = None


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
        required=False,  # Not required for special actions like --list-experiments
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
    parser.add_argument(
        "--experiment-info", type=str,
        help="Show information about a specific experiment"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--environment", choices=["default", "debug", "production"],
        default="default", help="Environment configuration"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="ai-from-scratch",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--tags", type=str, nargs="+", default=[],
        help="Tags to attach to the run"
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None,
        help="Path to checkpoint to load before training"
    )
    parser.add_argument(
        "--no-save-checkpoint", action="store_true",
        help="Skip saving final model checkpoint"
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

    # Validate experiment (required for normal training)
    if not args.experiment:
        print("Error: --experiment is required for training")
        print(f"Use --list-experiments to see available options")
        return 1
    
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
        setup_logging(
            level="DEBUG" if args.debug else "INFO",
            log_dir="outputs/logs",
            file_output=True,
            console_output=True
        )
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
        logger.info(f"Training {MODEL_NAME} on {args.experiment}")
        logger.info(
            f"Dataset: {dataset_config['dataset_name']} ({dataset_config['difficulty']})"
        )
        logger.info(
            f"Samples: {split_info['train_size']} train, {split_info.get('val_size', 0)} val"
        )
        logger.info(f"Learning rate: {training_config.learning_rate}")
        logger.info(f"Max epochs: {training_config.max_epochs}")
        logger.info(f"Device: {training_config.device}")
        logger.info("-" * 60)

        # Train the model
        training_result = trainer.train(model, data_split)

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Experiment: {args.experiment}")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Dataset: {dataset_config['dataset_name']}")
        logger.info("-" * 60)
        logger.info(f"Epochs trained: {training_result.epochs_trained}")
        logger.info(f"Training time: {training_result.total_training_time:.2f} seconds")
        logger.info(f"Converged: {'✓' if training_result.converged else '✗'}")
        logger.info(f"Final train accuracy: {training_result.final_train_accuracy:.4f}")

        if training_result.final_val_accuracy is not None:
            logger.info(
                f"Final validation accuracy: {training_result.final_val_accuracy:.4f}"
            )

        if training_result.final_test_accuracy is not None:
            logger.info(f"Final test accuracy: {training_result.final_test_accuracy:.4f}")

        # Performance vs expectation
        expected_acc = dataset_config["expected_accuracy"]
        actual_acc = training_result.final_train_accuracy

        if actual_acc >= expected_acc * 0.9:
            performance = "✓ MEETS EXPECTATIONS"
        elif actual_acc >= expected_acc * 0.7:
            performance = "~ BELOW EXPECTATIONS"
        else:
            performance = "✗ POOR PERFORMANCE"

        logger.info(f"Expected accuracy: {expected_acc:.3f}")
        logger.info(f"Performance: {performance}")
        logger.info("=" * 60)

        # Save additional information
        if training_result.best_model_path:
            logger.info(f"Best model saved: {training_result.best_model_path}")
        if training_result.final_model_path:
            logger.info(f"Final model saved: {training_result.final_model_path}")

        # Generate visualizations if requested
        if args.visualize:
            logger.info("\nGenerating visualizations...")
            plots_dir = Path(training_config.output_dir) / "visualizations"
            plots_dir.mkdir(exist_ok=True)

            if plot_training_history is not None:
                # Plot training history (loss and accuracy)
                plot_path = plots_dir / f"{args.experiment}_training_history.png"
                plot_training_history(
                    loss_history=training_result.loss_history,
                    accuracy_history=training_result.train_accuracy_history,
                    title=f"{MODEL_NAME} Training History",
                    save_path=str(plot_path),
                )
                logger.info(f"Training history plot saved: {plot_path}")
            else:
                logger.warning("plot_training_history not available")

            if plot_decision_boundary is not None and model_config["input_size"] == 2:
                boundary_path = plots_dir / f"{args.experiment}_decision_boundary.png"
                plot_decision_boundary(
                    model,
                    data_split.x_train,
                    data_split.y_train,
                    title=f"{MODEL_NAME} Decision Boundary",
                    save_path=str(boundary_path),
                )
                logger.info(f"Decision boundary plot saved: {boundary_path}")
            elif model_config["input_size"] == 2:
                logger.warning("plot_decision_boundary not available")

        # Educational summary
        logger.info(f"\nEducational Summary:")
        logger.info(f"Description: {dataset_config['description']}")

        if args.experiment in [
            "iris_binary",
            "linear_separable",
            "debug_small",
            "debug_linear",
        ]:
            logger.info(
                "✓ This experiment demonstrates Perceptron strengths on linearly separable data"
            )
        elif args.experiment in ["xor_problem", "circles_dataset", "mnist_subset"]:
            logger.info(
                "⚠ This experiment exposes Perceptron limitations on non-linearly separable data"
            )
            logger.info(
                "💡 These limitations motivated the development of multi-layer perceptrons (MLPs)"
            )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        import traceback
        logger.error(f"Training failed: {e}")
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
