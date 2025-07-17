#!/usr/bin/env python3
"""
Training script for the 01_perceptron model.

This script trains the classic Rosenblatt perceptron using the unified
engine framework. Supports multiple datasets including both linearly
separable (strengths) and non-separable (limitations) examples.
"""

import sys
import argparse
import traceback
from pathlib import Path
import torch

# Import shared packages
from utils import setup_logging, set_random_seed, get_logger
from data_utils import load_dataset
from engine.trainer import Trainer
from engine.base import DataSplit

# Import model-specific components
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
    x_features: torch.Tensor,
    y_target: torch.Tensor,
    validation_split: float = 0.2,
    test_split: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """
    Create train/validation/test data splits.

    Args:
        x_features: Input features
        y_target: Target labels
        validation_split: Fraction for validation set
        test_split: Fraction for test set
        random_state: Random seed

    Returns:
        DataSplit object with train/val/test splits
    """
    # Set random seed
    torch.manual_seed(random_state)

    n_samples = len(x_features)
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
    x_train, y_train = x_features[train_idx], y_target[train_idx]
    x_val, y_val = (
        (x_features[val_idx], y_target[val_idx])
        if val_idx is not None
        else (None, None)
    )
    x_test, y_test = (
        (x_features[test_idx], y_target[test_idx])
        if test_idx is not None
        else (None, None)
    )

    return DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def parse_args():
    """Parse command line arguments for the training script."""
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
        "--experiment-info",
        type=str,
        help="Show information about a specific experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--environment",
        choices=["default", "debug", "production"],
        default="default",
        help="Environment configuration",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ai-from-scratch",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--tags", type=str, nargs="+", default=[], help="Tags to attach to the run"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load before training",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Skip saving final model checkpoint",
    )
    return parser.parse_args()


def handle_special_args(args):
    """Handle special command line arguments like --list-experiments and --config-summary."""
    if args.list_experiments:
        print("\nAvailable Perceptron experiments:")
        print("-" * 50)
        for exp in ALL_EXPERIMENTS:
            try:
                dataset_config = get_dataset_config(exp)
                print(f"{exp:20} - {dataset_config['description']}")
            except Exception as e:  # pylint: disable=broad-except
                # Defensive: get_dataset_config may raise various errors,
                # we want to show all possible experiments
                print(f"{exp:20} - Error: {e}")
        return 0
    if args.config_summary:
        try:
            print_config_summary(args.experiment)
        except Exception as e:  # pylint: disable=broad-except
            # Defensive: print_config_summary may raise various errors,
            # we want to show a user-friendly message
            print(f"Error printing config summary: {e}")
            return 1
        return 0
    return None


def validate_experiment_arg(args):
    """Validate that a valid experiment name was provided."""
    if not args.experiment:
        print("Error: --experiment is required for training")
        print("Use --list-experiments to see available options")
        return 1
    if args.experiment not in ALL_EXPERIMENTS:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print(f"Available experiments: {ALL_EXPERIMENTS}")
        return 1
    return None


def prepare_training(args):
    """Prepare training configuration and overrides from command line arguments."""
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
    training_config = get_training_config(args.experiment, **overrides)
    model_config = get_model_config(args.experiment, **overrides)
    dataset_config = get_dataset_config(args.experiment)
    return training_config, model_config, dataset_config, overrides


def _setup_logging_and_seed(args, training_config):
    """Set up logging and random seed for training."""
    setup_logging(
        level="DEBUG" if args.debug else "INFO",
        log_dir="outputs/logs",
        file_output=True,
        console_output=True,
    )
    logger = get_logger(__name__)
    if training_config.random_seed is not None:
        set_random_seed(training_config.random_seed)
    return logger


def _load_and_prepare_data(logger, dataset_config):
    """Load and prepare dataset for training."""
    logger.info("Loading dataset...")
    x_features, y_target = load_dataset(
        dataset_config["dataset_name"], dataset_config["dataset_params"]
    )
    if not isinstance(x_features, torch.Tensor):
        x_features = torch.tensor(x_features, dtype=torch.float32)
    if not isinstance(y_target, torch.Tensor):
        y_target = torch.tensor(y_target, dtype=torch.float32)
    logger.info(
        "Dataset loaded: %s features, %d classes",
        x_features.shape,
        len(torch.unique(y_target)),
    )
    return x_features, y_target


def _create_model_and_trainer(logger, model_config, training_config):
    """Create the model and trainer instances."""
    logger.info("Creating Perceptron model...")
    model = create_perceptron(model_config)
    model_info = model.get_model_info()
    logger.info("Model created: %d parameters", model_info["total_parameters"])
    logger.info(
        "Architecture: %d -> %d", model_info["input_size"], model_info["output_size"]
    )
    logger.info("Activation: %s", model_info["activation"])
    logger.info("Initializing trainer...")
    trainer = Trainer(training_config)
    return model, trainer


def _log_training_results(logger, args, dataset_config, training_result):
    """Log training results and performance metrics."""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info("Experiment: %s", args.experiment)
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Dataset: %s", dataset_config["dataset_name"])
    logger.info("-" * 60)
    logger.info("Epochs trained: %d", training_result.epochs_trained)
    logger.info("Training time: %.2f seconds", training_result.total_training_time)
    logger.info("Converged: %s", "✓" if training_result.converged else "✗")
    logger.info("Final train accuracy: %.4f", training_result.final_train_accuracy)
    if training_result.final_val_accuracy is not None:
        logger.info(
            "Final validation accuracy: %.4f", training_result.final_val_accuracy
        )
    if training_result.final_test_accuracy is not None:
        logger.info("Final test accuracy: %.4f", training_result.final_test_accuracy)
    expected_acc = dataset_config["expected_accuracy"]
    actual_acc = training_result.final_train_accuracy
    if actual_acc >= expected_acc * 0.9:
        performance = "✓ MEETS EXPECTATIONS"
    elif actual_acc >= expected_acc * 0.7:
        performance = "~ BELOW EXPECTATIONS"
    else:
        performance = "✗ POOR PERFORMANCE"
    logger.info("Expected accuracy: %.3f", expected_acc)
    logger.info("Performance: %s", performance)
    logger.info("=" * 60)
    if training_result.best_model_path:
        logger.info("Best model saved: %s", training_result.best_model_path)
    if training_result.final_model_path:
        logger.info("Final model saved: %s", training_result.final_model_path)


def _log_data_splits(logger, split_info):
    """Log data split information."""
    logger.info("Data splits: %s", split_info)
    logger.info(
        "Samples: %d train, %d val",
        split_info["train_size"],
        split_info.get("val_size", 0),
    )


def run_training(args, training_config, model_config, dataset_config):
    """Run the complete training process."""
    logger = _setup_logging_and_seed(args, training_config)
    logger.info("Starting %s training", MODEL_NAME)
    logger.info("Experiment: %s", args.experiment)
    logger.info("Dataset: %s", dataset_config["dataset_name"])
    logger.info("Expected accuracy: %.3f", dataset_config["expected_accuracy"])
    logger.info("Difficulty: %s", dataset_config["difficulty"])

    x_features, y_target = _load_and_prepare_data(logger, dataset_config)

    logger.info("Creating data splits...")
    data_split = create_data_split(
        x_features,
        y_target,
        validation_split=training_config.validation_split,
        test_split=0.2,
        random_state=training_config.random_seed,
    )
    split_info = data_split.get_split_info()
    _log_data_splits(logger, split_info)

    model, trainer = _create_model_and_trainer(logger, model_config, training_config)

    logger.info("Starting training...")
    logger.info("Training %s on %s", MODEL_NAME, args.experiment)
    logger.info(
        "Dataset: %s (%s)", dataset_config["dataset_name"], dataset_config["difficulty"]
    )
    logger.info(
        "Learning rate: %f, Max epochs: %d, Device: %s",
        training_config.learning_rate,
        training_config.max_epochs,
        training_config.device,
    )
    logger.info("-" * 60)

    training_result = trainer.train(model, data_split)
    _log_training_results(logger, args, dataset_config, training_result)

    return logger, model, data_split, model_config, training_result


def generate_visualizations(
    args, logger, model, data_split, model_config, training_result, training_config
):
    """Generate visualizations if requested."""
    viz_args = (
        args,
        logger,
        model,
        data_split,
        model_config,
        training_result,
        training_config,
    )
    _generate_visualizations(*viz_args)


def _generate_visualizations(
    args, logger, model, data_split, model_config, training_result, training_config
):
    if args.visualize:
        logger.info("\nGenerating visualizations...")
        plots_dir = Path(training_config.output_dir) / "visualizations"
        plots_dir.mkdir(exist_ok=True)
        if plot_training_history is not None:
            plot_path = plots_dir / f"{args.experiment}_training_history.png"
            plot_training_history(
                loss_history=training_result.loss_history,
                accuracy_history=training_result.train_accuracy_history,
                title=f"{MODEL_NAME} Training History",
                save_path=str(plot_path),
            )
            logger.info("Training history plot saved: %s", plot_path)
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
            logger.info("Decision boundary plot saved: %s", boundary_path)
        elif model_config["input_size"] == 2:
            logger.warning("plot_decision_boundary not available")


def print_educational_summary(args, logger, dataset_config):
    """Print educational summary about the experiment."""
    logger.info("\nEducational Summary:")
    logger.info("Description: %s", dataset_config["description"])
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


def main():
    """Main entry point for the training script."""
    args = parse_args()
    special = handle_special_args(args)
    if special is not None:
        return special
    valid = validate_experiment_arg(args)
    if valid is not None:
        return valid
    try:
        training_config, model_config, dataset_config, _ = prepare_training(args)
        logger, model, data_split, model_config, training_result = run_training(
            args, training_config, model_config, dataset_config
        )
        generate_visualizations(
            args,
            logger,
            model,
            data_split,
            model_config,
            training_result,
            training_config,
        )
        print_educational_summary(args, logger, dataset_config)
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        # This is a top-level catch to ensure all errors are logged and surfaced
        logger.error("Training failed: %s", e)
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)
