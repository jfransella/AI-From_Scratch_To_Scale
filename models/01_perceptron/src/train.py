#!/usr/bin/env python3
"""
Training script for the 01_perceptron model.

This script trains the classic Rosenblatt perceptron using the unified
engine framework. Supports multiple datasets including both linearly
separable (strengths) and non-separable (limitations) examples.
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup path for shared packages
sys.path.insert(0, str(Path(__file__).parent.parent))
import setup_path  # pylint: disable=unused-import,wrong-import-position

# Handle torch imports gracefully
try:
    import torch

    if hasattr(torch, "__version__") and hasattr(torch, "tensor"):
        _TORCH_AVAILABLE = True
        TorchTensor = torch.Tensor
    else:
        # torch exists but is broken
        _TORCH_AVAILABLE = False
        torch = None
        TorchTensor = Any
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False
    TorchTensor = Any

import torch

from data_utils import load_dataset
from engine.base import DataSplit
from engine.trainer import Trainer

# Import shared packages
from utils import get_logger, set_random_seed, setup_logging

# Import model-specific components
try:
    from .config import (
        get_dataset_config,
        get_model_config,
        get_training_config,
        print_config_summary,
    )
    from .constants import ALL_EXPERIMENTS, MODEL_NAME
    from .model import Perceptron
except ImportError:
    # Fallback for direct imports (e.g., during testing)
    from config import (
        get_dataset_config,
        get_model_config,
        get_training_config,
        print_config_summary,
    )
    from constants import ALL_EXPERIMENTS, MODEL_NAME
    from model import Perceptron

# Optional plotting imports (handled gracefully if not installed)
try:
    from plotting import plot_decision_boundary, plot_training_history
except ImportError as e:
    plot_training_history = None
    plot_decision_boundary = None


def create_data_split(
    x_features: TorchTensor,
    y_target: TorchTensor,
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
    # Set random seed based on torch availability
    if _TORCH_AVAILABLE and torch is not None:
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

        # Create data splits using torch tensors
        x_train = x_features[train_idx]
        y_train = y_target[train_idx]
        x_val = x_features[val_idx] if val_idx is not None else None
        y_val = y_target[val_idx] if val_idx is not None else None
        x_test = x_features[test_idx] if test_idx is not None else None
        y_test = y_target[test_idx] if test_idx is not None else None

    else:
        # Use numpy when torch is not available
        import numpy as np

        np.random.seed(random_state)

        n_samples = len(x_features)
        indices = np.random.permutation(n_samples)

        # Calculate split sizes
        n_test = int(test_split * n_samples)
        n_val = int(validation_split * n_samples)
        n_train = n_samples - n_test - n_val

        # Split indices
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val] if n_val > 0 else None
        test_idx = indices[n_train + n_val :] if n_test > 0 else None

        # Create data splits using numpy arrays
        x_train = x_features[train_idx]
        y_train = y_target[train_idx]
        x_val = x_features[val_idx] if val_idx is not None else None
        y_val = y_target[val_idx] if val_idx is not None else None
        x_test = x_features[test_idx] if test_idx is not None else None
        y_test = y_target[test_idx] if test_idx is not None else None

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

    # Enhanced wandb arguments following integration plan
    parser.add_argument("--wandb-name", type=str, help="Override wandb run name")
    parser.add_argument("--wandb-tags", nargs="+", help="Additional wandb tags")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        help="Wandb mode",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ai-from-scratch-perceptron",
        help="Weights & Biases project name",
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

    # Handle enhanced wandb arguments
    if args.wandb_project:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_name:
        overrides["wandb_name"] = args.wandb_name
    if args.wandb_tags:
        # Extend existing tags with additional ones
        existing_tags = overrides.get("wandb_tags", [])
        overrides["wandb_tags"] = existing_tags + args.wandb_tags
    if args.wandb_mode:
        overrides["wandb_mode"] = args.wandb_mode

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
    import numpy as np

    logger.info("Loading dataset...")
    x_features, y_target = load_dataset(
        dataset_config["dataset_name"], dataset_config["dataset_params"]
    )

    # Handle type conversion when torch is available
    if _TORCH_AVAILABLE and torch is not None:
        if not isinstance(x_features, torch.Tensor):
            x_features = torch.tensor(
                x_features, dtype=torch.float32
            )  # inputs don't need gradients
        if not isinstance(y_target, torch.Tensor):
            y_target = torch.tensor(
                y_target, dtype=torch.float32
            )  # targets don't need gradients
    else:
        # When torch is not available, assume we have numpy arrays
        if not isinstance(x_features, np.ndarray):
            x_features = np.array(x_features, dtype=np.float32)
        if not isinstance(y_target, np.ndarray):
            y_target = np.array(y_target, dtype=np.float32)

    logger.info(
        "Dataset loaded: %s features, %d classes",
        x_features.shape,
        (
            len(np.unique(y_target))
            if hasattr(y_target, "numpy")
            else len(np.unique(y_target))
        ),
    )
    return x_features, y_target


def _create_model_and_trainer(logger, model_config: dict, training_config: dict):
    """Create the model and trainer instances."""
    logger.info("Creating Perceptron model...")

    # Filter model_config to only include Perceptron-specific parameters
    perceptron_params = {
        "input_size": model_config.get("input_size", 2),
        "learning_rate": model_config.get("learning_rate", 0.1),
        "max_epochs": model_config.get("max_epochs", 100),
        "tolerance": model_config.get("tolerance", 1e-6),
        "activation": model_config.get("activation", "step"),
        "init_method": model_config.get("init_method", "zeros"),
        "random_state": model_config.get("random_state", None),
    }

    model = Perceptron(**perceptron_params)
    model_info = model.get_model_info()
    logger.info("Model created: %d parameters", model_info["total_parameters"])
    logger.info(
        "Architecture: %d -> %d", model_info["input_size"], model_info["output_size"]
    )
    logger.info("Activation: %s", model_info["activation_function"])

    logger.info("Initializing trainer...")
    trainer = Trainer(training_config)

    # Initialize wandb through trainer if enabled
    if training_config.use_wandb:
        # Let the model handle wandb instead of trainer to avoid conflicts
        logger.info(
            f"🔄 Wandb will be managed by model - project: {training_config.wandb_project}"
        )
        training_config.use_wandb = False  # Disable trainer wandb

        # Initialize wandb through the model's BaseModel interface
        # Use custom name if provided, otherwise generate timestamp-based name
        if training_config.wandb_name:
            run_name = training_config.wandb_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{training_config.experiment_name}-{training_config.model_name}-{timestamp}"

        wandb_success = model.init_wandb(
            project=training_config.wandb_project,
            name=run_name,
            tags=training_config.wandb_tags,
            config=training_config.__dict__,
            notes=training_config.wandb_notes,
            mode=training_config.wandb_mode,
        )

        if wandb_success:
            logger.info("✅ Wandb integration activated via BaseModel")
            if training_config.wandb_watch_model:
                model.watch_model(
                    log=training_config.wandb_watch_log,
                    log_freq=training_config.wandb_watch_freq,
                )
                logger.info("📊 Model watching enabled")
            else:
                logger.info("📊 Model watching disabled in config")
        else:
            logger.warning("⚠️ Wandb setup failed, continuing without tracking")

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

        # Training history plot
        if plot_training_history is not None:
            plot_path = plots_dir / f"{args.experiment}_training_history.png"
            plot_training_history(
                loss_history=training_result.loss_history,
                accuracy_history=training_result.train_accuracy_history,
                title=f"{MODEL_NAME} Training History",
                save_path=str(plot_path),
            )
            logger.info("Training history plot saved: %s", plot_path)

            # Log to wandb if available
            if (
                hasattr(model, "log_image")
                and hasattr(model, "wandb_run")
                and model.wandb_run is not None
            ):
                model.log_image(
                    str(plot_path), caption=f"{args.experiment} Training History"
                )
                logger.info("Training history plot logged to wandb")
        else:
            logger.warning("plot_training_history not available")

        # Decision boundary plot
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

            # Log to wandb if available
            if (
                hasattr(model, "log_image")
                and hasattr(model, "wandb_run")
                and model.wandb_run is not None
            ):
                model.log_image(
                    str(boundary_path), caption=f"{args.experiment} Decision Boundary"
                )
                logger.info("Decision boundary plot logged to wandb")
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

    # Initialize variables for visualization
    viz_success = False
    viz_data = None

    try:
        training_config, model_config, dataset_config, _ = prepare_training(args)
        logger, model, data_split, model_config, training_result = run_training(
            args, training_config, model_config, dataset_config
        )
        # Store values for visualization
        viz_success = True
        viz_data = (
            args,
            logger,
            model,
            data_split,
            model_config,
            training_result,
            training_config,
        )

    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        # This is a top-level catch to ensure all errors are logged and surfaced
        print(f"ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1

    # Generate visualizations outside try block to ensure they always run
    if viz_success:
        try:
            generate_visualizations(*viz_data)
            print_educational_summary(args, logger, dataset_config)
        except Exception as e:
            print(f"Visualization error: {e}")
            print(f"Traceback: {traceback.format_exc()}")

    return 0


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)
