#!/usr/bin/env python3
"""
Training script for Multi-Layer Perceptron (MLP) experiments.

This script demonstrates the breakthrough capability of MLPs to solve
non-linearly separable problems, particularly the famous XOR problem
that single-layer perceptrons cannot handle.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path for shared package imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared packages
from utils import setup_logging, setup_device, set_random_seed, get_logger
from data_utils import generate_xor_dataset, generate_circles_dataset
from config import (
    get_experiment_config,
    list_available_experiments,
    get_experiment_info,
    apply_environment_overrides,
)
from model import MLP

# Constants
SEPARATOR_LENGTH = 60
SUB_SEPARATOR_LENGTH = 40

# Optional engine imports (for advanced implementations)
try:
    from engine import Trainer
    from engine.base import DataSplit

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


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
        xor_samples = 4
        xor_noise = 0.0
        xor_random_state = 42
        features, labels = generate_xor_dataset(
            n_samples=xor_samples, noise=xor_noise, random_state=xor_random_state
        )
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        # For XOR, use all data for training (only 4 samples)
        return x, y, x, y

    elif config.dataset_type == "circles":
        # Create circles dataset
        params = config.dataset_params
        default_circles_samples = 1000
        default_circles_noise = 0.1
        num_samples = params.get("num_samples", default_circles_samples)
        noise = params.get("noise", default_circles_noise)

        features, labels = generate_circles_dataset(n_samples=num_samples, noise=noise)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        # Split into train/test
        train_split_ratio = 0.8
        n_train = int(train_split_ratio * len(x))
        indices = torch.randperm(len(x))

        x_train = x[indices[:n_train]]
        y_train = y[indices[:n_train]]
        x_test = x[indices[n_train:]]
        y_test = y[indices[n_train:]]

        return x_train, y_train, x_test, y_test

    elif config.dataset_type in ["moons", "spirals"]:
        # For now, fall back to XOR for unsupported datasets
        LOGGER_NAME = "ai_from_scratch"
        logger = get_logger(LOGGER_NAME)
        logger.warning("%s dataset not yet implemented, using XOR", config.dataset_type)
        features, labels = generate_xor_dataset()
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y, x, y

    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")


def train_with_engine(config, args):
    """
    Train model using the engine framework (advanced pattern).

    Args:
        config: Configuration object
        args: Command line arguments

    Returns:
        Training results
    """
    if not HAS_ENGINE:
        raise ImportError("Engine framework not available")

    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)
    logger.info("Using engine-based training")

    # Set up device
    device = setup_device(args.device)

    # Create model
    model = MLP(
        input_size=config.input_size,
        hidden_layers=config.hidden_layers,
        output_size=config.output_size,
        activation=config.activation,
        weight_init=config.weight_init,
        device=device,
    )

    # Load data
    x_train, y_train, x_test, y_test = create_dataset(config)

    # Create data splits for engine
    data_split = DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,  # Use test as validation for now
        y_val=y_test,
        x_test=x_test,
        y_test=y_test,
    )

    # Create trainer
    trainer = Trainer(config)

    # Train model
    results = trainer.train(model, data_split)

    return results


def train_manually(config, args):
    """
    Train model manually (basic pattern).

    Args:
        config: Configuration object
        args: Command line arguments

    Returns:
        Training results
    """
    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)
    logger.info("Using manual training")

    # Set up device
    device = setup_device(args.device)

    # Create model
    model = MLP(
        input_size=config.input_size,
        hidden_layers=config.hidden_layers,
        output_size=config.output_size,
        activation=config.activation,
        weight_init=config.weight_init,
        device=device,
    )

    # Load data
    x_train, y_train, x_test, y_test = create_dataset(config)

    # Train model
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

    return results


def train_experiment(config, device="cpu"):
    """
    Train a single MLP experiment.

    Args:
        config: Experiment configuration
        device: Device to run on

    Returns:
        Tuple of (model, results)
    """
    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)

    logger.info("\n%s", "="*SEPARATOR_LENGTH)
    logger.info("Starting experiment: %s", config.name)
    logger.info("Description: %s", config.description)
    logger.info("%s", "="*SEPARATOR_LENGTH)

    # Set random seed for reproducibility
    set_random_seed(config.random_seed)

    # Create dataset
    logger.info("Creating %s dataset...", config.dataset_type)
    x_train, y_train, x_test, y_test = create_dataset(config)

    logger.info("Train set: %d samples", x_train.shape[0])
    logger.info("Test set: %d samples", x_test.shape[0])

    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Create model
    logger.info(
        "\nCreating MLP: %s -> %s -> %s",
        config.input_size, config.hidden_layers, config.output_size
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
    logger.info("Learning rate: %s", config.learning_rate)
    logger.info("Max epochs: %s", config.max_epochs)
    logger.info("Convergence threshold: %s", config.convergence_threshold)

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
    logger.info("\n%s", "="*SEPARATOR_LENGTH)
    logger.info("Experiment completed: %s", config.name)
    logger.info("Training time: %.2f seconds", training_time)
    logger.info("Final loss: %.6f", results['final_loss'])
    logger.info("Final train accuracy: %.4f", results['final_train_accuracy'])
    if "final_test_accuracy" in results:
        logger.info("Final test accuracy: %.4f", results['final_test_accuracy'])

    if results["converged"]:
        logger.info("✅ Converged at epoch %s", results['convergence_epoch'])
    else:
        logger.info("⚠️  Did not converge within %s epochs", config.max_epochs)

    # XOR-specific success message
    xor_success_threshold = 0.99
    if (
        config.dataset_type == "xor"
        and results["final_train_accuracy"] >= xor_success_threshold
    ):
        logger.info("🎉 Successfully solved the XOR problem!")
        logger.info("   This demonstrates MLP's ability to learn non-linear patterns")
        logger.info("   that single-layer perceptrons cannot handle.")

    logger.info("%s", "="*SEPARATOR_LENGTH)

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

    # Save results
    results_path = output_dir / f"{config.name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)
    logger.info("Results saved to %s", output_dir)


def run_educational_sequence():
    """
    Run a sequence of educational experiments.

    This demonstrates the progression from simple to complex problems.
    """
    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)
    logger.info("Running educational experiment sequence...")

    experiments = [
        "debug",  # 1. Verify everything works
        "quick_test",  # 2. Fast XOR solution
        "xor_breakthrough",  # 3. Proper XOR solution
        "circles_challenge",  # 4. More complex non-linear problem
    ]

    results = {}
    for exp_name in experiments:
        try:
            logger.info("\n%s", "="*SUB_SEPARATOR_LENGTH)
            logger.info("Running experiment: %s", exp_name)
            logger.info("%s", "="*SUB_SEPARATOR_LENGTH)

            config = get_experiment_config(exp_name)
            _model, result = train_experiment(config)
            results[exp_name] = result

        except Exception as e:
            logger.error("Failed to run experiment %s: %s", exp_name, e)
            results[exp_name] = {"error": str(e)}

    # Print summary
    logger.info("\n%s", "="*SEPARATOR_LENGTH)
    logger.info("EDUCATIONAL SEQUENCE COMPLETED")
    logger.info("%s", "="*SEPARATOR_LENGTH)

    for exp_name, result in results.items():
        if "error" in result:
            logger.info("%s: ❌ %s", exp_name, result['error'])
        else:
            status = "✅" if result.get("converged", False) else "⚠️"
            accuracy = result.get("final_train_accuracy", 0)
            logger.info("%s: %s %.4f", exp_name, status, accuracy)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        help="Experiment name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training",
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
        "--educational-sequence",
        action="store_true",
        help="Run educational experiment sequence",
    )
    parser.add_argument(
        "--use-engine",
        action="store_true",
        help="Use engine framework for training (if available)",
    )
    parser.add_argument(
        "--environment",
        choices=["default", "debug", "production"],
        default="default",
        help="Environment configuration",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def _handle_list_experiments():
    """Handle the --list-experiments command."""
    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)

    experiments = list_available_experiments()
    logger.info("\nAvailable MLP experiments:")
    logger.info("-" * SUB_SEPARATOR_LENGTH)
    for exp in experiments:
        try:
            info = get_experiment_info(exp)
            logger.info("%-25s - %s", exp, info['description'])
        except Exception as e:
            logger.info("%-25s - Error: %s", exp, e)


def _handle_experiment_info(experiment_name):
    """Handle the --experiment-info command."""
    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)

    try:
        info = get_experiment_info(experiment_name)
        logger.info("\nExperiment: %s", experiment_name)
        logger.info("Description: %s", info['description'])
        logger.info("Model: %s", info['model_name'])
        logger.info("Dataset: %s", info['dataset'])
        logger.info("Difficulty: %s", info['difficulty'])
        logger.info("Expected accuracy: %s", info['expected_accuracy'])
    except Exception as e:
        logger.error("Error getting experiment info: %s", e)


def _run_training_experiment(args):
    """Run the main training experiment."""
    logger_name = "ai_from_scratch"
    logger = get_logger(logger_name)

    # Load configuration
    config = get_experiment_config(args.experiment)

    # Apply environment overrides
    config = apply_environment_overrides(config, args.environment)

    logger.info("Starting training for experiment: %s", args.experiment)
    logger.info("Configuration: %s", config.name)

    # Choose training method
    if args.use_engine and HAS_ENGINE:
        train_with_engine(config, args)
    else:
        train_manually(config, args)

    logger.info("Training completed successfully!")


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging()

    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # Handle special commands
    if args.list_experiments:
        _handle_list_experiments()
        return

    if args.experiment_info:
        _handle_experiment_info(args.experiment_info)
        return

    if args.educational_sequence:
        run_educational_sequence()
        return

    # Validate experiment
    if not args.experiment:
        logger_name = "ai_from_scratch"
        logger = get_logger(logger_name)
        logger.error("Error: --experiment is required")
        logger.error("Use --list-experiments to see available experiments")
        return

    try:
        _run_training_experiment(args)
    except Exception as e:
        logger_name = "ai_from_scratch"
        logger = get_logger(logger_name)
        logger.error("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
