"""
Evaluation script for Perceptron model using unified evaluation infrastructure.

This script provides comprehensive evaluation of trained Perceptron models
using the shared evaluation engine, including metrics computation,
visualization generation, and detailed analysis.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

# Setup path for shared packages
sys.path.insert(0, str(Path(__file__).parent.parent))
import setup_path  # pylint: disable=unused-import,wrong-import-position

# Import model-specific components
# Always use absolute imports for better compatibility
from config import get_dataset_config, get_evaluation_config, get_model_config
from constants import ALL_EXPERIMENTS, MODEL_NAME
from model import Perceptron

from data_utils import load_dataset
from engine.evaluator import Evaluator

# Import shared packages
from utils import get_logger, setup_logging

# Optional plotting imports (handled gracefully if not installed)
try:
    from plotting import plot_confusion_matrix, plot_decision_boundary
except ImportError:
    plot_confusion_matrix = None
    plot_decision_boundary = None


def load_model_from_checkpoint(checkpoint_path: str, model_config: dict) -> Perceptron:
    """
    Load a Perceptron model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        model_config: Model configuration dictionary

    Returns:
        Loaded Perceptron model
    """
    logger = get_logger(__name__)

    try:
        # Try loading as a saved model first
        model = Perceptron.load_model(checkpoint_path)
        logger.info("Loaded saved model from %s", checkpoint_path)
        return model

    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to load saved model: %s", e)

        try:
            # Try loading as state dict
            model = Perceptron(**model_config)
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)

            logger.info("Loaded state dict from %s", checkpoint_path)
            return model

        except Exception as e2:  # pylint: disable=broad-except
            logger.error("Failed to load model: %s", e2)
            raise ValueError(f"Could not load model from {checkpoint_path}") from e2


def prepare_evaluation_data(dataset_config: dict, split: str = "test") -> tuple:
    """
    Prepare evaluation data.

    Args:
        dataset_config: Dataset configuration
        split: Data split to use ("test", "full", or "train")

    Returns:
        Tuple of (x_features, y_target) tensors
    """
    logger = get_logger(__name__)

    # Load dataset
    x_features, y_target = load_dataset(
        dataset_config["dataset_name"], dataset_config["dataset_params"]
    )

    # Convert to tensors if needed
    if not isinstance(x_features, torch.Tensor):
        x_features = torch.tensor(x_features, dtype=torch.float32)
    if not isinstance(y_target, torch.Tensor):
        y_target = torch.tensor(y_target, dtype=torch.float32)

    if split == "full":
        logger.info("Using full dataset: %d samples", len(x_features))
        return x_features, y_target
    if split == "train":
        # Use first 80% for train split evaluation
        n_train = int(0.8 * len(x_features))
        x_eval, y_eval = x_features[:n_train], y_target[:n_train]
        logger.info("Using train split: %d samples", len(x_eval))
        return x_eval, y_eval
    # test split
    # Use last 20% for test split evaluation
    n_train = int(0.8 * len(x_features))
    x_eval, y_eval = x_features[n_train:], y_target[n_train:]
    logger.info("Using test split: %d samples", len(x_eval))
    return x_eval, y_eval


def print_dataset_info(dataset_config: dict, experiment_name: str, num_samples: int):
    """Print dataset information section."""
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY: {experiment_name}")
    print(f"{'='*70}")

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {dataset_config['dataset_name']}")
    print(f"Description: {dataset_config['description']}")
    print(f"Difficulty: {dataset_config['difficulty']}")
    print(f"Expected Accuracy: {dataset_config['expected_accuracy']:.3f}")
    print(f"Samples Evaluated: {num_samples}")


def print_metrics_section(results):
    """Print performance metrics section."""
    print(f"\n{'-'*70}")
    print("PERFORMANCE METRICS")
    print(f"{'-'*70}")

    # Core metrics
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"Loss: {results.loss:.6f}")

    if results.precision is not None:
        print(f"Precision: {results.precision:.4f}")
    if results.recall is not None:
        print(f"Recall: {results.recall:.4f}")
    if results.f1_score is not None:
        print(f"F1-Score: {results.f1_score:.4f}")


def print_performance_analysis(results, dataset_config: dict):
    """Print performance analysis section."""
    expected_acc = dataset_config["expected_accuracy"]
    actual_acc = results.accuracy

    print(f"\n{'-'*70}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'-'*70}")

    if actual_acc >= expected_acc * 0.95:
        performance = "✓ EXCELLENT - Exceeds expectations"
    elif actual_acc >= expected_acc * 0.9:
        performance = "✓ GOOD - Meets expectations"
    elif actual_acc >= expected_acc * 0.8:
        performance = "~ ACCEPTABLE - Slightly below expectations"
    elif actual_acc >= expected_acc * 0.6:
        performance = "⚠ POOR - Well below expectations"
    else:
        performance = "✗ FAILED - Very poor performance"

    print(f"Performance: {performance}")
    print(f"Accuracy Gap: {actual_acc - expected_acc:+.3f}")


def print_educational_insights(experiment_name: str, results):
    """Print educational insights section."""
    actual_acc = results.accuracy

    print(f"\n{'-'*70}")
    print("EDUCATIONAL INSIGHTS")
    print(f"{'-'*70}")

    if experiment_name in [
        "iris_binary",
        "linear_separable",
        "debug_small",
        "debug_linear",
    ]:
        if actual_acc >= 0.9:
            print(
                "✓ This result demonstrates Perceptron's strength on "
                "linearly separable data"
            )
            print("✓ The high accuracy confirms the data is linearly separable")
        else:
            print("⚠ Lower than expected accuracy on linearly separable data")
            print("💡 Check training convergence and data quality")

    elif experiment_name in ["xor_problem", "circles_dataset", "mnist_subset"]:
        if actual_acc <= 0.6:
            print("✓ This result demonstrates Perceptron's fundamental limitations")
            print("✓ Poor performance confirms the data is NOT linearly separable")
            print("💡 This motivates the need for multi-layer perceptrons (MLPs)")
        else:
            print("🤔 Unexpectedly good performance on non-linearly separable data")
            print("💡 The dataset might be more separable than expected")


def print_confusion_matrix(results):
    """Print confusion matrix section if available."""
    if results.confusion_matrix is not None:
        print(f"\n{'-'*70}")
        print("CONFUSION MATRIX")
        print(f"{'-'*70}")
        cm = results.confusion_matrix
        if len(cm) == 2:  # Binary classification
            print(f"True Negatives: {cm[0][0]:4d}  | False Positives: {cm[0][1]:4d}")
            print(f"False Negatives: {cm[1][0]:4d} | True Positives: {cm[1][1]:4d}")
        else:
            for i, row in enumerate(cm):
                print(f"Class {i}: {row}")


def print_evaluation_summary(results, dataset_config: dict, experiment_name: str):
    """
    Print a comprehensive evaluation summary.

    Args:
        results: EvaluationResult object
        dataset_config: Dataset configuration
        experiment_name: Name of the experiment
    """
    print_dataset_info(dataset_config, experiment_name, results.num_samples)
    print_metrics_section(results)
    print_performance_analysis(results, dataset_config)
    print_educational_insights(experiment_name, results)
    print_confusion_matrix(results)
    print(f"{'='*70}\n")


def save_evaluation_results(results, output_path: str, experiment_name: str):
    """
    Save evaluation results to file.

    Args:
        results: EvaluationResult object
        output_path: Path to save results
        experiment_name: Name of the experiment
    """
    # Convert results to dictionary
    results_dict = results.to_dict()
    results_dict["experiment_name"] = experiment_name
    results_dict["model_name"] = MODEL_NAME

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    logger = get_logger(__name__)
    logger.info("Evaluation results saved to %s", output_file)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Perceptron model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=f"Experiment name. Available: {ALL_EXPERIMENTS}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "full"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate evaluation visualizations"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save model predictions to file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if args.experiment not in ALL_EXPERIMENTS:
        print("Error: Unknown experiment '%s'", args.experiment)
        print("Available experiments: %s", ALL_EXPERIMENTS)
        return False

    if not Path(args.checkpoint).exists():
        print("Error: Checkpoint file not found: %s", args.checkpoint)
        return False

    return True


def generate_visualizations(args, results, model, x_eval, y_eval, model_config):
    """Generate evaluation visualizations."""
    logger = get_logger(__name__)
    logger.info("Generating evaluation visualizations...")

    try:
        plots_dir = Path("outputs/visualizations")
        plots_dir.mkdir(exist_ok=True)

        # Plot confusion matrix if available
        if results.confusion_matrix is not None and plot_confusion_matrix is not None:
            cm_path = plots_dir / f"{args.experiment}_{args.split}_confusion_matrix.png"
            plot_confusion_matrix(results.confusion_matrix, str(cm_path))
            logger.info("Confusion matrix plot saved: %s", cm_path)

        # Plot decision boundary for 2D data
        if model_config["input_size"] == 2 and plot_decision_boundary is not None:
            boundary_path = (
                plots_dir / f"{args.experiment}_{args.split}_decision_boundary.png"
            )
            plot_decision_boundary(model, x_eval, y_eval, str(boundary_path))
            logger.info("Decision boundary plot saved: %s", boundary_path)

        # Plot prediction distribution
        if hasattr(results, "predictions") and results.predictions is not None:
            pred_path = plots_dir / f"{args.experiment}_{args.split}_predictions.png"
            # Additional plotting logic here
            logger.info("Prediction plots would be saved to: %s", pred_path)

    except ImportError:
        logger.warning("Plotting functions not available")
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error generating visualizations: %s", e)


def save_predictions(args, results):
    """Save model predictions to file."""
    if not (args.save_predictions and hasattr(results, "predictions")):
        return

    pred_file = f"outputs/predictions/{args.experiment}_{args.split}_predictions.json"
    pred_data = {
        "experiment": args.experiment,
        "split": args.split,
        "predictions": results.predictions,
        "ground_truth": results.ground_truth,
        "probabilities": (
            results.probabilities if hasattr(results, "probabilities") else None
        ),
    }

    pred_path = Path(pred_file)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, indent=2)

    logger = get_logger(__name__)
    logger.info("Predictions saved to %s", pred_path)


def main():
    """Main evaluation function."""
    args = parse_arguments()

    if not validate_arguments(args):
        return 1

    try:
        # Setup logging
        setup_logging(level="DEBUG" if args.debug else "INFO")
        logger = get_logger(__name__)

        logger.info("Starting %s evaluation", MODEL_NAME)
        logger.info("Experiment: %s", args.experiment)
        logger.info("Checkpoint: %s", args.checkpoint)
        logger.info("Data split: %s", args.split)

        # Load configurations
        evaluation_config = get_evaluation_config(
            args.experiment,
            verbose=args.verbose,
            store_predictions=args.save_predictions,
        )
        model_config = get_model_config(args.experiment)
        dataset_config = get_dataset_config(args.experiment)

        logger.info("Dataset: %s", dataset_config["dataset_name"])
        logger.info("Expected accuracy: %.3f", dataset_config["expected_accuracy"])

        # Load model
        logger.info("Loading model from checkpoint...")
        model = load_model_from_checkpoint(args.checkpoint, model_config)

        model_info = model.get_model_info()
        logger.info("Model loaded: %d parameters", model_info["total_parameters"])
        logger.info(
            "Architecture: %d -> %d",
            model_info["input_size"],
            model_info["output_size"],
        )

        # Prepare evaluation data
        logger.info("Loading evaluation data...")
        x_eval, y_eval = prepare_evaluation_data(dataset_config, args.split)

        # Create evaluator
        logger.info("Initializing evaluator...")
        evaluator = Evaluator(evaluation_config)

        # Run evaluation
        logger.info("Running evaluation...")
        print(f"\nEvaluating {MODEL_NAME} on {args.experiment}")
        print(f"Dataset: {dataset_config['dataset_name']} ({args.split} split)")
        print(f"Samples: {len(x_eval)}")
        print(f"Checkpoint: {Path(args.checkpoint).name}")
        print("-" * 50)

        results = evaluator.evaluate(
            model,
            x_eval,
            y_eval,
            dataset_name=f"{dataset_config['dataset_name']}_{args.split}",
        )

        # Print evaluation summary
        print_evaluation_summary(results, dataset_config, args.experiment)

        # Save results if output path specified
        if args.output:
            save_evaluation_results(results, args.output, args.experiment)

        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(args, results, model, x_eval, y_eval, model_config)

        # Save predictions if requested
        save_predictions(args, results)

        return 0

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Evaluation failed: %s", e)
        if args.debug:
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)
