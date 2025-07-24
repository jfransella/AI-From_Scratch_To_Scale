"""
Unified evaluation engine for AI From Scratch to Scale project.

Provides comprehensive evaluation capabilities including metrics computation,
result analysis, and performance benchmarking across different model types.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Handle torch imports gracefully
try:
    import torch

    if (
        hasattr(torch, "__version__")
        and hasattr(torch, "nn")
        and hasattr(torch, "tensor")
    ):
        import torch.nn as nn

        _TORCH_AVAILABLE = True
        TorchTensor = torch.Tensor
    else:
        # torch exists but is broken
        _TORCH_AVAILABLE = False
        torch = None
        nn = None
        TorchTensor = Any
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False
    TorchTensor = Any

import numpy as np

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from utils import get_logger
from .base import BaseModel, EvaluationResult, ModelAdapter


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Metrics to compute
    compute_accuracy: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    compute_f1: bool = True
    compute_confusion_matrix: bool = True

    # Per-class metrics
    compute_per_class: bool = True

    # Prediction storage
    store_predictions: bool = False
    store_probabilities: bool = False
    store_ground_truth: bool = False

    # Output configuration
    verbose: bool = True
    save_results: bool = False
    output_path: Optional[str] = None

    # Device
    device: str = "auto"


@dataclass
class ModelMetrics:
    """Container for comprehensive model metrics."""

    # Core metrics
    accuracy: float = 0.0
    loss: float = float("inf")

    # Classification metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Per-class metrics
    per_class_accuracy: Optional[Dict[str, float]] = None
    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None
    per_class_f1: Optional[Dict[str, float]] = None

    # Additional information
    num_samples: int = 0
    num_classes: int = 0
    class_names: Optional[List[str]] = None

    def summary(self) -> str:
        """Generate a human-readable summary of metrics."""
        lines = [
            "Model Evaluation Summary",
            "=" * 40,
            f"Samples: {self.num_samples}",
            f"Accuracy: {self.accuracy:.4f}",
            f"Loss: {self.loss:.6f}",
        ]

        if self.precision is not None:
            lines.append(f"Precision: {self.precision:.4f}")
        if self.recall is not None:
            lines.append(f"Recall: {self.recall:.4f}")
        if self.f1_score is not None:
            lines.append(f"F1-Score: {self.f1_score:.4f}")

        if self.per_class_accuracy and len(self.per_class_accuracy) > 1:
            lines.append("\nPer-Class Metrics:")
            for class_name, acc in self.per_class_accuracy.items():
                lines.append(f"  {class_name}: {acc:.4f}")

        return "\n".join(lines)


class Evaluator:
    """
    Comprehensive model evaluation engine.

    Provides detailed evaluation capabilities for any model implementing
    the BaseModel interface, including metrics computation, confusion
    matrices, and performance analysis.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator with configuration.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.logger.info(f"Evaluator initialized - Device: {self.device}")

    def evaluate(
        self,
        model: BaseModel,
        x: TorchTensor,
        y: TorchTensor,
        dataset_name: str = "unknown",
    ) -> EvaluationResult:
        """
        Evaluate a model on given data.

        Args:
            model: Model to evaluate
            x: Input data
            y: Ground truth labels
            dataset_name: Name of the dataset for metadata

        Returns:
            EvaluationResult with comprehensive metrics
        """
        self.logger.info(f"Starting evaluation on {len(x)} samples")
        start_time = time.time()

        # Move data to device
        x = x.to(self.device)
        y = y.to(self.device)

        # Move model to device
        if not isinstance(model, BaseModel):
            model = ModelAdapter(model, "unknown")
        model = model.to(str(self.device))

        # Set model to evaluation mode
        model.eval()

        with torch.no_grad():
            # Get predictions and probabilities
            predictions = model.predict(x)

            # Compute loss
            try:
                outputs = model.forward(x)
                loss = model.get_loss(outputs, y).item()
            except Exception as e:
                self.logger.warning(f"Could not compute loss: {e}")
                loss = float("inf")

            # Get probabilities if available
            probabilities = None
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(x)
                except Exception as e:
                    self.logger.warning(f"Could not get probabilities: {e}")

        # Convert to numpy for metrics computation
        y_true = y.cpu().numpy()
        y_pred = predictions.cpu().numpy()

        # Handle different prediction shapes - ensure both arrays have same dimensions
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze()
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.squeeze()

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred)

        # Create evaluation result
        result = EvaluationResult(
            accuracy=metrics.accuracy,
            loss=loss,
            precision=metrics.precision,
            recall=metrics.recall,
            f1_score=metrics.f1_score,
            confusion_matrix=(
                metrics.confusion_matrix.tolist()
                if metrics.confusion_matrix is not None
                else None
            ),
            per_class_accuracy=metrics.per_class_accuracy,
            per_class_precision=metrics.per_class_precision,
            per_class_recall=metrics.per_class_recall,
            num_samples=len(x),
            evaluation_time=time.time() - start_time,
            model_name=getattr(model, "model_type", "unknown"),
            dataset_name=dataset_name,
        )

        # Store predictions/probabilities if requested
        if self.config.store_predictions:
            result.predictions = y_pred.tolist()
        if self.config.store_probabilities and probabilities is not None:
            result.probabilities = probabilities.cpu().numpy().tolist()
        if self.config.store_ground_truth:
            result.ground_truth = y_true.tolist()

        # Verbose output
        if self.config.verbose:
            print("\nEvaluation Results:")
            print(f"{'=' * 50}")
            print(f"Dataset: {dataset_name}")
            print(f"Samples: {result.num_samples}")
            print(f"Accuracy: {result.accuracy:.4f}")
            print(f"Loss: {result.loss:.6f}")
            if result.precision:
                print(f"Precision: {result.precision:.4f}")
            if result.recall:
                print(f"Recall: {result.recall:.4f}")
            if result.f1_score:
                print(f"F1-Score: {result.f1_score:.4f}")
            print(f"Evaluation time: {result.evaluation_time:.2f}s")

            if metrics.confusion_matrix is not None:
                print("\nConfusion Matrix:")
                print(metrics.confusion_matrix)

        # Save results if requested
        if self.config.save_results and self.config.output_path:
            self._save_results(result, metrics)

        self.logger.info(f"Evaluation completed - Accuracy: {result.accuracy:.4f}")
        return result

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Compute comprehensive metrics."""
        metrics = ModelMetrics()

        # Basic info
        metrics.num_samples = len(y_true)
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        metrics.num_classes = len(unique_labels)

        # Core accuracy
        if self.config.compute_accuracy:
            metrics.accuracy = accuracy_score(y_true, y_pred)

        # Classification metrics
        try:
            if self.config.compute_precision:
                if metrics.num_classes > 2:
                    metrics.precision = precision_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                else:
                    metrics.precision = precision_score(y_true, y_pred, zero_division=0)

            if self.config.compute_recall:
                if metrics.num_classes > 2:
                    metrics.recall = recall_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                else:
                    metrics.recall = recall_score(y_true, y_pred, zero_division=0)

            if self.config.compute_f1:
                if metrics.num_classes > 2:
                    metrics.f1_score = f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                else:
                    metrics.f1_score = f1_score(y_true, y_pred, zero_division=0)

        except Exception as e:
            self.logger.warning(f"Error computing classification metrics: {e}")

        # Confusion matrix
        if self.config.compute_confusion_matrix:
            try:
                metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
            except Exception as e:
                self.logger.warning(f"Error computing confusion matrix: {e}")

        # Per-class metrics
        if self.config.compute_per_class and metrics.num_classes > 1:
            try:
                # Precision per class
                precisions = precision_score(
                    y_true, y_pred, average=None, zero_division=0
                )
                metrics.per_class_precision = {
                    f"class_{i}": p for i, p in enumerate(precisions)
                }

                # Recall per class
                recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
                metrics.per_class_recall = {
                    f"class_{i}": r for i, r in enumerate(recalls)
                }

                # F1 per class
                f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
                metrics.per_class_f1 = {f"class_{i}": f for i, f in enumerate(f1s)}

                # Accuracy per class (from confusion matrix)
                if metrics.confusion_matrix is not None:
                    per_class_acc = {}
                    for i in range(len(unique_labels)):
                        if i < metrics.confusion_matrix.shape[0]:
                            class_correct = metrics.confusion_matrix[i, i]
                            class_total = metrics.confusion_matrix[i, :].sum()
                            per_class_acc[f"class_{i}"] = (
                                class_correct / class_total if class_total > 0 else 0.0
                            )
                    metrics.per_class_accuracy = per_class_acc

            except Exception as e:
                self.logger.warning(f"Error computing per-class metrics: {e}")

        return metrics

    def _save_results(self, result: EvaluationResult, metrics: ModelMetrics):
        """Save evaluation results to file."""
        try:
            import json
            from pathlib import Path

            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "evaluation_result": result.to_dict(),
                        "metrics_summary": metrics.summary(),
                        "confusion_matrix": (
                            metrics.confusion_matrix.tolist()
                            if metrics.confusion_matrix is not None
                            else None
                        ),
                    },
                    f,
                    indent=2,
                )

            self.logger.info(f"Evaluation results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def compare_models(
        self,
        models: List[BaseModel],
        x: TorchTensor,
        y: TorchTensor,
        model_names: Optional[List[str]] = None,
        dataset_name: str = "unknown",
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple models on the same dataset.

        Args:
            models: List of models to compare
            x: Input data
            y: Ground truth labels
            model_names: Optional names for models
            dataset_name: Dataset name for metadata

        Returns:
            Dictionary mapping model names to evaluation results
        """
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(models))]

        if len(model_names) != len(models):
            raise ValueError("Number of model names must match number of models")

        results = {}

        self.logger.info(f"Comparing {len(models)} models on {dataset_name}")

        for model, name in zip(models, model_names):
            self.logger.info(f"Evaluating {name}")
            result = self.evaluate(model, x, y, f"{dataset_name}_{name}")
            results[name] = result

        # Print comparison if verbose
        if self.config.verbose:
            print("\nModel Comparison Results:")
            print(f"{'=' * 60}")
            print(f"{'Model':<15} {'Accuracy':<10} {'Loss':<10} {'F1-Score':<10}")
            print(f"{'-' * 60}")

            for name, result in results.items():
                f1_str = f"{result.f1_score:.4f}" if result.f1_score else "N/A"
                print(
                    f"{name:<15} {result.accuracy:<10.4f}"
                    f"{result.loss:<10.6f} {f1_str:<10}"
                )

        return results
