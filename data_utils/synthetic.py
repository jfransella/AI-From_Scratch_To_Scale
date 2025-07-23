"""
Synthetic dataset generators for AI From Scratch to Scale project.

Provides functions to generate synthetic datasets commonly used
to test and validate neural network architectures.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

from utils import get_logger, set_random_seed
from utils.exceptions import DataError


def generate_xor_dataset(
    n_samples: int = 1000, noise: float = 0.1, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset - classic non-linearly separable problem.

    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add to the data
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is labels

    Example:
        X, y = generate_xor_dataset(1000, noise=0.05)
        print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")
    """
    # Validate parameters
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")

    if random_state is not None:
        set_random_seed(random_state)

    logger = get_logger(__name__)
    logger.debug(f"Generating XOR dataset: {n_samples} samples, noise={noise}")

    try:
        # Generate base XOR patterns
        n_per_class = n_samples // 4

        # XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)

        idx = 0
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                start_idx = idx
                end_idx = min(idx + n_per_class, n_samples)

                # Generate samples around this corner
                X[start_idx:end_idx, 0] = x1 + np.random.normal(
                    0, noise, end_idx - start_idx
                )
                X[start_idx:end_idx, 1] = x2 + np.random.normal(
                    0, noise, end_idx - start_idx
                )

                # XOR logic
                y[start_idx:end_idx] = x1 ^ x2

                idx = end_idx
                if idx >= n_samples:
                    break
            if idx >= n_samples:
                break

        # Shuffle the dataset
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        # Ensure y is 2D for consistency
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        logger.info(f"Generated XOR dataset: {X.shape}, classes: {np.unique(y)}")
        return X, y

    except Exception as e:
        raise DataError(f"Failed to generate XOR dataset: {e}")


def generate_circles_dataset(
    n_samples: int = 1000,
    noise: float = 0.1,
    factor: float = 0.8,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles dataset - another non-linearly separable problem.

    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        factor: Scale factor between inner and outer circle
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is labels

    Example:
        X, y = generate_circles_dataset(1000, factor=0.5)
    """
    if random_state is not None:
        set_random_seed(random_state)

    logger = get_logger(__name__)
    logger.debug(f"Generating circles dataset: {n_samples} samples")

    try:
        # Generate samples for outer circle (class 1)
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out

        # Outer circle
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        outer_circ_x = np.cos(linspace_out)
        outer_circ_y = np.sin(linspace_out)

        # Inner circle
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        inner_circ_x = np.cos(linspace_in) * factor
        inner_circ_y = np.sin(linspace_in) * factor

        # Combine circles
        X = np.vstack(
            [
                np.column_stack([outer_circ_x, outer_circ_y]),
                np.column_stack([inner_circ_x, inner_circ_y]),
            ]
        )

        # Labels: outer circle = 1, inner circle = 0
        y = np.hstack(
            [np.ones(n_samples_out, dtype=int), np.zeros(n_samples_in, dtype=int)]
        )

        # Add noise
        if noise > 0:
            X += np.random.normal(0, noise, X.shape)

        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        # Ensure y is 2D for consistency
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        logger.info(f"Generated circles dataset: {X.shape}, classes: {np.unique(y)}")
        return X, y

    except Exception as e:
        raise DataError(f"Failed to generate circles dataset: {e}")


def generate_linear_dataset(
    n_samples: int = 1000,
    n_features: int = 2,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate linearly separable dataset.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes
        noise: Amount of noise to add
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is labels

    Example:
        X, y = generate_linear_dataset(1000, n_features=3, n_classes=3)
    """
    if random_state is not None:
        set_random_seed(random_state)

    logger = get_logger(__name__)
    logger.debug(
        f"Generating linear dataset: {n_samples} samples, {n_features} features, {n_classes} classes"
    )

    try:
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_redundant=0,
            n_informative=n_features,
            n_clusters_per_class=1,
            n_classes=n_classes,
            flip_y=noise,
            random_state=random_state,
        )

        # Ensure y is 2D for consistency
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        logger.info(f"Generated linear dataset: {X.shape}, classes: {np.unique(y)}")
        return X, y

    except ImportError:
        # Fallback implementation without sklearn
        logger.warning("scikit-learn not available, using simple fallback")
        return _generate_simple_linear_dataset(n_samples, n_features, n_classes, noise)
    except Exception as e:
        raise DataError(f"Failed to generate linear dataset: {e}")


def generate_classification_dataset(
    dataset_type: str, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate classification dataset of specified type.

    Args:
        dataset_type: Type of dataset ('xor', 'circles', 'linear')
        **kwargs: Additional arguments passed to specific generator

    Returns:
        Tuple of (X, y) where X is features and y is labels

    Example:
        X, y = generate_classification_dataset('xor', n_samples=500)
        X, y = generate_classification_dataset('circles', factor=0.3)
    """
    generators = {
        "xor": generate_xor_dataset,
        "circles": generate_circles_dataset,
        "linear": generate_linear_dataset,
    }

    if dataset_type not in generators:
        available = list(generators.keys())
        raise DataError(f"Unknown dataset type: {dataset_type}. Available: {available}")

    logger = get_logger(__name__)
    logger.info(f"Generating {dataset_type} dataset with args: {kwargs}")

    return generators[dataset_type](**kwargs)


def get_dataset_properties(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Get properties of a dataset.

    Args:
        X: Feature matrix
        y: Label vector

    Returns:
        Dictionary with dataset properties

    Example:
        props = get_dataset_properties(X, y)
        print(f"Dataset has {props['n_classes']} classes")
    """
    try:
        properties = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1] if X.ndim > 1 else 1,
            "n_classes": len(np.unique(y)),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
            "feature_ranges": {
                "min": np.min(X, axis=0).tolist(),
                "max": np.max(X, axis=0).tolist(),
                "mean": np.mean(X, axis=0).tolist(),
                "std": np.std(X, axis=0).tolist(),
            },
            "is_balanced": _check_class_balance(y),
        }

        return properties

    except Exception as e:
        raise DataError(f"Failed to compute dataset properties: {e}")


def visualize_2d_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Dataset Visualization",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize 2D dataset with class colors.

    Args:
        X: Feature matrix (must be 2D)
        y: Label vector
        title: Plot title
        save_path: Path to save plot (optional)

    Example:
        X, y = generate_xor_dataset()
        visualize_2d_dataset(X, y, "XOR Dataset")
    """
    if X.shape[1] != 2:
        raise DataError(f"Can only visualize 2D datasets, got {X.shape[1]}D")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))

        # Plot each class with different color
        classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))

        for i, cls in enumerate(classes):
            mask = y == cls
            plt.scatter(
                X[mask, 0], X[mask, 1], c=[colors[i]], label=f"Class {cls}", alpha=0.7
            )

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger = get_logger(__name__)
            logger.info(f"Saved plot to {save_path}")

        plt.show()

    except ImportError:
        logger = get_logger(__name__)
        logger.warning("matplotlib not available, cannot visualize dataset")
    except Exception as e:
        raise DataError(f"Failed to visualize dataset: {e}")


# Private helper functions


def _generate_simple_linear_dataset(
    n_samples: int, n_features: int, n_classes: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple fallback for linear dataset generation without sklearn."""
    logger = get_logger(__name__)
    logger.debug("Using simple linear dataset generation")

    # Generate random data
    X = np.random.randn(n_samples, n_features)

    # Create linear decision boundary
    weights = np.random.randn(n_features)
    scores = X @ weights

    if n_classes == 2:
        y = (scores > 0).astype(int)
    else:
        # Divide score range into n_classes bins
        y = np.digitize(
            scores, bins=np.linspace(scores.min(), scores.max(), n_classes + 1)[1:-1]
        )

    # Add noise
    if noise > 0:
        n_flip = int(noise * n_samples)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        y[flip_indices] = np.random.choice(n_classes, n_flip)

    # Ensure y is 2D for consistency
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y


def _check_class_balance(y: np.ndarray, threshold: float = 0.1) -> bool:
    """Check if classes are balanced within threshold."""
    unique, counts = np.unique(y, return_counts=True)

    if len(unique) < 2:
        return True

    proportions = counts / len(y)
    expected_proportion = 1.0 / len(unique)

    # Check if all proportions are within threshold of expected
    deviations = np.abs(proportions - expected_proportion)
    return np.all(deviations <= threshold)
