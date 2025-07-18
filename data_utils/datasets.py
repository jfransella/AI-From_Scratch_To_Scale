"""
Dataset loading functionality for AI From Scratch to Scale project.

Provides unified dataset loading for both synthetic and real datasets
used across different model implementations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from utils import get_logger, set_random_seed
from utils.exceptions import DataError
from .synthetic import (
    generate_xor_dataset,
    generate_circles_dataset,
    generate_linear_dataset
)


def load_dataset(dataset_name: str, dataset_params: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset by name with unified interface.
    
    Args:
        dataset_name: Name of dataset to load
        dataset_params: Dictionary of dataset-specific parameters (optional)
        **kwargs: Additional arguments passed to dataset loader
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
        
    Supported datasets:
        - Synthetic: xor_problem, circles_dataset, linear_separable, debug_small, debug_linear
        - Real: iris_binary, breast_cancer_binary, mnist_subset
        
    Example:
        X, y = load_dataset('iris_binary')
        X, y = load_dataset('xor_problem', {'n_samples': 1000, 'noise': 0.1})
        X, y = load_dataset('xor_problem', n_samples=1000, noise=0.1)
    """
    logger = get_logger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Merge dataset_params into kwargs
    if dataset_params is not None:
        kwargs.update(dataset_params)
    
    # Synthetic datasets
    if dataset_name == 'xor_problem' or dataset_name == 'xor':
        return generate_xor_dataset(
            n_samples=kwargs.get('n_samples', 1000),
            noise=kwargs.get('noise', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
    
    elif dataset_name == 'circles_dataset' or dataset_name == 'circles':
        return generate_circles_dataset(
            n_samples=kwargs.get('n_samples', 1000),
            noise=kwargs.get('noise', 0.1),
            factor=kwargs.get('factor', 0.8),
            random_state=kwargs.get('random_state', 42)
        )
    
    elif dataset_name == 'linear_separable':
        return generate_linear_dataset(
            n_samples=kwargs.get('n_samples', 1000),
            n_features=kwargs.get('n_features', 2),
            n_classes=kwargs.get('n_classes', 2),
            noise=kwargs.get('noise', 0.05),
            random_state=kwargs.get('random_state', 42)
        )
    
    elif dataset_name == 'debug_small':
        return generate_linear_dataset(
            n_samples=100,
            n_features=2,
            n_classes=2,
            noise=0.01,
            random_state=42
        )
    
    elif dataset_name == 'debug_linear':
        return generate_linear_dataset(
            n_samples=200,
            n_features=2,
            n_classes=2,
            noise=0.05,
            random_state=42
        )
    
    elif dataset_name == 'simple_linear':
        return generate_linear_dataset(
            n_samples=kwargs.get('n_samples', 200),
            n_features=kwargs.get('n_features', 2),
            n_classes=kwargs.get('n_classes', 2),
            noise=kwargs.get('noise', 0.05),
            random_state=kwargs.get('random_state', 42)
        )
    
    elif dataset_name == 'noisy_linear':
        return generate_linear_dataset(
            n_samples=kwargs.get('n_samples', 500),
            n_features=kwargs.get('n_features', 2),
            n_classes=kwargs.get('n_classes', 2),
            noise=kwargs.get('noise', 0.15),
            random_state=kwargs.get('random_state', 42)
        )
    
    # Real datasets
    elif dataset_name == 'iris_binary':
        # Only pass supported parameters
        iris_kwargs = {}
        if 'random_state' in kwargs:
            iris_kwargs['random_state'] = kwargs['random_state']
        return _load_iris_binary(**iris_kwargs)
    
    elif dataset_name == 'iris_setosa_versicolor':
        # Only pass supported parameters
        iris_kwargs = {}
        if 'random_state' in kwargs:
            iris_kwargs['random_state'] = kwargs['random_state']
        return _load_iris_setosa_versicolor(**iris_kwargs)
    
    elif dataset_name == 'iris_versicolor_virginica':
        # Only pass supported parameters
        iris_kwargs = {}
        if 'random_state' in kwargs:
            iris_kwargs['random_state'] = kwargs['random_state']
        return _load_iris_versicolor_virginica(**iris_kwargs)
    
    elif dataset_name == 'breast_cancer_binary' or dataset_name == 'breast_cancer':
        # Only pass supported parameters  
        cancer_kwargs = {}
        if 'random_state' in kwargs:
            cancer_kwargs['random_state'] = kwargs['random_state']
        return _load_breast_cancer_binary(**cancer_kwargs)
    
    elif dataset_name == 'mnist_subset' or dataset_name == 'mnist_binary':
        # Pass relevant parameters for MNIST
        mnist_kwargs = {}
        if 'digits' in kwargs:
            mnist_kwargs['digits'] = kwargs['digits']
        if 'max_samples_per_class' in kwargs:
            mnist_kwargs['max_samples_per_class'] = kwargs['max_samples_per_class']
        if 'random_state' in kwargs:
            mnist_kwargs['random_state'] = kwargs['random_state']
        return _load_mnist_subset(**mnist_kwargs)
    
    else:
        available = get_available_datasets()
        raise DataError(f"Unknown dataset: {dataset_name}. Available: {available}")


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dictionary with dataset information
        
    Example:
        info = get_dataset_info('iris_binary')
        print(f"Classes: {info['n_classes']}, Features: {info['n_features']}")
    """
    dataset_info = {
        'xor_problem': {
            'type': 'synthetic',
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': False,
            'description': 'XOR problem - classic non-linearly separable dataset'
        },
        'circles_dataset': {
            'type': 'synthetic', 
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': False,
            'description': 'Concentric circles - non-linearly separable dataset'
        },
        'linear_separable': {
            'type': 'synthetic',
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': True,
            'description': 'Linearly separable synthetic dataset'
        },
        'debug_small': {
            'type': 'synthetic',
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': True,
            'description': 'Small linearly separable dataset for debugging'
        },
        'debug_linear': {
            'type': 'synthetic',
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': True,
            'description': 'Medium linearly separable dataset for debugging'
        },
        'simple_linear': {
            'type': 'synthetic',
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': True,
            'description': 'Simple linearly separable dataset for ADALINE demonstrations'
        },
        'noisy_linear': {
            'type': 'synthetic',
            'n_classes': 2,
            'n_features': 2,
            'linearly_separable': True,
            'description': 'Noisy linearly separable dataset for convergence studies'
        },
        'iris_binary': {
            'type': 'real',
            'n_classes': 2,
            'n_features': 4,
            'linearly_separable': True,
            'description': 'Iris dataset binary classification (setosa vs non-setosa)'
        },
        'iris_setosa_versicolor': {
            'type': 'real',
            'n_classes': 2,
            'n_features': 4,
            'linearly_separable': True,
            'description': 'Iris dataset (setosa vs versicolor) - linearly separable'
        },
        'iris_versicolor_virginica': {
            'type': 'real',
            'n_classes': 2,
            'n_features': 4,
            'linearly_separable': False,
            'description': 'Iris dataset (versicolor vs virginica) - not linearly separable'
        },
        'breast_cancer_binary': {
            'type': 'real',
            'n_classes': 2,
            'n_features': 30,
            'linearly_separable': True,
            'description': 'Breast cancer Wisconsin dataset'
        },
        'mnist_subset': {
            'type': 'real',
            'n_classes': 2,
            'n_features': 784,
            'linearly_separable': False,
            'description': 'MNIST subset (0 vs 1) - flattened images'
        }
    }
    
    if dataset_name not in dataset_info:
        available = list(dataset_info.keys())
        raise DataError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return dataset_info[dataset_name]


def list_available_datasets() -> List[str]:
    """
    List all available datasets.
    
    Returns:
        List of dataset names
        
    Example:
        datasets = list_available_datasets()
        print(f"Available datasets: {datasets}")
    """
    return [
        'xor_problem', 'circles_dataset', 'linear_separable',
        'debug_small', 'debug_linear', 'simple_linear', 'noisy_linear',
        'iris_binary', 'iris_setosa_versicolor', 'iris_versicolor_virginica',
        'breast_cancer_binary', 'mnist_subset'
    ]


def get_available_datasets() -> List[str]:
    """Alias for list_available_datasets()."""
    return list_available_datasets()


def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate dataset format and basic properties.
    
    Args:
        X: Feature matrix
        y: Label vector
        
    Raises:
        DataError: If dataset is invalid
        
    Example:
        validate_dataset(X, y)  # Raises error if invalid
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise DataError("X and y must be numpy arrays")
    
    if X.ndim != 2:
        raise DataError(f"X must be 2D array, got {X.ndim}D")
    
    if y.ndim != 1:
        raise DataError(f"y must be 1D array, got {y.ndim}D")
    
    if X.shape[0] != y.shape[0]:
        raise DataError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    if X.shape[0] == 0:
        raise DataError("Dataset cannot be empty")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise DataError("X contains NaN or infinite values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise DataError("y contains NaN or infinite values")
    
    logger = get_logger(__name__)
    logger.debug(f"Dataset validation passed: {X.shape}, {len(np.unique(y))} classes")


# Private helper functions for loading real datasets

def _load_iris_binary(random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset for binary classification (setosa vs non-setosa)."""
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        logger = get_logger(__name__)
        logger.debug("Loading Iris dataset")
        
        # Load full iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Convert to binary: setosa (0) vs non-setosa (1)
        y_binary = (y != 0).astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Shuffle dataset
        if random_state is not None:
            set_random_seed(random_state)
        
        indices = np.random.permutation(len(X_scaled))
        X_scaled = X_scaled[indices]
        y_binary = y_binary[indices]
        
        logger.info(f"Loaded Iris binary: {X_scaled.shape}, classes: {np.unique(y_binary)}")
        return X_scaled, y_binary
        
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("scikit-learn not available, generating synthetic iris-like dataset")
        return _generate_iris_like_dataset()


def _load_iris_setosa_versicolor(random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset for binary classification (setosa vs versicolor)."""
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        logger = get_logger(__name__)
        logger.debug("Loading Iris dataset (setosa vs versicolor)")
        
        # Load full iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Filter for setosa (0) and versicolor (1) only
        mask = (y == 0) | (y == 1)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Convert to binary: setosa (0) vs versicolor (1)
        y_binary = y_filtered  # Already 0 and 1
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Shuffle dataset
        if random_state is not None:
            set_random_seed(random_state)
        
        indices = np.random.permutation(len(X_scaled))
        X_scaled = X_scaled[indices]
        y_binary = y_binary[indices]
        
        logger.info(f"Loaded Iris setosa vs versicolor: {X_scaled.shape}, classes: {np.unique(y_binary)}")
        return X_scaled, y_binary
        
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("scikit-learn not available, generating synthetic iris-like dataset")
        return _generate_iris_like_dataset()


def _load_iris_versicolor_virginica(random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset for binary classification (versicolor vs virginica)."""
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        logger = get_logger(__name__)
        logger.debug("Loading Iris dataset (versicolor vs virginica)")
        
        # Load full iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Filter for versicolor (1) and virginica (2) only
        mask = (y == 1) | (y == 2)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Convert to binary: versicolor (0) vs virginica (1)
        y_binary = (y_filtered == 2).astype(int)  # 0 for versicolor, 1 for virginica
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Shuffle dataset
        if random_state is not None:
            set_random_seed(random_state)
        
        indices = np.random.permutation(len(X_scaled))
        X_scaled = X_scaled[indices]
        y_binary = y_binary[indices]
        
        logger.info(f"Loaded Iris versicolor vs virginica: {X_scaled.shape}, classes: {np.unique(y_binary)}")
        return X_scaled, y_binary
        
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("scikit-learn not available, generating synthetic iris-like dataset")
        return _generate_iris_like_dataset()


def _load_breast_cancer_binary(random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load breast cancer dataset for binary classification."""
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        
        logger = get_logger(__name__)
        logger.debug("Loading breast cancer dataset")
        
        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Standardize features (important for perceptron)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Shuffle dataset
        if random_state is not None:
            set_random_seed(random_state)
        
        indices = np.random.permutation(len(X_scaled))
        X_scaled = X_scaled[indices]
        y = y[indices]
        
        logger.info(f"Loaded breast cancer: {X_scaled.shape}, classes: {np.unique(y)}")
        return X_scaled, y
        
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("scikit-learn not available, generating synthetic cancer-like dataset")
        return _generate_cancer_like_dataset()


def _load_mnist_subset(digits: Tuple[int, int] = (0, 1), 
                       max_samples_per_class: int = 1000,
                       random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST subset for binary classification."""
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler
        
        logger = get_logger(__name__)
        logger.debug(f"Loading MNIST subset: digits {digits}")
        
        # Load MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Filter for specific digits
        mask = np.isin(y, digits)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Convert to binary labels
        y_binary = (y_filtered == digits[1]).astype(int)
        
        # Sample subset if too large
        if len(X_filtered) > max_samples_per_class * 2:
            if random_state is not None:
                set_random_seed(random_state)
            
            indices = []
            for label in [0, 1]:
                label_indices = np.where(y_binary == label)[0]
                if len(label_indices) > max_samples_per_class:
                    selected = np.random.choice(label_indices, max_samples_per_class, replace=False)
                    indices.extend(selected)
                else:
                    indices.extend(label_indices)
            
            indices = np.array(indices)
            X_filtered = X_filtered[indices]
            y_binary = y_binary[indices]
        
        # Normalize pixel values
        X_normalized = X_filtered / 255.0
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_normalized)
        
        # Shuffle
        if random_state is not None:
            set_random_seed(random_state)
        
        indices = np.random.permutation(len(X_scaled))
        X_scaled = X_scaled[indices]
        y_binary = y_binary[indices]
        
        logger.info(f"Loaded MNIST subset: {X_scaled.shape}, classes: {np.unique(y_binary)}")
        return X_scaled, y_binary
        
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("scikit-learn not available, generating synthetic MNIST-like dataset")
        return _generate_mnist_like_dataset()


def _generate_iris_like_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic iris-like dataset when sklearn is not available."""
    logger = get_logger(__name__)
    logger.info("Generating synthetic iris-like dataset")
    
    # Create linearly separable dataset with 4 features
    X, y = generate_linear_dataset(
        n_samples=150,
        n_features=4,
        n_classes=2,
        noise=0.1,
        random_state=42
    )
    
    return X, y


def _generate_cancer_like_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic cancer-like dataset when sklearn is not available."""
    logger = get_logger(__name__)
    logger.info("Generating synthetic cancer-like dataset")
    
    # Create linearly separable dataset with 30 features
    X, y = generate_linear_dataset(
        n_samples=569,
        n_features=30,
        n_classes=2,
        noise=0.1,
        random_state=42
    )
    
    return X, y


def _generate_mnist_like_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic MNIST-like dataset when sklearn is not available."""
    logger = get_logger(__name__)
    logger.info("Generating synthetic MNIST-like dataset")
    
    # Create non-linearly separable dataset with 784 features (28x28)
    X = np.random.randn(2000, 784)
    
    # Create somewhat complex decision boundary
    weights1 = np.random.randn(784, 50)
    hidden = np.tanh(X @ weights1)
    weights2 = np.random.randn(50)
    scores = hidden @ weights2
    
    y = (scores > 0).astype(int)
    
    # Add some noise to make it more realistic
    flip_indices = np.random.choice(len(y), int(0.1 * len(y)), replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    
    return X, y 