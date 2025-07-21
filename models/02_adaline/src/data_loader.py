"""
Unified dataset loading for ADALINE model.

This module provides standardized dataset loading functionality that replaces
the scattered dataset loading functions across ADALINE's train.py, evaluate.py,
and compare_with_perceptron.py files.
"""

import torch
import numpy as np
from typing import Tuple
import logging

# Import unified dataset loading system
try:
    from data_utils.datasets import load_dataset, get_dataset_info
    HAS_DATA_UTILS = True
except ImportError:
    HAS_DATA_UTILS = False

logger = logging.getLogger(__name__)


def load_adaline_dataset(dataset_name: str, 
                         return_numpy: bool = False,
                         device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load dataset for ADALINE using unified data_utils system.
    
    Args:
        dataset_name: Name of dataset to load
        return_numpy: If True, return numpy arrays instead of torch tensors
        device: Device to place tensors on
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
        
    Raises:
        ImportError: If data_utils is not available
        ValueError: If dataset is not supported
        
    Example:
        X, y = load_adaline_dataset('iris_setosa_versicolor')
        X, y = load_adaline_dataset('debug_small', return_numpy=True)
    """
    if not HAS_DATA_UTILS:
        raise ImportError(
            "data_utils package not available. "
            "Please install the project in editable mode: pip install -e ."
        )
    
    try:
        # Load dataset using unified system
        X, y = load_dataset(dataset_name)
        
        if return_numpy:
            # Return as numpy arrays (for ADALINE's numpy-based model)
            x_data = np.array(X, dtype=np.float32)
            y_data = np.array(y, dtype=np.float32).reshape(-1, 1)
            
            logger.info(f"Loaded {dataset_name}: {x_data.shape}, classes: {np.unique(y)}")
            return x_data, y_data
        else:
            # Return as torch tensors (for evaluation and comparison)
            x_data = torch.tensor(X, dtype=torch.float32, device=device)
            y_data = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
            
            logger.info(f"Loaded {dataset_name}: {x_data.shape}, classes: {torch.unique(y_data).cpu().numpy()}")
            return x_data, y_data
            
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise


def get_adaline_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset for ADALINE.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dictionary with dataset information
        
    Example:
        info = get_adaline_dataset_info('iris_setosa_versicolor')
        print(f"Features: {info['n_features']}, Separable: {info['linearly_separable']}")
    """
    if not HAS_DATA_UTILS:
        raise ImportError("data_utils package not available")
    
    return get_dataset_info(dataset_name)


def validate_adaline_dataset(dataset_name: str) -> bool:
    """
    Validate that a dataset is suitable for ADALINE training.
    
    Args:
        dataset_name: Name of dataset to validate
        
    Returns:
        True if dataset is valid for ADALINE
        
    Raises:
        ValueError: If dataset is not suitable
    """
    try:
        info = get_adaline_dataset_info(dataset_name)
        
        # Check that it's binary classification
        if info['n_classes'] != 2:
            raise ValueError(f"ADALINE requires binary classification. {dataset_name} has {info['n_classes']} classes")
        
        # Log warning for non-linearly separable datasets
        if not info['linearly_separable']:
            logger.warning(f"Dataset {dataset_name} is not linearly separable. ADALINE may not converge.")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed for {dataset_name}: {e}")
        raise


def load_adaline_train_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data for ADALINE (returns numpy arrays).
    
    This is the primary function for ADALINE's train.py to replace
    the custom load_dataset_data() function.
    
    Args:
        dataset_name: Name of dataset to load
        
    Returns:
        Tuple of (x_data, y_data) as numpy arrays
    """
    # Validate dataset first
    validate_adaline_dataset(dataset_name)
    
    # Load data as numpy arrays for ADALINE's numpy-based implementation
    return load_adaline_dataset(dataset_name, return_numpy=True)


def load_adaline_eval_data(dataset_name: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load evaluation data for ADALINE (returns torch tensors).
    
    This is the primary function for ADALINE's evaluate.py and 
    compare_with_perceptron.py to replace their load_dataset_data() functions.
    
    Args:
        dataset_name: Name of dataset to load
        device: Device to place tensors on
        
    Returns:
        Tuple of (x_data, y_data) as torch tensors
    """
    # Validate dataset first
    validate_adaline_dataset(dataset_name)
    
    # Load data as torch tensors for evaluation
    return load_adaline_dataset(dataset_name, return_numpy=False, device=device) 