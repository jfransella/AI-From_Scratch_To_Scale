"""
DataLoader configuration and creation utilities for AI From Scratch to Scale project.

This module implements the DataLoader standards defined in the project strategy
to provide optimized data loading configurations based on dataset size.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from utils import get_logger
from .base_datasets import BaseDataset


# DataLoader Configuration Standards from Project Strategy
DATALOADER_CONFIGS = {
    'small_datasets': {  # < 10K samples
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'drop_last': False,
        'persistent_workers': False
    },
    'medium_datasets': {  # 10K - 100K samples
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': False,
        'persistent_workers': True
    },
    'large_datasets': {  # > 100K samples
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'drop_last': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    }
}

# Dataset size thresholds
SMALL_DATASET_THRESHOLD = 10000
LARGE_DATASET_THRESHOLD = 100000


class TensorDataset(Dataset):
    """
    PyTorch Dataset wrapper for numpy arrays and tensors.
    
    Provides a simple interface to wrap data arrays for use with DataLoader.
    """
    
    def __init__(self, X: Union[torch.Tensor, 'np.ndarray'], 
                 y: Union[torch.Tensor, 'np.ndarray'], 
                 transform: Optional[callable] = None):
        """
        Initialize TensorDataset.
        
        Args:
            X: Feature data
            y: Target data
            transform: Optional transformation function
        """
        import numpy as np
        
        # Convert to tensors if needed
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float() if X.dtype != torch.float32 else X
            
        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        else:
            self.y = y.long() if y.dtype != torch.long else y
        
        self.transform = transform
        
        # Ensure same length
        assert len(self.X) == len(self.y), f"X and y must have same length: {len(self.X)} vs {len(self.y)}"
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        sample = (self.X[idx], self.y[idx])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class BaseDatasetWrapper(Dataset):
    """
    PyTorch Dataset wrapper for BaseDataset instances.
    
    Allows BaseDataset instances to be used with PyTorch DataLoader.
    """
    
    def __init__(self, base_dataset: BaseDataset):
        """
        Initialize wrapper.
        
        Args:
            base_dataset: BaseDataset instance to wrap
        """
        self.base_dataset = base_dataset
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        return self.base_dataset[idx]


def get_dataloader_config(dataset_size: int, 
                         custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get appropriate DataLoader configuration based on dataset size.
    
    Args:
        dataset_size: Number of samples in dataset
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        DataLoader configuration dictionary
        
    Example:
        config = get_dataloader_config(5000)  # Returns small_datasets config
        config = get_dataloader_config(50000, {'batch_size': 64})  # Custom override
    """
    logger = get_logger(__name__)
    
    # Determine dataset category
    if dataset_size < SMALL_DATASET_THRESHOLD:
        category = 'small_datasets'
    elif dataset_size < LARGE_DATASET_THRESHOLD:
        category = 'medium_datasets'
    else:
        category = 'large_datasets'
    
    # Get base configuration
    config = DATALOADER_CONFIGS[category].copy()
    
    # Apply custom overrides
    if custom_config:
        config.update(custom_config)
    
    logger.debug(f"Selected {category} config for {dataset_size} samples: {config}")
    
    return config


def create_dataloader(dataset: Union[Dataset, BaseDataset, Tuple[torch.Tensor, torch.Tensor]], 
                     batch_size: Optional[int] = None,
                     shuffle: Optional[bool] = None,
                     custom_config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """
    Create optimized DataLoader with automatic configuration.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Override batch size (optional)
        shuffle: Override shuffle setting (optional)
        custom_config: Custom DataLoader configuration
        
    Returns:
        Configured DataLoader instance
        
    Example:
        # From tensors
        loader = create_dataloader((X_tensor, y_tensor))
        
        # From BaseDataset
        dataset = create_synthetic_dataset('xor')
        loader = create_dataloader(dataset, batch_size=32)
    """
    logger = get_logger(__name__)
    
    # Convert input to PyTorch Dataset if needed
    if isinstance(dataset, tuple) and len(dataset) == 2:
        dataset = TensorDataset(dataset[0], dataset[1])
    elif isinstance(dataset, BaseDataset):
        dataset = BaseDatasetWrapper(dataset)
    elif not isinstance(dataset, Dataset):
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
    
    # Get automatic configuration based on dataset size
    config = get_dataloader_config(len(dataset), custom_config)
    
    # Override specific parameters if provided
    if batch_size is not None:
        config['batch_size'] = batch_size
    if shuffle is not None:
        config['shuffle'] = shuffle
    
    # Adjust num_workers for small datasets or Windows
    import platform
    if len(dataset) < 1000 or platform.system() == 'Windows':
        config['num_workers'] = 0  # Avoid multiprocessing overhead
        config['persistent_workers'] = False
    
    logger.info(f"Created DataLoader for {len(dataset)} samples with config: {config}")
    
    return DataLoader(dataset, **config)


def create_train_val_test_loaders(dataset: Union[BaseDataset, Dataset, Tuple[torch.Tensor, torch.Tensor]], 
                                 train_split: float = 0.7,
                                 val_split: float = 0.15,
                                 test_split: float = 0.15,
                                 batch_size: Optional[int] = None,
                                 custom_config: Optional[Dict[str, Any]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with automatic splitting.
    
    Args:
        dataset: Dataset to split and create loaders for
        train_split: Fraction for training (default: 0.7)
        val_split: Fraction for validation (default: 0.15)
        test_split: Fraction for testing (default: 0.15)
        batch_size: Override batch size (optional)
        custom_config: Custom DataLoader configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        dataset = create_synthetic_dataset('circles', n_samples=1000)
        train_loader, val_loader, test_loader = create_train_val_test_loaders(dataset)
    """
    logger = get_logger(__name__)
    
    # Validate split ratios
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_split + val_split + test_split}")
    
    # Convert to PyTorch Dataset if needed
    if isinstance(dataset, tuple) and len(dataset) == 2:
        dataset = TensorDataset(dataset[0], dataset[1])
    elif isinstance(dataset, BaseDataset):
        dataset = BaseDatasetWrapper(dataset)
    elif not isinstance(dataset, Dataset):
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size  # Ensure exact total
    
    logger.info(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create loaders with appropriate configurations
    train_config = custom_config.copy() if custom_config else {}
    train_config['shuffle'] = True  # Always shuffle training data
    
    val_test_config = custom_config.copy() if custom_config else {}
    val_test_config['shuffle'] = False  # Don't shuffle validation/test data
    
    train_loader = create_dataloader(train_dataset, batch_size, custom_config=train_config)
    val_loader = create_dataloader(val_dataset, batch_size, custom_config=val_test_config)
    test_loader = create_dataloader(test_dataset, batch_size, custom_config=val_test_config)
    
    return train_loader, val_loader, test_loader


def create_data_loaders(X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
                       X_test: Optional[torch.Tensor] = None, y_test: Optional[torch.Tensor] = None,
                       batch_size: Optional[int] = None,
                       custom_config: Optional[Dict[str, Any]] = None) -> Tuple[DataLoader, ...]:
    """
    Create DataLoaders from separate train/val/test tensors.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        X_test, y_test: Test data (optional)
        batch_size: Override batch size (optional)
        custom_config: Custom DataLoader configuration
        
    Returns:
        Tuple of DataLoaders (train_loader, [val_loader], [test_loader])
        
    Example:
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
    """
    logger = get_logger(__name__)
    
    loaders = []
    
    # Training loader (always shuffle)
    train_config = custom_config.copy() if custom_config else {}
    train_config['shuffle'] = True
    train_loader = create_dataloader((X_train, y_train), batch_size, custom_config=train_config)
    loaders.append(train_loader)
    
    # Validation loader (don't shuffle)
    if X_val is not None and y_val is not None:
        val_config = custom_config.copy() if custom_config else {}
        val_config['shuffle'] = False
        val_loader = create_dataloader((X_val, y_val), batch_size, custom_config=val_config)
        loaders.append(val_loader)
    
    # Test loader (don't shuffle)
    if X_test is not None and y_test is not None:
        test_config = custom_config.copy() if custom_config else {}
        test_config['shuffle'] = False
        test_loader = create_dataloader((X_test, y_test), batch_size, custom_config=test_config)
        loaders.append(test_loader)
    
    logger.info(f"Created {len(loaders)} DataLoaders")
    
    return tuple(loaders)


def get_dataset_info_from_loader(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Extract dataset information from a DataLoader.
    
    Args:
        dataloader: DataLoader to analyze
        
    Returns:
        Dictionary with dataset information
        
    Example:
        info = get_dataset_info_from_loader(train_loader)
        print(f"Batch size: {info['batch_size']}, Total batches: {info['num_batches']}")
    """
    dataset = dataloader.dataset
    
    info = {
        'dataset_size': len(dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'shuffle': dataloader.sampler is not None,
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory
    }
    
    # Try to get sample shape
    try:
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 2:
            X_batch, y_batch = sample_batch
            info.update({
                'feature_shape': tuple(X_batch.shape[1:]),
                'target_shape': tuple(y_batch.shape[1:]) if y_batch.dim() > 1 else (),
                'feature_dtype': str(X_batch.dtype),
                'target_dtype': str(y_batch.dtype)
            })
    except Exception:
        pass  # Skip if unable to get sample
    
    return info 