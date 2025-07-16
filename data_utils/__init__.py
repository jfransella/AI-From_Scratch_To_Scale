"""
Data utilities package for AI From Scratch to Scale project.

This package provides dataset loading, preprocessing, and data loader functionality
used across all models in the project.

Key components:
- Dataset loading for synthetic and real datasets
- Data loader creation with proper batching
- Data preprocessing and transformations
- Dataset factory for easy dataset creation
"""

from .datasets import (
    load_dataset,
    get_dataset_info,
    list_available_datasets
)
# from .loaders import (
#     create_data_loaders,
#     create_train_val_test_loaders
# )
# from .preprocessing import (
#     StandardScaler,
#     MinMaxScaler,
#     normalize_data,
#     train_test_split_data
# )
from .synthetic import (
    generate_xor_dataset,
    generate_circles_dataset,
    generate_linear_dataset,
    generate_classification_dataset
)

__version__ = "1.0.0"
__all__ = [
    # Main dataset loading
    "load_dataset",
    "get_dataset_info", 
    "list_available_datasets",
    # # Data loaders
    # "create_data_loaders",
    # "create_train_val_test_loaders",
    # # Preprocessing
    # "StandardScaler",
    # "MinMaxScaler",
    # "normalize_data",
    # "train_test_split_data",
    # Synthetic datasets
    "generate_xor_dataset",
    "generate_circles_dataset", 
    "generate_linear_dataset",
    "generate_classification_dataset"
] 