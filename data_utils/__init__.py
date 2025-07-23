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

from .datasets import load_dataset, get_dataset_info, list_available_datasets
from .base_datasets import (
    BaseDataset,
    SyntheticDataset,
    RealDataset,
    create_dataset,
    create_synthetic_dataset,
    create_real_dataset,
)
from .loaders import (
    create_data_loaders,
    create_train_val_test_loaders,
    create_dataloader,
    get_dataloader_config,
    TensorDataset,
    BaseDatasetWrapper,
)
from .cache import (
    DatasetCache,
    cache_dataset,
    load_cached_dataset,
    get_cache_stats,
    cleanup_cache,
    get_cache,
)
from .metadata import (
    DatasetMetadata,
    DatasetShape,
    DatasetStatistics,
    PreprocessingInfo,
    DatasetCompatibility,
    MetadataValidator,
    create_metadata_from_data,
    validate_dataset_metadata,
    save_metadata_to_file,
    load_metadata_from_file,
)
from .organization import (
    ProjectOrganizer,
    analyze_project_organization,
    create_project_directories,
    generate_organization_report,
)

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
    generate_classification_dataset,
)

__version__ = "1.0.0"
__all__ = [
    # Main dataset loading
    "load_dataset",
    "get_dataset_info",
    "list_available_datasets",
    # BaseDataset classes
    "BaseDataset",
    "SyntheticDataset",
    "RealDataset",
    "create_dataset",
    "create_synthetic_dataset",
    "create_real_dataset",
    # Data loaders
    "create_data_loaders",
    "create_train_val_test_loaders",
    "create_dataloader",
    "get_dataloader_config",
    "TensorDataset",
    "BaseDatasetWrapper",
    # Enhanced caching
    "DatasetCache",
    "cache_dataset",
    "load_cached_dataset",
    "get_cache_stats",
    "cleanup_cache",
    "get_cache",
    # Metadata schema
    "DatasetMetadata",
    "DatasetShape",
    "DatasetStatistics",
    "PreprocessingInfo",
    "DatasetCompatibility",
    "MetadataValidator",
    "create_metadata_from_data",
    "validate_dataset_metadata",
    "save_metadata_to_file",
    "load_metadata_from_file",
    # File organization
    "ProjectOrganizer",
    "analyze_project_organization",
    "create_project_directories",
    "generate_organization_report",
    # # Preprocessing
    # "StandardScaler",
    # "MinMaxScaler",
    # "normalize_data",
    # "train_test_split_data",
    # Synthetic datasets
    "generate_xor_dataset",
    "generate_circles_dataset",
    "generate_linear_dataset",
    "generate_classification_dataset",
]
