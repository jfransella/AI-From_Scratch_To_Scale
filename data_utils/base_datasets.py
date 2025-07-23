"""
Concrete BaseDataset implementations for AI From Scratch to Scale project.

This module implements the BaseDataset classes defined in the project strategy
to provide a unified interface for dataset handling across all models.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from utils import get_logger
from utils.exceptions import DataError
from .cache import cache_dataset, load_cached_dataset
from .metadata import create_metadata_from_data, DatasetMetadata


class BaseDataset(ABC):
    """
    Abstract base class for all datasets following project strategy.

    This class provides the interface defined in docs/strategy/Dataset_Strategy.md
    for consistent dataset handling across all models.
    """

    def __init__(self, data_dir: Union[str, Path], split: str = 'train',
                 transform: Optional[Callable] = None, **kwargs):
        """
        Initialize BaseDataset.

        Args:
            data_dir: Directory containing dataset files
            split: Data split ('train', 'val', 'test')
            transform: Optional transformation function
            **kwargs: Additional dataset-specific parameters
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.kwargs = kwargs
        self.logger = get_logger(__name__)

        # Initialize metadata
        self.metadata = self._load_metadata()

        # Load data
        self.data = self._load_data()

        # Generate or load dataset metadata
        self.dataset_metadata = self._generate_or_load_metadata()

        self.logger.info(f"Loaded {self.__class__.__name__} - Split: {split}, Samples: {len(self)}")

    @abstractmethod
    def _load_data(self) -> Any:
        """Load data from storage - implement in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_data()")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata if available."""
        metadata_path = self.data_dir / f"{self.data_dir.name}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")

        return self._generate_default_metadata()

    def _generate_default_metadata(self) -> Dict[str, Any]:
        """Generate default metadata structure."""
        return {
            "dataset_name": self.data_dir.name,
            "version": "1.0.0",
            "splits": {},
            "preprocessing": {},
            "created_at": None,
            "checksum": None
        }

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        if isinstance(self.data, (list, tuple)):
            return len(self.data)
        elif isinstance(self.data, dict) and 'X' in self.data:
            return len(self.data['X'])
        elif hasattr(self.data, '__len__'):
            return len(self.data)
        else:
            raise DataError("Cannot determine dataset length")

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get a single sample from the dataset."""
        if isinstance(self.data, dict):
            sample = (self.data['X'][idx], self.data['y'][idx])
        elif isinstance(self.data, (list, tuple)) and len(self.data) == 2:
            sample = (self.data[0][idx], self.data[1][idx])
        else:
            raise DataError("Dataset format not supported for indexing")

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        info = {
            "name": self.data_dir.name,
            "split": self.split,
            "size": len(self),
            "metadata": self.metadata
        }

        if isinstance(self.data, dict):
            if 'X' in self.data and 'y' in self.data:
                info.update({
                    "n_features": self.data['X'].shape[1] if self.data['X'].ndim > 1 else 1,
                    "n_classes": len(np.unique(self.data['y'])),
                    "feature_shape": self.data['X'].shape[1:],
                    "class_distribution": dict(zip(*np.unique(self.data['y'], return_counts=True)))
                })

        return info

    def _generate_or_load_metadata(self) -> DatasetMetadata:
        """Generate or load comprehensive dataset metadata."""
        try:
            # Try to load existing metadata first
            metadata_path = self.data_dir / f"{self.data_dir.name}_dataset_metadata.json"
            if metadata_path.exists():
                try:
                    from .metadata import load_metadata_from_file
                    return load_metadata_from_file(metadata_path, validate=False)
                except Exception as e:
                    self.logger.warning(f"Failed to load existing metadata: {e}")

            # Generate new metadata from data
            # Use the actual dataset name, not the directory name
            dataset_name = getattr(self, 'dataset_name', None) or getattr(self, 'dataset_type', self.data_dir.name)
            dataset_metadata = create_metadata_from_data(
                data=self.data,
                name=dataset_name,
                dataset_type=self.kwargs.get('dataset_type', 'unknown'),
                description=f"Dataset loaded from {self.data_dir} (split: {self.split})"
            )

            # Add split information
            dataset_metadata.add_split_info(
                split_name=self.split,
                n_samples=len(self.data['X']) if 'X' in self.data else len(self.data)
            )

            # Save the generated metadata
            self.save_dataset_metadata(dataset_metadata, metadata_path)

            return dataset_metadata

        except Exception as e:
            self.logger.warning(f"Failed to generate metadata: {e}")
            # Return minimal metadata as fallback
            return DatasetMetadata(
                name=self.data_dir.name,
                dataset_type='unknown',
                category='unknown',
                description=f"Fallback metadata for {self.data_dir.name}"
            )

    def save_dataset_metadata(self, dataset_metadata: DatasetMetadata,
                              metadata_path: Optional[Path] = None) -> None:
        """Save comprehensive dataset metadata to file."""
        if metadata_path is None:
            metadata_path = self.data_dir / f"{self.data_dir.name}_dataset_metadata.json"

        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_metadata.save_to_file(metadata_path)
            self.logger.debug(f"Saved dataset metadata to {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dataset metadata: {e}")

    def save_metadata(self, metadata_path: Optional[Path] = None) -> None:
        """Save basic metadata to file (legacy method)."""
        if metadata_path is None:
            metadata_path = self.data_dir / f"{self.data_dir.name}_metadata.json"

        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            self.logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")


class SyntheticDataset(BaseDataset):
    """
    Concrete implementation for synthetic datasets.

    Handles generated datasets like XOR, circles, linear separable data.
    Supports caching and parameter-based generation.
    """

    def __init__(self, dataset_type: str, data_dir: Union[str, Path] = None,
                 split: str = 'train', transform: Optional[Callable] = None, **kwargs):
        """
        Initialize SyntheticDataset.

        Args:
            dataset_type: Type of synthetic dataset ('xor', 'circles', 'linear')
            data_dir: Directory for caching (optional, defaults to data/generated)
            split: Data split (used for consistent generation)
            transform: Optional transformation function
            **kwargs: Parameters for dataset generation
        """
        self.dataset_type = dataset_type

        if data_dir is None:
            data_dir = Path("data") / "generated" / dataset_type

        # Store generation parameters
        self.generation_params = kwargs.copy()
        self.generation_params['dataset_type'] = dataset_type
        self.generation_params['split'] = split

        # Set dataset_type for metadata generation
        kwargs['dataset_type'] = 'synthetic'

        super().__init__(data_dir, split, transform, **kwargs)

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load or generate synthetic data using enhanced caching."""
        # Try to load from enhanced cache first
        cached_data = load_cached_dataset('synthetic', self.dataset_type, self.generation_params)
        if cached_data is not None:
            self.logger.debug(f"Loaded {self.dataset_type} from enhanced cache")
            return cached_data

        # Generate new data
        data = self._generate_data()

        # Cache using enhanced caching system
        cache_dataset(data, 'synthetic', self.dataset_type, self.generation_params, {
            'dataset_type': self.dataset_type,
            'split': self.split,
            'generation_params': self.generation_params
        })

        return data

    def _generate_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic data based on dataset type."""
        from .synthetic import (
            generate_xor_dataset,
            generate_circles_dataset,
            generate_linear_dataset
        )

        # Set random seed for consistent splits
        random_state = self.generation_params.get('random_state', 42)
        if self.split == 'val':
            random_state += 1
        elif self.split == 'test':
            random_state += 2

        self.logger.info(f"Generating {self.dataset_type} data for {self.split} split")

        if self.dataset_type in ['xor', 'xor_problem']:
            X, y = generate_xor_dataset(
                n_samples=self.generation_params.get('n_samples', 1000),
                noise=self.generation_params.get('noise', 0.1),
                random_state=random_state
            )
        elif self.dataset_type in ['circles', 'circles_dataset']:
            X, y = generate_circles_dataset(
                n_samples=self.generation_params.get('n_samples', 1000),
                noise=self.generation_params.get('noise', 0.1),
                factor=self.generation_params.get('factor', 0.8),
                random_state=random_state
            )
        elif self.dataset_type in ['linear', 'linear_separable']:
            X, y = generate_linear_dataset(
                n_samples=self.generation_params.get('n_samples', 1000),
                n_features=self.generation_params.get('n_features', 2),
                n_classes=self.generation_params.get('n_classes', 2),
                noise=self.generation_params.get('noise', 0.1),
                random_state=random_state
            )
        else:
            raise DataError(f"Unknown synthetic dataset type: {self.dataset_type}")

        return {'X': X, 'y': y}

    def _get_cache_path(self) -> Path:
        """Get cache file path based on generation parameters."""
        # Create parameter hash for unique caching
        param_str = json.dumps(self.generation_params, sort_keys=True)
        param_hash = hash(param_str) & 0x7FFFFFFF  # Ensure positive

        cache_dir = Path("data") / "cache" / "synthetic"
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / f"{self.dataset_type}_{param_hash}_{self.split}.npz"

    def _load_from_cache(self, cache_path: Path) -> Dict[str, np.ndarray]:
        """Load data from cache file."""
        data = np.load(cache_path)
        self.logger.debug(f"Loaded {self.dataset_type} from cache: {cache_path}")
        return {'X': data['X'], 'y': data['y']}

    def _save_to_cache(self, data: Dict[str, np.ndarray], cache_path: Path) -> None:
        """Save data to cache file."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, X=data['X'], y=data['y'])
            self.logger.debug(f"Cached {self.dataset_type} to: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")


class RealDataset(BaseDataset):
    """
    Concrete implementation for real datasets.

    Handles real-world datasets like Iris, MNIST, breast cancer.
    Supports preprocessing, standardization, and caching.
    """

    def __init__(self, dataset_name: str, data_dir: Union[str, Path] = None,
                 split: str = 'train', transform: Optional[Callable] = None,
                 download: bool = True, **kwargs):
        """
        Initialize RealDataset.

        Args:
            dataset_name: Name of real dataset ('iris', 'mnist', 'breast_cancer')
            data_dir: Directory containing dataset (optional, defaults to data/real)
            split: Data split ('train', 'val', 'test')
            transform: Optional transformation function
            download: Whether to download dataset if not found
            **kwargs: Dataset-specific parameters
        """
        self.dataset_name = dataset_name
        self.download = download

        if data_dir is None:
            data_dir = Path("data") / "real" / dataset_name

        # Set dataset_type for metadata generation
        kwargs['dataset_type'] = 'real'

        super().__init__(data_dir, split, transform, **kwargs)

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load real dataset using enhanced caching."""
        # Create cache parameters for this specific dataset configuration
        cache_params = {
            'split': self.split,
            'random_state': self.kwargs.get('random_state', 42),
            'dataset_name': self.dataset_name
        }

        # Try to load from enhanced cache first
        cached_data = load_cached_dataset('real', self.dataset_name, cache_params)
        if cached_data is not None:
            self.logger.debug(f"Loaded {self.dataset_name} from enhanced cache")
            return cached_data

        # Load and process raw data
        data = self._load_and_process_raw_data()

        # Cache using enhanced caching system
        cache_dataset(data, 'real', self.dataset_name, cache_params, {
            'dataset_name': self.dataset_name,
            'split': self.split,
            'download': self.download,
            'processing_params': self.kwargs
        })

        return data

    def _load_and_process_raw_data(self) -> Dict[str, np.ndarray]:
        """Load and process raw dataset."""
        from .datasets import (
            _load_iris_setosa_versicolor,
            _load_iris_versicolor_virginica,
            _load_breast_cancer_binary,
            _load_mnist_subset
        )

        self.logger.info(f"Loading {self.dataset_name} dataset")

        # Load based on dataset name
        if self.dataset_name == 'iris_setosa_versicolor':
            X, y = _load_iris_setosa_versicolor(
                random_state=self.kwargs.get('random_state', 42)
            )
        elif self.dataset_name == 'iris_versicolor_virginica':
            X, y = _load_iris_versicolor_virginica(
                random_state=self.kwargs.get('random_state', 42)
            )
        elif self.dataset_name == 'breast_cancer':
            X, y = _load_breast_cancer_binary(
                random_state=self.kwargs.get('random_state', 42)
            )
        elif self.dataset_name == 'mnist_subset':
            X, y = _load_mnist_subset(
                digits=self.kwargs.get('digits', (0, 1)),
                max_samples_per_class=self.kwargs.get('max_samples_per_class', 1000),
                random_state=self.kwargs.get('random_state', 42)
            )
        else:
            raise DataError(f"Unknown real dataset: {self.dataset_name}")

        # Split data if needed
        if self.split in ['val', 'test']:
            X, y = self._create_split(X, y)

        return {'X': X, 'y': y}

    def _create_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create validation or test split from full data."""
        from sklearn.model_selection import train_test_split

        random_state = self.kwargs.get('random_state', 42)

        if self.split == 'val':
            # Use 20% for validation
            _, X_split, _, y_split = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )
        elif self.split == 'test':
            # Use 20% for test (different from validation)
            _, X_split, _, y_split = train_test_split(
                X, y, test_size=0.2, random_state=random_state + 1, stratify=y
            )
        else:
            # Training split - use 60%
            X_split, _, y_split, _ = train_test_split(
                X, y, test_size=0.4, random_state=random_state, stratify=y
            )

        return X_split, y_split

    def _get_cache_path(self) -> Path:
        """Get cache file path for processed data."""
        cache_dir = Path("data") / "cache" / "real"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Include key parameters in filename
        params = [
            self.dataset_name,
            self.split,
            str(self.kwargs.get('random_state', 42))
        ]
        filename = "_".join(params) + ".npz"

        return cache_dir / filename

    def _load_from_cache(self, cache_path: Path) -> Dict[str, np.ndarray]:
        """Load processed data from cache."""
        data = np.load(cache_path)
        self.logger.debug(f"Loaded {self.dataset_name} from cache: {cache_path}")
        return {'X': data['X'], 'y': data['y']}

    def _save_to_cache(self, data: Dict[str, np.ndarray], cache_path: Path) -> None:
        """Save processed data to cache."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, X=data['X'], y=data['y'])
            self.logger.debug(f"Cached {self.dataset_name} to: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")


# Factory functions for easy dataset creation

def create_dataset(dataset_type: str, dataset_name: str, **kwargs) -> BaseDataset:
    """
    Factory function to create appropriate dataset instance.

    Args:
        dataset_type: 'synthetic' or 'real'
        dataset_name: Name of the specific dataset
        **kwargs: Additional parameters

    Returns:
        Appropriate BaseDataset instance

    Example:
        dataset = create_dataset('synthetic', 'xor', n_samples=500)
        dataset = create_dataset('real', 'iris_setosa_versicolor', split='train')
    """
    if dataset_type == 'synthetic':
        return SyntheticDataset(dataset_name, **kwargs)
    elif dataset_type == 'real':
        return RealDataset(dataset_name, **kwargs)
    else:
        raise DataError(f"Unknown dataset type: {dataset_type}")


def create_synthetic_dataset(dataset_name: str, **kwargs) -> SyntheticDataset:
    """Create synthetic dataset instance."""
    return SyntheticDataset(dataset_name, **kwargs)


def create_real_dataset(dataset_name: str, **kwargs) -> RealDataset:
    """Create real dataset instance."""
    return RealDataset(dataset_name, **kwargs)
