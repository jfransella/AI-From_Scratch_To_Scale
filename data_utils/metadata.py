"""
Dataset metadata schema and management for AI From Scratch to Scale project.

This module implements a comprehensive metadata schema system following the project
strategy specification for standardized dataset information and validation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field_item

import numpy as np

from utils import get_logger
from utils.exceptions import DataError


@dataclass
class DatasetShape:
    """Dataset shape information."""
    n_samples: int
    n_features: int
    feature_shape: tuple
    target_shape: tuple

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'feature_shape': list(self.feature_shape),
            'target_shape': list(self.target_shape)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetShape':
        return cls(
            n_samples=data['n_samples'],
            n_features=data['n_features'],
            feature_shape=tuple(data['feature_shape']),
            target_shape=tuple(data['target_shape'])
        )


@dataclass
class DatasetStatistics:
    """Dataset statistical information."""
    feature_means: List[float]
    feature_stds: List[float]
    feature_mins: List[float]
    feature_maxs: List[float]
    target_distribution: Dict[str, int]
    class_balance: Dict[str, float]
    missing_values: int = 0
    outliers_detected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetStatistics':
        return cls(**data)


@dataclass
class PreprocessingInfo:
    """Preprocessing steps applied to dataset."""
    steps: List[Dict[str, Any]] = field_item(default_factory=list)
    normalization: Optional[str] = None
    scaling: Optional[str] = None
    feature_selection: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None

    def add_step(self, step_name: str, parameters: Dict[str, Any], description: str = ""):
        """Add a preprocessing step."""
        self.steps.append({
            'name': step_name,
            'parameters': parameters,
            'description': description,
            'applied_at': datetime.now().isoformat()
        })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingInfo':
        return cls(**data)


@dataclass
class DatasetCompatibility:
    """Dataset compatibility information with models."""
    model_types: List[str] = field_item(default_factory=list)
    frameworks: List[str] = field_item(default_factory=list)
    min_memory_mb: float = 0.0
    recommended_batch_sizes: Dict[str, int] = field_item(default_factory=dict)
    performance_notes: List[str] = field_item(default_factory=list)

    def add_model_compatibility(self, model_type: str, batch_size: int, notes: str = ""):
        """Add model compatibility info."""
        if model_type not in self.model_types:
            self.model_types.append(model_type)
        self.recommended_batch_sizes[model_type] = batch_size
        if notes:
            self.performance_notes.append(f"{model_type}: {notes}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetCompatibility':
        return cls(**data)


@dataclass
class DatasetMetadata:
    """
    Comprehensive dataset metadata following project strategy specification.

    This class provides a standardized metadata schema for all datasets
    in the AI From Scratch to Scale project.
    """

    # Core identification
    name: str
    version: str = "1.0.0"
    dataset_type: str = "unknown"  # 'synthetic', 'real', 'processed'
    category: str = "unknown"  # 'binary_classification', 'multi_class', etc.

    # Dataset information
    description: str = ""
    source: str = ""
    license: str = ""
    citation: Optional[str] = None

    # Technical specifications
    shape: Optional[DatasetShape] = None
    statistics: Optional[DatasetStatistics] = None
    preprocessing: Optional[PreprocessingInfo] = None
    compatibility: Optional[DatasetCompatibility] = None

    # Data quality
    quality_score: float = 0.0  # 0-100 quality score
    validation_results: Dict[str, Any] = field_item(default_factory=dict)
    known_issues: List[str] = field_item(default_factory=list)

    # Versioning and tracking
    created_at: str = field_item(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field_item(default_factory=lambda: datetime.now().isoformat())
    checksum: str = ""
    file_size_bytes: int = 0

    # Splits information
    splits: Dict[str, Dict[str, Any]] = field_item(default_factory=dict)

    # Custom metadata
    custom_fields: Dict[str, Any] = field_item(default_factory=dict)

    def update_timestamp(self):
        """Update the modified timestamp."""
        self.modified_at = datetime.now().isoformat()

    def add_split_info(self, split_name: str, n_samples: int,
                       split_ratio: float = None, **kwargs):
        """Add information about a data split."""
        self.splits[split_name] = {
            'n_samples': n_samples,
            'split_ratio': split_ratio,
            'created_at': datetime.now().isoformat(),
            **kwargs
        }
        self.update_timestamp()

    def add_validation_result(self, test_name: str, result: Any,
                              passed: bool, details: str = ""):
        """Add a validation test result."""
        self.validation_results[test_name] = {
            'result': result,
            'passed': passed,
            'details': details,
            'tested_at': datetime.now().isoformat()
        }
        self.update_timestamp()

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score based on various factors."""
        score = 0.0
        max_score = 100.0

        # Basic information completeness (20 points)
        info_score = 0
        if self.description:
            info_score += 5
        if self.source:
            info_score += 5
        if self.shape:
            info_score += 5
        if self.statistics:
            info_score += 5
        score += info_score

        # Data quality factors (40 points)
        if self.statistics:
            # No missing values bonus
            if self.statistics.missing_values == 0:
                score += 10
            else:
                # Penalize missing values
                missing_ratio = self.statistics.missing_values / (self.shape.n_samples if self.shape else 1)
                score += max(0, 10 - (missing_ratio * 20))

            # Class balance bonus (for classification)
            if self.statistics.class_balance:
                balance_values = list(self.statistics.class_balance.values())
                if balance_values:
                    balance_score = 1.0 - np.std(balance_values)  # Higher score for balanced classes
                    score += balance_score * 15

            # Low outliers bonus
            if hasattr(self.statistics, 'outliers_detected'):
                outlier_ratio = self.statistics.outliers_detected / (self.shape.n_samples if self.shape else 1)
                score += max(0, 15 - (outlier_ratio * 30))

        # Validation results (25 points)
        if self.validation_results:
            passed_tests = sum(1 for result in self.validation_results.values() if result.get('passed', False))
            total_tests = len(self.validation_results)
            if total_tests > 0:
                score += (passed_tests / total_tests) * 25

        # Documentation and metadata completeness (15 points)
        doc_score = 0
        if self.preprocessing and self.preprocessing.steps:
            doc_score += 3
        if self.compatibility:
            doc_score += 3
        if self.license:
            doc_score += 3
        if self.citation:
            doc_score += 3
        if len(self.splits) > 1:
            doc_score += 3  # Multiple splits documented
        score += doc_score

        # Penalize known issues
        issue_penalty = min(10, len(self.known_issues) * 2)
        score -= issue_penalty

        self.quality_score = max(0.0, min(max_score, score))
        return self.quality_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)

        # Convert nested dataclass objects
        if self.shape:
            data['shape'] = self.shape.to_dict()
        if self.statistics:
            data['statistics'] = self.statistics.to_dict()
        if self.preprocessing:
            data['preprocessing'] = self.preprocessing.to_dict()
        if self.compatibility:
            data['compatibility'] = self.compatibility.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create from dictionary with proper deserialization."""
        # Handle nested objects
        if 'shape' in data and data['shape']:
            data['shape'] = DatasetShape.from_dict(data['shape'])
        if 'statistics' in data and data['statistics']:
            data['statistics'] = DatasetStatistics.from_dict(data['statistics'])
        if 'preprocessing' in data and data['preprocessing']:
            data['preprocessing'] = PreprocessingInfo.from_dict(data['preprocessing'])
        if 'compatibility' in data and data['compatibility']:
            data['compatibility'] = DatasetCompatibility.from_dict(data['compatibility'])

        return cls(**data)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save metadata to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        except Exception as e:
            raise DataError(f"Failed to save metadata to {file_path}: {e}")

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'DatasetMetadata':
        """Load metadata from JSON file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataError(f"Metadata file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise DataError(f"Failed to load metadata from {file_path}: {e}")


class MetadataValidator:
    """
    Validator for dataset metadata following project standards.
    """

    REQUIRED_FIELDS = ['name', 'dataset_type', 'category']
    VALID_DATASET_TYPES = ['synthetic', 'real', 'processed', 'augmented']
    VALID_CATEGORIES = [
        'binary_classification', 'multi_class_classification',
        'regression', 'time_series', 'text', 'image', 'graph'
    ]

    def __init__(self):
        self.logger = get_logger(__name__)

    def validate_metadata(self, metadata: DatasetMetadata) -> Dict[str, Any]:
        """
        Validate dataset metadata comprehensively.

        Args:
            metadata: DatasetMetadata instance to validate

        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'score': 0.0,
            'details': {}
        }

        # Required fields validation
        self._validate_required_fields(metadata, results)

        # Type validation
        self._validate_types(metadata, results)

        # Data consistency validation
        self._validate_data_consistency(metadata, results)

        # Quality validation
        self._validate_quality_indicators(metadata, results)

        # Schema compliance validation
        self._validate_schema_compliance(metadata, results)

        # Calculate overall validation score
        results['score'] = self._calculate_validation_score(results)
        results['valid'] = len(results['errors']) == 0

        self.logger.debug(f"Metadata validation completed: {results['score']:.1f}/100")

        return results

    def _validate_required_fields(self, metadata: DatasetMetadata, results: Dict):
        """Validate required fields are present and non-empty."""
        for field_item in self.REQUIRED_FIELDS:
            value = getattr(metadata, field_item, None)
            if not value or (isinstance(value, str) and not value.strip()):
                results['errors'].append(f"Required field '{field_item}' is missing or empty")

    def _validate_types(self, metadata: DatasetMetadata, results: Dict):
        """Validate field_item types and values."""
        # Dataset type validation
        if metadata.dataset_type not in self.VALID_DATASET_TYPES:
            results['errors'].append(
                f"Invalid dataset_type '{metadata.dataset_type}'. "
                f"Must be one of: {self.VALID_DATASET_TYPES}"
            )

        # Category validation
        if metadata.category not in self.VALID_CATEGORIES:
            results['warnings'].append(
                f"Unknown category '{metadata.category}'. "
                f"Standard categories: {self.VALID_CATEGORIES}"
            )

        # Quality score validation
        if not (0.0 <= metadata.quality_score <= 100.0):
            results['errors'].append(
                f"Quality score must be between 0-100, got {metadata.quality_score}"
            )

    def _validate_data_consistency(self, metadata: DatasetMetadata, results: Dict):
        """Validate internal data consistency."""
        # Shape and statistics consistency
        if metadata.shape and metadata.statistics:
            if len(metadata.statistics.feature_means) != metadata.shape.n_features:
                results['errors'].append(
                    "Statistics feature count doesn't match shape n_features"
                )

        # Splits consistency
        if metadata.splits and metadata.shape:
            total_split_samples = sum(
                split_info.get('n_samples', 0)
                for split_info in metadata.splits.values()
            )
            if total_split_samples > metadata.shape.n_samples:
                results['warnings'].append(
                    "Total split samples exceed dataset size"
                )

    def _validate_quality_indicators(self, metadata: DatasetMetadata, results: Dict):
        """Validate quality indicators and suggest improvements."""
        if not metadata.description:
            results['warnings'].append("Missing dataset description")

        if not metadata.source:
            results['warnings'].append("Missing dataset source information")

        if not metadata.statistics:
            results['warnings'].append("Missing statistical information")

        if not metadata.preprocessing:
            results['warnings'].append("No preprocessing information documented")

        if metadata.known_issues:
            results['warnings'].append(
                f"Dataset has {len(metadata.known_issues)} known issues"
            )

    def _validate_schema_compliance(self, metadata: DatasetMetadata, results: Dict):
        """Validate compliance with project schema standards."""
        # Check timestamp formats
        try:
            datetime.fromisoformat(metadata.created_at)
            datetime.fromisoformat(metadata.modified_at)
        except ValueError:
            results['errors'].append("Invalid timestamp format (should be ISO format)")

        # Check version format
        if not self._is_valid_version(metadata.version):
            results['warnings'].append(
                f"Version '{metadata.version}' doesn't follow semantic versioning"
            )

    def _is_valid_version(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        try:
            parts = version.split('.')
            return len(parts) == 3 and all(part.isdigit() for part in parts)
        except Exception:
            return False

    def _calculate_validation_score(self, results: Dict) -> float:
        """Calculate overall validation score."""
        base_score = 100.0

        # Deduct for errors (major issues)
        error_penalty = len(results['errors']) * 20

        # Deduct for warnings (minor issues)
        warning_penalty = len(results['warnings']) * 5

        score = max(0.0, base_score - error_penalty - warning_penalty)
        return score


def create_metadata_from_data(data: Dict[str, np.ndarray],
                              name: str,
                              dataset_type: str = "unknown",
                              description: str = "",
                              **kwargs) -> DatasetMetadata:
    """
    Create metadata from dataset data arrays.

    Args:
        data: Dictionary with 'X' and 'y' arrays
        name: Dataset name
        dataset_type: Type of dataset
        description: Dataset description
        **kwargs: Additional metadata fields

    Returns:
        DatasetMetadata instance
    """
    logger = get_logger(__name__)

    # Extract data arrays
    X = data.get('X')
    y = data.get('y')

    if X is None or y is None:
        raise DataError("Data must contain 'X' and 'y' arrays")

    # Create shape information
    shape = DatasetShape(
        n_samples=len(X),
        n_features=X.shape[1] if X.ndim > 1 else 1,
        feature_shape=X.shape[1:] if X.ndim > 1 else (1,),
        target_shape=y.shape[1:] if y.ndim > 1 else ()
    )

    # Calculate statistics
    unique_values, counts = np.unique(y, return_counts=True)
    statistics = DatasetStatistics(
        feature_means=[float(x) for x in np.mean(X, axis=0)],
        feature_stds=[float(x) for x in np.std(X, axis=0)],
        feature_mins=[float(x) for x in np.min(X, axis=0)],
        feature_maxs=[float(x) for x in np.max(X, axis=0)],
        target_distribution={str(float(val)): int(count) for val, count in zip(unique_values, counts)},
        class_balance={}
    )

    # Calculate class balance
    unique_values, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    statistics.class_balance = {
        str(float(val)): float(count) / float(total_samples)
        for val, count in zip(unique_values, counts)
    }

    # Create compatibility info
    compatibility = DatasetCompatibility(
        frameworks=['numpy', 'pytorch'],
        min_memory_mb=X.nbytes / (1024 * 1024)
    )

    # Determine category from data
    if len(unique_values) == 2:
        category = "binary_classification"
    elif len(unique_values) > 2 and y.dtype in [np.int32, np.int64]:
        category = "multi_class_classification"
    else:
        category = "regression"

    # Create metadata
    metadata = DatasetMetadata(
        name=name,
        dataset_type=dataset_type,
        category=category,
        description=description,
        shape=shape,
        statistics=statistics,
        compatibility=compatibility,
        **kwargs
    )

    # Calculate quality score
    metadata.calculate_quality_score()

    logger.debug(f"Created metadata for {name}: {shape.n_samples} samples, "
                 f"quality score {metadata.quality_score:.1f}")

    return metadata


def validate_dataset_metadata(metadata: DatasetMetadata) -> Dict[str, Any]:
    """
    Convenience function to validate dataset metadata.

    Args:
        metadata: DatasetMetadata to validate

    Returns:
        Validation results
    """
    validator = MetadataValidator()
    return validator.validate_metadata(metadata)


def save_metadata_to_file(metadata: DatasetMetadata,
                          file_path: Union[str, Path]) -> None:
    """Save metadata to file with validation."""
    # Validate before saving
    validation_results = validate_dataset_metadata(metadata)

    if not validation_results['valid']:
        logger = get_logger(__name__)
        logger.warning(f"Saving metadata with validation errors: {validation_results['errors']}")

    metadata.save_to_file(file_path)


def load_metadata_from_file(file_path: Union[str, Path],
                            validate: bool = True) -> DatasetMetadata:
    """Load and optionally validate metadata from file."""
    metadata = DatasetMetadata.load_from_file(file_path)

    if validate:
        validation_results = validate_dataset_metadata(metadata)
        if not validation_results['valid']:
            logger = get_logger(__name__)
            logger.warning(f"Loaded metadata has validation issues: {validation_results['errors']}")

    return metadata
