"""
Unit tests for data_utils BaseDataset classes.

Tests the new BaseDataset, SyntheticDataset, and RealDataset implementations
for strategy compliance.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from data_utils import (
    SyntheticDataset, RealDataset,
    create_dataset, create_synthetic_dataset, create_real_dataset
)
from utils.exceptions import DataError


class TestSyntheticDataset:
    """Test SyntheticDataset functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_xor_dataset(self):
        """Test XOR dataset creation."""
        dataset = create_synthetic_dataset('xor', n_samples=100, data_dir=self.temp_dir)
        
        assert len(dataset) == 100
        assert hasattr(dataset, 'data')
        assert 'X' in dataset.data
        assert 'y' in dataset.data
        assert dataset.data['X'].shape == (100, 2)
        assert dataset.data['y'].shape == (100, 1)
        
        # Check XOR pattern
        X, y = dataset.data['X'], dataset.data['y'].flatten()
        unique_labels = np.unique(y)
        assert len(unique_labels) == 2  # Binary classification
    
    def test_create_circles_dataset(self):
        """Test circles dataset creation."""
        dataset = create_synthetic_dataset('circles', n_samples=200, data_dir=self.temp_dir)
        
        assert len(dataset) == 200
        assert dataset.data['X'].shape == (200, 2)
        assert dataset.data['y'].shape == (200, 1)
    
    def test_create_linear_dataset(self):
        """Test linear dataset creation."""
        dataset = create_synthetic_dataset('linear', n_samples=150, n_features=3, data_dir=self.temp_dir)
        
        assert len(dataset) == 150
        assert dataset.data['X'].shape == (150, 3)
        assert dataset.data['y'].shape == (150, 1)
    
    def test_caching_functionality(self):
        """Test that caching works correctly."""
        # First creation
        dataset1 = create_synthetic_dataset('xor', n_samples=50, data_dir=self.temp_dir, random_state=42)
        
        # Second creation with same parameters should use cache
        dataset2 = create_synthetic_dataset('xor', n_samples=50, data_dir=self.temp_dir, random_state=42)
        
        # Data should be identical (from cache)
        np.testing.assert_array_equal(dataset1.data['X'], dataset2.data['X'])
        np.testing.assert_array_equal(dataset1.data['y'], dataset2.data['y'])
    
    def test_different_parameters_no_cache(self):
        """Test that different parameters create different datasets."""
        dataset1 = create_synthetic_dataset('xor', n_samples=50, data_dir=self.temp_dir, random_state=42)
        dataset2 = create_synthetic_dataset('xor', n_samples=100, data_dir=self.temp_dir, random_state=42)
        
        assert len(dataset1) != len(dataset2)
    
    def test_dataset_indexing(self):
        """Test dataset indexing functionality."""
        dataset = create_synthetic_dataset('xor', n_samples=10, data_dir=self.temp_dir)
        
        # Test single sample access
        sample = dataset[0]
        assert len(sample) == 2  # (X, y)
        assert isinstance(sample[0], np.ndarray)
        # y might be scalar from indexing 2D array
        assert isinstance(sample[1], (np.ndarray, np.integer, int))
    
    def test_dataset_metadata_generation(self):
        """Test automatic metadata generation."""
        dataset = create_synthetic_dataset('xor', n_samples=100, data_dir=self.temp_dir)
        
        assert hasattr(dataset, 'dataset_metadata')
        metadata = dataset.dataset_metadata
        
        assert metadata.name == 'xor'  # Should be the dataset type, not temp dir name
        assert metadata.dataset_type == 'synthetic'
        assert metadata.shape.n_samples == 100
        assert metadata.shape.n_features == 2
        assert metadata.quality_score >= 0
    
    def test_invalid_dataset_type(self):
        """Test handling of invalid dataset types."""
        with pytest.raises(DataError):
            create_synthetic_dataset('invalid_type', data_dir=self.temp_dir)


class TestRealDataset:
    """Test RealDataset functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_iris_dataset(self):
        """Test Iris dataset creation."""
        dataset = create_real_dataset('iris_setosa_versicolor', data_dir=self.temp_dir)
        
        assert len(dataset) > 0
        assert hasattr(dataset, 'data')
        assert 'X' in dataset.data
        assert 'y' in dataset.data
        assert dataset.data['X'].shape[1] == 4  # 4 features for iris
    
    def test_dataset_splits(self):
        """Test different data splits."""
        train_dataset = create_real_dataset('iris_setosa_versicolor', split='train', data_dir=self.temp_dir)
        val_dataset = create_real_dataset('iris_setosa_versicolor', split='val', data_dir=self.temp_dir)
        test_dataset = create_real_dataset('iris_setosa_versicolor', split='test', data_dir=self.temp_dir)
        
        # Training split should be larger than val/test for typical splits
        assert len(train_dataset) >= len(val_dataset)
        assert len(train_dataset) >= len(test_dataset)
        
        # All should be non-empty
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) > 0
    
    def test_real_dataset_metadata(self):
        """Test real dataset metadata generation."""
        dataset = create_real_dataset('iris_setosa_versicolor', data_dir=self.temp_dir)
        
        assert hasattr(dataset, 'dataset_metadata')
        metadata = dataset.dataset_metadata
        
        assert metadata.dataset_type == 'real'
        assert metadata.shape.n_samples > 0
        assert metadata.shape.n_features == 4  # Iris has 4 features
        assert metadata.statistics is not None


class TestDatasetFactoryFunctions:
    """Test dataset factory functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_dataset_synthetic(self):
        """Test create_dataset function with synthetic type."""
        dataset = create_dataset('synthetic', 'xor', n_samples=50, data_dir=self.temp_dir)
        
        assert isinstance(dataset, SyntheticDataset)
        assert len(dataset) == 50
    
    def test_create_dataset_real(self):
        """Test create_dataset function with real type."""
        dataset = create_dataset('real', 'iris_setosa_versicolor', data_dir=self.temp_dir)
        
        assert isinstance(dataset, RealDataset)
        assert len(dataset) > 0
    
    def test_create_dataset_invalid_type(self):
        """Test create_dataset with invalid type."""
        with pytest.raises(DataError):
            create_dataset('invalid', 'xor', data_dir=self.temp_dir)


class TestDatasetInfo:
    """Test dataset information methods."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_get_info_synthetic(self):
        """Test get_info for synthetic dataset."""
        dataset = create_synthetic_dataset('xor', n_samples=100, data_dir=self.temp_dir)
        info = dataset.get_info()
        
        assert 'name' in info
        assert 'split' in info
        assert 'size' in info
        assert 'n_features' in info
        assert 'n_classes' in info
        assert info['size'] == 100
        assert info['n_features'] == 2
        assert info['n_classes'] == 2
    
    def test_get_info_real(self):
        """Test get_info for real dataset."""
        dataset = create_real_dataset('iris_setosa_versicolor', data_dir=self.temp_dir)
        info = dataset.get_info()
        
        assert 'name' in info
        assert 'size' in info
        assert 'n_features' in info
        assert info['n_features'] == 4  # Iris features
    
    def test_dataset_transforms(self):
        """Test dataset with transforms."""
        def dummy_transform(sample):
            x, y = sample
            return x * 2, y  # Simple transform
        
        dataset = create_synthetic_dataset('xor', n_samples=10, transform=dummy_transform, data_dir=self.temp_dir)
        
        # Get original data
        original_sample = dataset.data['X'][0], dataset.data['y'][0]
        
        # Get transformed sample
        transformed_sample = dataset[0]
        
        # Check transform was applied
        np.testing.assert_array_equal(transformed_sample[0], original_sample[0] * 2)
        np.testing.assert_array_equal(transformed_sample[1], original_sample[1])


class TestErrorHandling:
    """Test error handling in dataset creation."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with pytest.raises((ValueError, Exception)):  # Could be ValueError or DataError
            create_synthetic_dataset('xor', n_samples=-10)
    
    def test_missing_dataset_name(self):
        """Test handling of missing dataset name."""
        with pytest.raises(DataError):
            create_real_dataset('nonexistent_dataset')


if __name__ == "__main__":
    pytest.main([__file__]) 