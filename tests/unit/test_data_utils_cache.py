"""
Unit tests for data_utils enhanced caching system.

Tests the DatasetCache, cache management, TTL functionality, and cleanup operations.
"""

import pytest
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta

from data_utils.cache import (
    DatasetCache, CacheEntry, 
    cache_dataset, load_cached_dataset,
    get_cache_stats, cleanup_cache
)


class TestDatasetCache:
    """Test DatasetCache functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = DatasetCache(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.base_dir == self.temp_dir
        assert self.cache.base_dir.exists()
        
        # Check category directories were created
        for category in ['synthetic', 'real', 'processed', 'models']:
            assert (self.temp_dir / category).exists()
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.cache.generate_cache_key('synthetic', 'xor', {'n_samples': 100})
        key2 = self.cache.generate_cache_key('synthetic', 'xor', {'n_samples': 100})
        key3 = self.cache.generate_cache_key('synthetic', 'xor', {'n_samples': 200})
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        assert key1 != key3
        
        # Keys should follow naming convention
        assert key1.startswith('synthetic_xor_')
        assert len(key1.split('_')) == 3  # category_identifier_hash
    
    def test_save_and_load_cache(self):
        """Test saving and loading data to/from cache."""
        # Create test data
        data = {
            'X': np.random.random((100, 2)),
            'y': np.random.randint(0, 2, (100, 1))
        }
        
        # Save to cache
        cache_key = 'test_key'
        entry = self.cache.save_to_cache(data, 'synthetic', cache_key)
        
        assert isinstance(entry, CacheEntry)
        assert entry.cache_path.exists()
        
        # Load from cache
        result = self.cache.load_from_cache('synthetic', cache_key)
        assert result is not None
        
        loaded_data, loaded_entry = result
        
        # Verify data integrity
        np.testing.assert_array_equal(loaded_data['X'], data['X'])
        np.testing.assert_array_equal(loaded_data['y'], data['y'])
        
        # Verify entry information
        assert isinstance(loaded_entry, CacheEntry)
        assert loaded_entry.category == 'synthetic'
        assert loaded_entry.key == cache_key
    
    def test_cache_entry_validation(self):
        """Test cache entry TTL validation."""
        # Create test data
        data = {'X': np.array([[1, 2]]), 'y': np.array([[0]])}
        
        # Save to cache
        cache_key = 'ttl_test'
        entry = self.cache.save_to_cache(data, 'synthetic', cache_key)
        
        # Should be valid immediately
        assert entry.is_valid(ttl_hours=1.0)
        
        # Should be invalid with very short TTL
        assert not entry.is_valid(ttl_hours=0.0)
    
    def test_cache_metadata_tracking(self):
        """Test cache metadata tracking."""
        initial_stats = self.cache.get_cache_stats()
        initial_entries = initial_stats['total_entries']
        
        # Add data to cache
        data = {'X': np.random.random((50, 3)), 'y': np.random.randint(0, 2, (50, 1))}
        self.cache.save_to_cache(data, 'synthetic', 'metadata_test')
        
        # Check metadata was updated
        updated_stats = self.cache.get_cache_stats()
        assert updated_stats['total_entries'] == initial_entries + 1
        assert updated_stats['total_size_mb'] > initial_stats['total_size_mb']
        assert 'synthetic' in updated_stats['categories']
        assert updated_stats['categories']['synthetic']['count'] >= 1
    
    def test_cache_compression(self):
        """Test cache compression functionality."""
        # Create moderately large data
        data = {
            'X': np.random.random((1000, 10)),
            'y': np.random.randint(0, 2, (1000, 1))
        }
        
        # Save with compression (default for synthetic)
        cache_key = 'compression_test'
        entry = self.cache.save_to_cache(data, 'synthetic', cache_key)
        
        # File should exist and be compressed (.npz extension)
        assert entry.cache_path.suffix == '.npz'
        assert entry.cache_path.exists()
        
        # Should be able to load back
        result = self.cache.load_from_cache('synthetic', cache_key)
        assert result is not None
        
        loaded_data, _ = result
        np.testing.assert_array_equal(loaded_data['X'], data['X'])


class TestCacheManagement:
    """Test cache management operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = DatasetCache(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cache_stats(self):
        """Test cache statistics generation."""
        stats = self.cache.get_cache_stats()
        
        assert 'total_entries' in stats
        assert 'total_size_mb' in stats
        assert 'categories' in stats
        assert 'cache_dir' in stats
        assert 'max_size_gb' in stats
        assert 'usage_percentage' in stats
        
        # Initially should be empty
        assert stats['total_entries'] == 0
        assert stats['total_size_mb'] == 0.0
    
    def test_clear_cache_category(self):
        """Test clearing specific cache category."""
        # Add test data to multiple categories
        test_data = {'X': np.array([[1, 2]]), 'y': np.array([[0]])}
        
        self.cache.save_to_cache(test_data, 'synthetic', 'test1')
        self.cache.save_to_cache(test_data, 'real', 'test2')
        
        # Clear only synthetic category
        result = self.cache.clear_cache(category='synthetic', confirm=True)
        
        assert 'synthetic' in result['categories_cleared']
        assert result['entries_removed'] >= 1
        
        # Synthetic should be gone, real should remain
        assert self.cache.load_from_cache('synthetic', 'test1') is None
        assert self.cache.load_from_cache('real', 'test2') is not None
    
    def test_clear_all_cache(self):
        """Test clearing entire cache."""
        # Add test data
        test_data = {'X': np.array([[1, 2]]), 'y': np.array([[0]])}
        self.cache.save_to_cache(test_data, 'synthetic', 'test1')
        self.cache.save_to_cache(test_data, 'real', 'test2')
        
        # Clear all
        result = self.cache.clear_cache(confirm=True)
        
        assert len(result['categories_cleared']) >= 2
        assert result['entries_removed'] >= 2
        
        # All should be gone
        assert self.cache.load_from_cache('synthetic', 'test1') is None
        assert self.cache.load_from_cache('real', 'test2') is None
    
    def test_clear_cache_safety(self):
        """Test cache clearing safety mechanism."""
        with pytest.raises(ValueError):
            # Should require confirmation
            self.cache.clear_cache(confirm=False)


class TestConvenienceFunctions:
    """Test convenience functions for caching."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Set up temporary cache for testing
        import data_utils.cache as cache_module
        self.original_cache = cache_module._global_cache
        cache_module._global_cache = DatasetCache(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original cache
        import data_utils.cache as cache_module
        cache_module._global_cache = self.original_cache
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cache_dataset_function(self):
        """Test cache_dataset convenience function."""
        data = {'X': np.random.random((10, 2)), 'y': np.random.randint(0, 2, (10, 1))}
        
        # Cache data
        cache_key = cache_dataset(data, 'synthetic', 'test_dataset', {'param1': 'value1'})
        
        assert isinstance(cache_key, str)
        assert cache_key.startswith('synthetic_test_dataset_')
    
    def test_load_cached_dataset_function(self):
        """Test load_cached_dataset convenience function."""
        data = {'X': np.random.random((10, 2)), 'y': np.random.randint(0, 2, (10, 1))}
        params = {'param1': 'value1'}
        
        # Cache data first
        cache_dataset(data, 'synthetic', 'test_dataset', params)
        
        # Load cached data
        loaded_data = load_cached_dataset('synthetic', 'test_dataset', params)
        
        assert loaded_data is not None
        np.testing.assert_array_equal(loaded_data['X'], data['X'])
        np.testing.assert_array_equal(loaded_data['y'], data['y'])
    
    def test_get_cache_stats_function(self):
        """Test get_cache_stats convenience function."""
        stats = get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'total_entries' in stats
        assert 'total_size_mb' in stats
    
    def test_cleanup_cache_function(self):
        """Test cleanup_cache convenience function."""
        # Add some data
        data = {'X': np.array([[1, 2]]), 'y': np.array([[0]])}
        cache_dataset(data, 'synthetic', 'cleanup_test')
        
        # Run cleanup
        result = cleanup_cache(force=True)
        
        assert isinstance(result, dict)
        assert 'total_checked' in result


class TestErrorHandling:
    """Test error handling in cache operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = DatasetCache(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_category(self):
        """Test handling of invalid cache category."""
        data = {'X': np.array([[1, 2]]), 'y': np.array([[0]])}
        
        with pytest.raises(ValueError):
            self.cache.save_to_cache(data, 'invalid_category', 'test_key')
    
    def test_load_nonexistent_cache(self):
        """Test loading non-existent cache entry."""
        result = self.cache.load_from_cache('synthetic', 'nonexistent_key')
        assert result is None
    
    def test_corrupted_cache_handling(self):
        """Test handling of corrupted cache files."""
        # Create a corrupted cache file
        cache_path = self.cache.get_cache_path('synthetic', 'corrupted_test')
        
        # Write invalid data
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(b'corrupted data')
        
        # Should handle gracefully
        result = self.cache.load_from_cache('synthetic', 'corrupted_test')
        assert result is None
        
        # Corrupted file should be removed
        assert not cache_path.exists()


if __name__ == "__main__":
    pytest.main([__file__]) 