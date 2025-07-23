"""
Enhanced caching strategy for AI From Scratch to Scale project.

This module implements advanced caching with timestamp validation, cache management,
and performance optimization following the project strategy specification.
"""

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils import get_logger
from utils.exceptions import DataError


# Cache configuration
CACHE_CONFIG = {
    "base_dir": Path("data") / "cache",
    "max_cache_size_gb": 5.0,  # Maximum cache size in GB
    "default_ttl_hours": 24 * 7,  # Default TTL: 1 week
    "cleanup_frequency_hours": 24,  # Cleanup every 24 hours
    "compression": True,  # Use compressed storage
    "metadata_filename": ".cache_metadata.json",
    "lock_filename": ".cache_lock",
}

# Cache categories with specific configurations
CACHE_CATEGORIES = {
    "synthetic": {
        "subdir": "synthetic",
        "ttl_hours": 24 * 30,  # 30 days for synthetic data
        "compression": True,
        "auto_cleanup": True,
    },
    "real": {
        "subdir": "real",
        "ttl_hours": 24 * 90,  # 90 days for real data
        "compression": True,
        "auto_cleanup": False,  # Don't auto-cleanup real data
    },
    "processed": {
        "subdir": "processed",
        "ttl_hours": 24 * 14,  # 14 days for processed data
        "compression": True,
        "auto_cleanup": True,
    },
    "models": {
        "subdir": "models",
        "ttl_hours": 24 * 180,  # 180 days for model checkpoints
        "compression": False,  # Don't compress models
        "auto_cleanup": False,
    },
}


class CacheEntry:
    """
    Represents a single cache entry with metadata.
    """

    def __init__(
        self,
        cache_path: Path,
        category: str,
        key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize cache entry.

        Args:
            cache_path: Path to cached file
            category: Cache category ('synthetic', 'real', etc.)
            key: Unique cache key
            metadata: Optional metadata dict
        """
        self.cache_path = cache_path
        self.category = category
        self.key = key
        self.metadata = metadata or {}

        # Auto-detect metadata if file exists
        if self.cache_path.exists() and not self.metadata:
            self.metadata = self._detect_metadata()

    def _detect_metadata(self) -> Dict[str, Any]:
        """Detect metadata from existing cache file."""
        try:
            stat = self.cache_path.stat()
            return {
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_bytes": stat.st_size,
                "checksum": self._calculate_checksum(),
            }
        except Exception:
            return {}

    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of cache file."""
        try:
            hasher = hashlib.sha256()
            with open(self.cache_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def is_valid(self, ttl_hours: Optional[float] = None) -> bool:
        """Check if cache entry is still valid."""
        if not self.cache_path.exists():
            return False

        # Check TTL if specified
        if ttl_hours is not None:
            try:
                stat = self.cache_path.stat()
                age_hours = (time.time() - stat.st_mtime) / 3600
                if age_hours > ttl_hours:
                    return False
            except Exception:
                return False

        return True

    def get_size_mb(self) -> float:
        """Get cache entry size in MB."""
        try:
            return self.cache_path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0

    def get_age_hours(self) -> float:
        """Get cache entry age in hours."""
        try:
            stat = self.cache_path.stat()
            return (time.time() - stat.st_mtime) / 3600
        except Exception:
            return float("inf")


class DatasetCache:
    """
    Advanced dataset caching system with timestamp validation and management.

    This class provides comprehensive caching functionality following the project
    strategy specification for optimal performance and storage management.
    """

    def __init__(self, base_dir: Optional[Path] = None, config: Optional[Dict] = None):
        """
        Initialize dataset cache.

        Args:
            base_dir: Base cache directory (optional)
            config: Cache configuration overrides (optional)
        """
        self.config = CACHE_CONFIG.copy()
        if config:
            self.config.update(config)

        if base_dir:
            self.config["base_dir"] = Path(base_dir)

        self.base_dir = Path(str(self.config["base_dir"]))
        self.logger = get_logger(__name__)

        # Initialize cache directory structure
        self._initialize_cache_structure()

        # Load cache metadata
        self.metadata = self._load_cache_metadata()

        # Schedule cleanup if needed
        self._check_cleanup_schedule()

    def _initialize_cache_structure(self) -> None:
        """Initialize cache directory structure."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)

            # Create category subdirectories
            for category, category_config in CACHE_CATEGORIES.items():
                subdir = self.base_dir / str(category_config["subdir"])
                subdir.mkdir(parents=True, exist_ok=True)

            self.logger.debug("Initialized cache structure at %s", self.base_dir)

        except Exception as e:
            self.logger.error("Failed to initialize cache structure: %s", e)
            raise DataError(f"Cache initialization failed: {e}") from e

    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        metadata_path = self.base_dir / str(self.config["metadata_filename"])

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self.logger.debug("Loaded cache metadata")
                return metadata
            except Exception as e:
                self.logger.warning("Failed to load cache metadata: %s", e)

        # Create default metadata
        metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "last_cleanup": None,
            "total_entries": 0,
            "total_size_mb": 0.0,
            "categories": {},
        }

        self._save_cache_metadata(metadata)
        return metadata

    def _save_cache_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save cache metadata to disk."""
        metadata_path = self.base_dir / str(self.config["metadata_filename"])

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.debug("Saved cache metadata")
        except Exception as e:
            self.logger.warning("Failed to save cache metadata: %s", e)

    def generate_cache_key(
        self, category: str, identifier: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate unique cache key for dataset.

        Args:
            category: Cache category
            identifier: Dataset identifier
            params: Optional parameters for key generation

        Returns:
            Unique cache key
        """
        # Create parameter string
        if params:
            param_str = json.dumps(params, sort_keys=True, default=str)
        else:
            param_str = ""

        # Combine all components
        key_components = [category, identifier, param_str]
        key_string = "|".join(key_components)

        # Generate hash
        hasher = hashlib.sha256()
        hasher.update(key_string.encode("utf-8"))
        cache_key = hasher.hexdigest()[:16]  # Use first 16 chars

        return f"{category}_{identifier}_{cache_key}"

    def get_cache_path(
        self, category: str, cache_key: str, create_dirs: bool = True
    ) -> Path:
        """
        Get cache file path for given category and key.

        Args:
            category: Cache category
            cache_key: Unique cache key
            create_dirs: Whether to create parent directories

        Returns:
            Path to cache file
        """
        if category not in CACHE_CATEGORIES:
            raise ValueError(f"Unknown cache category: {category}")

        category_config = CACHE_CATEGORIES[category]
        cache_dir = self.base_dir / str(category_config["subdir"])

        if create_dirs:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Add appropriate file extension
        if category_config["compression"]:
            cache_path = cache_dir / f"{cache_key}.npz"
        else:
            cache_path = cache_dir / f"{cache_key}.pkl"

        return cache_path

    def save_to_cache(
        self,
        data: Dict[str, np.ndarray],
        category: str,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """
        Save data to cache with metadata.

        Args:
            data: Data dictionary to cache
            category: Cache category
            cache_key: Unique cache key
            metadata: Optional metadata

        Returns:
            CacheEntry for saved data
        """
        cache_path = self.get_cache_path(category, cache_key)
        category_config = CACHE_CATEGORIES[category]

        try:
            # Save data with appropriate format
            if category_config["compression"]:
                np.savez_compressed(cache_path, **data)
            else:
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)

            # Create cache entry
            entry_metadata = metadata or {}
            entry_metadata.update(
                {
                    "cached_at": datetime.now().isoformat(),
                    "category": category,
                    "cache_key": cache_key,
                    "compression": category_config["compression"],
                }
            )

            cache_entry = CacheEntry(cache_path, category, cache_key, entry_metadata)

            # Update global metadata
            self._update_cache_metadata(cache_entry, "add")

            self.logger.debug(
                "Cached data to %s (%.2f MB)", cache_path, cache_entry.get_size_mb()
            )

            return cache_entry

        except Exception as e:
            self.logger.error("Failed to save to cache: %s", e)
            if cache_path.exists():
                cache_path.unlink()  # Clean up partial file
            raise DataError(f"Cache save failed: {e}") from e

    def load_from_cache(
        self, category: str, cache_key: str, validate_ttl: bool = True
    ) -> Optional[Tuple[Dict[str, np.ndarray], CacheEntry]]:
        """
        Load data from cache with validation.

        Args:
            category: Cache category
            cache_key: Unique cache key
            validate_ttl: Whether to validate TTL

        Returns:
            Tuple of (data, cache_entry) or None if not found/invalid
        """
        cache_path = self.get_cache_path(category, cache_key, create_dirs=False)

        if not cache_path.exists():
            return None

        # Create cache entry and validate
        cache_entry = CacheEntry(cache_path, category, cache_key)

        if validate_ttl:
            category_config = CACHE_CATEGORIES[category]
            if not cache_entry.is_valid(float(category_config["ttl_hours"])):
                self.logger.debug("Cache entry expired: %s", cache_key)
                return None

        try:
            # Load data with appropriate format
            category_config = CACHE_CATEGORIES[category]

            if category_config["compression"]:
                loaded_data = np.load(cache_path)
                data = {key: loaded_data[key] for key in loaded_data.files}
            else:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)

            self.logger.debug(
                "Loaded from cache: %s (%.2f MB)", cache_key, cache_entry.get_size_mb()
            )

            return data, cache_entry

        except Exception as e:
            self.logger.warning("Failed to load from cache %s: %s", cache_key, e)
            # Remove corrupted cache file
            try:
                cache_path.unlink()
                self._update_cache_metadata(cache_entry, "remove")
            except Exception:
                pass
            return None

    def _update_cache_metadata(self, cache_entry: CacheEntry, operation: str) -> None:
        """Update global cache metadata."""
        try:
            if operation == "add":
                self.metadata["total_entries"] += 1
                self.metadata["total_size_mb"] += cache_entry.get_size_mb()

                # Update category stats
                if cache_entry.category not in self.metadata["categories"]:
                    self.metadata["categories"][cache_entry.category] = {
                        "count": 0,
                        "size_mb": 0.0,
                    }

                cat_stats = self.metadata["categories"][cache_entry.category]
                cat_stats["count"] += 1
                cat_stats["size_mb"] += cache_entry.get_size_mb()

            elif operation == "remove":
                self.metadata["total_entries"] = max(
                    0, self.metadata["total_entries"] - 1
                )
                self.metadata["total_size_mb"] = max(
                    0.0, self.metadata["total_size_mb"] - cache_entry.get_size_mb()
                )

                # Update category stats
                if cache_entry.category in self.metadata["categories"]:
                    cat_stats = self.metadata["categories"][cache_entry.category]
                    cat_stats["count"] = max(0, cat_stats["count"] - 1)
                    cat_stats["size_mb"] = max(
                        0.0, cat_stats["size_mb"] - cache_entry.get_size_mb()
                    )

            self._save_cache_metadata(self.metadata)

        except Exception as e:
            self.logger.warning("Failed to update cache metadata: %s", e)

    def cleanup_expired_entries(self, force: bool = False) -> Dict[str, Any]:
        """
        Clean up expired cache entries.

        Args:
            force: Force cleanup regardless of schedule

        Returns:
            Dictionary with cleanup statistics
        """
        if not force and not self._should_cleanup():
            return {"skipped": True, "reason": "not_scheduled"}

        self.logger.info("Starting cache cleanup")

        stats = {
            "total_checked": 0,
            "expired_removed": 0,
            "corrupted_removed": 0,
            "space_freed_mb": 0.0,
            "categories": {},
        }

        # Clean up each category
        for category, category_config in CACHE_CATEGORIES.items():
            if not category_config["auto_cleanup"]:
                continue

            cat_stats = self._cleanup_category(category, category_config)
            stats["categories"][category] = cat_stats

            # Update totals
            stats["total_checked"] += cat_stats["checked"]
            stats["expired_removed"] += cat_stats["expired"]
            stats["corrupted_removed"] += cat_stats["corrupted"]
            stats["space_freed_mb"] += cat_stats["space_freed_mb"]

        # Update cleanup timestamp
        self.metadata["last_cleanup"] = datetime.now().isoformat()
        self._save_cache_metadata(self.metadata)

        self.logger.info(
            "Cache cleanup completed: %d expired, %d corrupted, %.2f MB freed",
            stats["expired_removed"],
            stats["corrupted_removed"],
            stats["space_freed_mb"],
        )

        return stats

    def _cleanup_category(self, category: str, category_config: Dict) -> Dict[str, Any]:
        """Clean up specific category."""
        cache_dir = self.base_dir / str(category_config["subdir"])

        stats = {"checked": 0, "expired": 0, "corrupted": 0, "space_freed_mb": 0.0}

        if not cache_dir.exists():
            return stats

        # Find cache files
        for cache_file in cache_dir.iterdir():
            if cache_file.is_file() and not cache_file.name.startswith("."):
                stats["checked"] += 1

                try:
                    # Create cache entry
                    cache_key = cache_file.stem
                    cache_entry = CacheEntry(cache_file, category, cache_key)

                    # Check if expired
                    if not cache_entry.is_valid(float(category_config["ttl_hours"])):
                        size_mb = cache_entry.get_size_mb()
                        cache_file.unlink()
                        stats["expired"] += 1
                        stats["space_freed_mb"] += size_mb
                        self.logger.debug("Removed expired cache: %s", cache_key)

                except Exception:
                    # Remove corrupted files
                    try:
                        size_mb = cache_file.stat().st_size / (1024 * 1024)
                        cache_file.unlink()
                        stats["corrupted"] += 1
                        stats["space_freed_mb"] += size_mb
                        self.logger.debug(
                            "Removed corrupted cache: %s", cache_file.name
                        )
                    except Exception:
                        pass

        return stats

    def _should_cleanup(self) -> bool:
        """Check if cleanup should be performed."""
        if not self.metadata.get("last_cleanup"):
            return True

        try:
            last_cleanup = datetime.fromisoformat(self.metadata["last_cleanup"])
            hours_since = (datetime.now() - last_cleanup).total_seconds() / 3600
            return hours_since >= float(self.config["cleanup_frequency_hours"])
        except Exception:
            return True

    def _check_cleanup_schedule(self) -> None:
        """Check and perform scheduled cleanup."""
        if self._should_cleanup():
            try:
                self.cleanup_expired_entries()
            except Exception as e:
                self.logger.warning("Scheduled cleanup failed: %s", e)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Refresh metadata from disk
        current_stats = {"total_entries": 0, "total_size_mb": 0.0, "categories": {}}

        for category, category_config in CACHE_CATEGORIES.items():
            cache_dir = self.base_dir / str(category_config["subdir"])

            cat_stats = {
                "count": 0,
                "size_mb": 0.0,
                "ttl_hours": category_config["ttl_hours"],
                "auto_cleanup": category_config["auto_cleanup"],
            }

            if cache_dir.exists():
                for cache_file in cache_dir.iterdir():
                    if cache_file.is_file() and not cache_file.name.startswith("."):
                        try:
                            size_mb = cache_file.stat().st_size / (1024 * 1024)
                            cat_stats["count"] += 1
                            cat_stats["size_mb"] += size_mb
                        except Exception:
                            pass

            current_stats["categories"][category] = cat_stats
            current_stats["total_entries"] += cat_stats["count"]
            current_stats["total_size_mb"] += cat_stats["size_mb"]

        # Add system info
        current_stats.update(
            {
                "cache_dir": str(self.base_dir),
                "last_cleanup": self.metadata.get("last_cleanup"),
                "max_size_gb": self.config["max_cache_size_gb"],
                "usage_percentage": (float(current_stats["total_size_mb"]) / 1024)
                / float(self.config["max_cache_size_gb"])
                * 100,
            }
        )

        return current_stats

    def clear_cache(
        self, category: Optional[str] = None, confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Clear cache entries.

        Args:
            category: Specific category to clear (optional, clears all if None)
            confirm: Confirmation flag for safety

        Returns:
            Dictionary with clear statistics
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear cache")

        stats = {"categories_cleared": [], "entries_removed": 0, "space_freed_mb": 0.0}

        if category:
            # Clear specific category
            if category not in CACHE_CATEGORIES:
                raise ValueError(f"Unknown category: {category}")

            cache_dir = self.base_dir / str(CACHE_CATEGORIES[category]["subdir"])
            if cache_dir.exists():
                removed_count, space_freed = self._clear_directory(cache_dir)
                stats["categories_cleared"] = [category]
                stats["entries_removed"] = removed_count
                stats["space_freed_mb"] = space_freed
        else:
            # Clear all categories
            for cat_name, cat_config in CACHE_CATEGORIES.items():
                cache_dir = self.base_dir / str(cat_config["subdir"])
                if cache_dir.exists():
                    removed_count, space_freed = self._clear_directory(cache_dir)
                    stats["categories_cleared"].append(cat_name)
                    stats["entries_removed"] += removed_count
                    stats["space_freed_mb"] += space_freed

        # Reset metadata
        self.metadata["total_entries"] = 0
        self.metadata["total_size_mb"] = 0.0
        self.metadata["categories"] = {}
        self._save_cache_metadata(self.metadata)

        self.logger.info(
            "Cache cleared: %d entries, %.2f MB freed",
            stats["entries_removed"],
            stats["space_freed_mb"],
        )

        return stats

    def _clear_directory(self, directory: Path) -> Tuple[int, float]:
        """Clear all files in directory and return stats."""
        removed_count = 0
        space_freed_mb = 0.0

        for file_path in directory.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    file_path.unlink()
                    removed_count += 1
                    space_freed_mb += size_mb
                except Exception:
                    pass

        return removed_count, space_freed_mb


# Global cache instance
_global_cache: Optional[DatasetCache] = None


def get_cache() -> DatasetCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DatasetCache()
    return _global_cache


def cache_dataset(
    data: Dict[str, np.ndarray],
    category: str,
    identifier: str,
    params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Convenience function to cache dataset.

    Args:
        data: Data to cache
        category: Cache category
        identifier: Dataset identifier
        params: Optional parameters
        metadata: Optional metadata

    Returns:
        Cache key for retrieval
    """
    cache = get_cache()
    cache_key = cache.generate_cache_key(category, identifier, params)
    cache.save_to_cache(data, category, cache_key, metadata)
    return cache_key


def load_cached_dataset(
    category: str, identifier: str, params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Convenience function to load cached dataset.

    Args:
        category: Cache category
        identifier: Dataset identifier
        params: Optional parameters

    Returns:
        Cached data or None if not found
    """
    cache = get_cache()
    cache_key = cache.generate_cache_key(category, identifier, params)
    result = cache.load_from_cache(category, cache_key)
    return result[0] if result else None


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache().get_cache_stats()


def cleanup_cache(force: bool = False) -> Dict[str, int]:
    """Clean up expired cache entries."""
    return get_cache().cleanup_expired_entries(force=force)
