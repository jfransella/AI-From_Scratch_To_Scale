"""
Shared utilities package for AI From Scratch to Scale project.

This package provides foundational functionality used across all models:
- Exception handling
- Logging setup
- Random seed management  
- Device management
- General utilities
"""

from .exceptions import (
    AIFromScratchError,
    ModelError,
    DataError,
    ConfigError,
    TrainingError
)
from .logging import setup_logging, get_logger
from .seeds import set_random_seed, get_random_seed
from .device import setup_device, get_device_info
from .general import ensure_dir, save_json, load_json, format_time

__version__ = "1.0.0"
__all__ = [
    # Exceptions
    "AIFromScratchError",
    "ModelError", 
    "DataError",
    "ConfigError",
    "TrainingError",
    # Logging
    "setup_logging",
    "get_logger",
    # Seeds
    "set_random_seed",
    "get_random_seed", 
    # Device
    "setup_device",
    "get_device_info",
    # General
    "ensure_dir",
    "save_json",
    "load_json",
    "format_time"
] 