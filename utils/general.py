"""
General utility functions for AI From Scratch to Scale project.

Provides common utilities for file operations, JSON handling,
time formatting, and other general-purpose functions.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to create

    Returns:
        Path object for the directory

    Example:
        ensure_dir("outputs/logs")
        ensure_dir(Path("models/checkpoints"))
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Dict[str, Any], filepath: Union[str, Path],
              indent: int = 2, ensure_directory: bool = True) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save the file
        indent: JSON indentation for readability
        ensure_directory: Whether to create directory if it doesn't exist

    Example:
        save_json({"epoch": 10, "loss": 0.25}, "outputs/results.json")
    """
    filepath = Path(filepath)

    if ensure_directory:
        ensure_dir(filepath.parent)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary loaded from the file

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON

    Example:
        data = load_json("outputs/results.json")
        epoch = data["epoch"]
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Invalid JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def format_time(seconds: float, precision: int = 2) -> str:
    """
    Format time duration in a human-readable format.

    Args:
        seconds: Time duration in seconds
        precision: Decimal precision for seconds

    Returns:
        Formatted time string

    Example:
        format_time(125.5)  # "2m 5.50s"
        format_time(3661)   # "1h 1m 1.00s"
    """
    if seconds < 60:
        return f"{seconds:.{precision}f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.{precision}f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds:.{precision}f}s"


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        filepath: Path to the file

    Returns:
        File size in bytes

    Example:
        size = get_file_size("model.pth")
        print(f"Model size: {size / (1024**2):.1f} MB")
    """
    return Path(filepath).stat().st_size


def format_bytes(size_bytes: int, precision: int = 1) -> str:
    """
    Format byte size in human-readable format.

    Args:
        size_bytes: Size in bytes
        precision: Decimal precision

    Returns:
        Formatted size string

    Example:
        format_bytes(1536)      # "1.5 KB"
        format_bytes(1048576)   # "1.0 MB"
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.{precision}f} {units[unit_index]}"


def safe_filename(filename: str, replacement: str = "_") -> str:
    """
    Convert a string to a safe filename by replacing invalid characters.

    Args:
        filename: Original filename
        replacement: Character to replace invalid characters

    Returns:
        Safe filename string

    Example:
        safe_filename("model: experiment_1")  # "model_ experiment_1"
    """
    # Characters not allowed in Windows filenames
    invalid_chars = '<>:"/\\|?*'

    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, replacement)

    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')

    # Ensure not empty
    if not safe_name:
        safe_name = "untitled"

    return safe_name


def create_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Create a timestamp string.

    Args:
        format_string: Strftime format string

    Returns:
        Formatted timestamp string

    Example:
        timestamp = create_timestamp()  # "20240315_143022"
        timestamp = create_timestamp("%Y-%m-%d")  # "2024-03-15"
    """
    return time.strftime(format_string)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Searches upward from current directory to find the project root
    (directory containing the workspace file or README).

    Returns:
        Path to project root

    Example:
        root = get_project_root()
        models_dir = root / "models"
    """
    current = Path.cwd()

    # Look for workspace file or README
    markers = [
        "ai-from-scratch-to-scale.code-workspace",
        "README.md",
        ".git"
    ]

    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    # Fallback to current directory
    return Path.cwd()


def count_files(directory: Union[str, Path], pattern: str = "*",
                recursive: bool = True) -> int:
    """
    Count files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively

    Returns:
        Number of matching files

    Example:
        python_files = count_files("src", "*.py")
        all_files = count_files("outputs", recursive=True)
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        return 0

    if recursive:
        return len(list(dir_path.rglob(pattern)))
    else:
        return len(list(dir_path.glob(pattern)))


def cleanup_old_files(directory: Union[str, Path], pattern: str = "*",
                      max_age_days: int = 30, dry_run: bool = False) -> int:
    """
    Clean up old files in a directory.

    Args:
        directory: Directory to clean
        pattern: Glob pattern to match
        max_age_days: Maximum age in days
        dry_run: If True, only count files without deleting

    Returns:
        Number of files cleaned/would be cleaned

    Example:
        # Clean log files older than 7 days
        cleaned = cleanup_old_files("outputs/logs", "*.log", max_age_days=7)
    """
    dir_path = Path(directory)
    logger = logging.getLogger(__name__)

    if not dir_path.exists():
        return 0

    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    cleaned_count = 0

    for file_path in dir_path.rglob(pattern):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            if dry_run:
                logger.info(f"Would delete: {file_path}")
            else:
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
                    continue
            cleaned_count += 1

    if dry_run:
        logger.info(f"Would clean {cleaned_count} files from {directory}")
    else:
        logger.info(f"Cleaned {cleaned_count} files from {directory}")

    return cleaned_count


def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration

    Example:
        base = {"lr": 0.01, "batch_size": 32}
        override = {"lr": 0.001}
        merged = merge_configs(base, override)  # {"lr": 0.001, "batch_size": 32}
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def validate_config_keys(config: Dict[str, Any],
                         required_keys: list,
                         optional_keys: Optional[list] = None) -> None:
    """
    Validate that a configuration has required keys.

    Args:
        config: Configuration dictionary
        required_keys: Keys that must be present
        optional_keys: Keys that are allowed but not required

    Raises:
        ValueError: If required keys are missing or unknown keys are present

    Example:
        validate_config_keys(config, ["model_name", "learning_rate"])
    """
    from .exceptions import ConfigError

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigError(f"Missing required configuration keys: {missing_keys}")

    if optional_keys is not None:
        allowed_keys = set(required_keys + optional_keys)
        unknown_keys = [key for key in config.keys() if key not in allowed_keys]
        if unknown_keys:
            raise ConfigError(f"Unknown configuration keys: {unknown_keys}")


# Context manager for timing operations
class Timer:
    """
    Context manager for timing operations.

    Example:
        with Timer() as timer:
            # Do some work
            time.sleep(1)
        print(f"Operation took {timer.elapsed:.2f} seconds")
    """

    def __init__(self, logger: Optional[logging.Logger] = None,
                 message: Optional[str] = None):
        self.logger = logger
        self.message = message
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time

        if self.logger and self.message:
            self.logger.info(f"{self.message}: {format_time(self.elapsed)}")
        elif self.logger:
            self.logger.info(f"Operation completed in {format_time(self.elapsed)}")


# Progress tracking utility
class ProgressTracker:
    """
    Simple progress tracking utility.

    Example:
        tracker = ProgressTracker(total=100, logger=logger)
        for i in range(100):
            # Do work
            tracker.update(1)
    """

    def __init__(self, total: int, logger: Optional[logging.Logger] = None,
                 update_interval: int = 10):
        self.total = total
        self.current = 0
        self.logger = logger or logging.getLogger(__name__)
        self.update_interval = update_interval
        self.start_time = time.time()

    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current += increment

        if self.current % self.update_interval == 0 or self.current == self.total:
            self._log_progress()

    def _log_progress(self):
        """Log current progress."""
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time

        if self.current > 0:
            estimated_total = elapsed * (self.total / self.current)
            remaining = estimated_total - elapsed

            self.logger.info(
                f"Progress: {self.current}/{self.total} ({percentage:.1f}%) - "
                f"Elapsed: {format_time(elapsed)} - "
                f"Remaining: {format_time(remaining)}"
            )
        else:
            self.logger.info(f"Progress: {self.current}/{self.total} ({percentage:.1f}%)")

    def finish(self):
        """Mark progress as finished."""
        self.current = self.total
        elapsed = time.time() - self.start_time
        self.logger.info(f"Completed {self.total} items in {format_time(elapsed)}")
