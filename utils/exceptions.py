"""
Custom exception hierarchy for AI From Scratch to Scale project.

Provides structured error handling across all models and shared packages.
"""

from typing import Optional


class AIFromScratchError(Exception):
    """
    Base exception for all project-specific errors.
    
    All custom exceptions in this project should inherit from this class
    to enable easy identification and handling of project-specific issues.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the exception message with optional details."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ModelError(AIFromScratchError):
    """
    Model-specific errors (architecture, forward pass, initialization, etc.).
    
    Examples:
    - Invalid input dimensions
    - Model producing NaN values
    - Architecture configuration errors
    - Parameter initialization failures
    """
    pass


class DataError(AIFromScratchError):
    """
    Data loading and preprocessing errors.
    
    Examples:
    - Dataset not found
    - Invalid data format
    - Preprocessing failures
    - Data loader configuration errors
    """
    pass


class ConfigError(AIFromScratchError):
    """
    Configuration validation and loading errors.
    
    Examples:
    - Unknown experiment name
    - Invalid hyperparameter values
    - Missing required configuration keys
    - Configuration file parsing errors
    """
    pass


class TrainingError(AIFromScratchError):
    """
    Training loop and optimization errors.
    
    Examples:
    - Optimizer configuration failures
    - Loss computation errors
    - Gradient explosion/vanishing
    - Checkpoint saving/loading failures
    """
    pass


class DeviceError(AIFromScratchError):
    """
    Device setup and management errors.
    
    Examples:
    - CUDA not available when requested
    - Device memory allocation failures
    - Tensor device mismatch errors
    """
    pass


class PlottingError(AIFromScratchError):
    """
    Visualization and plotting errors.
    
    Examples:
    - Plot generation failures
    - File saving errors
    - Invalid plot configuration
    """
    pass


# Exception handling utilities
def handle_gracefully(exception_class=AIFromScratchError, default_return=None, log_error=True):
    """
    Decorator for graceful error handling.
    
    Args:
        exception_class: Type of exception to catch
        default_return: Value to return if exception occurs
        log_error: Whether to log the error
    
    Example:
        @handle_gracefully(DataError, default_return=[])
        def load_dataset(name):
            # ... implementation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_class as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
            except Exception as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                if isinstance(e, AIFromScratchError):
                    raise
                raise exception_class(f"Unexpected error in {func.__name__}: {e}") from e
        return wrapper
    return decorator


def validate_and_raise(condition: bool, error_class: type, message: str, details: Optional[str] = None):
    """
    Validate a condition and raise an exception if it fails.
    
    Args:
        condition: Boolean condition to check
        error_class: Exception class to raise
        message: Error message
        details: Optional additional details
    
    Example:
        validate_and_raise(
            x.dim() == 2,
            ModelError,
            "Expected 2D input tensor",
            f"Got tensor with shape {x.shape}"
        )
    """
    if not condition:
        raise error_class(message, details) 