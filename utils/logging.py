"""
Logging utilities for AI From Scratch to Scale project.

Provides structured logging setup for both console and file output,
supporting the dual logging system (Python logging + wandb).
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up structured logging for the project.
    
    Creates both console and file handlers with appropriate formatting.
    This supports the "narrative log" part of the dual logging system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path (overrides log_dir)
        log_dir: Directory for log files (creates timestamped file)
        format_string: Custom format string (uses default if None)
        include_timestamp: Whether to include timestamps in console output
        console_output: Whether to log to console
        file_output: Whether to log to file
    
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_logging(level="DEBUG", log_dir="outputs/logs")
        logger.info("Training started")
    """
    try:
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create logger
        logger = logging.getLogger("ai_from_scratch")
        logger.setLevel(_get_log_level(level))
        
        # Create formatters
        if format_string is None:
            if include_timestamp:
                format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            else:
                format_string = "%(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(
            format_string,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(_get_log_level(level))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            log_path = _get_log_file_path(log_file, log_dir)
            if log_path:
                # Ensure log directory exists
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
                file_handler.setLevel(_get_log_level(level))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"Logging to file: {log_path}")
        
        # Log setup completion
        logger.info(f"Logging initialized - Level: {level}")
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("ai_from_scratch")
        logger.error(f"Failed to setup advanced logging: {e}")
        logger.info("Using basic logging configuration")
        return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module or component.
    
    Args:
        name: Logger name (uses calling module if None)
    
    Returns:
        Logger instance
    
    Example:
        logger = get_logger("model.perceptron")
        logger.debug("Forward pass completed")
    """
    if name is None:
        name = "ai_from_scratch"
    
    return logging.getLogger(name)


def log_experiment_start(logger: logging.Logger, config: dict):
    """
    Log the start of an experiment with key configuration details.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    
    Example:
        log_experiment_start(logger, config)
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT STARTED")
    logger.info("=" * 60)
    
    # Log key configuration
    key_params = [
        'model_name', 'experiment', 'dataset', 'learning_rate', 
        'batch_size', 'epochs', 'seed', 'device'
    ]
    
    for param in key_params:
        if param in config:
            logger.info(f"{param.replace('_', ' ').title()}: {config[param]}")
    
    logger.info("-" * 60)


def log_experiment_end(logger: logging.Logger, results: dict):
    """
    Log the end of an experiment with final results.
    
    Args:
        logger: Logger instance
        results: Results dictionary
    
    Example:
        log_experiment_end(logger, {"final_loss": 0.045, "accuracy": 0.95})
    """
    logger.info("-" * 60)
    logger.info("EXPERIMENT COMPLETED")
    
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
    
    logger.info("=" * 60)


def log_epoch_progress(logger: logging.Logger, epoch: int, total_epochs: int,
                       train_loss: float, val_loss: Optional[float] = None,
                       metrics: Optional[dict] = None):
    """
    Log progress for a training epoch.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss (optional)
        metrics: Additional metrics dictionary (optional)
    
    Example:
        log_epoch_progress(logger, 10, 100, 0.25, 0.30, {"accuracy": 0.85})
    """
    progress = f"Epoch {epoch:3d}/{total_epochs}"
    loss_info = f"Train Loss: {train_loss:.4f}"
    
    if val_loss is not None:
        loss_info += f", Val Loss: {val_loss:.4f}"
    
    if metrics:
        metric_strs = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                      for k, v in metrics.items()]
        loss_info += f", {', '.join(metric_strs)}"
    
    logger.info(f"{progress} - {loss_info}")


def _get_log_level(level: Union[str, int]) -> int:
    """Convert string level to logging constant."""
    if isinstance(level, int):
        return level
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    return level_map.get(level.upper(), logging.INFO)


def _get_log_file_path(log_file: Optional[Union[str, Path]], 
                      log_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """Determine the log file path."""
    if log_file:
        return Path(log_file)
    
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_{timestamp}.log"
        return Path(log_dir) / filename
    
    return None


# Utility decorators for logging
def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls.
    
    Example:
        @log_function_call()
        def train_model():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            func_logger.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Completed {func.__name__}")
                return result
            except Exception as e:
                func_logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Example:
        @log_execution_time()
        def train_epoch():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            func_logger = logger or get_logger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
                raise
        return wrapper
    return decorator 