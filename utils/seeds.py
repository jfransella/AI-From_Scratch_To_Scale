"""
Random seed management utilities for reproducible experiments.

Provides unified seed setting across NumPy, PyTorch, and Python's random module
to ensure consistent results across different runs of the same experiment.
"""

import logging
import os
import random
from typing import Optional

import numpy as np
_NUMPY_AVAILABLE = True

# Handle torch imports gracefully
try:
    import torch
    if hasattr(torch, '__version__') and hasattr(torch, 'manual_seed'):
        _TORCH_AVAILABLE = True
    else:
        # torch exists but is broken
        _TORCH_AVAILABLE = False
        torch = None
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


def set_random_seed(seed: int, deterministic: bool = True) -> int:
    """
    Set random seed for all available random number generators.

    Sets seeds for:
    - Python's built-in random module
    - NumPy (if available)
    - PyTorch (if available)
    - Environment variables for deterministic behavior

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic behavior (slower but reproducible)

    Returns:
        The seed that was set

    Example:
        set_random_seed(42)  # Ensures reproducible results
    """
    logger = logging.getLogger(__name__)

    # Python random
    random.seed(seed)
    logger.debug(f"Set Python random seed: {seed}")

    # NumPy
    if _NUMPY_AVAILABLE:
        np.random.seed(seed)
        logger.debug(f"Set NumPy random seed: {seed}")

    # PyTorch
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        logger.debug(f"Set PyTorch random seed: {seed}")

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            logger.debug(f"Set PyTorch CUDA random seed: {seed}")

        if deterministic:
            # Enable deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("Enabled PyTorch deterministic mode")

    # Environment variables for additional determinism
    if deterministic:
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.debug(f"Set PYTHONHASHSEED: {seed}")

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")
    return seed


def get_random_seed() -> Optional[int]:
    """
    Get a random seed value from the system.

    Uses the system's random number generator to create a seed value
    suitable for use with set_random_seed().

    Returns:
        Random seed value

    Example:
        seed = get_random_seed()
        set_random_seed(seed)
    """
    # Use system random to generate a seed
    import time
    seed = int(time.time() * 1000000) % (2**32)
    return seed


def create_random_state(seed: int):
    """
    Create random state objects for different libraries.

    Args:
        seed: Random seed value

    Returns:
        Dictionary containing random state objects

    Example:
        states = create_random_state(42)
        rng = states['numpy']  # NumPy random state
    """
    states = {}

    # Python random state
    py_random = random.Random(seed)
    states['python'] = py_random

    # NumPy random state
    if _NUMPY_AVAILABLE:
        np_random = np.random.RandomState(seed)
        states['numpy'] = np_random

    # PyTorch generator
    if _TORCH_AVAILABLE:
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        states['torch'] = torch_gen

    return states


def verify_seed_consistency(seed: int, iterations: int = 10) -> bool:
    """
    Verify that random seed setting produces consistent results.

    Runs a test to ensure that setting the same seed produces
    identical random sequences across multiple runs.

    Args:
        seed: Seed to test
        iterations: Number of test iterations

    Returns:
        True if seed produces consistent results

    Example:
        is_consistent = verify_seed_consistency(42)
        assert is_consistent, "Random seeding is not working properly"
    """
    logger = logging.getLogger(__name__)

    # Store original sequences
    original_sequences = {}

    # Generate original sequences
    set_random_seed(seed)

    # Python random sequence
    original_sequences['python'] = [random.random() for _ in range(iterations)]

    # NumPy sequence
    if _NUMPY_AVAILABLE:
        original_sequences['numpy'] = [np.random.random() for _ in range(iterations)]

    # PyTorch sequence
    if _TORCH_AVAILABLE:
        original_sequences['torch'] = [torch.rand(1).item() for _ in range(iterations)]

    # Test consistency by regenerating
    set_random_seed(seed)

    # Check Python random
    python_seq = [random.random() for _ in range(iterations)]
    if python_seq != original_sequences['python']:
        logger.error("Python random seed inconsistency detected")
        return False

    # Check NumPy
    if _NUMPY_AVAILABLE:
        numpy_seq = [np.random.random() for _ in range(iterations)]
        if not all(abs(a - b) < 1e-10 for a, b in zip(numpy_seq, original_sequences['numpy'])):
            logger.error("NumPy random seed inconsistency detected")
            return False

    # Check PyTorch
    if _TORCH_AVAILABLE:
        torch_seq = [torch.rand(1).item() for _ in range(iterations)]
        if not all(abs(a - b) < 1e-10 for a, b in zip(torch_seq, original_sequences['torch'])):
            logger.error("PyTorch random seed inconsistency detected")
            return False

    logger.info(f"Random seed {seed} verified as consistent")
    return True


def get_library_versions() -> dict:
    """
    Get version information for random number generation libraries.

    Returns:
        Dictionary with library versions

    Example:
        versions = get_library_versions()
        print(f"Using PyTorch version: {versions['torch']}")
    """
    versions = {}

    if _NUMPY_AVAILABLE:
        versions['numpy'] = np.__version__
    else:
        versions['numpy'] = "Not available"

    if _TORCH_AVAILABLE:
        versions['torch'] = torch.__version__
    else:
        versions['torch'] = "Not available"

    return versions


def log_seed_info(logger: logging.Logger, seed: int):
    """
    Log comprehensive seed information for debugging.

    Args:
        logger: Logger instance
        seed: Current seed value

    Example:
        log_seed_info(logger, 42)
    """
    logger.info("Random Seed Configuration:")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")

    versions = get_library_versions()
    for lib, version in versions.items():
        logger.info(f"  {lib.title()}: {version}")

    if _TORCH_AVAILABLE:
        logger.info(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  PyTorch deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"  PyTorch benchmark: {torch.backends.cudnn.benchmark}")


# Seed management context manager
class SeedContext:
    """
    Context manager for temporary seed changes.

    Example:
        with SeedContext(42):
            # Use seed 42
            result = some_random_operation()
        # Original seed restored
    """

    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_states = {}

    def __enter__(self):
        # Save current states
        self.original_states['python'] = random.getstate()

        if _NUMPY_AVAILABLE:
            self.original_states['numpy'] = np.random.get_state()

        if _TORCH_AVAILABLE:
            self.original_states['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.original_states['torch_cuda'] = torch.cuda.get_rng_state()

        # Set new seed
        set_random_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        random.setstate(self.original_states['python'])

        if _NUMPY_AVAILABLE and 'numpy' in self.original_states:
            np.random.set_state(self.original_states['numpy'])

        if _TORCH_AVAILABLE and 'torch' in self.original_states:
            torch.set_rng_state(self.original_states['torch'])
            if torch.cuda.is_available() and 'torch_cuda' in self.original_states:
                torch.cuda.set_rng_state(self.original_states['torch_cuda'])
