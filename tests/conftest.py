"""
Pytest configuration for AI From Scratch to Scale tests.

This file sets up test fixtures and handles import paths for all tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add models directory to path
models_dir = project_root / "models"
sys.path.insert(0, str(models_dir))


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions and classes"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "smoke: Smoke tests for basic functionality"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that should be run separately"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add default markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# Test fixtures
@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for tests."""
    import tempfile
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_config():
    """Return a mock configuration for testing."""
    return {
        'input_size': 4,
        'learning_rate': 0.1,
        'max_epochs': 10,
        'tolerance': 1e-6,
        'activation': 'step',
        'init_method': 'zeros',
        'random_state': 42
    }


@pytest.fixture(scope="function")
def mock_data():
    """Return mock data for testing."""
    import torch
    return {
        'X': torch.randn(100, 4),
        'y': torch.randint(0, 2, (100,)).float(),
        'X_test': torch.randn(20, 4),
        'y_test': torch.randint(0, 2, (20,)).float()
    } 