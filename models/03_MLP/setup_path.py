"""
Standard setup_path.py for 03_mlp model.

This module ensures that shared packages (utils, engine, data_utils, plotting)
can be imported properly in the MLP model directory. This fixes editable
installation issues by manually adding the project root to Python path.

Usage:
    # At the top of any script in the mlp directory:
    import setup_path  # This fixes the import paths

    # Now you can import shared packages:
    import utils
    import engine
    import data_utils
    import plotting
"""

import sys
from pathlib import Path


def setup_shared_packages_path():
    """Add project root to Python path for shared package imports."""
    # Get the current file's directory (models/03_mlp/)
    current_dir = Path(__file__).parent.absolute()
    
    # Go up two levels to reach project root
    # models/03_mlp/ -> models/ -> project_root/
    project_root = current_dir.parent.parent
    
    # Add project root to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def verify_imports(silent=True):
    """Verify that all shared packages can be imported."""
    try:
        import utils  # noqa: F401
        import engine  # noqa: F401
        import data_utils  # noqa: F401
        import plotting  # noqa: F401
        
        if not silent:
            print("✅ All shared packages available for import")
            print("   - utils: Available")
            print("   - engine: Available")
            print("   - data_utils: Available")
            print("   - plotting: Available")
        
        return True
    except ImportError as e:
        if not silent:
            print(f"❌ Import error: {e}")
        return False


# Automatically setup when imported
project_root = setup_shared_packages_path()

# Verify imports work (silently by default)
verify_imports(silent=True)
