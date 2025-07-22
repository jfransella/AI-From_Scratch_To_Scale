"""
Standard setup_path.py template for AI From Scratch to Scale models.

This module ensures that shared packages (utils, engine, data_utils, plotting)
can be imported properly in any model directory. Copy this file to each model
directory to fix editable installation issues.

Usage:
    # At the top of any script in a model directory:
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
    """Add the project root to Python path for shared package access."""
    # Get the project root (3 levels up from model/src or 2 levels up from model)
    current_file = Path(__file__).resolve()
    
    # Try to find project root by looking for characteristic files
    project_root = None
    for parent in current_file.parents:
        if (parent / "utils" / "__init__.py").exists() and \
           (parent / "engine" / "__init__.py").exists() and \
           (parent / "data_utils" / "__init__.py").exists() and \
           (parent / "plotting" / "__init__.py").exists():
            project_root = parent
            break
    
    if project_root is None:
        # Fallback: assume standard directory structure
        project_root = current_file.parent.parent.parent
    
    # Add to Python path if not already there
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

# Export useful functions
__all__ = ['setup_shared_packages_path', 'verify_imports', 'project_root']
