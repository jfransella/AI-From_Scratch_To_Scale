"""
Setup Python path for shared packages.

This module ensures that the shared packages (utils, engine, data_utils, plotting)
can be imported properly in the 01_perceptron model.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path so we can import shared packages
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify imports work
try:
    import utils
    import engine
    import data_utils
    import plotting
    print("✅ All shared packages available for import")
except ImportError as e:
    print(f"❌ Import error: {e}")
