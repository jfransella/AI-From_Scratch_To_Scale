#!/usr/bin/env python3
"""
Test file to verify imports work correctly.
This helps VS Code detect the correct Python environment.
"""

# Setup path for shared packages
import setup_path  # pylint: disable=unused-import

from data_utils import load_dataset
from engine.evaluator import Evaluator
from plotting import plot_confusion_matrix

# Test all the problematic imports
from utils import get_logger, setup_logging

print("✅ All imports successful!")
print("If you can see this without errors, the environment is working correctly.")

# Show Python info
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
