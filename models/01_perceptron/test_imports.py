#!/usr/bin/env python3
"""
Test file to verify imports work correctly.
This helps VS Code detect the correct Python environment.
"""

# Test all the problematic imports
from utils import setup_logging, get_logger
from data_utils import load_dataset
from engine.evaluator import Evaluator
from plotting import plot_confusion_matrix

print("✅ All imports successful!")
print("If you can see this without errors, the environment is working correctly.")

# Show Python info
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}") 