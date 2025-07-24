"""
Environment validation script for 01_perceptron.

This script validates that all required packages and shared infrastructure
are properly installed and working.
"""

print("🔍 Validating 01_Perceptron Environment")
print("=" * 50)

# Setup path for shared packages
import setup_path

# Test core ML libraries
print("\n📦 Testing Core ML Libraries:")
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    import torch

    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Pandas: {pd.__version__}")
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")

except ImportError as e:
    print(f"❌ Core ML library import failed: {e}")

# Test shared infrastructure
print("\n🏗️  Testing Shared Infrastructure:")
try:
    import data_utils
    import engine
    import plotting
    import utils

    print(f"✅ Utils: {utils.__version__}")
    print("✅ Engine: Available")
    print("✅ Data Utils: Available")
    print("✅ Plotting: Available")

except ImportError as e:
    print(f"❌ Shared infrastructure import failed: {e}")

# Test basic functionality
print("\n🧪 Testing Basic Functionality:")
try:
    # Test tensor operations
    x = torch.randn(2, 3)
    print(f"✅ PyTorch tensor creation: {x.shape}")

    # Test utils functions
    from utils import set_random_seed, setup_logging

    logger = setup_logging("test")
    set_random_seed(42)
    print("✅ Utils functions working")

    # Test numpy-torch integration
    np_array = np.array([1, 2, 3])
    torch_tensor = torch.from_numpy(np_array)
    print(f"✅ NumPy-PyTorch integration: {torch_tensor}")

except (ImportError, RuntimeError, ValueError) as e:
    print(f"❌ Basic functionality test failed: {e}")

print("\n🎉 Environment validation complete!")
print("=" * 50)
print("✅ Ready to run 01_Perceptron model!")
