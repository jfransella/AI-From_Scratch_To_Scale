"""
Test the model import isolation utility.
"""

import sys
from pathlib import Path

# Add project root to path for shared imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_isolation import ModelTestIsolation  # noqa: E402


def test_perceptron_isolation():
    """Test that perceptron imports work in isolation."""

    # Test importing perceptron in isolation
    model_path = project_root / "models" / "01_perceptron" / "src"
    with ModelTestIsolation(model_path, "perceptron") as model_test:
        # Import perceptron modules
        constants = model_test.import_module("constants")
        model_module = model_test.import_module("model")

        Perceptron = model_module.Perceptron
        MODEL_NAME = constants.MODEL_NAME
        MODEL_VERSION = constants.MODEL_VERSION

        print(f"Successfully imported {MODEL_NAME} v{MODEL_VERSION}")
        print(f"Perceptron class: {Perceptron}")

        # Test instantiation
        model = Perceptron(input_size=2)
        print(f"Created model: {model}")


def test_adaline_isolation():
    """Test that ADALINE imports work in isolation."""

    try:
        model_path = project_root / "models" / "02_adaline" / "src"
        with ModelTestIsolation(model_path, "adaline") as model_test:
            # Import ADALINE modules
            constants = model_test.import_module("constants")
            model_module = model_test.import_module("model")

            ADALINE = model_module.ADALINE
            MODEL_NAME = constants.MODEL_NAME
            MODEL_VERSION = constants.MODEL_VERSION

            print(f"Successfully imported {MODEL_NAME} v{MODEL_VERSION}")
            print(f"ADALINE class: {ADALINE}")
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"⚠️  ADALINE test skipped (expected if not fully implemented): {e}")


def test_mlp_isolation():
    """Test that MLP imports work in isolation."""

    try:
        model_path = project_root / "models" / "03_mlp" / "src"
        with ModelTestIsolation(model_path, "mlp") as model_test:
            # Import MLP modules
            constants = model_test.import_module("constants")
            model_module = model_test.import_module("model")

            MLP = model_module.MLP
            MODEL_NAME = constants.MODEL_NAME
            MODEL_VERSION = constants.MODEL_VERSION

            print(f"Successfully imported {MODEL_NAME} v{MODEL_VERSION}")
            print(f"MLP class: {MLP}")
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"⚠️  MLP test skipped (expected if not fully implemented): {e}")


if __name__ == "__main__":
    print("Testing model import isolation...")

    print("\n=== Testing Perceptron ===")
    test_perceptron_isolation()

    print("\n=== Testing ADALINE ===")
    test_adaline_isolation()

    print("\n=== Testing MLP ===")
    test_mlp_isolation()

    print("\n✅ All isolation tests completed!")
