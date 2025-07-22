#!/usr/bin/env python3
"""Debug script for testing perceptron wandb integration."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_torch():
    """Test torch availability."""
    print("🔍 Testing PyTorch...")
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")

        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0])
        print(f"✅ Basic tensor operations work: {x}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False


def test_wandb():
    """Test wandb availability."""
    print("\n🔍 Testing Wandb...")
    try:
        import wandb

        print(f"✅ Wandb available")
        return True
    except Exception as e:
        print(f"❌ Wandb error: {e}")
        return False


def test_perceptron_basic():
    """Test basic perceptron functionality."""
    print("\n🔍 Testing Perceptron Basic...")
    try:
        # Set up path for local imports
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))

        from src.model import Perceptron

        model = Perceptron(input_size=2)
        print(f"✅ Perceptron created: {model}")

        # Check if it has BaseModel methods
        if hasattr(model, "init_wandb"):
            print("✅ Wandb methods available via BaseModel")
            return True
        else:
            print("❌ No wandb methods found")
            return False

    except Exception as e:
        print(f"❌ Perceptron error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_training():
    """Test simple perceptron training."""
    print("\n🔍 Testing Simple Training...")
    try:
        import torch
        from src.model import Perceptron

        # Simple linearly separable data
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([0, 1], dtype=torch.float32)

        model = Perceptron(input_size=2, max_epochs=10)

        # Test historical training (no gradients)
        if hasattr(model, "fit_historical"):
            print("🏛️ Testing historical training...")
            results = model.fit_historical(X, y, verbose=False)
            print(f"✅ Historical training: accuracy={results['final_accuracy']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_wandb_integration():
    """Test wandb integration with perceptron."""
    print("\n🔍 Testing Wandb Integration...")
    try:
        import torch
        from src.model import Perceptron

        model = Perceptron(input_size=2)

        # Test wandb initialization
        if hasattr(model, "init_wandb"):
            success = model.init_wandb(
                project="ai-from-scratch-perceptron-test",
                name="debug-test",
                tags=["debug", "test"],
                mode="disabled",  # Disable actual wandb for testing
            )
            print(f"✅ Wandb init test: {success}")

            # Test other wandb methods
            if hasattr(model, "log_metrics"):
                print("✅ log_metrics method available")
            if hasattr(model, "finish_wandb"):
                print("✅ finish_wandb method available")

            return True
        else:
            print("❌ No wandb methods available")
            return False

    except Exception as e:
        print(f"❌ Wandb integration error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all debug tests."""
    print("🧪 PERCEPTRON WANDB DEBUG TESTS")
    print("=" * 50)

    tests = [
        ("PyTorch", test_torch),
        ("Wandb", test_wandb),
        ("Perceptron Basic", test_perceptron_basic),
        ("Simple Training", test_simple_training),
        ("Wandb Integration", test_wandb_integration),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False

    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")

    all_passed = all(results.values())
    print(
        f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
