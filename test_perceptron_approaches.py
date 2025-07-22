#!/usr/bin/env python3
"""
Test script comparing Historical (1957) vs Modern (gradient-based) Perceptron approaches.

This demonstrates the key educational distinction:
- Historical: Error-driven rule updates (no gradients)
- Modern: Gradient-based optimization (with gradients)
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent / "src"))


def create_simple_dataset():
    """Create a simple linearly separable dataset."""
    # Simple 2D linearly separable data
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )

    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    return X, y


def test_historical_approach():
    """Test the original 1957 Perceptron Learning Rule."""
    print("🏛️ HISTORICAL PERCEPTRON (1957) - Rosenblatt's Original Algorithm")
    print("=" * 70)
    print("• No gradients, no backpropagation")
    print("• Error-driven weight updates only")
    print("• Step function activation")
    print("• Rule-based learning (no calculus!)")
    print()

    try:
        from model import Perceptron

        # Create simple dataset
        X, y = create_simple_dataset()
        print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Classes: {torch.unique(y).tolist()}")

        # Create perceptron with historical settings
        model = Perceptron(
            input_size=2,
            learning_rate=0.1,
            max_epochs=50,
            activation="step",  # Historical step function
        )

        print(f"\n🧠 Created Historical Perceptron:")
        print(f"   Input size: {model.input_size}")
        print(f"   Learning rate: {model.learning_rate}")
        print(f"   Activation: {model.activation}")
        print(f"   Algorithm: Original 1957 Rosenblatt rule")

        # Train with historical method
        print(f"\n🚀 Training with Historical Algorithm...")
        results = model.fit_historical(X, y, verbose=True)

        # Test final predictions
        print(f"\n🔬 Final Predictions (Historical):")
        with torch.no_grad():
            outputs = model.linear(X)
            predictions = (outputs >= 0.0).float().squeeze()
            for i, (inp, pred, target) in enumerate(zip(X, predictions, y)):
                correct = "✅" if pred == target else "❌"
                print(
                    f"   {inp.tolist()} → {pred.item():.0f} (target: {target.item():.0f}) {correct}"
                )

        print(f"\n📈 Historical Results:")
        print(f"   Final Accuracy: {results['final_accuracy']:.4f}")
        print(f"   Epochs Trained: {results['epochs_trained']}")
        print(f"   Converged: {results['converged']}")
        print(f"   Total Errors: {results['total_errors']}")
        print(f"   Algorithm: {results['algorithm']}")

        if results["final_accuracy"] >= 0.99:
            print(f"🎯 Perfect! Historical perceptron found linear separator!")

        return True

    except Exception as e:
        print(f"❌ Historical test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_modern_approach():
    """Test modern gradient-based approach."""
    print("\n\n🔧 MODERN PERCEPTRON (Gradient-Based) - Contemporary Implementation")
    print("=" * 70)
    print("• Uses gradients and backpropagation")
    print("• Differentiable loss functions")
    print("• Modern PyTorch optimization")
    print("• Compatible with unified trainer")
    print()

    try:
        from config import get_model_config, get_training_config
        from model import Perceptron

        # Create simple dataset
        X, y = create_simple_dataset()
        print(f"📊 Same Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Get modern configuration
        config = get_model_config("debug_small")

        # Create perceptron for modern training
        model = Perceptron(
            input_size=2,
            learning_rate=0.1,
            max_epochs=50,
            activation="sigmoid",  # Differentiable activation
        )

        print(f"\n🧠 Created Modern Perceptron:")
        print(f"   Input size: {model.input_size}")
        print(f"   Learning rate: {model.learning_rate}")
        print(f"   Activation: {model.activation} (differentiable)")
        print(f"   Algorithm: Gradient-based optimization")

        # Manual gradient-based training (simplified)
        print(f"\n🚀 Training with Gradient-Based Algorithm...")

        optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate)

        for epoch in range(20):  # Fewer epochs for demo
            # Forward pass
            outputs = model.forward(X)
            loss = model.get_loss(outputs, y)

            # Backward pass (gradients!)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                predictions = (torch.sigmoid(outputs) >= 0.5).float().squeeze()
                accuracy = (predictions == y).float().mean().item()

            if epoch % 5 == 0:
                print(
                    f"   Epoch {epoch:2d}: Loss={loss.item():.6f}, Accuracy={accuracy:.4f}"
                )

        # Test final predictions
        print(f"\n🔬 Final Predictions (Modern):")
        with torch.no_grad():
            outputs = model.forward(X)
            predictions = (torch.sigmoid(outputs) >= 0.5).float().squeeze()
            for i, (inp, pred, target) in enumerate(zip(X, predictions, y)):
                correct = "✅" if pred == target else "❌"
                print(
                    f"   {inp.tolist()} → {pred.item():.0f} (target: {target.item():.0f}) {correct}"
                )

        print(f"\n📈 Modern Results:")
        print(f"   Final Loss: {loss.item():.6f}")
        print(f"   Final Accuracy: {accuracy:.4f}")
        print(f"   Uses Gradients: ✅")
        print(f"   Differentiable: ✅")

        return True

    except Exception as e:
        print(f"❌ Modern test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main comparison function."""
    print("🎓 Perceptron Learning Algorithms: Historical vs Modern")
    print("=" * 70)
    print("Educational Comparison of 1957 Original vs Contemporary Implementation")
    print()

    # Test historical approach
    historical_success = test_historical_approach()

    # Test modern approach
    modern_success = test_modern_approach()

    # Summary
    print("\n\n📊 EDUCATIONAL SUMMARY")
    print("=" * 50)
    print("🏛️ Historical (1957):")
    print("   ✓ Historically accurate")
    print("   ✓ No gradients needed")
    print("   ✓ Error-driven learning")
    print("   ✓ Educational transparency")
    print("   ✗ Not differentiable")

    print("\n🔧 Modern (Current):")
    print("   ✓ Framework integration")
    print("   ✓ Gradient-based optimization")
    print("   ✓ Differentiable and smooth")
    print("   ✓ Wandb/trainer compatible")
    print("   ✗ Less historically accurate")

    print(f"\n🎯 RECOMMENDATION:")
    print(f"   • Use HISTORICAL for educational accuracy")
    print(f"   • Use MODERN for infrastructure integration")
    print(f"   • Both approaches provide valuable insights!")

    if historical_success and modern_success:
        print(f"\n✨ Both approaches working successfully!")
        return 0
    else:
        print(f"\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
