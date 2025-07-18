#!/usr/bin/env python3
"""
Demonstration of unified training engine with perceptron and MLP models.

This script showcases how the shared infrastructure (engine, utils, data_utils, plotting)
works seamlessly across different model implementations, demonstrating the "scratch to scale"
philosophy with consistent APIs and shared functionality.
"""

import sys
from pathlib import Path

from data_utils import generate_xor_dataset

# Import shared packages
from engine import Trainer, TrainingConfig
from engine.base import DataSplit, ModelAdapter
from utils import set_random_seed, setup_logging


def import_models():
    """Dynamically import model classes."""
    # Add model paths
    sys.path.insert(0, str(Path("models/01_perceptron/src")))
    sys.path.insert(0, str(Path("models/03_MLP/src")))

    try:
        from model import MLP, Perceptron

        return Perceptron, MLP
    except ImportError:
        print("⚠️  Could not import models - running in demo mode")
        return None, None


def create_xor_data():
    """Create XOR dataset for demonstration."""
    features, labels = generate_xor_dataset(n_samples=4, noise=0.0, random_state=42)

    import torch

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    return DataSplit(x_train=x, y_train=y, x_test=x, y_test=y)


def demo_unified_training():
    """Demonstrate unified training with different models."""

    # Setup
    setup_logging(level="INFO")
    set_random_seed(42)

    print("🚀 AI From Scratch to Scale - Unified Training Demo")
    print("=" * 60)

    # Create XOR dataset
    print("📊 Creating XOR dataset...")
    data = create_xor_data()
    print(f"   Dataset: {data.get_split_info()}")

    # Model configurations
    models_config = [
        {
            "name": "perceptron",
            "model": Perceptron(
                input_size=2, activation="step", weight_init="random_normal"
            ),
            "config_overrides": {
                "experiment_name": "unified_perceptron_xor",
                "model_name": "perceptron",
                "learning_rate": 0.1,
                "max_epochs": 100,
                "convergence_threshold": 1e-4,
                "verbose": True,
            },
        },
        {
            "name": "mlp",
            "model": MLP(
                input_size=2, hidden_layers=[3], output_size=1, activation="sigmoid"
            ),
            "config_overrides": {
                "experiment_name": "unified_mlp_xor",
                "model_name": "mlp",
                "learning_rate": 0.5,
                "max_epochs": 200,
                "convergence_threshold": 1e-6,
                "verbose": True,
            },
        },
    ]

    results = {}

    # Train each model with unified engine
    for model_config in models_config:
        model_name = model_config["name"]
        model = model_config["model"]
        overrides = model_config["config_overrides"]

        print(f"\n🧠 Training {model_name.upper()} with Unified Engine")
        print("-" * 50)

        # Create training configuration
        training_config = TrainingConfig(
            experiment_name=overrides["experiment_name"],
            model_name=overrides["model_name"],
            dataset_name="xor",
            learning_rate=overrides["learning_rate"],
            max_epochs=overrides["max_epochs"],
            convergence_threshold=overrides["convergence_threshold"],
            verbose=overrides["verbose"],
            save_best_model=True,
            output_dir=f"outputs/unified_demo/{model_name}",
        )

        # Create trainer and train
        trainer = Trainer(training_config)

        # Adapt model to work with engine (if needed)
        adapted_model = ModelAdapter(model, model_name)

        # Train the model
        result = trainer.train(adapted_model, data)
        results[model_name] = result

        # Print results
        print(f"✅ {model_name.upper()} Training Complete!")
        print(f"   Final Accuracy: {result.final_train_accuracy:.4f}")
        print(f"   Final Loss: {result.final_loss:.6f}")
        print(f"   Epochs: {result.epochs_trained}")
        print(f"   Converged: {result.converged}")
        if result.converged:
            print(f"   Convergence Epoch: {result.convergence_epoch}")

    # Compare results
    print(f"\n📊 UNIFIED TRAINING COMPARISON")
    print("=" * 60)
    print(
        f"{'Model':<12} {'Accuracy':<10} {'Loss':<12} {'Epochs':<8} {'Converged':<10}"
    )
    print("-" * 60)

    for model_name, result in results.items():
        converged_str = "✅ Yes" if result.converged else "⚠️  No"
        print(
            f"{model_name.upper():<12} {result.final_train_accuracy:<10.4f} "
            f"{result.final_loss:<12.6f} {result.epochs_trained:<8} {converged_str:<10}"
        )

    # Key insights
    print(f"\n🎯 KEY INSIGHTS")
    print("=" * 60)

    perceptron_result = results.get("perceptron")
    mlp_result = results.get("mlp")

    if perceptron_result and mlp_result:
        if mlp_result.final_train_accuracy > perceptron_result.final_train_accuracy:
            print("✅ MLP outperformed Perceptron on XOR problem")
            print("   This demonstrates the breakthrough of multi-layer networks")
            print("   for solving non-linearly separable problems!")

        if mlp_result.converged and not perceptron_result.converged:
            print("✅ MLP converged while Perceptron struggled")
            print("   Hidden layers enable learning complex patterns")

    print(f"\n🔧 UNIFIED INFRASTRUCTURE BENEFITS")
    print("=" * 60)
    print("✅ Consistent training loop across all models")
    print("✅ Unified experiment tracking and logging")
    print("✅ Standardized model saving and loading")
    print("✅ Common evaluation metrics and reporting")
    print("✅ Shared visualization and plotting capabilities")

    print(f"\n🎓 This demonstrates the 'AI From Scratch to Scale' philosophy:")
    print("   - Start with simple models (Perceptron)")
    print("   - Build shared, reusable infrastructure")
    print("   - Scale to complex models (MLP) with same tools")
    print("   - Maintain educational clarity throughout")

    return results


if __name__ == "__main__":
    try:
        results = demo_unified_training()
        print(f"\n🎉 Demo completed successfully!")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure to run from project root and install with: pip install -e .")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
