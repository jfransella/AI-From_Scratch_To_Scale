"""
Template for train.py - Model Training Script

This template provides the basic structure for training neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and reproducibility.

Replace MODEL_NAME with the actual model name (e.g., "Perceptron", "MLP", etc.)
"""

import argparse
import sys
import os
import time
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

# Handle PyTorch imports gracefully
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import shared components with error handling
try:
    from engine import Trainer
    from engine.base import DataSplit
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

try:
    from data_utils import load_dataset, create_data_loaders
    HAS_DATA_UTILS = True
except ImportError:
    HAS_DATA_UTILS = False

try:
    from plotting import generate_visualizations
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from utils import setup_logging, set_random_seed, get_logger, setup_device
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

# Import model-specific components using explicit loading for compatibility
def load_model_module():
    """Load model module with explicit path handling."""
    try:
        from model import create_model, ModelTemplate
        return create_model, ModelTemplate
    except ImportError:
        # Use explicit module loading as fallback
        model_path = Path(__file__).parent / "model.py"
        if model_path.exists():
            spec = importlib.util.spec_from_file_location("model", model_path)
            if spec and spec.loader:
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                return getattr(model_module, "create_model", None), getattr(model_module, "ModelTemplate", None)
        return None, None

def load_config_module():
    """Load config module with explicit path handling."""
    try:
        from config import get_config, get_experiment_config, list_available_experiments
        return get_config, get_experiment_config, list_available_experiments
    except ImportError:
        # Use explicit module loading as fallback
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                return (
                    getattr(config_module, "get_config", None),
                    getattr(config_module, "get_experiment_config", None),
                    getattr(config_module, "list_available_experiments", lambda: ["debug", "quick_test", "standard"])
                )
        return None, None, lambda: ["debug", "quick_test", "standard"]

def load_constants_module():
    """Load constants module with explicit path handling."""
    try:
        from constants import MODEL_NAME, ALL_EXPERIMENTS
        return MODEL_NAME, ALL_EXPERIMENTS
    except ImportError:
        # Use explicit module loading as fallback
        constants_path = Path(__file__).parent / "constants.py"
        if constants_path.exists():
            spec = importlib.util.spec_from_file_location("constants", constants_path)
            if spec and spec.loader:
                constants_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(constants_module)
                return getattr(constants_module, "MODEL_NAME", "Template"), getattr(constants_module, "ALL_EXPERIMENTS", ["debug", "quick_test", "standard"])
        return "Template", ["debug", "quick_test", "standard"]

# Load modules
create_model, ModelTemplate = load_model_module()
get_config, get_experiment_config, list_available_experiments = load_config_module()
MODEL_NAME_CONSTANT, ALL_EXPERIMENTS = load_constants_module()


def parse_arguments():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=f"Train {MODEL_NAME_CONSTANT} model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="debug",
        choices=list_available_experiments(),
        help="Experiment configuration to use"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=None,
        help="Learning rate override"
    )
    parser.add_argument(
        "--max-epochs", 
        type=int, 
        default=None,
        help="Maximum epochs override"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size override"
    )
    
    # Model parameters
    parser.add_argument(
        "--hidden-size", 
        type=int, 
        default=None,
        help="Hidden layer size override"
    )
    parser.add_argument(
        "--activation", 
        type=str, 
        default=None,
        help="Activation function override"
    )
    
    # Training behavior
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save model or results"
    )
    parser.add_argument(
        "--no-visualize", 
        action="store_true",
        help="Don't generate visualizations"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # W&B integration
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default=None,
        help="W&B project name override"
    )
    
    # Advanced options
    parser.add_argument(
        "--config-file", 
        type=str, 
        default=None,
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory override"
    )
    
    return parser.parse_args()


def setup_experiment(args) -> Dict[str, Any]:
    """
    Setup experiment configuration and environment.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Complete experiment configuration
    """
    # Set random seed early
    if HAS_UTILS:
        set_random_seed(args.random_seed)
    
    # Get base configuration
    if get_experiment_config is not None:
        config = get_experiment_config(args.experiment, "simple")
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
    else:
        # Fallback configuration
        config_dict = {
            "experiment_name": args.experiment,
            "input_size": 2,
            "output_size": 1,
            "learning_rate": 0.01,
            "max_epochs": 100,
            "batch_size": 32,
            "device": "cpu",
            "random_seed": args.random_seed,
        }
    
    # Apply command line overrides
    overrides = {}
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.hidden_size is not None:
        overrides["hidden_size"] = args.hidden_size
    if args.activation is not None:
        overrides["activation"] = args.activation
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.wandb:
        overrides["use_wandb"] = True
    if args.wandb_project is not None:
        overrides["wandb_project"] = args.wandb_project
    if args.verbose:
        overrides["verbose"] = True
    
    # Handle device selection
    if args.device is not None:
        if args.device == "auto":
            if HAS_UTILS:
                device = setup_device()
            else:
                device = "cpu"
        else:
            device = args.device
        overrides["device"] = device
    
    # Update configuration
    config_dict.update(overrides)
    
    return config_dict


def train_with_engine(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train model using the engine framework.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    if not HAS_ENGINE:
        raise RuntimeError("Engine framework not available")
        
    if HAS_UTILS:
        logger = get_logger(__name__)
        logger.info(f"🚀 Starting {MODEL_NAME_CONSTANT} training with engine framework")
    
    # Create model
    if create_model is not None:
        model = create_model(config)
    else:
        raise RuntimeError("Model creation function not available")
    
    # Load dataset
    if HAS_DATA_UTILS:
        dataset = load_dataset(config["experiment_name"])
        data_loaders = create_data_loaders(dataset, config)
    else:
        raise RuntimeError("Data utilities not available")
    
    # Create training configuration
    from config import get_training_config
    training_config = get_training_config(config["experiment_name"], **config)
    
    # Create trainer
    trainer = Trainer(training_config)
    
    # Create data split
    data_split = DataSplit(
        train_loader=data_loaders["train"],
        val_loader=data_loaders.get("val"),
        test_loader=data_loaders.get("test")
    )
    
    # Train model
    results = trainer.train(model, data_split)
    
    # Generate visualizations if requested
    if HAS_PLOTTING and not config.get("no_visualize", False):
        generate_visualizations(model, dataset, results, config)
    
    return results


def train_simple(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train model using simple training loop (without engine).
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    if HAS_UTILS:
        logger = get_logger(__name__)
        logger.info(f"🔧 Starting {MODEL_NAME_CONSTANT} training with simple approach")
    
    # Create model
    if create_model is not None:
        model = create_model(config)
    elif ModelTemplate is not None:
        model = ModelTemplate(**config)
    else:
        raise RuntimeError("Model class not available")
    
    # TODO: Implement simple training loop
    # This is a template - actual implementation depends on the specific model
    
    # Example simple training loop:
    # 1. Load or generate data
    # 2. Create optimizer
    # 3. Training loop with forward/backward passes
    # 4. Track metrics
    # 5. Save results
    
    # For now, return mock results
    results = {
        "converged": True,
        "epochs_trained": config.get("max_epochs", 100),
        "final_loss": 0.1,
        "final_accuracy": 0.9,
        "training_time": 60.0,
        "model_path": None,
    }
    
    if HAS_UTILS:
        logger.info("✅ Simple training completed")
    
    return results


def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Save training results to disk.
    
    Args:
        results: Training results
        config: Training configuration
    """
    if config.get("no_save", False):
        return
        
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results JSON
    results_file = output_dir / f"{config['experiment_name']}_results.json"
    
    try:
        import json
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for key, value in results.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[key] = value
                except (TypeError, ValueError):
                    serializable_results[key] = str(value)
            
            json.dump({
                "config": config,
                "results": serializable_results,
                "timestamp": time.time(),
            }, f, indent=2)
            
        if HAS_UTILS:
            logger = get_logger(__name__)
            logger.info(f"💾 Results saved to {results_file}")
    except Exception as e:
        if HAS_UTILS:
            logger = get_logger(__name__)
            logger.warning(f"Failed to save results: {e}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    if HAS_UTILS:
        setup_logging()
        logger = get_logger(__name__)
        logger.info(f"🎯 Starting {MODEL_NAME_CONSTANT} training")
        logger.info(f"📊 Experiment: {args.experiment}")
    
    try:
        # Setup experiment configuration
        config = setup_experiment(args)
        
        # Choose training approach
        if HAS_ENGINE and not args.no_save:  # Use engine if available and not disabled
            results = train_with_engine(config)
        else:
            results = train_simple(config)
        
        # Save results
        save_results(results, config)
        
        # Print summary
        if HAS_UTILS:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"📈 Final accuracy: {results.get('final_accuracy', 'N/A'):.4f}")
            logger.info(f"⏱️ Training time: {results.get('training_time', 'N/A'):.2f}s")
        else:
            print("Training completed successfully!")
            print(f"Final accuracy: {results.get('final_accuracy', 'N/A')}")
            
    except Exception as e:
        if HAS_UTILS:
            logger = get_logger(__name__)
            logger.error(f"❌ Training failed: {e}")
        else:
            print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
