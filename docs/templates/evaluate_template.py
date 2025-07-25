#!/usr/bin/env python3
"""
Evaluation script template for neural network models.
Supports both simple and engine-based evaluation patterns.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import importlib.util

# Graceful imports
try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available. Limited functionality.")
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
for path_name in ['engine', 'utils', 'data_utils', 'plotting']:
    path = project_root / path_name
    if path.exists():
        sys.path.insert(0, str(path))

# Add model source
model_dir = Path(__file__).parent
sys.path.insert(0, str(model_dir))

# Import configurations and constants
try:
    # Explicit module loading for compatibility
    config_spec = importlib.util.spec_from_file_location("config", model_dir / "config.py")
    if config_spec and config_spec.loader:
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        get_training_config = getattr(config_module, 'get_training_config', None)
        SimpleExperimentConfig = getattr(config_module, 'SimpleExperimentConfig', None)
    else:
        get_training_config = None
        SimpleExperimentConfig = None
    
    constants_spec = importlib.util.spec_from_file_location("constants", model_dir / "constants.py")
    if constants_spec and constants_spec.loader:
        constants_module = importlib.util.module_from_spec(constants_spec)
        constants_spec.loader.exec_module(constants_module)
        MODEL_CONFIG = getattr(constants_module, 'MODEL_CONFIG', {})
        DATASET_SPECS = getattr(constants_module, 'DATASET_SPECS', {})
    else:
        MODEL_CONFIG = {}
        DATASET_SPECS = {}
    
    ENGINE_AVAILABLE = get_training_config is not None
    
except Exception as e:
    warnings.warn(f"Error loading configuration: {e}")
    get_training_config = None
    SimpleExperimentConfig = None
    MODEL_CONFIG = {}
    DATASET_SPECS = {}
    ENGINE_AVAILABLE = False

# Import model
try:
    from model import ModelTemplate, SimpleModelTemplate
    MODEL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Model not available: {e}")
    ModelTemplate = None
    SimpleModelTemplate = None
    MODEL_AVAILABLE = False

# Engine imports
if ENGINE_AVAILABLE:
    try:
        from evaluator import Evaluator
        from base import EvaluationConfig
        EVALUATOR_AVAILABLE = True
    except ImportError:
        EVALUATOR_AVAILABLE = False
else:
    EVALUATOR_AVAILABLE = False

# Utility imports
try:
    from logger import get_logger
    from device import get_device
    from seeds import set_seed
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Data imports
try:
    from datasets import get_dataset
    from loaders import create_data_loaders
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

# Plotting imports
try:
    from model_analysis import plot_confusion_matrix, plot_classification_report
    from training_plots import plot_metrics_comparison
    PLOTTING_UTILS_AVAILABLE = True
except ImportError:
    PLOTTING_UTILS_AVAILABLE = False


def setup_logging(verbose: bool = False) -> Any:
    """Setup logging with graceful fallback."""
    if UTILS_AVAILABLE:
        return get_logger("evaluate", level="DEBUG" if verbose else "INFO")
    else:
        import logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("evaluate")


def load_model_checkpoint(checkpoint_path: Path, config: Any) -> nn.Module:
    """Load model from checkpoint."""
    if not TORCH_AVAILABLE or not MODEL_AVAILABLE:
        raise RuntimeError("PyTorch and model implementation required")
    
    # Create model
    if ENGINE_AVAILABLE and hasattr(config, 'model_config'):
        model = ModelTemplate(config.model_config)
    else:
        model = SimpleModelTemplate(config)
    
    # Load weights
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def create_synthetic_data(config: Any, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for evaluation."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for data creation")
    
    if hasattr(config, 'model_config'):
        input_size = config.model_config.input_size
        output_size = config.model_config.output_size
    else:
        input_size = getattr(config, 'input_size', 2)
        output_size = getattr(config, 'output_size', 2)
    
    # Create random data
    X = torch.randn(num_samples, input_size)
    
    # Create synthetic labels (linearly separable for binary case)
    if output_size == 2:
        y = (X.sum(dim=1) > 0).long()
    else:
        y = torch.randint(0, output_size, (num_samples,))
    
    return X, y


def evaluate_model_simple(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, float]:
    """Simple model evaluation."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for evaluation")
    
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=1)
        
        # Calculate metrics
        accuracy = (predictions == y).float().mean().item()
        
        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, y).item()
        
        # Per-class accuracy if binary
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'num_samples': len(y)
        }
        
        if len(torch.unique(y)) == 2:  # Binary classification
            tp = ((predictions == 1) & (y == 1)).sum().item()
            tn = ((predictions == 0) & (y == 0)).sum().item()
            fp = ((predictions == 1) & (y == 0)).sum().item()
            fn = ((predictions == 0) & (y == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            })
    
    return metrics


def evaluate_model_engine(
    model: nn.Module,
    data_loader: Any,
    config: Any,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Engine-based model evaluation."""
    if not EVALUATOR_AVAILABLE:
        raise RuntimeError("Engine evaluator not available")
    
    eval_config = EvaluationConfig(
        batch_size=getattr(config, 'batch_size', 32),
        device=device,
        metrics=['accuracy', 'loss', 'confusion_matrix']
    )
    
    evaluator = Evaluator(eval_config)
    results = evaluator.evaluate(model, data_loader)
    
    return results


def plot_evaluation_results(
    metrics: Dict[str, Any],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot evaluation results."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    # Metrics bar plot
    metric_names = []
    metric_values = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key != 'num_samples':
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(value)
    
    axes[0, 0].bar(metric_names, metric_values)
    axes[0, 0].set_title('Evaluation Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Confusion matrix if available
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        if hasattr(cm, 'numpy'):
            cm = cm.numpy()
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
    else:
        axes[0, 1].text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Confusion Matrix')
    
    # Performance summary
    summary_text = f"""
    Accuracy: {metrics.get('accuracy', 0):.3f}
    Loss: {metrics.get('loss', 0):.3f}
    Samples: {metrics.get('num_samples', 0)}
    """
    
    if 'precision' in metrics:
        summary_text += f"""
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1-Score: {metrics['f1_score']:.3f}
    """
    
    axes[1, 0].text(0.1, 0.5, summary_text, transform=axes[1, 0].transAxes,
                   fontsize=12, verticalalignment='center')
    axes[1, 0].set_title('Summary')
    axes[1, 0].axis('off')
    
    # Model info if available
    info_text = "Model Information:\n"
    if MODEL_CONFIG:
        for key, value in MODEL_CONFIG.items():
            if isinstance(value, (str, int, float)):
                info_text += f"{key}: {value}\n"
    
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('Model Info')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Evaluation plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--experiment", type=str, default="debug",
                       help="Experiment name")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples for synthetic data")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save evaluation plots")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show plots")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting model evaluation")
    
    # Setup device
    if UTILS_AVAILABLE and args.device == "auto":
        device = get_device()
    else:
        device = args.device if args.device != "auto" else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    if UTILS_AVAILABLE:
        set_seed(42)
    elif TORCH_AVAILABLE:
        torch.manual_seed(42)
    
    try:
        # Load configuration
        if ENGINE_AVAILABLE and get_training_config:
            config = get_training_config(args.experiment)
            logger.info(f"Loaded engine config for experiment: {args.experiment}")
        elif SimpleExperimentConfig:
            config = SimpleExperimentConfig()
            logger.info("Using simple experiment configuration")
        else:
            raise RuntimeError("No configuration system available")
        
        # Determine checkpoint path
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            # Try common checkpoint locations
            output_dir = Path("outputs/models")
            checkpoint_path = output_dir / f"{args.experiment}_best.pth"
            
            if not checkpoint_path.exists():
                checkpoint_path = output_dir / "best_model.pth"
            
            if not checkpoint_path.exists():
                logger.warning("No checkpoint found, creating random model")
                checkpoint_path = None
        
        # Load or create model
        if checkpoint_path and checkpoint_path.exists():
            model = load_model_checkpoint(checkpoint_path, config)
            logger.info(f"Loaded model from: {checkpoint_path}")
        else:
            # Create new model for testing
            if ENGINE_AVAILABLE and hasattr(config, 'model_config'):
                model = ModelTemplate(config.model_config)
            else:
                model = SimpleModelTemplate(config)
            logger.warning("Created new model (no checkpoint found)")
        
        model = model.to(device)
        
        # Create or load data
        if DATA_UTILS_AVAILABLE:
            try:
                dataset = get_dataset(args.experiment)
                _, test_loader = create_data_loaders(
                    dataset, 
                    batch_size=args.batch_size,
                    train_split=0.8
                )
                logger.info("Using dataset from data_utils")
                use_data_loader = True
            except Exception as e:
                logger.warning(f"Failed to load dataset: {e}")
                use_data_loader = False
        else:
            use_data_loader = False
        
        if not use_data_loader:
            # Create synthetic data
            X, y = create_synthetic_data(config, args.num_samples)
            logger.info(f"Created synthetic data: {X.shape}, {y.shape}")
        
        # Evaluate model
        if EVALUATOR_AVAILABLE and use_data_loader:
            # Engine-based evaluation
            metrics = evaluate_model_engine(model, test_loader, config, device)
            logger.info("Completed engine-based evaluation")
        else:
            # Simple evaluation
            metrics = evaluate_model_simple(model, X, y, device)
            logger.info("Completed simple evaluation")
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            elif isinstance(value, int):
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Log to W&B if available
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=f"model_evaluation_{args.experiment}",
                    name=f"eval_{args.experiment}",
                    config=vars(args)
                )
                wandb.log(metrics)
                logger.info("Logged results to W&B")
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
        
        # Create plots
        if PLOTTING_AVAILABLE:
            save_path = None
            if args.save_plots:
                output_dir = Path("outputs/visualizations")
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"evaluation_{args.experiment}.png"
            
            plot_evaluation_results(
                metrics, 
                save_path=save_path, 
                show=not args.no_show
            )
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
