"""
ADALINE Training Script.

Training script for ADALINE model with Delta Rule learning algorithm.
Follows the Simple implementation pattern.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_experiment_config, validate_config, list_experiments
from model import create_adaline
from constants import get_experiment_info
from data_loader import load_adaline_train_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ADALINE model")
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="debug_small",
        help="Experiment name to run"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override number of epochs"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations"
    )
    
    # Enhanced wandb arguments
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable wandb experiment tracking"
    )
    
    parser.add_argument(
        "--wandb-project", 
        type=str,
        help="Override wandb project name"
    )
    
    parser.add_argument(
        "--wandb-name", 
        type=str,
        help="Override wandb run name"
    )
    
    parser.add_argument(
        "--wandb-tags", 
        nargs="+",
        help="Additional wandb tags"
    )
    
    parser.add_argument(
        "--wandb-mode", 
        type=str,
        choices=["online", "offline", "disabled"],
        help="Wandb mode"
    )
    
    parser.add_argument(
        "--list-experiments", 
        action="store_true",
        help="List available experiments"
    )
    
    parser.add_argument(
        "--experiment-info", 
        type=str,
        help="Get detailed info about specific experiment"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()





def train_adaline(config, x_data, y_data, use_wandb=False):
    """Train ADALINE model with optional wandb tracking."""
    print(f"Training ADALINE on {config.dataset} dataset")
    print(f"Configuration: {config}")
    
    if use_wandb and config.use_wandb:
        # Use the enhanced wrapper with wandb integration
        from adaline_wrapper import ADALINEWrapper
        
        # Create wrapper model
        model = ADALINEWrapper(config)
        
        # Train with wandb tracking
        print("🔬 Training with wandb tracking enabled")
        results = model.fit_with_wandb(x_data, y_data, config)
        
    else:
        # Use simple model without wandb
        model = create_adaline(config)
        
        # Train model normally
        results = model.fit(x_data, y_data)
    
    # Print results
    print(f"Training completed:")
    print(f"  - Converged: {results['converged']}")
    print(f"  - Final MSE: {results['final_mse']:.6f}")
    print(f"  - Epochs trained: {results['epochs_trained']}")
    
    return model, results


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # List experiments if requested
    if args.list_experiments:
        experiments = list_experiments()
        print("Available experiments:")
        for name, desc in experiments.items():
            print(f"  {name}: {desc}")
        return
    
    # Get experiment info if requested
    if args.experiment_info:
        try:
            info = get_experiment_info(args.experiment_info)
            print(f"Experiment: {args.experiment_info}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        except ValueError as e:
            print(f"Error: {e}")
        return
    
    # Get configuration
    try:
        config = get_experiment_config(args.experiment)
        
        # Override parameters if provided
        if args.epochs:
            config.epochs = args.epochs
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        
        # Handle wandb configuration
        if args.wandb:
            config.use_wandb = True
        
        if args.wandb_project:
            config.wandb_project = args.wandb_project
            
        if args.wandb_name:
            config.wandb_name = args.wandb_name
            
        if args.wandb_tags:
            config.wandb_tags.extend(args.wandb_tags)
            
        if args.wandb_mode:
            config.wandb_mode = args.wandb_mode
        
        # Validate configuration
        config = validate_config(config)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-experiments to see available experiments")
        return
    
    # Load data
    try:
        x_data, y_data = load_adaline_train_data(config.dataset)
        print(f"Loaded {len(x_data)} samples for {config.dataset}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    # Train model
    try:
        model, results = train_adaline(config, x_data, y_data, use_wandb=args.wandb)
        
        # Save model if requested
        if config.save_model:
            output_dir = Path("outputs/models")
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"{config.name}_model.npz"
            
            # Save based on model type
            if hasattr(model, 'pure_adaline'):
                # Save from wrapper's pure ADALINE
                np.savez(model_path, 
                         weights=model.pure_adaline.weights, 
                         bias=model.pure_adaline.bias)
            else:
                # Save simple model
                np.savez(model_path, weights=model.weights, bias=model.bias)
            print(f"Model saved to {model_path}")
        
        # Generate visualizations if requested
        if args.visualize:
            try:
                from visualize import save_all_adaline_plots
                plots = save_all_adaline_plots(model, x_data, y_data)
                print(f"Generated {len(plots)} visualization plots")
            except ImportError:
                print("Visualization module not available")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if args.debug:
            raise


if __name__ == "__main__":
    main() 