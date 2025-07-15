"""
Template for train.py - Model Training Script

This template provides the basic structure for training neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and reproducibility.

Replace [MODEL_NAME] with the actual model name (e.g., "Perceptron", "MLP", etc.)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import shared components
from engine import Trainer
from data_utils import load_dataset, create_data_loaders
from plotting import generate_visualizations
from utils import setup_logging, set_random_seed

# Import model-specific components
from model import [MODEL_NAME], create_model
from config import get_config
from constants import MODEL_NAME as MODEL_NAME_CONSTANT

# Set up logging
logger = setup_logging(__name__)


def parse_arguments():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=f'Train {MODEL_NAME_CONSTANT} model')
    
    # Required arguments
    parser.add_argument('--experiment', required=True, type=str,
                       help='Experiment name (e.g., iris-hard, xor)')
    
    # Optional training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Model and data options
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to load before training')
    parser.add_argument('--no-save-checkpoint', action='store_true',
                       help='Skip saving final model checkpoint')
    
    # Logging and monitoring
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='ai-from-scratch',
                       help='Weights & Biases project name')
    parser.add_argument('--tags', type=str, nargs='+', default=[],
                       help='Tags to attach to the run')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate and save visualizations')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """
    Set up the appropriate device for training.
    
    Args:
        device_arg (str): Device argument from command line
        
    Returns:
        torch.device: Configured device
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def load_data(config: dict) -> tuple:
    """
    Load and prepare data for training.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader objects
    """
    logger.info(f"Loading data for experiment: {config['experiment']}")
    
    # Load dataset based on experiment configuration
    X, y = load_dataset(
        dataset_name=config['dataset'],
        experiment=config['experiment'],
        **config.get('dataset_params', {})
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y,
        batch_size=config['batch_size'],
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15),
        test_split=config.get('test_split', 0.15),
        shuffle=config.get('shuffle', True),
        random_state=config.get('seed', 42)
    )
    
    logger.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
               f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model (nn.Module): Model to optimize
        config (dict): Configuration dictionary
        
    Returns:
        optim.Optimizer: Configured optimizer
    """
    optimizer_type = config.get('optimizer', 'adam').lower()
    learning_rate = config['learning_rate']
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")
    return optimizer


def create_loss_function(config: dict) -> nn.Module:
    """
    Create loss function for training.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        nn.Module: Loss function
    """
    loss_type = config.get('loss_function', 'crossentropy').lower()
    
    if loss_type == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'bce':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Get configuration
    config = get_config(args.experiment)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set up reproducibility
    if config.get('seed') is not None:
        set_random_seed(config['seed'])
    
    # Set up device
    device = setup_device(args.device)
    config['device'] = device
    
    # Log experiment details
    logger.info(f"Starting training for {MODEL_NAME_CONSTANT}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Configuration: {config}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, config)
    criterion = create_loss_function(config)
    
    # Set up Weights & Biases logging (if enabled)
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{MODEL_NAME_CONSTANT}_{args.experiment}",
            config=config,
            tags=[MODEL_NAME_CONSTANT, args.experiment] + args.tags
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_wandb=use_wandb,
        config=config
    )
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs']
    )
    
    # Save final checkpoint (unless disabled)
    if not args.no_save_checkpoint:
        checkpoint_path = f"outputs/models/{MODEL_NAME_CONSTANT}_{args.experiment}_final.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        model.save_checkpoint(checkpoint_path, epoch=config['epochs'], 
                            optimizer_state=optimizer.state_dict())
        
        # Save to wandb if enabled
        if use_wandb:
            wandb.save(checkpoint_path)
    
    # Generate visualizations (if requested)
    if args.visualize:
        logger.info("Generating visualizations...")
        visualization_plots = generate_visualizations(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            training_history=training_history,
            config=config,
            experiment_name=args.experiment
        )
        
        # Save visualizations
        viz_dir = "outputs/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        for plot_name, plot_data in visualization_plots.items():
            plot_path = os.path.join(viz_dir, f"{plot_name}_{args.experiment}.png")
            plot_data.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization: {plot_path}")
            
            # Save to wandb if enabled
            if use_wandb:
                wandb.log({plot_name: wandb.Image(plot_path)})
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Log final results
    logger.info(f"Training completed. Final test metrics: {test_metrics}")
    
    if use_wandb:
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})
        wandb.finish()
    
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main() 