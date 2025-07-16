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
from utils import setup_logging, set_random_seed, get_logger

# Import model-specific components
from model import create_model
from config import get_config
from constants import MODEL_NAME as MODEL_NAME_CONSTANT, ALL_EXPERIMENTS


def parse_arguments():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=f'Train {MODEL_NAME_CONSTANT} model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --experiment xor                    # Basic training
  python train.py --experiment xor --visualize        # With visualizations
  python train.py --experiment xor --debug            # Debug mode
  python train.py --list-experiments                  # Show available experiments
  python train.py --experiment-info xor               # Show experiment details
        """
    )
    
    # Required arguments
    parser.add_argument('--experiment', required=True, type=str,
                       help='Experiment name (e.g., xor, iris-hard)')
    
    # Optional training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Environment and debugging
    parser.add_argument('--environment', choices=["default", "debug", "production"],
                       default="default", help="Environment configuration")
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with reduced epochs')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    # Model and data options
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to load before training')
    parser.add_argument('--no-save-checkpoint', action='store_true',
                       help='Skip saving final model checkpoint')
    
    # Logging and monitoring
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
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
    
    # Educational and information commands
    parser.add_argument('--list-experiments', action='store_true',
                       help='List available experiments and exit')
    parser.add_argument('--experiment-info', type=str,
                       help='Show information about a specific experiment')
    parser.add_argument('--config-summary', action='store_true',
                       help='Print configuration summary and exit')
    
    return parser.parse_args()


def print_available_experiments():
    """Print list of available experiments."""
    print(f"\nAvailable {MODEL_NAME_CONSTANT} experiments:")
    print("-" * 50)
    for exp in ALL_EXPERIMENTS:
        try:
            config = get_config(exp)
            print(f"{exp:20} - {config.get('description', 'No description')}")
        except Exception as e:
            print(f"{exp:20} - Error: {e}")


def print_experiment_info(experiment_name: str):
    """Print detailed information about a specific experiment."""
    try:
        config = get_config(experiment_name)
        print(f"\nExperiment: {experiment_name}")
        print(f"Description: {config.get('description', 'No description')}")
        print(f"Dataset: {config.get('dataset', 'Unknown')}")
        print(f"Architecture: {config.get('architecture', 'Standard')}")
        print(f"Learning rate: {config.get('learning_rate', 'Default')}")
        print(f"Max epochs: {config.get('epochs', 'Default')}")
        print(f"Expected accuracy: {config.get('expected_accuracy', 'Not specified')}")
    except Exception as e:
        print(f"Error getting experiment info: {e}")


def print_config_summary(experiment_name: str):
    """Print configuration summary for an experiment."""
    try:
        config = get_config(experiment_name)
        print(f"\nConfiguration Summary for {experiment_name}:")
        print("=" * 50)
        for key, value in config.items():
            print(f"{key:20}: {value}")
    except Exception as e:
        print(f"Error printing config summary: {e}")


def build_config_overrides(args):
    """
    Build configuration overrides from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        dict: Configuration overrides
    """
    overrides = {}
    
    # Training parameter overrides
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.device != 'auto':
        overrides['device'] = args.device
    
    # Environment overrides
    if args.environment == 'debug':
        overrides['verbose'] = True
        overrides['log_freq'] = 1
        # Reduce epochs for debug mode
        if 'epochs' in overrides:
            overrides['epochs'] = min(overrides['epochs'], 20)
        else:
            overrides['epochs'] = 20
    
    # Wandb overrides
    if args.wandb:
        overrides['use_wandb'] = True
        overrides['wandb_project'] = args.wandb_project
        overrides['wandb_tags'] = args.tags
    
    return overrides


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
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    return device


def load_data(config: dict) -> tuple:
    """
    Load and prepare data for training.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader objects
    """
    logger = get_logger("ai_from_scratch")
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
    
    logger = get_logger("ai_from_scratch")
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


def log_training_results(training_result, config, args):
    """
    Log comprehensive training results with educational context.
    
    Args:
        training_result: Training result object
        config (dict): Configuration used
        args: Command line arguments
    """
    logger = get_logger("ai_from_scratch")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Model: {MODEL_NAME_CONSTANT}")
    logger.info(f"Dataset: {config.get('dataset', 'Unknown')}")
    logger.info("-" * 60)
    logger.info(f"Epochs trained: {training_result.get('epochs_trained', 'Unknown')}")
    logger.info(f"Training time: {training_result.get('total_training_time', 0):.2f} seconds")
    logger.info(f"Converged: {'✓' if training_result.get('converged', False) else '✗'}")
    logger.info(f"Final train accuracy: {training_result.get('final_train_accuracy', 0):.4f}")
    
    if training_result.get('final_val_accuracy') is not None:
        logger.info(f"Final validation accuracy: {training_result['final_val_accuracy']:.4f}")
    
    if training_result.get('final_test_accuracy') is not None:
        logger.info(f"Final test accuracy: {training_result['final_test_accuracy']:.4f}")
    
    # Performance vs expectation (if available)
    expected_acc = config.get('expected_accuracy')
    actual_acc = training_result.get('final_train_accuracy', 0)
    
    if expected_acc is not None:
        if actual_acc >= expected_acc * 0.9:
            performance = "✓ MEETS EXPECTATIONS"
        elif actual_acc >= expected_acc * 0.7:
            performance = "~ BELOW EXPECTATIONS"
        else:
            performance = "✗ WELL BELOW EXPECTATIONS"
        
        logger.info(f"Performance: {performance}")
        logger.info(f"Expected: {expected_acc:.3f}, Actual: {actual_acc:.3f}")
    
    # Special success messages for educational milestones
    if args.experiment == "xor" and actual_acc >= 0.99:
        logger.info("\n🎉 XOR PROBLEM SOLVED! 🎉")
        logger.info("This demonstrates the model's ability to learn non-linear patterns!")
    
    logger.info("=" * 60)


def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Handle informational commands first
    if args.list_experiments:
        print_available_experiments()
        return 0
    
    if args.experiment_info:
        print_experiment_info(args.experiment_info)
        return 0
    
    if args.config_summary:
        print_config_summary(args.experiment)
        return 0
    
    # Validate experiment
    if args.experiment not in ALL_EXPERIMENTS:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print(f"Available experiments: {ALL_EXPERIMENTS}")
        print("Use --list-experiments to see all available experiments")
        return 1
    
    try:
        # Setup enhanced logging
        log_level = "DEBUG" if args.debug else args.log_level
        setup_logging(
            level=log_level,
            log_dir="outputs/logs",
            file_output=True,
            console_output=True
        )
        logger = get_logger("ai_from_scratch")
        
        # Build configuration overrides
        overrides = build_config_overrides(args)
        
        # Get configuration
        config = get_config(args.experiment, **overrides)
        
        # Set up reproducibility
        if config.get('seed') is not None:
            set_random_seed(config['seed'])
        
        # Set up device
        device = setup_device(args.device)
        config['device'] = device
        
        logger.info(f"Using device: {device}")
        
        # Log experiment details
        logger.info(f"Starting training for {MODEL_NAME_CONSTANT}")
        logger.info(f"Experiment: {args.experiment}")
        logger.info(f"Environment: {args.environment}")
        if args.debug:
            logger.info("Debug mode enabled - reduced epochs and verbose logging")
        
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
        use_wandb = args.wandb
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
        training_result = trainer.train(
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
                training_history=training_result,
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
        
        # Log comprehensive results
        log_training_results(training_result, config, args)
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Log final results
        logger.info(f"Training completed. Final test metrics: {test_metrics}")
        
        if use_wandb:
            wandb.log({"test_" + k: v for k, v in test_metrics.items()})
            wandb.finish()
        
        logger.info("Training script completed successfully!")
        
    except Exception as e:
        logger = get_logger("ai_from_scratch")
        logger.error(f"Error during training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 