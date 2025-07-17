# pylint: skip-file
# flake8: noqa
# type: ignore
"""
Template for train.py - Model Training Script

This template provides the basic structure for training neural network models
in the "AI From Scratch to Scale" project. Each model should follow this pattern
for consistency and reproducibility.

Replace MODEL_NAME with the actual model name (e.g., "Perceptron", "MLP", etc.)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import shared components
try:
    from engine import Trainer
    from engine.base import DataSplit
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

from data_utils import load_dataset, create_data_loaders
from plotting import generate_visualizations
from utils import setup_logging, set_random_seed, get_logger, setup_device

# Import model-specific components
from model import create_model
from config import get_config, get_experiment_config, list_available_experiments
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
        overrides['max_epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    if args.seed is not None:
        overrides['random_seed'] = args.seed
    if args.device != 'auto':
        overrides['device'] = args.device
    
    # Environment overrides
    if args.environment == 'debug':
        overrides['verbose'] = True
        overrides['log_freq'] = 1
        # Reduce epochs for debug mode
        if 'max_epochs' in overrides:
            overrides['max_epochs'] = min(overrides['max_epochs'], 20)
        else:
            overrides['max_epochs'] = 20
    
    # Wandb overrides
    if args.wandb:
        overrides['use_wandb'] = True
        overrides['wandb_project'] = args.wandb_project
        overrides['wandb_tags'] = args.tags
    
    return overrides


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
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, test_loader


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 0.0)
    momentum = config.get('momentum', 0.9)
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('epsilon', 1e-8)
        )
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def create_loss_function(config: dict) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loss function
    """
    loss_type = config.get('loss_function', 'crossentropy')
    
    if loss_type.lower() == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_type.lower() == 'mse':
        return nn.MSELoss()
    elif loss_type.lower() == 'bce':
        return nn.BCELoss()
    elif loss_type.lower() == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


def log_training_results(training_result, config, args):
    """Log training results."""
    logger = get_logger("ai_from_scratch")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.get('experiment', 'Unknown')}")
    logger.info(f"Model: {config.get('model_name', 'Unknown')}")
    logger.info(f"Final Loss: {training_result.get('final_loss', 'N/A'):.6f}")
    logger.info(f"Final Accuracy: {training_result.get('final_accuracy', 'N/A'):.4f}")
    logger.info(f"Epochs Trained: {training_result.get('epochs_trained', 'N/A')}")
    logger.info(f"Converged: {training_result.get('converged', 'N/A')}")
    logger.info(f"Training Time: {training_result.get('training_time', 'N/A')}")
    logger.info("=" * 60)


def save_training_results(training_result, config, args):
    """Save training results to file."""
    import json
    from datetime import datetime
    
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"training_results_{timestamp}.json")
    
    # Prepare results for saving
    save_data = {
        'experiment': config.get('experiment'),
        'model_name': config.get('model_name'),
        'timestamp': timestamp,
        'results': training_result,
        'config': config
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger = get_logger("ai_from_scratch")
    logger.info(f"Training results saved to: {results_file}")


def train_with_engine(config: dict, args) -> dict:
    """
    Train model using engine framework (advanced pattern).
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        dict: Training results
    """
    if not HAS_ENGINE:
        raise ImportError("Engine framework not available")
    
    logger = get_logger("ai_from_scratch")
    logger.info("Using engine-based training")
    
    # Set up device
    device = setup_device(args.device)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    # Train model
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time
    
    # Add training time to results
    results['training_time'] = training_time
    
    return results


def train_manually(config: dict, args) -> dict:
    """
    Train model manually (basic pattern).
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        dict: Training results
    """
    logger = get_logger("ai_from_scratch")
    logger.info("Using manual training")
    
    # Set up device
    device = setup_device(args.device)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, config)
    criterion = create_loss_function(config)
    
    # Training loop
    start_time = time.time()
    max_epochs = config.get('max_epochs', 100)
    log_freq = config.get('log_freq', 10)
    verbose = config.get('verbose', True)
    
    model.train()
    training_history = {'loss': [], 'accuracy': []}
    
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        training_history['loss'].append(avg_loss)
        training_history['accuracy'].append(accuracy)
        
        if verbose and (epoch + 1) % log_freq == 0:
            logger.info(f'Epoch {epoch+1}/{max_epochs}: '
                       f'Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}')
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    test_accuracy = correct / total
    
    return {
        'final_loss': test_loss,
        'final_accuracy': test_accuracy,
        'epochs_trained': max_epochs,
        'training_time': training_time,
        'converged': True,  # Simplified for template
        'training_history': training_history
    }


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Handle special commands
    if args.list_experiments:
        print_available_experiments()
        return
    
    if args.experiment_info:
        print_experiment_info(args.experiment_info)
        return
    
    if args.config_summary:
        print_config_summary(args.experiment)
        return
    
    # Set up logging
    setup_logging(level=args.log_level)
    logger = get_logger("ai_from_scratch")
    
    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)
    
    # Get configuration
    config = get_config(args.experiment, args.environment)
    
    # Apply command line overrides
    overrides = build_config_overrides(args)
    config.update(overrides)
    
    # Log configuration
    logger.info(f"Starting training for experiment: {args.experiment}")
    logger.info(f"Configuration: {config}")
    
    try:
        # Choose training method
        if HAS_ENGINE and config.get('use_engine', False):
            results = train_with_engine(config, args)
        else:
            results = train_manually(config, args)
        
        # Log and save results
        log_training_results(results, config, args)
        save_training_results(results, config, args)
        
        # Generate visualizations if requested
        if args.visualize:
            try:
                generate_visualizations(results, config)
                logger.info("Visualizations generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 