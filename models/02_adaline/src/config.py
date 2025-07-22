"""
ADALINE Configuration Management.

Simple dataclass-based configuration following the Simple implementation pattern
as demonstrated in 03_mlp.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
try:
    from .constants import EXPERIMENTS, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS
except ImportError:
    from constants import EXPERIMENTS, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS


@dataclass
class ADALINEConfig:
    """Configuration for ADALINE experiments."""
    
    # Experiment metadata
    name: str
    description: str
    
    # Model architecture
    input_size: int = 2
    output_size: int = 1
    
    # Training parameters
    learning_rate: float = DEFAULT_LEARNING_RATE
    epochs: int = DEFAULT_MAX_EPOCHS
    tolerance: float = 1e-6
    
    # Data parameters
    dataset: str = "simple_linear"
    batch_size: int = 32
    train_split: float = 0.8
    
    # Logging and output
    log_interval: int = 50
    save_model: bool = True
    visualize: bool = False
    
    # Enhanced wandb integration (following Wandb Integration Plan)
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    
    # Advanced wandb features
    wandb_watch_model: bool = False
    wandb_watch_log: str = "gradients"  # "gradients", "parameters", "all"
    wandb_watch_freq: int = 100
    
    # Artifact configuration
    wandb_log_checkpoints: bool = True
    wandb_log_visualizations: bool = True
    wandb_log_datasets: bool = False
    
    # Group and sweep support
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_sweep_id: Optional[str] = None
    
    # Reproducibility
    random_seed: int = 42


def get_experiment_config(experiment_name: str) -> ADALINEConfig:
    """Get configuration for specific experiment."""
    
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    exp_config = EXPERIMENTS[experiment_name]
    
    # Determine input size based on dataset
    dataset_name = str(exp_config["dataset"])
    input_size = get_dataset_input_size(dataset_name)
    
    # Create base config
    config = ADALINEConfig(
        name=experiment_name,
        description=str(exp_config["description"]),
        input_size=input_size,
        dataset=str(exp_config["dataset"]),
        epochs=int(exp_config["epochs"]), 
        learning_rate=float(exp_config["learning_rate"])
    )
    
    # Apply wandb defaults based on integration plan
    config = apply_wandb_defaults(config)
    
    return config


def apply_wandb_defaults(config: ADALINEConfig) -> ADALINEConfig:
    """Apply wandb defaults according to integration plan standards."""
    
    # Auto-generate project name if not set
    if config.wandb_project is None:
        config.wandb_project = "ai-from-scratch-adaline"
    
    # Auto-generate run name if not set
    if config.wandb_name is None:
        config.wandb_name = f"adaline-{config.name}"
    
    # Auto-generate tags if empty
    if not config.wandb_tags:
        config.wandb_tags = [
            "adaline",
            "module-1", 
            "foundation",
            "simple",
            config.dataset,
            "delta-rule"
        ]
    
    # Add experiment-specific tags
    if "strength" in config.name.lower():
        config.wandb_tags.append("strength")
    elif "weakness" in config.name.lower():
        config.wandb_tags.append("weakness")
    elif "debug" in config.name.lower():
        config.wandb_tags.append("debug")
    
    # Auto-generate notes if not set
    if config.wandb_notes is None:
        config.wandb_notes = (
            f"ADALINE training on {config.dataset} dataset using Delta Rule algorithm. "
            f"Experiment: {config.description}"
        )
    
    # Set group for organization
    if config.wandb_group is None:
        config.wandb_group = "module-1-foundations"
    
    # Set job type
    if config.wandb_job_type is None:
        config.wandb_job_type = "train"
    
    return config


def create_wandb_config_dict(config: ADALINEConfig) -> Dict[str, any]:
    """Create wandb config dictionary with all relevant parameters."""
    return {
        # Experiment info
        "experiment_name": config.name,
        "description": config.description,
        "dataset": config.dataset,
        
        # Model architecture
        "model_name": "ADALINE",
        "input_size": config.input_size,
        "output_size": config.output_size,
        "activation": "linear",
        "learning_algorithm": "delta-rule",
        
        # Training parameters
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "tolerance": config.tolerance,
        "batch_size": config.batch_size,
        
        # Data parameters
        "train_split": config.train_split,
        "random_seed": config.random_seed,
        
        # Logging parameters
        "log_interval": config.log_interval,
        "save_model": config.save_model,
        "visualize": config.visualize,
    }


def get_dataset_input_size(dataset_name: str) -> int:
    """Get input size for a given dataset."""
    dataset_sizes = {
        "simple_linear": 2,
        "linearly_separable": 2,
        "noisy_linear": 2,
        "debug_small": 2,
        "debug_linear": 2,
        "iris_binary": 4,
        "iris_setosa_versicolor": 4,
        "iris_versicolor_virginica": 4,
        "breast_cancer_binary": 30,
        "mnist_subset": 784,
        "xor_problem": 2,
        "circles_dataset": 2
    }
    
    return dataset_sizes.get(dataset_name, 2)  # Default to 2 if unknown


def list_experiments() -> Dict[str, str]:
    """List all available experiments."""
    return {name: config["description"] for name, config in EXPERIMENTS.items()}


def validate_config(config: ADALINEConfig) -> ADALINEConfig:
    """Validate configuration parameters."""
    from constants import validate_learning_rate
    
    # Validate learning rate
    config.learning_rate = validate_learning_rate(config.learning_rate)
    
    # Validate epochs
    if config.epochs < 1:
        config.epochs = 1
    elif config.epochs > 10000:
        config.epochs = 10000
    
    # Validate tolerance
    if config.tolerance <= 0:
        config.tolerance = 1e-6
    
    # Validate train split
    if config.train_split <= 0 or config.train_split >= 1:
        config.train_split = 0.8
    
    return config 