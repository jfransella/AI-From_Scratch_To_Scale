"""
ADALINE Configuration Management.

Simple dataclass-based configuration following the Simple implementation pattern
as demonstrated in 03_mlp.
"""

from dataclasses import dataclass
from typing import Dict
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
    
    return ADALINEConfig(
        name=experiment_name,
        description=str(exp_config["description"]),
        input_size=input_size,
        dataset=str(exp_config["dataset"]),
        epochs=int(exp_config["epochs"]), 
        learning_rate=float(exp_config["learning_rate"])
    )


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