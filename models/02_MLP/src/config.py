"""
Configuration management for Multi-Layer Perceptron (MLP) experiments.

This module provides experiment configurations for training MLPs on various
non-linearly separable datasets, with a focus on demonstrating the breakthrough
of solving the XOR problem that single-layer perceptrons cannot handle.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging


@dataclass
class MLPExperimentConfig:
    """Configuration for a single MLP experiment."""
    
    # Experiment metadata
    name: str
    description: str
    
    # Architecture configuration
    architecture_name: str
    input_size: int
    hidden_layers: List[int]
    output_size: int
    activation: str = "sigmoid"
    weight_init: str = "xavier_normal"
    
    # Training configuration
    learning_rate: float = 0.1
    max_epochs: int = 1000
    convergence_threshold: float = 1e-6
    patience: int = 50
    batch_size: Optional[int] = None
    weight_decay: float = 0.0
    momentum: float = 0.0
    
    # Dataset configuration
    dataset_type: str = "xor"
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training behavior
    early_stopping: bool = True
    save_model: bool = True
    save_history: bool = True
    verbose: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Output configuration
    output_dir: str = "outputs"
    save_plots: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


# =============================================================================
# PREDEFINED EXPERIMENTS
# =============================================================================

def get_xor_breakthrough_config() -> MLPExperimentConfig:
    """
    The classic XOR experiment - demonstrating MLP's ability to solve
    non-linearly separable problems that perceptrons cannot handle.
    """
    return MLPExperimentConfig(
        name="xor_breakthrough",
        description="Solve XOR problem - the classic demonstration of MLP superiority over perceptrons",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[2],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=0.5,
        max_epochs=1000,
        convergence_threshold=1e-6,
        dataset_type="xor",
        early_stopping=True,
        random_seed=42
    )


def get_quick_test_config() -> MLPExperimentConfig:
    """Quick test configuration for rapid development feedback."""
    return MLPExperimentConfig(
        name="quick_test",
        description="Fast training on XOR for development and debugging",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[3],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=1.0,
        max_epochs=100,
        convergence_threshold=1e-4,
        dataset_type="xor",
        early_stopping=False,
        verbose=True,
        random_seed=42
    )


def get_circles_challenge_config() -> MLPExperimentConfig:
    """
    Concentric circles dataset - tests MLP's ability to learn complex
    non-linear decision boundaries.
    """
    return MLPExperimentConfig(
        name="circles_challenge",
        description="Learn to separate concentric circles - complex non-linear patterns",
        architecture_name="small",
        input_size=2,
        hidden_layers=[8],
        output_size=1,
        activation="relu",
        weight_init="he_normal",
        learning_rate=0.01,
        max_epochs=2000,
        convergence_threshold=1e-6,
        dataset_type="circles",
        dataset_params={"num_samples": 1000, "noise": 0.1},
        early_stopping=True,
        random_seed=42
    )


def get_moons_config() -> MLPExperimentConfig:
    """
    Two moons dataset - another classic non-linear classification problem.
    """
    return MLPExperimentConfig(
        name="moons_classification",
        description="Classify two interleaving half-circles (moons)",
        architecture_name="medium",
        input_size=2,
        hidden_layers=[8, 4],
        output_size=1,
        activation="relu",
        weight_init="he_normal",
        learning_rate=0.01,
        max_epochs=2000,
        convergence_threshold=1e-6,
        dataset_type="moons",
        dataset_params={"num_samples": 1000, "noise": 0.1},
        early_stopping=True,
        random_seed=42
    )


def get_spirals_config() -> MLPExperimentConfig:
    """
    Two spirals dataset - challenging non-linear problem requiring deep networks.
    """
    return MLPExperimentConfig(
        name="spirals_challenge",
        description="Learn to separate intertwined spirals - requires deeper networks",
        architecture_name="deep",
        input_size=2,
        hidden_layers=[16, 8, 4],
        output_size=1,
        activation="relu",
        weight_init="he_normal",
        learning_rate=0.001,
        max_epochs=5000,
        convergence_threshold=1e-6,
        dataset_type="spirals",
        dataset_params={"num_samples": 1000, "noise": 0.0},
        early_stopping=True,
        patience=100,
        random_seed=42
    )


def get_architecture_comparison_config() -> MLPExperimentConfig:
    """
    Compare different MLP architectures on XOR problem.
    """
    return MLPExperimentConfig(
        name="architecture_comparison",
        description="Compare minimal vs small architectures on XOR",
        architecture_name="small",
        input_size=2,
        hidden_layers=[4],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=0.3,
        max_epochs=1000,
        convergence_threshold=1e-6,
        dataset_type="xor",
        early_stopping=True,
        random_seed=42
    )


def get_activation_study_config() -> MLPExperimentConfig:
    """
    Study different activation functions on XOR problem.
    """
    return MLPExperimentConfig(
        name="activation_study",
        description="Test ReLU vs Sigmoid activations on XOR",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[4],
        output_size=1,
        activation="relu",
        weight_init="he_normal",
        learning_rate=0.01,
        max_epochs=2000,
        convergence_threshold=1e-6,
        dataset_type="xor",
        early_stopping=True,
        random_seed=42
    )


def get_learning_rate_study_config() -> MLPExperimentConfig:
    """
    Study the effect of different learning rates.
    """
    return MLPExperimentConfig(
        name="lr_study",
        description="Study learning rate effects on convergence",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[3],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=0.1,  # Will be varied in training script
        max_epochs=1000,
        convergence_threshold=1e-6,
        dataset_type="xor",
        early_stopping=False,  # Want to see full training curves
        random_seed=42
    )


def get_debug_config() -> MLPExperimentConfig:
    """Debug configuration with minimal training for troubleshooting."""
    return MLPExperimentConfig(
        name="debug",
        description="Minimal configuration for debugging and development",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[2],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=1.0,
        max_epochs=10,
        convergence_threshold=1e-2,
        dataset_type="xor",
        early_stopping=False,
        verbose=True,
        random_seed=42
    )


# =============================================================================
# EXPERIMENT REGISTRY
# =============================================================================

EXPERIMENT_CONFIGS = {
    "xor_breakthrough": get_xor_breakthrough_config,
    "quick_test": get_quick_test_config,
    "circles": get_circles_challenge_config,
    "moons": get_moons_config,
    "spirals": get_spirals_config,
    "arch_comparison": get_architecture_comparison_config,
    "activation_study": get_activation_study_config,
    "lr_study": get_learning_rate_study_config,
    "debug": get_debug_config
}


def get_experiment_config(experiment_name: str) -> MLPExperimentConfig:
    """
    Get configuration for a named experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        MLPExperimentConfig: Configuration for the experiment
        
    Raises:
        KeyError: If experiment name is not found
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise KeyError(f"Unknown experiment '{experiment_name}'. Available: {available}")
    
    return EXPERIMENT_CONFIGS[experiment_name]()


def list_available_experiments() -> List[str]:
    """Get list of available experiment names."""
    return list(EXPERIMENT_CONFIGS.keys())


def get_experiment_info(experiment_name: str) -> Dict[str, str]:
    """
    Get information about an experiment without creating the full config.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dict with experiment metadata
    """
    config = get_experiment_config(experiment_name)
    return {
        "name": config.name,
        "description": config.description,
        "architecture": f"{config.hidden_layers} hidden layers",
        "dataset": config.dataset_type,
        "max_epochs": str(config.max_epochs),
        "learning_rate": str(config.learning_rate)
    }


# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

def get_debug_override_config() -> Dict[str, Any]:
    """Configuration overrides for debug/development environment."""
    return {
        "max_epochs": 10,
        "convergence_threshold": 1e-2,
        "verbose": True,
        "save_plots": False,
        "early_stopping": False
    }


def get_production_override_config() -> Dict[str, Any]:
    """Configuration overrides for production environment."""
    return {
        "verbose": False,
        "save_plots": True,
        "early_stopping": True
    }


def apply_environment_overrides(config: MLPExperimentConfig, 
                                environment: str = "default") -> MLPExperimentConfig:
    """
    Apply environment-specific configuration overrides.
    
    Args:
        config: Base configuration
        environment: Environment type ("debug", "production", "default")
        
    Returns:
        Modified configuration
    """
    if environment == "debug":
        overrides = get_debug_override_config()
    elif environment == "production":
        overrides = get_production_override_config()
    else:
        return config  # No overrides for default
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logging.warning(f"Unknown config attribute for override: {key}")
    
    return config


# =============================================================================
# EDUCATIONAL EXPERIMENT SEQUENCES
# =============================================================================

def get_educational_sequence() -> List[str]:
    """
    Get recommended sequence of experiments for educational purposes.
    
    Returns:
        List of experiment names in recommended order
    """
    return [
        "debug",            # 1. Verify everything works
        "quick_test",       # 2. Fast XOR solution
        "xor_breakthrough",  # 3. Proper XOR solution
        "arch_comparison",  # 4. Compare architectures
        "activation_study",  # 5. Study activation functions
        "circles",          # 6. More complex non-linear problem
        "moons",            # 7. Another non-linear challenge
        "spirals"           # 8. Most challenging dataset
    ]


def get_research_sequence() -> List[str]:
    """
    Get experiment sequence for research/analysis purposes.
    
    Returns:
        List of experiment names for research
    """
    return [
        "xor_breakthrough",
        "lr_study",
        "activation_study",
        "arch_comparison",
        "circles",
        "moons",
        "spirals"
    ] 