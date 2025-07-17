"""
Configuration management for Multi-Layer Perceptron (MLP) experiments.

This module provides experiment configurations for training MLPs on various
non-linearly separable datasets, with a focus on demonstrating the breakthrough
of solving the XOR problem that single-layer perceptrons cannot handle.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Optional engine imports (for advanced implementations)
try:
    from engine import TrainingConfig, EvaluationConfig

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# Local imports
try:
    from constants import ALL_EXPERIMENTS, DATASET_CONFIGS
except ImportError:
    ALL_EXPERIMENTS = []
    DATASET_CONFIGS = {}


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
# Engine-based Configuration (Advanced Pattern)
# =============================================================================

if HAS_ENGINE:

    def get_training_config(experiment_name: str, **overrides) -> TrainingConfig:
        """
        Get TrainingConfig for MLP experiment using engine framework.

        This function provides engine-based configuration similar to 01_Perceptron.

        Args:
            experiment_name: Name of the experiment to run
            **overrides: Any parameter overrides

        Returns:
            TrainingConfig instance configured for the experiment
        """
        # Base configuration for all MLP experiments
        base_config = {
            "experiment_name": experiment_name,
            "model_name": "MLP",
            "dataset_name": "synthetic",
            "learning_rate": 0.1,
            "max_epochs": 1000,
            "batch_size": None,  # Full batch for MLP
            "optimizer_type": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "lr_scheduler": None,
            "convergence_threshold": 1e-6,
            "patience": 50,
            "early_stopping": True,
            "validation_split": 0.2,
            "validation_freq": 1,
            "save_best_model": True,
            "save_final_model": True,
            "checkpoint_freq": 0,
            "output_dir": "outputs",
            "log_freq": 10,
            "verbose": True,
            "use_wandb": False,
            "wandb_project": "ai-from-scratch-mlp",
            "wandb_tags": ["mlp", experiment_name],
            "random_seed": 42,
            "device": "cpu",
        }

        # Experiment-specific overrides
        experiment_configs = {
            "xor_breakthrough": {
                "max_epochs": 1000,
                "learning_rate": 0.5,
                "convergence_threshold": 1e-6,
                "patience": 200,
                "wandb_tags": ["mlp", "breakthrough", "xor"],
            },
            "quick_test": {
                "max_epochs": 100,
                "learning_rate": 1.0,
                "convergence_threshold": 1e-4,
                "patience": 20,
                "wandb_tags": ["mlp", "debug", "quick"],
            },
            "circles_challenge": {
                "max_epochs": 2000,
                "learning_rate": 0.01,
                "convergence_threshold": 1e-6,
                "patience": 100,
                "wandb_tags": ["mlp", "challenge", "circles"],
            },
            "moons_classification": {
                "max_epochs": 2000,
                "learning_rate": 0.01,
                "convergence_threshold": 1e-6,
                "patience": 100,
                "wandb_tags": ["mlp", "classification", "moons"],
            },
            "spirals_challenge": {
                "max_epochs": 5000,
                "learning_rate": 0.001,
                "convergence_threshold": 1e-6,
                "patience": 200,
                "wandb_tags": ["mlp", "challenge", "spirals"],
            },
        }

        # Apply experiment-specific overrides
        if experiment_name in experiment_configs:
            base_config.update(experiment_configs[experiment_name])

        # Apply user overrides
        base_config.update(overrides)

        # Create and return TrainingConfig
        return TrainingConfig(**base_config)

    def get_evaluation_config(experiment_name: str, **overrides) -> EvaluationConfig:
        """
        Get EvaluationConfig for MLP experiment using engine framework.

        Args:
            experiment_name: Name of the experiment
            **overrides: Any parameter overrides

        Returns:
            EvaluationConfig instance configured for the experiment
        """
        # Base evaluation configuration
        base_config = {
            "compute_accuracy": True,
            "compute_precision": True,
            "compute_recall": True,
            "compute_f1": True,
            "compute_confusion_matrix": True,
            "compute_per_class": True,
            "store_predictions": True,
            "store_probabilities": True,
            "store_ground_truth": True,
            "verbose": True,
            "save_results": True,
            "output_path": f"outputs/{experiment_name}_evaluation.json",
            "device": "cpu",
        }

        # Apply user overrides
        base_config.update(overrides)

        # Create and return EvaluationConfig
        return EvaluationConfig(**base_config)


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
        description={
            "Solve XOR problem - the classic demonstration of MLP superiority over perceptrons"
        },
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
        random_seed=42,
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
        random_seed=42,
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
        random_seed=42,
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
        random_seed=42,
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
        random_seed=42,
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
        random_seed=42,
    )


def get_activation_study_config() -> MLPExperimentConfig:
    """
    Study different activation functions on XOR problem.
    """
    return MLPExperimentConfig(
        name="activation_study",
        description="Compare different activation functions on XOR",
        architecture_name="small",
        input_size=2,
        hidden_layers=[4],
        output_size=1,
        activation="relu",  # Will be overridden
        weight_init="he_normal",
        learning_rate=0.01,
        max_epochs=1000,
        convergence_threshold=1e-6,
        dataset_type="xor",
        early_stopping=True,
        random_seed=42,
    )


def get_learning_rate_study_config() -> MLPExperimentConfig:
    """
    Study different learning rates on XOR problem.
    """
    return MLPExperimentConfig(
        name="learning_rate_study",
        description="Compare different learning rates on XOR",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[2],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=0.1,  # Will be overridden
        max_epochs=1000,
        convergence_threshold=1e-6,
        dataset_type="xor",
        early_stopping=True,
        random_seed=42,
    )


def get_debug_config() -> MLPExperimentConfig:
    """
    Debug configuration for development and testing.
    """
    return MLPExperimentConfig(
        name="debug",
        description="Debug configuration for development",
        architecture_name="minimal",
        input_size=2,
        hidden_layers=[2],
        output_size=1,
        activation="sigmoid",
        weight_init="xavier_normal",
        learning_rate=0.5,
        max_epochs=50,
        convergence_threshold=1e-4,
        dataset_type="xor",
        early_stopping=False,
        verbose=True,
        random_seed=42,
    )


# =============================================================================
# Configuration Factory Functions
# =============================================================================


def get_experiment_config(
    experiment_name: str, _config_type: str = "simple"
) -> MLPExperimentConfig:
    """
    Get experiment configuration based on type.

    Args:
        experiment_name: Name of the experiment
        config_type: Type of configuration ("simple" or "engine")

    Returns:
        Configuration object
    """
    # Map experiment names to config functions
    config_functions = {
        "xor_breakthrough": get_xor_breakthrough_config,
        "quick_test": get_quick_test_config,
        "circles_challenge": get_circles_challenge_config,
        "moons_classification": get_moons_config,
        "spirals_challenge": get_spirals_config,
        "architecture_comparison": get_architecture_comparison_config,
        "activation_study": get_activation_study_config,
        "learning_rate_study": get_learning_rate_study_config,
        "debug": get_debug_config,
    }

    if experiment_name not in config_functions:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    return config_functions[experiment_name]()


def list_available_experiments() -> List[str]:
    """List available experiments."""
    return [
        "xor_breakthrough",
        "quick_test",
        "circles_challenge",
        "moons_classification",
        "spirals_challenge",
        "architecture_comparison",
        "activation_study",
        "learning_rate_study",
        "debug",
    ]


def get_experiment_info(experiment_name: str) -> Dict[str, str]:
    """Get information about a specific experiment."""
    config = get_experiment_config(experiment_name)
    return {
        "name": config.name,
        "description": config.description,
        "model_name": "MLP",
        "dataset": config.dataset_type,
        "difficulty": "medium",
        "expected_accuracy": 0.95,
    }


def apply_environment_overrides(
    config: MLPExperimentConfig, environment: str = "default"
) -> MLPExperimentConfig:
    """
    Apply environment-specific overrides to configuration.

    Args:
        config: Base configuration
        environment: Environment name

    Returns:
        Modified configuration
    """
    if environment == "debug":
        config.max_epochs = min(config.max_epochs, 20)
        config.verbose = True
        config.save_plots = False
    elif environment == "production":
        config.verbose = False
        config.save_plots = True
        config.save_model = True

    return config


# =============================================================================
# Legacy Compatibility
# =============================================================================


def get_config(experiment_name: str, _env: str = "default") -> Dict[str, Any]:
    """
    Legacy compatibility function that returns a dictionary config.

    This function provides backward compatibility with older code that expects
    a dictionary configuration.

    Args:
        experiment_name: Name of the experiment to run
        env: Environment (ignored in new implementation)

    Returns:
        Dictionary with complete configuration (flattened)
    """
    config = get_experiment_config(experiment_name)

    # Convert to dictionary
    config_dict = {
        "name": config.name,
        "description": config.description,
        "architecture_name": config.architecture_name,
        "input_size": config.input_size,
        "hidden_layers": config.hidden_layers,
        "output_size": config.output_size,
        "activation": config.activation,
        "weight_init": config.weight_init,
        "learning_rate": config.learning_rate,
        "max_epochs": config.max_epochs,
        "convergence_threshold": config.convergence_threshold,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "momentum": config.momentum,
        "dataset_type": config.dataset_type,
        "dataset_params": config.dataset_params,
        "early_stopping": config.early_stopping,
        "save_model": config.save_model,
        "save_history": config.save_history,
        "verbose": config.verbose,
        "random_seed": config.random_seed,
        "output_dir": config.output_dir,
        "save_plots": config.save_plots,
    }

    return config_dict


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def validate_experiment(experiment_name: str) -> bool:
    """Validate that an experiment name is supported."""
    return experiment_name in ALL_EXPERIMENTS


def get_model_config(_experiment_name: str, **overrides) -> Dict[str, Any]:
    """Get model-specific configuration."""
    base_config = {
        "input_size": 2,
        "output_size": 1,
        "learning_rate": 0.1,
        "max_epochs": 1000,
    }
    base_config.update(overrides)
    return base_config


def get_dataset_config(experiment_name: str) -> Dict[str, Any]:
    """Get dataset-specific configuration."""
    return DATASET_CONFIGS.get(experiment_name, {
        "dataset_name": experiment_name,
        "dataset_params": {},
        "description": f"Dataset for {experiment_name} experiment"
    })


if __name__ == "__main__":
    # Test configuration loading
    print("Testing MLP configuration system...")

    # Test simple config
    simple_config = get_experiment_config("xor_breakthrough", "simple")
    print(f"Simple config: {simple_config.name}")

    # Test engine config (if available)
    if HAS_ENGINE:
        engine_config = get_experiment_config("xor_breakthrough", "engine")
        print(f"Engine config: {engine_config.name}")

    # Test environment overrides
    modified_config = apply_environment_overrides(simple_config, "debug")
    print(f"Debug config max_epochs: {modified_config.max_epochs}")

    print("Configuration system test completed!")
