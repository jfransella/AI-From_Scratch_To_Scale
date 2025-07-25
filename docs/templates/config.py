"""
Template for config.py - Configuration Management

This template provides a comprehensive configuration management system for the
"AI From Scratch to Scale" project. It implements both simple and advanced
configuration patterns to support different implementation needs.

CONFIGURATION PATTERNS:
1. Simple Pattern: Dataclass-based configuration (like 03_MLP)
2. Advanced Pattern: Engine-based configuration (like 01_Perceptron)

Replace MODEL_NAME with the actual model name (e.g., "Perceptron", "MLP", etc.)
"""

import os
import json
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Optional engine imports (for advanced implementations)
try:
    from engine import TrainingConfig, EvaluationConfig
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# Handle PyTorch imports gracefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import constants using explicit module loading for compatibility
def load_constants():
    """Load constants module with explicit path handling."""
    try:
        # Try direct import first
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
                return getattr(constants_module, "MODEL_NAME", "Template"), getattr(constants_module, "ALL_EXPERIMENTS", [])
        # Final fallback
        return "Template", ["debug", "quick_test", "standard", "production"]

MODEL_NAME, ALL_EXPERIMENTS = load_constants()


# =============================================================================
# SIMPLE CONFIGURATION PATTERN (Dataclass-based)
# =============================================================================

@dataclass
class SimpleExperimentConfig:
    """Simple configuration for a single experiment (like 03_MLP)."""
    
    # Experiment metadata
    name: str
    description: str
    
    # Architecture configuration
    architecture_name: str
    input_size: int
    hidden_size: Optional[int] = None
    output_size: int = 1
    activation: str = "relu"
    weight_init: str = "xavier_normal"
    
    # Training configuration
    learning_rate: float = 0.01
    max_epochs: int = 100
    convergence_threshold: float = 1e-6
    patience: int = 20
    batch_size: Optional[int] = None
    weight_decay: float = 0.0
    momentum: float = 0.0
    
    # Dataset configuration
    dataset_type: str = "synthetic"
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
        
        # Validate parameters
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_epochs > 0, "Max epochs must be positive"
        assert self.input_size > 0, "Input size must be positive"
        assert self.output_size > 0, "Output size must be positive"
        assert 0 <= self.convergence_threshold <= 1, "Convergence threshold must be between 0 and 1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# =============================================================================
# ADVANCED CONFIGURATION PATTERN (Engine-based)
# =============================================================================

if HAS_ENGINE:
    def get_training_config(experiment_name: str, **overrides) -> TrainingConfig:
        """
        Get TrainingConfig for experiment using engine framework.
        
        This function provides engine-based configuration similar to 01_Perceptron.
        
        Args:
            experiment_name: Name of the experiment to run
            **overrides: Any parameter overrides
            
        Returns:
            TrainingConfig instance configured for the experiment
        """
        # Base configuration for all experiments
        base_config = {
            "experiment_name": experiment_name,
            "model_name": MODEL_NAME,
            "dataset_name": "synthetic",
            "learning_rate": 0.01,
            "max_epochs": 100,
            "batch_size": 32,
            "optimizer_type": "adam",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "lr_scheduler": None,
            "convergence_threshold": 1e-6,
            "patience": 20,
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
            "wandb_project": f"ai-from-scratch-{MODEL_NAME.lower()}",
            "wandb_tags": [MODEL_NAME.lower(), experiment_name],
            "random_seed": 42,
            "device": "cpu",
        }
        
        # Experiment-specific overrides
        experiment_configs = {
            "debug": {
                "max_epochs": 20,
                "learning_rate": 0.1,
                "log_freq": 1,
                "verbose": True,
                "patience": 10,
                "batch_size": 16,
            },
            "quick_test": {
                "max_epochs": 50,
                "learning_rate": 0.1,
                "log_freq": 5,
                "patience": 20,
                "batch_size": 32,
            },
            "standard": {
                "max_epochs": 100,
                "learning_rate": 0.01,
                "convergence_threshold": 1e-6,
                "patience": 30,
                "batch_size": 32,
            },
            "production": {
                "max_epochs": 200,
                "learning_rate": 0.001,
                "convergence_threshold": 1e-8,
                "patience": 50,
                "batch_size": 64,
                "use_wandb": True,
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
        Get EvaluationConfig for experiment using engine framework.
        
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
# CONFIGURATION FACTORY FUNCTIONS
# =============================================================================

def get_experiment_config(experiment_name: str, config_type: str = "simple") -> Union[SimpleExperimentConfig, TrainingConfig]:
    """
    Get experiment configuration based on type.
    
    Args:
        experiment_name: Name of the experiment
        config_type: "simple" or "advanced"
        
    Returns:
        Configuration object
    """
    if config_type == "simple":
        # Return simple dataclass configuration
        return SimpleExperimentConfig(
            name=experiment_name,
            description=f"Standard {MODEL_NAME} experiment",
            architecture_name="standard",
            input_size=2,
            output_size=1,
            learning_rate=0.01,
            max_epochs=100,
        )
    elif config_type == "advanced" and HAS_ENGINE:
        # Return engine-based configuration
        return get_training_config(experiment_name)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


def list_available_experiments() -> List[str]:
    """List all available experiments."""
    return ALL_EXPERIMENTS


def get_experiment_info(experiment_name: str) -> Dict[str, str]:
    """Get information about a specific experiment."""
    return {
        "name": experiment_name,
        "description": f"Standard {MODEL_NAME} experiment",
        "difficulty": "medium",
        "expected_accuracy": "0.8-0.9"
    }


def apply_wandb_defaults(config: Union[SimpleExperimentConfig, TrainingConfig], wandb_sweep_config: Optional[Dict[str, Any]] = None) -> Union[SimpleExperimentConfig, TrainingConfig]:
    """
    Apply W&B sweep configuration overrides.
    
    This function follows the pattern from successful implementations
    to integrate with W&B hyperparameter sweeps.
    
    Args:
        config: Base configuration object
        wandb_sweep_config: W&B sweep configuration dictionary
        
    Returns:
        Updated configuration object
    """
    if wandb_sweep_config is None:
        return config
    
    # Apply sweep overrides based on config type
    if isinstance(config, SimpleExperimentConfig):
        # Update dataclass fields directly
        for key, value in wandb_sweep_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    elif HAS_ENGINE and isinstance(config, TrainingConfig):
        # For TrainingConfig, create new instance with updates
        config_dict = config.__dict__.copy()
        config_dict.update(wandb_sweep_config)
        config = TrainingConfig(**config_dict)
    
    return config


def validate_config(config: Union[SimpleExperimentConfig, TrainingConfig]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if isinstance(config, SimpleExperimentConfig):
        # Validation is handled in __post_init__
        return True
    elif HAS_ENGINE and isinstance(config, TrainingConfig):
        # Add engine-specific validation
        assert config.learning_rate > 0, "Learning rate must be positive"
        assert config.max_epochs > 0, "Max epochs must be positive"
        assert 0 <= config.convergence_threshold <= 1, "Convergence threshold must be between 0 and 1"
        return True
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def get_device_config() -> str:
    """Get the appropriate device configuration."""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def get_config(experiment_name: str, env: str = "default") -> Dict[str, Any]:
    """
    Get complete configuration for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        env: Environment name
        
    Returns:
        Configuration dictionary
    """
    # Get base configuration
    config = get_experiment_config(experiment_name, "simple")
    
    # Apply environment overrides
    config = apply_environment_overrides(config, env)
    
    # Convert to dictionary
    return config.to_dict()


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


def get_model_config(experiment_name: str, **overrides) -> Dict[str, Any]:
    """Get model-specific configuration."""
    base_config = {
        "input_size": 2,
        "output_size": 1,
        "learning_rate": 0.01,
        "max_epochs": 100,
    }
    base_config.update(overrides)
    return base_config


def get_dataset_config(experiment_name: str) -> Dict[str, Any]:
    """Get dataset-specific configuration."""
    return {
        "dataset_name": experiment_name,
        "dataset_params": {},
        "description": f"Dataset for {experiment_name} experiment"
    } 