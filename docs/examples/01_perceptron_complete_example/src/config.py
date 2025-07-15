"""
Configuration management for Perceptron model.

This file implements the hierarchical configuration system for the Perceptron
model, demonstrating how to use the template configuration system effectively.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the template directory to the path to import the config system
template_dir = Path(__file__).parent.parent.parent.parent / "templates"
sys.path.insert(0, str(template_dir))

try:
    from config import create_config, LinearConfig
except ImportError:
    # Fallback for when templates are not available
    print("Warning: Could not import template configuration system")
    
    class LinearConfig:
        """Fallback LinearConfig class when templates are not available."""
        def __init__(self, **kwargs):
            # Set default attributes
            self.model_name = "Perceptron"
            self.experiment = "default"
            self.learning_rate = 0.1
            self.epochs = 100
            self.input_size = 2
            self.output_size = 1
            self.dataset = "synthetic"
            self.expected_accuracy = 0.8
            self.should_converge = True
            self.description = "Default configuration"
            
            # Override with provided kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def create_config(model_name: str, model_type: str, experiment_name: str, **kwargs) -> LinearConfig:
        return LinearConfig(model_name=model_name, experiment=experiment_name, **kwargs)

from constants import (
    MODEL_NAME,
    MODEL_VERSION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    ALL_EXPERIMENTS,
    DATASET_SPECS,
    get_experiment_info,
    get_expected_performance
)


def get_perceptron_config(experiment_name: str, env: str = "default") -> LinearConfig:
    """
    Get configuration for Perceptron experiments.
    
    This function creates a complete configuration by:
    1. Starting with LinearConfig base class
    2. Adding Perceptron-specific defaults
    3. Applying experiment-specific overrides
    4. Applying environment-specific overrides
    
    Args:
        experiment_name: Name of the experiment to run
        env: Environment (default, dev, prod, debug)
        
    Returns:
        Complete configuration object
        
    Raises:
        ValueError: If experiment name is not recognized
    """
    
    # Validate experiment name
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available experiments: {ALL_EXPERIMENTS}")
    
    # Get experiment info
    exp_info = get_experiment_info(experiment_name)
    expected_perf = get_expected_performance(experiment_name)
    
    # Create base configuration with Perceptron-specific defaults
    config = create_config(
        model_name=MODEL_NAME,
        model_type="linear",
        experiment_name=experiment_name,
        env=env,
        
        # Perceptron-specific overrides
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=1,  # Classic perceptron uses online learning
        activation="step",  # Classic perceptron activation
        
        # Architecture parameters from experiment
        input_size=exp_info.get("input_size", 2),
        output_size=exp_info.get("output_size", 1),
        
        # No hidden layers for perceptron
        hidden_size=None,
        num_layers=1,
        
        # Perceptron doesn't use these features
        dropout=0.0,
        batch_norm=False,
        weight_decay=0.0,
        
        # Experiment-specific dataset configuration
        dataset=_get_dataset_name(experiment_name),
        dataset_params=_get_dataset_params(experiment_name),
        
        # Evaluation metrics
        eval_metrics=["accuracy", "precision", "recall"],
        
        # Visualization settings
        plot_types=_get_plot_types(experiment_name),
        
        # Expected performance for validation
        expected_accuracy=expected_perf["expected_accuracy"],
        should_converge=expected_perf["should_converge"],
        
        # Model metadata
        model_version=MODEL_VERSION,
        description=f"Perceptron on {experiment_name}: {exp_info['description']}"
    )
    
    return config


def _get_dataset_name(experiment_name: str) -> str:
    """Get the dataset name for a given experiment."""
    dataset_mapping = {
        "and_gate": "and_gate",
        "or_gate": "or_gate",
        "xor_gate": "xor_gate",
        "iris_easy": "iris",
        "iris_hard": "iris",
        "mnist_binary": "mnist",
        "circles": "circles",
        "moons": "moons",
        "debug_small": "and_gate",  # Use simple dataset for debug
        "debug_overfit": "and_gate"
    }
    return dataset_mapping.get(experiment_name, "synthetic")


def _get_dataset_params(experiment_name: str) -> Dict[str, Any]:
    """Get dataset parameters for a given experiment."""
    dataset_params = {
        "and_gate": {
            "n_samples": 1000,
            "noise": 0.1,
            "gate_type": "and"
        },
        "or_gate": {
            "n_samples": 1000,
            "noise": 0.1,
            "gate_type": "or"
        },
        "xor_gate": {
            "n_samples": 1000,
            "noise": 0.1,
            "gate_type": "xor"
        },
        "iris_easy": {
            "classes": ["setosa", "versicolor", "virginica"],
            "binary_target": "setosa"  # Setosa vs others
        },
        "iris_hard": {
            "classes": ["versicolor", "virginica"]  # Harder separation
        },
        "mnist_binary": {
            "classes": [0, 1],
            "flatten": True,
            "normalize": True
        },
        "circles": {
            "n_samples": 1000,
            "noise": 0.1,
            "factor": 0.5
        },
        "moons": {
            "n_samples": 1000,
            "noise": 0.1
        },
        "debug_small": {
            "n_samples": 100,
            "noise": 0.05,
            "gate_type": "and"
        },
        "debug_overfit": {
            "n_samples": 4,
            "noise": 0.0,
            "gate_type": "and"
        }
    }
    return dataset_params.get(experiment_name, {})


def _get_plot_types(experiment_name: str) -> List[str]:
    """Get appropriate plot types for a given experiment."""
    base_plots = ["loss_curve"]
    
    # Add decision boundary for 2D problems
    if experiment_name in ["and_gate", "or_gate", "xor_gate", "circles", "moons"]:
        base_plots.append("decision_boundary")
    
    # Add weight evolution for interesting cases
    if experiment_name in ["xor_gate", "iris_hard", "circles"]:
        base_plots.append("weight_evolution")
    
    # Add convergence analysis for all non-debug experiments
    if not experiment_name.startswith("debug"):
        base_plots.append("convergence_analysis")
    
    return base_plots


def get_available_experiments() -> List[str]:
    """Get list of available experiments for Perceptron."""
    return ALL_EXPERIMENTS.copy()


def get_experiment_description(experiment_name: str) -> str:
    """Get description of a specific experiment."""
    if experiment_name not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    exp_info = get_experiment_info(experiment_name)
    return exp_info.get("description", f"Experiment: {experiment_name}")


def validate_config(config: LinearConfig) -> None:
    """
    Validate a configuration object.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required attributes
    required_attrs = [
        "model_name", "experiment", "learning_rate", "epochs",
        "input_size", "output_size", "dataset"
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Missing required configuration attribute: {attr}")
    
    # Check value ranges
    if config.learning_rate <= 0 or config.learning_rate > 1:
        raise ValueError("Learning rate must be between 0 and 1")
    
    if config.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if config.input_size <= 0:
        raise ValueError("Input size must be positive")
    
    if config.output_size <= 0:
        raise ValueError("Output size must be positive")
    
    # Check experiment-specific constraints
    if hasattr(config, "expected_accuracy"):
        if config.expected_accuracy < 0 or config.expected_accuracy > 1:
            raise ValueError("Expected accuracy must be between 0 and 1")


def print_config_summary(config: LinearConfig) -> None:
    """Print a summary of the configuration."""
    print(f"\nPerceptron Configuration Summary")
    print(f"=" * 40)
    print(f"Model: {config.model_name}")
    print(f"Experiment: {config.experiment}")
    print(f"Dataset: {config.dataset}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print(f"Input Size: {config.input_size}")
    print(f"Output Size: {config.output_size}")
    
    if hasattr(config, "expected_accuracy"):
        print(f"Expected Accuracy: {config.expected_accuracy:.1%}")
    
    if hasattr(config, "should_converge"):
        print(f"Should Converge: {'Yes' if config.should_converge else 'No'}")
    
    if hasattr(config, "description"):
        print(f"Description: {config.description}")
    
    print(f"=" * 40)


# Convenience function for backward compatibility
def get_config(experiment_name: str) -> LinearConfig:
    """
    Get configuration for a specific experiment.
    
    This is the main entry point for getting configurations.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Complete configuration object
    """
    return get_perceptron_config(experiment_name)


if __name__ == "__main__":
    # Example usage and testing
    print("Perceptron Configuration System")
    print("=" * 40)
    
    # Show available experiments
    print("Available experiments:")
    for exp in get_available_experiments():
        desc = get_experiment_description(exp)
        print(f"  {exp}: {desc}")
    
    print("\nExample configurations:")
    
    # Test a few example configurations
    test_experiments = ["and_gate", "xor_gate", "iris_easy", "debug_small"]
    
    for exp in test_experiments:
        try:
            config = get_config(exp)
            validate_config(config)
            print(f"\n✓ {exp} configuration valid")
            print_config_summary(config)
        except Exception as e:
            print(f"\n✗ {exp} configuration failed: {e}")
    
    print("\nConfiguration system test completed!") 