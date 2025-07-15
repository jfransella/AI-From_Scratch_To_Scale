"""
Template for config.py - Advanced Configuration Management

This template provides a comprehensive configuration management system for the
"AI From Scratch to Scale" project. It implements a hierarchical configuration
system with inheritance, validation, and environment-specific overrides.

CONFIGURATION INHERITANCE HIERARCHY:
1. BaseConfig - Universal defaults for all models
2. ModelTypeConfig - Specific to model categories (e.g., LinearConfig, CNNConfig)
3. ModelConfig - Specific to individual models (e.g., PerceptronConfig, MLPConfig)
4. ExperimentConfig - Specific to experiment/dataset combinations
5. EnvironmentConfig - Runtime overrides (dev, prod, debug)

Replace [MODEL_NAME] with the actual model name (e.g., "Perceptron", "MLP", etc.)
"""

import os
import json
import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


# =============================================================================
# Base Configuration Classes
# =============================================================================

@dataclass
class BaseConfig:
    """
    Base configuration with universal defaults for all models.
    
    This is the foundation of the configuration hierarchy. All other configs
    inherit from this class and can override any parameters.
    """
    
    # Model metadata
    model_name: str = "[MODEL_NAME]"
    model_type: str = "base"  # linear, cnn, rnn, transformer, etc.
    version: str = "1.0.0"
    description: str = "Base configuration template"
    
    # Architecture parameters
    input_size: int = 2
    output_size: int = 1
    hidden_size: Optional[int] = None
    num_layers: int = 1
    activation: str = "relu"  # relu, tanh, sigmoid, leaky_relu, gelu
    dropout: float = 0.0
    batch_norm: bool = False
    
    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"  # adam, sgd, rmsprop, adagrad
    loss_function: str = "crossentropy"  # crossentropy, mse, bce, huber
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9  # For Adam
    beta2: float = 0.999  # For Adam
    epsilon: float = 1e-8  # For Adam
    
    # Learning rate scheduling
    scheduler: Optional[str] = None  # step, cosine, exponential, plateau
    lr_decay_factor: float = 0.1
    lr_decay_steps: int = 50
    lr_patience: int = 10  # For plateau scheduler
    
    # Regularization
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    gradient_clip: Optional[float] = None
    noise_std: float = 0.0  # Input noise for regularization
    
    # Data parameters
    dataset: str = "synthetic"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    
    # Data augmentation
    augmentation: bool = False
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    
    # Device and performance
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = False
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    # Logging and monitoring
    log_level: str = "INFO"
    log_interval: int = 10
    save_interval: int = 50
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_best_only: bool = True
    save_last: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Visualization
    plot_types: List[str] = field(default_factory=lambda: ['loss_curve'])
    plot_interval: int = 20
    plot_dir: str = "plots"
    
    # Experiment tracking
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    tensorboard_dir: str = "runs"
    
    # Evaluation
    eval_interval: int = 5
    eval_metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == "auto":
            self.device = self._get_best_device()
        
        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Validate configuration
        self._validate()
    
    def _get_best_device(self) -> str:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _validate(self) -> None:
        """Validate configuration parameters."""
        # Validate ranges
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Number of epochs must be positive"
        assert self.input_size > 0, "Input size must be positive"
        assert self.output_size > 0, "Output size must be positive"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
        
        # Validate splits
        splits = [self.train_split, self.val_split, self.test_split]
        assert abs(sum(splits) - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # Validate choices
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
        assert self.optimizer in valid_optimizers, f"Optimizer must be one of {valid_optimizers}"
        
        valid_losses = ['crossentropy', 'mse', 'bce', 'huber']
        assert self.loss_function in valid_losses, f"Loss function must be one of {valid_losses}"
        
        valid_activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu']
        assert self.activation in valid_activations, f"Activation must be one of {valid_activations}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# =============================================================================
# Model Type Configurations
# =============================================================================

@dataclass
class LinearConfig(BaseConfig):
    """Configuration for linear models (Perceptron, ADALINE, etc.)."""
    
    model_type: str = "linear"
    activation: str = "linear"  # Linear models often use linear activation
    batch_norm: bool = False  # Not typically used in simple linear models
    learning_rate: float = 0.1  # Often need higher LR for linear models
    

@dataclass
class MLPConfig(BaseConfig):
    """Configuration for Multi-Layer Perceptron models."""
    
    model_type: str = "mlp"
    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.2
    batch_norm: bool = True
    

@dataclass
class CNNConfig(BaseConfig):
    """Configuration for Convolutional Neural Networks."""
    
    model_type: str = "cnn"
    
    # CNN-specific parameters
    num_filters: int = 32
    filter_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    num_conv_layers: int = 2
    
    # Image-specific parameters
    input_channels: int = 1
    image_size: int = 28
    
    # Often need different training settings
    learning_rate: float = 0.001
    batch_size: int = 64
    

@dataclass
class RNNConfig(BaseConfig):
    """Configuration for Recurrent Neural Networks."""
    
    model_type: str = "rnn"
    
    # RNN-specific parameters
    hidden_size: int = 128
    num_layers: int = 1
    rnn_type: str = "LSTM"  # LSTM, GRU, RNN
    bidirectional: bool = False
    
    # Sequence-specific parameters
    sequence_length: int = 100
    vocab_size: int = 10000
    embedding_dim: int = 100
    
    # Different defaults for sequence models
    batch_size: int = 32
    learning_rate: float = 0.001
    

@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for Transformer models."""
    
    model_type: str = "transformer"
    
    # Transformer-specific parameters
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    
    # Attention parameters
    attention_dropout: float = 0.1
    
    # Often need specific training settings
    learning_rate: float = 0.0001
    batch_size: int = 16
    warmup_steps: int = 4000
    

@dataclass
class GenerativeConfig(BaseConfig):
    """Configuration for generative models (VAE, GAN, etc.)."""
    
    model_type: str = "generative"
    
    # Generative-specific parameters
    latent_dim: int = 100
    
    # GAN-specific
    discriminator_lr: float = 0.0002
    generator_lr: float = 0.0002
    
    # VAE-specific
    kl_weight: float = 1.0
    reconstruction_loss: str = "mse"
    

# =============================================================================
# Experiment Configurations
# =============================================================================

def get_experiment_config(base_config: BaseConfig, experiment_name: str) -> BaseConfig:
    """
    Get experiment-specific configuration by merging base config with experiment overrides.
    
    Args:
        base_config: Base configuration to start from
        experiment_name: Name of the experiment
        
    Returns:
        Updated configuration with experiment-specific parameters
    """
    
    # Define experiment-specific overrides
    experiment_overrides = {
        
        # === Strength Experiments ===
        'xor': {
            'description': f'{base_config.model_name} on XOR problem - classic non-linear test',
            'dataset': 'xor',
            'dataset_params': {'n_samples': 1000, 'noise': 0.1},
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.1,
            'epochs': 200,
            'batch_size': 16,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
            'eval_metrics': ['accuracy', 'precision', 'recall'],
        },
        
        'and_gate': {
            'description': f'{base_config.model_name} on AND gate - simple linear problem',
            'dataset': 'and_gate',
            'dataset_params': {'n_samples': 1000, 'noise': 0.05},
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.1,
            'epochs': 100,
            'batch_size': 16,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
        },
        
        'iris_easy': {
            'description': f'{base_config.model_name} on Iris dataset - Setosa vs others',
            'dataset': 'iris',
            'dataset_params': {'classes': ['setosa', 'versicolor', 'virginica'], 'binary_target': 'setosa'},
            'input_size': 4,
            'output_size': 1,
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 32,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'feature_importance'],
        },
        
        'iris_hard': {
            'description': f'{base_config.model_name} on Iris dataset - Versicolor vs Virginica',
            'dataset': 'iris',
            'dataset_params': {'classes': ['versicolor', 'virginica']},
            'input_size': 4,
            'output_size': 1,
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 16,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'feature_importance'],
        },
        
        'iris_multiclass': {
            'description': f'{base_config.model_name} on Iris dataset - all three classes',
            'dataset': 'iris',
            'dataset_params': {'classes': ['setosa', 'versicolor', 'virginica']},
            'input_size': 4,
            'output_size': 3,
            'learning_rate': 0.01,
            'epochs': 150,
            'batch_size': 32,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'confusion_matrix'],
        },
        
        'mnist_binary': {
            'description': f'{base_config.model_name} on MNIST - 0s vs 1s',
            'dataset': 'mnist',
            'dataset_params': {'classes': [0, 1], 'flatten': True},
            'input_size': 784,
            'output_size': 1,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'sample_predictions'],
        },
        
        'mnist_multiclass': {
            'description': f'{base_config.model_name} on MNIST - all 10 digits',
            'dataset': 'mnist',
            'dataset_params': {'classes': list(range(10)), 'flatten': True},
            'input_size': 784,
            'output_size': 10,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'confusion_matrix', 'sample_predictions'],
        },
        
        'fashion_mnist': {
            'description': f'{base_config.model_name} on Fashion-MNIST',
            'dataset': 'fashion_mnist',
            'dataset_params': {'flatten': True},
            'input_size': 784,
            'output_size': 10,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'confusion_matrix'],
        },
        
        'cifar10': {
            'description': f'{base_config.model_name} on CIFAR-10',
            'dataset': 'cifar10',
            'dataset_params': {'flatten': False},
            'input_size': 3072 if base_config.model_type == 'linear' else 32,
            'input_channels': 3,
            'image_size': 32,
            'output_size': 10,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 128,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'confusion_matrix'],
        },
        
        # === Weakness Experiments ===
        'circles': {
            'description': f'{base_config.model_name} on concentric circles - non-linear challenge',
            'dataset': 'circles',
            'dataset_params': {'n_samples': 1000, 'noise': 0.1, 'factor': 0.5},
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.01,
            'epochs': 300,
            'batch_size': 32,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
        },
        
        'moons': {
            'description': f'{base_config.model_name} on moons dataset - non-linear challenge',
            'dataset': 'moons',
            'dataset_params': {'n_samples': 1000, 'noise': 0.1},
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.01,
            'epochs': 200,
            'batch_size': 32,
            'loss_function': 'bce',
            'plot_types': ['loss_curve', 'decision_boundary'],
        },
        
        # === Sequence Experiments ===
        'shakespeare': {
            'description': f'{base_config.model_name} on Shakespeare text generation',
            'dataset': 'shakespeare',
            'dataset_params': {'sequence_length': 100, 'char_level': True},
            'sequence_length': 100,
            'vocab_size': 256,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'text_samples'],
        },
        
        'imdb': {
            'description': f'{base_config.model_name} on IMDb sentiment analysis',
            'dataset': 'imdb',
            'dataset_params': {'max_length': 512, 'vocab_size': 10000},
            'sequence_length': 512,
            'vocab_size': 10000,
            'output_size': 2,
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32,
            'loss_function': 'crossentropy',
            'plot_types': ['loss_curve', 'attention_weights'],
        },
        
        # === Memory/Associative Experiments ===
        'hopfield_patterns': {
            'description': f'{base_config.model_name} on binary patterns - associative memory',
            'dataset': 'hopfield_patterns',
            'dataset_params': {'pattern_size': 25, 'num_patterns': 3},
            'input_size': 25,
            'output_size': 25,
            'learning_rate': 0.1,
            'epochs': 1,  # Hopfield networks train in one shot
            'batch_size': 1,
            'loss_function': 'mse',
            'plot_types': ['pattern_recovery', 'energy_landscape'],
        },
        
        # === Debug/Development Experiments ===
        'debug_small': {
            'description': f'{base_config.model_name} debug configuration - small dataset',
            'dataset': 'xor',
            'dataset_params': {'n_samples': 100, 'noise': 0.1},
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.1,
            'epochs': 10,
            'batch_size': 8,
            'loss_function': 'bce',
            'log_interval': 1,
            'plot_types': ['loss_curve'],
        },
        
        'debug_overfit': {
            'description': f'{base_config.model_name} debug configuration - overfit test',
            'dataset': 'xor',
            'dataset_params': {'n_samples': 4, 'noise': 0.0},  # Minimal dataset
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 0.1,
            'epochs': 1000,
            'batch_size': 4,
            'loss_function': 'bce',
            'log_interval': 100,
            'early_stopping_patience': 1000,  # Disable early stopping
            'plot_types': ['loss_curve'],
        },
    }
    
    # Get experiment overrides
    if experiment_name not in experiment_overrides:
        available_experiments = list(experiment_overrides.keys())
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {available_experiments}")
    
    # Create new config with overrides
    config_dict = base_config.to_dict()
    config_dict.update(experiment_overrides[experiment_name])
    
    # Add experiment metadata
    config_dict['experiment'] = experiment_name
    
    return base_config.from_dict(config_dict)


# =============================================================================
# Environment-Specific Configurations
# =============================================================================

def apply_environment_config(config: BaseConfig, env: str = "default") -> BaseConfig:
    """
    Apply environment-specific configuration overrides.
    
    Args:
        config: Base configuration
        env: Environment name (default, dev, prod, debug)
        
    Returns:
        Configuration with environment-specific overrides
    """
    
    env_overrides = {
        'dev': {
            'epochs': min(config.epochs, 10),  # Faster training for development
            'log_interval': 1,
            'save_interval': 5,
            'plot_interval': 5,
            'log_level': 'DEBUG',
            'deterministic': True,
            'wandb_project': f"{config.wandb_project}_dev" if config.wandb_project else None,
        },
        
        'debug': {
            'epochs': 5,
            'batch_size': min(config.batch_size, 8),
            'log_interval': 1,
            'save_interval': 1,
            'plot_interval': 1,
            'log_level': 'DEBUG',
            'deterministic': True,
            'num_workers': 0,  # Easier debugging with single-threaded
        },
        
        'prod': {
            'log_level': 'INFO',
            'deterministic': True,
            'cudnn_benchmark': True,
            'mixed_precision': True,
            'compile_model': True,
        },
        
        'fast': {
            'epochs': max(1, config.epochs // 10),
            'log_interval': max(1, config.log_interval // 2),
            'save_interval': max(1, config.save_interval // 2),
            'early_stopping_patience': max(3, config.early_stopping_patience // 2),
            'mixed_precision': True,
        },
    }
    
    if env in env_overrides:
        config_dict = config.to_dict()
        config_dict.update(env_overrides[env])
        return config.from_dict(config_dict)
    
    return config


# =============================================================================
# Factory Functions
# =============================================================================

def create_config(
    model_name: str,
    model_type: str,
    experiment_name: str,
    env: str = "default",
    **kwargs
) -> BaseConfig:
    """
    Factory function to create complete configuration.
    
    This function implements the full configuration inheritance hierarchy:
    1. Start with model-type-specific base config
    2. Apply experiment-specific overrides
    3. Apply environment-specific overrides
    4. Apply any additional kwargs
    
    Args:
        model_name: Name of the model (e.g., "Perceptron", "MLP")
        model_type: Type of model (linear, mlp, cnn, rnn, transformer, generative)
        experiment_name: Name of the experiment
        env: Environment (default, dev, prod, debug)
        **kwargs: Additional overrides
        
    Returns:
        Complete configuration object
    """
    
    # Step 1: Create base config based on model type
    type_configs = {
        'linear': LinearConfig,
        'mlp': MLPConfig,
        'cnn': CNNConfig,
        'rnn': RNNConfig,
        'transformer': TransformerConfig,
        'generative': GenerativeConfig,
    }
    
    config_class = type_configs.get(model_type, BaseConfig)
    base_config = config_class(model_name=model_name)
    
    # Step 2: Apply experiment-specific overrides
    config = get_experiment_config(base_config, experiment_name)
    
    # Step 3: Apply environment-specific overrides
    config = apply_environment_config(config, env)
    
    # Step 4: Apply additional kwargs
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = config.from_dict(config_dict)
    
    return config


def get_available_experiments() -> List[str]:
    """Get list of all available experiments."""
    # This mirrors the experiments defined in get_experiment_config
    return [
        'xor', 'and_gate', 'iris_easy', 'iris_hard', 'iris_multiclass',
        'mnist_binary', 'mnist_multiclass', 'fashion_mnist', 'cifar10',
        'circles', 'moons', 'shakespeare', 'imdb', 'hopfield_patterns',
        'debug_small', 'debug_overfit'
    ]


def get_available_model_types() -> List[str]:
    """Get list of all available model types."""
    return ['linear', 'mlp', 'cnn', 'rnn', 'transformer', 'generative']


def print_config_hierarchy():
    """Print the configuration inheritance hierarchy documentation."""
    print("""
    CONFIGURATION INHERITANCE HIERARCHY
    ===================================
    
    1. BaseConfig (Universal defaults)
       ├── model_name, version, description
       ├── Architecture: input_size, output_size, hidden_size, etc.
       ├── Training: learning_rate, batch_size, epochs, optimizer, etc.
       ├── Data: dataset, splits, augmentation, etc.
       ├── Regularization: dropout, weight_decay, etc.
       ├── Device: device, mixed_precision, etc.
       └── Monitoring: logging, checkpointing, visualization, etc.
    
    2. ModelTypeConfig (Category-specific defaults)
       ├── LinearConfig (Perceptron, ADALINE)
       ├── MLPConfig (Multi-Layer Perceptron)
       ├── CNNConfig (LeNet, AlexNet, ResNet)
       ├── RNNConfig (RNN, LSTM, GRU)
       ├── TransformerConfig (Transformer, BERT)
       └── GenerativeConfig (VAE, GAN, DDPM)
    
    3. ExperimentConfig (Dataset/task-specific overrides)
       ├── Strength experiments (show model capabilities)
       ├── Weakness experiments (expose model limitations)
       └── Debug experiments (development and testing)
    
    4. EnvironmentConfig (Runtime overrides)
       ├── dev (fast iteration)
       ├── debug (detailed logging)
       ├── prod (optimized performance)
       └── fast (quick testing)
    
    5. Additional kwargs (final overrides)
    
    USAGE EXAMPLES:
    ==============
    
    # Basic usage
    config = create_config('Perceptron', 'linear', 'xor')
    
    # With environment
    config = create_config('MLP', 'mlp', 'mnist_multiclass', env='dev')
    
    # With additional overrides
    config = create_config('CNN', 'cnn', 'cifar10', env='prod', 
                          learning_rate=0.0001, batch_size=256)
    
    # Direct instantiation
    config = MLPConfig(model_name='CustomMLP', hidden_size=128)
    config = get_experiment_config(config, 'iris_multiclass')
    config = apply_environment_config(config, 'debug')
    """)


# =============================================================================
# Utility Functions
# =============================================================================

def print_config(config: BaseConfig) -> None:
    """Print configuration in a readable format."""
    print(f"\n{'='*60}")
    print(f"Configuration for {config.model_name}")
    print(f"Experiment: {getattr(config, 'experiment', 'default')}")
    print(f"Environment: {getattr(config, 'environment', 'default')}")
    print(f"{'='*60}")
    
    # Group parameters by category
    categories = {
        'Model': ['model_name', 'model_type', 'version', 'description'],
        'Architecture': ['input_size', 'output_size', 'hidden_size', 'num_layers', 'activation'],
        'Training': ['learning_rate', 'batch_size', 'epochs', 'optimizer', 'loss_function'],
        'Data': ['dataset', 'train_split', 'val_split', 'test_split'],
        'Regularization': ['dropout', 'weight_decay', 'l1_penalty', 'l2_penalty'],
        'Device': ['device', 'mixed_precision', 'compile_model'],
        'Monitoring': ['log_level', 'log_interval', 'save_interval', 'plot_types'],
    }
    
    for category, params in categories.items():
        print(f"\n{category}:")
        for param in params:
            if hasattr(config, param):
                value = getattr(config, param)
                if value is not None:
                    print(f"  {param}: {value}")
    
    print(f"\n{'='*60}\n")


def validate_all_experiments(model_name: str, model_type: str) -> Dict[str, bool]:
    """
    Validate all experiments for a given model.
    
    Args:
        model_name: Name of the model
        model_type: Type of the model
        
    Returns:
        Dictionary mapping experiment names to validation results
    """
    results = {}
    experiments = get_available_experiments()
    
    for experiment in experiments:
        try:
            config = create_config(model_name, model_type, experiment)
            results[experiment] = True
        except Exception as e:
            results[experiment] = False
            print(f"Validation failed for {experiment}: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("=== Configuration System Demo ===\n")
    
    # Show hierarchy
    print_config_hierarchy()
    
    # Show available options
    print("Available model types:", get_available_model_types())
    print("Available experiments:", get_available_experiments())
    
    # Example configurations
    print("\n=== Example Configurations ===")
    
    # Linear model example
    config1 = create_config('Perceptron', 'linear', 'xor', env='dev')
    print_config(config1)
    
    # MLP example
    config2 = create_config('MLP', 'mlp', 'mnist_multiclass', env='prod', 
                           hidden_size=256, dropout=0.3)
    print_config(config2)
    
    # Validation test
    print("\n=== Validation Test ===")
    results = validate_all_experiments('TestModel', 'mlp')
    print(f"Validation results: {sum(results.values())}/{len(results)} passed") 