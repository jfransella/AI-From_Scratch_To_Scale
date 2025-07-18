# AI Development Guide

This guide provides comprehensive instructions for developing AI models in the "AI From Scratch to Scale" project, following our unified approach that supports both engine-based and simple implementation patterns.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Development Patterns](#development-patterns)
3. [Model Implementation](#model-implementation)
4. [Configuration Management](#configuration-management)
5. [Training and Evaluation](#training-and-evaluation)
6. [Testing and Validation](#testing-and-validation)
7. [Documentation Standards](#documentation-standards)
8. [Code Quality](#code-quality)
9. [Educational Integration](#educational-integration)

## Project Overview

The "AI From Scratch to Scale" project implements neural networks from first principles, progressing from simple perceptrons to complex modern architectures. Our goal is to provide both educational value and production-ready implementations.

### Key Principles

- **Educational Focus**: Each model demonstrates fundamental concepts
- **Historical Accuracy**: Implementations match original papers
- **Unified Framework**: Support both simple and advanced patterns
- **Production Ready**: Scalable and maintainable code
- **Comprehensive Testing**: Thorough validation and benchmarking

## Development Patterns

We support two main development patterns to accommodate different needs:

### Pattern 1: Simple Implementation (Basic)

**Use when**: Quick prototyping, educational demonstrations, or when engine framework is not available.

**Characteristics**:

- Direct PyTorch implementation
- Simple configuration with dataclasses
- Manual training loops
- Self-contained functionality
- Minimal dependencies

**Example**: `models/03_MLP/` (simple implementation)

**Key Files**:

- `model.py`: Direct PyTorch implementation
- `config.py`: Dataclass-based configuration
- `train.py`: Manual training loops
- `constants.py`: Model metadata and validation

### Pattern 2: Engine-Based Implementation (Advanced)

**Use when**: Production deployments, unified training pipelines, or advanced features needed.

**Characteristics**:

- Inherits from `BaseModel` interface
- Uses unified `TrainingConfig` and `EvaluationConfig`
- Integrated with engine framework
- Advanced logging and monitoring
- Comprehensive experiment management

**Example**: `models/01_Perceptron/` (engine-integrated)

**Key Files**:

- `model.py`: Engine-integrated implementation
- `config.py`: Engine-based configuration functions
- `train.py`: Engine-based training
- `constants.py`: Comprehensive metadata and validation

### Template Support

Our templates now support both patterns with improved alignment:

- `docs/templates/model.py`: Includes both simple and advanced versions
- `docs/templates/config.py`: Supports both dataclass and engine configurations
- `docs/templates/train.py`: Provides both manual and engine-based training
- `docs/templates/constants.py`: Comprehensive metadata and validation

## Model Implementation

### Basic Model Structure

Every model should follow this structure:

```python
class ModelName(nn.Module):
    """Model description with historical context."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize architecture
        self._build_model()
        self._initialize_weights()
        
    def forward(self, x):
        """Forward pass implementation."""
        pass
    
    def predict(self, x):
        """Make predictions."""
        pass
    
    def get_model_info(self):
        """Get model information."""
        pass
```

### Advanced Model Structure (Engine Integration)

For engine-based implementations:

```python
class ModelNameAdvanced(ModelName, BaseModel):
    """Advanced version with engine integration."""
    
    def fit(self, x_data, y_target):
        """Required by BaseModel interface."""
        pass
    
    def get_loss(self, outputs, targets):
        """Compute loss for engine framework."""
        pass
```

### Required Methods

All models must implement:

1. **`__init__()`**: Model initialization
2. **`forward()`**: Forward pass
3. **`predict()`**: Prediction interface
4. **`get_model_info()`**: Model metadata

Optional methods for advanced features:

1. **`predict_proba()`**: Probability predictions
2. **`save_checkpoint()`**: Model persistence
3. **`load_from_checkpoint()`**: Model loading
4. **`get_loss()`**: Loss computation (for engine)

### Historical Context

Each model should include:

- **Original paper reference**
- **Year introduced and authors**
- **Key innovations and contributions**
- **Problems solved**
- **Limitations**
- **Historical significance**

## Configuration Management

### Simple Configuration (Dataclass Pattern)

```python
@dataclass
class SimpleExperimentConfig:
    name: str
    description: str
    input_size: int
    output_size: int
    learning_rate: float = 0.01
    max_epochs: int = 100
    # ... other parameters
```

### Advanced Configuration (Engine Pattern)

```python
def get_training_config(experiment_name: str, **overrides) -> TrainingConfig:
    """Get engine-based configuration."""
    base_config = {
        "experiment_name": experiment_name,
        "model_name": "ModelName",
        "learning_rate": 0.01,
        "max_epochs": 100,
        # ... other parameters
    }
    return TrainingConfig(**base_config)
```

### Configuration Best Practices

1. **Validation**: Always validate configuration parameters
2. **Defaults**: Provide sensible defaults for all parameters
3. **Documentation**: Document all configuration options
4. **Environment Support**: Support different environments (debug, production)
5. **Backward Compatibility**: Maintain compatibility with existing code

## Training and Evaluation

### Simple Training Pattern

```python
def train_manually(config, args):
    """Manual training implementation."""
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    criterion = create_loss_function(config)
    
    for epoch in range(config.max_epochs):
        # Training loop implementation
        pass
    
    return results
```

### Engine-Based Training Pattern

```python
def train_with_engine(config, args):
    """Engine-based training implementation."""
    model = create_model(config)
    trainer = Trainer(model=model, config=config)
    results = trainer.train()
    return results
```

### Training Script Structure

Every training script should support:

1. **Command-line interface** with argparse
2. **Experiment configuration** loading
3. **Device management** (CPU/GPU)
4. **Logging and monitoring**
5. **Result saving**
6. **Visualization support**

### Evaluation Standards

All models should support:

1. **Accuracy metrics** (classification)
2. **Loss metrics** (regression)
3. **Confusion matrices** (multi-class)
4. **Precision/Recall/F1** (binary classification)
5. **Custom metrics** (model-specific)

## Testing and Validation

### Unit Testing

Every model should have comprehensive unit tests:

```python
def test_model_creation():
    """Test model initialization."""
    model = ModelName(input_size=2, output_size=1)
    assert model.input_size == 2
    assert model.output_size == 1

def test_forward_pass():
    """Test forward pass functionality."""
    model = ModelName(input_size=2, output_size=1)
    x = torch.randn(1, 2)
    output = model(x)
    assert output.shape == (1, 1)
```

### Integration Testing

Test complete training workflows:

```python
def test_training_workflow():
    """Test complete training workflow."""
    config = get_experiment_config("test_experiment")
    model, results = train_experiment(config)
    assert results["converged"] or results["epochs_trained"] > 0
```

### Validation Standards

1. **Parameter validation**: All inputs validated
2. **Shape validation**: Tensor shapes checked
3. **Range validation**: Parameter bounds enforced
4. **Type validation**: Correct data types required

## Documentation Standards

### Model Documentation

Every model should include:

1. **Historical context** and significance
2. **Mathematical formulation**
3. **Implementation details**
4. **Usage examples**
5. **Performance benchmarks**

### Code Documentation

Follow these standards:

1. **Docstrings**: All functions and classes documented
2. **Type hints**: All parameters and return values typed
3. **Comments**: Complex logic explained
4. **Examples**: Usage examples provided

### README Files

Each model directory should contain:

1. **Overview**: What the model does
2. **Installation**: Setup instructions
3. **Usage**: How to use the model
4. **Examples**: Code examples
5. **Results**: Performance benchmarks
6. **References**: Academic references

## Code Quality

### Style Guidelines

Follow these standards:

1. **PEP 8**: Python style guide compliance
2. **Type hints**: All functions typed
3. **Docstrings**: Comprehensive documentation
4. **Error handling**: Proper exception handling
5. **Logging**: Appropriate logging levels

### Quality Tools

Use these tools for code quality:

1. **pylint**: Code analysis and style checking
2. **black**: Code formatting
3. **mypy**: Type checking
4. **pytest**: Testing framework
5. **coverage**: Test coverage measurement

### Performance Standards

1. **Memory efficiency**: Minimize memory usage
2. **Computational efficiency**: Optimize algorithms
3. **Scalability**: Handle large datasets
4. **Reproducibility**: Deterministic results

## Educational Integration

### Learning Objectives

Each model should support specific learning objectives:

1. **Conceptual understanding**: What the model does
2. **Mathematical foundations**: How it works
3. **Implementation details**: How to code it
4. **Historical context**: Why it matters
5. **Practical applications**: Where it's used

### Experiment Design

Design experiments that demonstrate:

1. **Model strengths**: Where the model excels
2. **Model limitations**: Where it struggles
3. **Parameter sensitivity**: How parameters affect performance
4. **Comparison studies**: How it compares to other models

### Visualization Requirements

Provide visualizations for:

1. **Training curves**: Loss and accuracy over time
2. **Decision boundaries**: For 2D classification problems
3. **Parameter sensitivity**: How parameters affect results
4. **Model comparisons**: Side-by-side comparisons

## Template Usage

### Creating a New Model

1. **Copy templates**: Use `docs/templates/` as starting point
2. **Choose pattern**: Decide between simple or advanced pattern
3. **Customize model**: Implement model-specific logic
4. **Configure experiments**: Define relevant experiments
5. **Add tests**: Create comprehensive test suite
6. **Document**: Write README and docstrings

### Template Customization

Templates support customization through:

1. **Placeholder replacement**: Replace template placeholders
2. **Pattern selection**: Choose simple or advanced pattern
3. **Feature selection**: Enable/disable optional features
4. **Integration options**: Select integration level

### Template Alignment

The templates have been updated to align with successful implementations:

- **Model Template**: Matches patterns from 01_Perceptron and 03_MLP
- **Config Template**: Supports both dataclass and engine patterns
- **Train Template**: Provides both manual and engine-based training
- **Constants Template**: Comprehensive metadata and validation

## Best Practices

### Development Workflow

1. **Start with templates**: Use provided templates
2. **Choose appropriate pattern**: Simple for quick prototyping, advanced for production
3. **Implement incrementally**: Build and test step by step
4. **Validate thoroughly**: Test all functionality
5. **Document completely**: Write comprehensive documentation
6. **Optimize carefully**: Profile and optimize performance

### Code Organization

1. **Modular design**: Separate concerns into modules
2. **Clear interfaces**: Well-defined function signatures
3. **Consistent naming**: Follow naming conventions
4. **Error handling**: Graceful error handling
5. **Logging**: Appropriate logging throughout

### Testing Strategy

1. **Unit tests**: Test individual components
2. **Integration tests**: Test complete workflows
3. **Performance tests**: Test scalability
4. **Regression tests**: Prevent breaking changes
5. **Educational tests**: Verify learning objectives

## Conclusion

This guide provides a comprehensive framework for developing AI models in our project. By following these guidelines, you'll create models that are both educational and production-ready, supporting our mission to build AI from scratch to scale.

For specific implementation details, refer to the individual model directories and the template files in `docs/templates/`.
