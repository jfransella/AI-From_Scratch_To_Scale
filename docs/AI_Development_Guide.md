# **AI Development Guide**
## **Quick Start for AI Assistants**

This guide consolidates all essential information needed to develop the "AI From Scratch to Scale" project. Use this as your primary reference when implementing new models or working on shared infrastructure.

---

## **🎯 Project Overview**

**Mission**: Build 25 key neural network architectures chronologically to understand the evolution of AI, from Perceptron to modern transformers.

**Philosophy**: "Learning by building" - prioritize clarity and educational value over optimization.

**Approach**: Separate shared infrastructure from model-specific code, progress from NumPy implementations to framework-based solutions.

---

## **📁 Project Structure**

```
ai-from-scratch-to-scale\
├── data_utils\          # SHARED: Dataset loading & transformations
├── engine\              # SHARED: Training/evaluation engine with wandb
├── plotting\            # SHARED: Visualization generation
├── utils\               # SHARED: Logging, seeds, general utilities
├── tests\               # SHARED: Automated tests
├── models\              # Individual model implementations
│   ├── 01_perceptron\
│   │   ├── src\
│   │   │   ├── constants.py    # Fixed values for this model
│   │   │   ├── config.py       # Hyperparameters & settings
│   │   │   ├── model.py        # Model architecture
│   │   │   ├── train.py        # Training script
│   │   │   └── evaluate.py     # Evaluation script
│   │   ├── notebooks\          # Analysis notebooks
│   │   ├── outputs\            # Generated files
│   │   ├── requirements.txt    # Model-specific dependencies
│   │   └── README.md
│   └── 02_adaline\
└── requirements-dev.txt    # Development dependencies
```

---

## **🔧 Development Workflow**

### **Setting Up a New Model**

1. **Create Directory Structure**:
   ```powershell
   New-Item -ItemType Directory -Force -Path "models\XX_modelname\src"
   New-Item -ItemType Directory -Force -Path "models\XX_modelname\notebooks"
   New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\logs"
   New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\models"
   New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\visualizations"
   ```

2. **Create Virtual Environment**:
   ```powershell
   Set-Location models\XX_modelname
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install -r ..\..\requirements-dev.txt  # For development
   pip install -e ..\..  # Install shared packages
   ```

### **Standard Development Commands**

```powershell
# Training
python src\train.py --experiment iris-hard --visualize --epochs 100

# Evaluation
python src\evaluate.py --checkpoint path\to\model.pth --experiment iris-hard

# Testing
pytest ..\..\tests\  # Run all tests

# Formatting & Linting
black src\
flake8 src\
```

---

## **📝 Coding Standards**

### **Style Guidelines**
- **PEP 8** compliance with **black** formatting (88 character line length)
- **snake_case** for variables and functions
- **PascalCase** for classes
- **ALL_CAPS** for constants

### **File Structure Requirements**

**model.py**:
- Define model class inheriting from appropriate base (torch.nn.Module for frameworks)
- Include proper docstrings describing the model's purpose
- Keep architecture definition clean and well-commented
- Follow naming conventions for layers and methods

**train.py**:
- Standard argument parsing with required `--experiment` parameter
- Import and use shared `engine.Trainer` class for training logic
- Load configuration from `config.py` based on experiment name
- Handle model instantiation, data loading, and training orchestration
- Include proper error handling and logging

**evaluate.py**:
- Load pre-trained model from checkpoint
- Use shared `engine.Evaluator` for evaluation logic
- Support both local and wandb checkpoint loading
- Generate evaluation metrics and visualizations

### **Documentation Requirements**
- **Google-style docstrings** for all functions and classes
- **README.md** for each model with setup instructions and key findings
- **Inline comments** explaining complex logic (the "why", not the "what")

---

## **🗂️ Configuration Management**

### **config.py Approach**
- **Single function**: `get_config(experiment_name: str) -> dict`
- **Base configuration**: Default values for learning_rate, batch_size, epochs, seed, device
- **Experiment-specific overrides**: Each experiment can override base values
- **Dataset specification**: Include dataset name and any dataset-specific parameters
- **Return dictionary**: Consistent structure across all models

### **constants.py Purpose**
- **Model-specific fixed values**: Architecture parameters that don't change
- **File path constants**: Standardized directory paths for outputs (use Windows path separators)
- **Model metadata**: Name, version, historical context
- **Avoid hardcoding**: Keep all magic numbers and paths in one place

### **Configuration Principles**
- **No hardcoded values** in training or model scripts
- **Centralized management** through config.py
- **Easy experimentation** by adding new experiment configurations
- **Historical accuracy** - use parameters close to original implementations where possible

---

## **📊 Dataset Strategy**

### **Two-Phase Approach**
- **Strength Datasets**: Demonstrate where the model excels and validates the core innovation
- **Weakness Datasets**: Expose limitations that motivate the next model in the sequence
- **Educational Value**: Each dataset choice should teach something specific about the model

### **Dataset Loading Principles**
- **Shared data_utils**: Use centralized loading functions for consistency
- **Experiment-driven**: Dataset selection controlled by experiment name in config
- **Progressive Complexity**: Start simple (synthetic), move to real-world, then complex
- **Historical Context**: Use datasets appropriate to the model's historical period when possible

### **Standard Patterns**
- **Synthetic Data**: Generated datasets for controlled experiments (XOR, circles, etc.)
- **Classic Datasets**: Iris, MNIST for historical accuracy and comparison
- **Modern Datasets**: CIFAR-10, ImageNet subsets for complex models
- **Task-Specific**: Segmentation masks, text corpora based on model purpose

---

## **🧪 Testing Strategy**

### **Test Categories**
1. **Unit Tests**: Test individual functions and model components in isolation
2. **Integration Tests**: Verify shared components work together correctly
3. **Smoke Tests**: End-to-end training pipeline validation (single epoch)

### **Testing Approach**
- **Shared Infrastructure**: Focus testing on `engine\`, `data_utils\`, `plotting\`, `utils\`
- **Model-Specific**: Test model initialization, forward pass, and basic training
- **Data Loading**: Verify dataset loading functions work correctly
- **Visualization**: Test plot generation without requiring human validation

### **Test Organization**
- **Centralized Tests**: All tests in `tests\` directory at project root
- **Naming Convention**: `test_[component]_[functionality].py`
- **Pytest Framework**: Use pytest for all testing with appropriate fixtures
- **CI Integration**: Tests run automatically on pull requests

---

## **📈 Visualization & Logging**

### **Dual Logging System**
- **Python logging**: Human-readable narrative for console and log files
- **Weights & Biases (wandb)**: Structured metrics database for analysis
- **Shared utils**: Use centralized logging setup for consistency

### **Logging Principles**
- **Structured Information**: Log key metrics, epoch progress, and important events
- **Appropriate Levels**: Use INFO for progress, DEBUG for detailed information
- **Consistent Format**: Standardized logging patterns across all models
- **Historical Narrative**: Logs should tell the story of the training process

### **Visualization Strategy**
- **Flag-Activated**: Use `--visualize` flag to generate plots (not automatic)
- **Separation of Concerns**: Generate plots separate from training logic
- **Standard Locations**: Save to `outputs\visualizations\` directory
- **Educational Focus**: Visualizations should support learning objectives

---

## **🔄 Development Patterns**

### **Training Architecture**
- **Use Shared Engine**: Delegate training logic to `engine.Trainer` class
- **Standard Training Loop**: Train phase → Validation phase → Logging → Repeat
- **Proper Mode Setting**: Use `model.train()` and `model.eval()` appropriately
- **Gradient Management**: Clear gradients, compute loss, backpropagate, update weights
- **Progress Tracking**: Log metrics each epoch, save checkpoints, handle early stopping

### **Evaluation Architecture**
- **Use Shared Engine**: Delegate evaluation logic to `engine.Evaluator` class
- **No Gradient Computation**: Wrap evaluation in `torch.no_grad()` context
- **Comprehensive Metrics**: Compute accuracy, loss, and task-specific metrics
- **Prediction Collection**: Save predictions for detailed analysis
- **Visualization Integration**: Generate plots when `--visualize` flag is used

### **Error Handling Patterns**
- **Graceful Degradation**: Handle missing datasets, checkpoints, or configuration gracefully
- **Informative Messages**: Provide clear error messages that help debug issues
- **Validation**: Validate inputs, configurations, and model outputs
- **Recovery**: Allow partial failures without crashing entire training runs

---

## **🚀 Quick Commands Reference**

### **Creating a New Model**
```powershell
# 1. Set up directory structure
New-Item -ItemType Directory -Force -Path "models\XX_modelname\src"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\notebooks"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\logs"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\models"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\visualizations"

# 2. Create and activate virtual environment
Set-Location models\XX_modelname
python -m venv .venv
.venv\Scripts\activate

# 3. Create model-specific requirements.txt
# (Add only libraries needed for THIS model)

# 4. Install dependencies
pip install -r requirements.txt
pip install -r ..\..\requirements-dev.txt  # Development tools
pip install -e ..\..  # Shared packages
```

### **Standard Run Commands**
```powershell
# Basic training
python src\train.py --experiment iris-hard

# Training with visualization
python src\train.py --experiment iris-hard --visualize

# Training with custom epochs
python src\train.py --experiment iris-hard --epochs 50

# Evaluation
python src\evaluate.py --checkpoint outputs\models\model.pth --experiment iris-hard

# Testing
pytest ..\..\tests\test_modelname.py

# Code formatting
black src\
flake8 src\
```

---

## **🎯 Model Implementation Checklist**

### **Before Starting**
- [ ] Review historical context and original paper
- [ ] Identify strength and weakness datasets
- [ ] Plan experiment configurations

### **Implementation Phase**
- [ ] Create directory structure
- [ ] Implement `constants.py` with fixed values
- [ ] Implement `config.py` with experiment configurations
- [ ] Implement `model.py` with clear architecture
- [ ] Implement `train.py` with argument parsing
- [ ] Implement `evaluate.py` for model assessment
- [ ] Write unit tests for key functions
- [ ] Create model-specific `README.md`

### **Validation Phase**
- [ ] Test on strength datasets (should succeed)
- [ ] Test on weakness datasets (should fail/struggle)
- [ ] Generate visualizations with `--visualize`
- [ ] Run automated tests
- [ ] Format code with black
- [ ] Check code with flake8

### **Documentation Phase**
- [ ] Create analysis notebooks
- [ ] Document key findings and limitations
- [ ] Link to next model motivation

---

## **📚 Quick Links to Full Documentation**

- **[Project Charter](strategy/Project_Charter.md)**: Full project scope and roadmap
- **[Codebase Architecture](technical/Codebase_Architecture.md)**: Detailed technical architecture
- **[Coding Standards](technical/Coding_Standards.md)**: Complete style guide
- **[Notebook Strategy](strategy/Notebook_Strategy.md)**: Jupyter notebook guidelines
- **[Dataset Strategy](strategy/Dataset_Strategy.md)**: Complete dataset specifications

---

## **📋 Detailed Workflow Examples**

### **Complete Model Implementation Workflow**

#### **Example: Implementing a New MLP Model**

**Step 1: Research and Planning (30 minutes)**
```powershell
# Create project directory
New-Item -ItemType Directory -Force -Path "models\03_mlp\src"
New-Item -ItemType Directory -Force -Path "models\03_mlp\notebooks"
New-Item -ItemType Directory -Force -Path "models\03_mlp\outputs\logs"
New-Item -ItemType Directory -Force -Path "models\03_mlp\outputs\models"
New-Item -ItemType Directory -Force -Path "models\03_mlp\outputs\visualizations"
Set-Location models\03_mlp

# Research the model
# - Read original paper (if available)
# - Understand key innovations
# - Identify strength datasets (XOR, MNIST)
# - Identify weakness datasets (CIFAR-10)
```

**Step 2: Environment Setup (15 minutes)**
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Verify activation
python -c "import sys; print(sys.executable)"
# Should show: ...\models\03_mlp\.venv\Scripts\python.exe

# Create requirements.txt
@"
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
wandb>=0.13.0
"@ | Out-File -FilePath requirements.txt -Encoding utf8

# Install dependencies
pip install -r requirements.txt
pip install -r ..\..\requirements-dev.txt
pip install -e ..\..

# Test installation
python -c "from data_utils import load_dataset; print('Shared packages OK')"
```

**Step 3: Implement Constants (10 minutes)**
```python
# src/constants.py
"""
Constants for Multi-Layer Perceptron (MLP) model.
"""

# Model metadata
MODEL_NAME = "MLP"
MODEL_VERSION = "1.0.0"
YEAR_INTRODUCED = 1986
PAPER_TITLE = "Learning representations by back-propagating errors"
AUTHORS = ["David Rumelhart", "Geoffrey Hinton", "Ronald Williams"]

# Architecture constants
DEFAULT_HIDDEN_SIZES = [64, 32]
DEFAULT_ACTIVATION = "relu"
DEFAULT_OUTPUT_ACTIVATION = "softmax"

# Training constants
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# File paths (Windows style)
OUTPUTS_DIR = "outputs"
MODELS_DIR = "outputs\\models"
LOGS_DIR = "outputs\\logs"
PLOTS_DIR = "outputs\\visualizations"

# Experiment configurations
STRENGTH_EXPERIMENTS = ["xor", "iris_multiclass", "mnist_multiclass"]
WEAKNESS_EXPERIMENTS = ["cifar10"]  # Should struggle due to lack of spatial awareness

# Validation ranges
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1024
MIN_EPOCHS = 1
MAX_EPOCHS = 1000
```

**Step 4: Configure Experiments (20 minutes)**
```python
# src/config.py - Use template and customize
from templates.config import create_config

def get_config(experiment_name: str):
    """Get configuration for MLP experiments."""
    return create_config(
        model_name="MLP",
        model_type="mlp",
        experiment_name=experiment_name,
        env="default",
        # MLP-specific overrides
        hidden_size=64,
        num_layers=2,
        activation="relu",
        dropout=0.2,
        batch_norm=True
    )

# Test configuration
if __name__ == "__main__":
    config = get_config("xor")
    print(f"Config loaded: {config.experiment}")
```

**Step 5: Implement Model Architecture (45 minutes)**
```python
# src/model.py
"""
Multi-Layer Perceptron (MLP) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable hidden layers.
    
    Key Innovation: Hidden layers with non-linear activations allow
    learning of non-linear functions like XOR.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build layers
        layers = []
        input_size = config.input_size
        
        # Hidden layers
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_size, config.hidden_size))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_size))
            layers.append(self._get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            input_size = config.hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, config.output_size))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def get_features(self, x):
        """Get intermediate features for analysis."""
        features = []
        for layer in self.network:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                features.append(x.detach())
        return features

# Test model creation
if __name__ == "__main__":
    from config import get_config
    config = get_config("xor")
    model = MLP(config)
    print(f"Model created: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

**Step 6: Implement Training Script (30 minutes)**
```python
# src/train.py
"""
Training script for MLP model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import get_config
from model import MLP
from data_utils import load_dataset
from engine import Trainer
from utils import setup_logging, set_random_seed

def main():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., xor, iris_multiclass)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.experiment)
    
    # Apply command-line overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    
    # Setup environment
    if args.debug:
        config.log_level = 'DEBUG'
        config.epochs = min(config.epochs, 5)
    
    setup_logging(level=config.log_level)
    set_random_seed(config.seed)
    
    # Create model
    model = MLP(config)
    
    # Load data
    train_loader, val_loader = load_dataset(
        config.dataset,
        config.dataset_params,
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=args.wandb,
        generate_plots=args.visualize
    )
    
    # Train model
    print(f"Starting training for {config.experiment}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    training_history = trainer.train()
    
    print(f"Training completed!")
    print(f"Final train loss: {training_history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {training_history['val_loss'][-1]:.4f}")
    
    # Save model
    model_path = Path(config.checkpoint_dir) / f"{config.experiment}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
```

**Step 7: Test Basic Functionality (15 minutes)**
```powershell
# Test configuration
python src\train.py --experiment debug_small --epochs 2 --debug
# Should complete quickly without errors

# Test model on strength dataset
python src\train.py --experiment xor --epochs 50 --visualize
# Should achieve good performance on XOR problem

# Test model on weakness dataset
python src\train.py --experiment cifar10 --epochs 10 --visualize
# Should show limitations with complex image data
```

**Step 8: Implement Evaluation Script (20 minutes)**
```python
# src/evaluate.py
"""
Evaluation script for MLP model.
"""

import argparse
import torch
from pathlib import Path

from config import get_config
from model import MLP
from data_utils import load_dataset
from engine import Evaluator
from utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Evaluate MLP model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name for dataset')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate evaluation visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.experiment)
    setup_logging(level=config.log_level)
    
    # Create model and load checkpoint
    model = MLP(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Load test data
    _, _, test_loader = load_dataset(
        config.dataset,
        config.dataset_params,
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        config=config,
        test_loader=test_loader,
        generate_plots=args.visualize
    )
    
    # Evaluate model
    print(f"Evaluating model on {config.experiment}")
    results = evaluator.evaluate()
    
    print(f"Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
```

**Step 9: Testing and Validation (30 minutes)**
```powershell
# Run automated tests
pytest ..\..\tests\unit\test_mlp.py -v

# Test all experiment configurations
$experiments = @("xor", "iris_multiclass", "mnist_multiclass", "cifar10")
foreach ($exp in $experiments) {
    Write-Host "Testing $exp..."
    python src\train.py --experiment $exp --epochs 5 --debug
}

# Format code
black src\
flake8 src\

# Check for common issues
python -c "from src.model import MLP; print('Model imports OK')"
python -c "from src.config import get_config; print('Config imports OK')"
python -c "from src.train import main; print('Training imports OK')"
```

**Step 10: Create Documentation (25 minutes)**
```markdown
# models/03_mlp/README.md

# Multi-Layer Perceptron (MLP)

Implementation of the Multi-Layer Perceptron, the foundation of modern deep learning.

## Key Innovation
- Hidden layers with non-linear activations
- Backpropagation algorithm for training
- Ability to learn non-linear functions

## Quick Start
```powershell
# Activate environment
.venv\Scripts\activate

# Train on XOR problem (strength)
python src\train.py --experiment xor --visualize

# Train on CIFAR-10 (weakness)
python src\train.py --experiment cifar10 --epochs 50 --visualize
```

## Results Summary
- **Strengths**: Solves XOR, handles multi-class classification
- **Weaknesses**: Poor performance on complex image data
- **Next Model**: CNN to handle spatial data properly
```

### **Advanced Command-Line Usage Examples**

#### **Systematic Experimentation Workflow**
```powershell
# 1. Quick smoke test
python src\train.py --experiment debug_small --epochs 1 --debug

# 2. Hyperparameter sweep
$learning_rates = @(0.1, 0.01, 0.001)
$batch_sizes = @(16, 32, 64)

foreach ($lr in $learning_rates) {
    foreach ($bs in $batch_sizes) {
        $exp_name = "xor_lr${lr}_bs${bs}"
        python src\train.py --experiment xor --learning-rate $lr --batch-size $bs --epochs 100 --wandb
    }
}

# 3. Full evaluation suite
$experiments = @("xor", "iris_multiclass", "mnist_multiclass", "cifar10")
foreach ($exp in $experiments) {
    # Train
    python src\train.py --experiment $exp --epochs 100 --visualize --wandb
    
    # Evaluate
    $checkpoint = "outputs\models\${exp}_model.pth"
    python src\evaluate.py --checkpoint $checkpoint --experiment $exp --visualize
}
```

#### **Development and Debugging Workflow**
```powershell
# Development mode with fast iteration
python src\train.py --experiment debug_small --epochs 5 --debug --visualize

# Memory profiling
python -m memory_profiler src\train.py --experiment mnist_multiclass --epochs 1

# Performance profiling
python -m cProfile -o profile_output.prof src\train.py --experiment xor --epochs 10
python -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats(10)"

# GPU monitoring during training
# In separate terminal:
nvidia-smi -l 1

# Monitor log files in real-time
Get-Content -Path "outputs\logs\training.log" -Wait
```

#### **Batch Processing and Automation**
```powershell
# Batch training script
$experiments = @(
    @{name="xor"; epochs=200; lr=0.1},
    @{name="iris_multiclass"; epochs=100; lr=0.01},
    @{name="mnist_multiclass"; epochs=50; lr=0.001}
)

foreach ($exp in $experiments) {
    Write-Host "Training $($exp.name)..."
    python src\train.py --experiment $exp.name --epochs $exp.epochs --learning-rate $exp.lr --wandb
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $($exp.name) completed successfully"
    } else {
        Write-Host "✗ $($exp.name) failed"
    }
}

# Automated testing after changes
function Test-ModelImplementation {
    Write-Host "Running automated tests..."
    
    # 1. Unit tests
    pytest ..\..\tests\unit\test_mlp.py -v
    if ($LASTEXITCODE -ne 0) { return $false }
    
    # 2. Integration tests
    pytest ..\..\tests\integration\test_training_pipeline.py -v
    if ($LASTEXITCODE -ne 0) { return $false }
    
    # 3. Smoke tests
    python src\train.py --experiment debug_small --epochs 1 --debug
    if ($LASTEXITCODE -ne 0) { return $false }
    
    Write-Host "✓ All tests passed"
    return $true
}

# Run tests
if (Test-ModelImplementation) {
    Write-Host "Ready for production training"
} else {
    Write-Host "Fix issues before continuing"
}
```

### **Troubleshooting Integration Examples**

#### **Common Issues During Implementation**
```powershell
# Issue: Import errors
python -c "import sys; print('\n'.join(sys.path))"
python -c "from src.model import MLP; print('Model OK')"
# Fix: Ensure virtual environment is activated and packages installed

# Issue: CUDA out of memory
python src\train.py --experiment mnist_multiclass --batch-size 16  # Reduce batch size
# Or use gradient accumulation:
python src\train.py --experiment mnist_multiclass --batch-size 8 --gradient-accumulation-steps 4

# Issue: Training not converging
python src\train.py --experiment xor --learning-rate 0.1 --epochs 500 --visualize
# Check learning rate and increase epochs

# Issue: Model overfitting
python src\train.py --experiment mnist_multiclass --dropout 0.5 --weight-decay 0.01
# Add regularization
```

#### **Performance Debugging Example**
```powershell
# Step 1: Baseline performance
python src\train.py --experiment mnist_multiclass --epochs 10 --debug
# Note: training time per epoch

# Step 2: Optimize data loading
python src\train.py --experiment mnist_multiclass --epochs 10 --debug
# Modify config to increase num_workers

# Step 3: Profile bottlenecks
python -m torch.profiler src\train.py --experiment mnist_multiclass --epochs 1
# Analyze profiler output

# Step 4: GPU utilization
nvidia-smi dmon -s pucvmet -d 1
# Run training and monitor GPU usage
```

### **Integration with Shared Infrastructure**

#### **Using Shared Components Effectively**
```python
# Example: Leveraging engine.Trainer with custom callbacks
from engine import Trainer
from engine.callbacks import EarlyStopping, LearningRateScheduler

trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[
        EarlyStopping(patience=config.early_stopping_patience),
        LearningRateScheduler(scheduler_type='cosine')
    ]
)

# Example: Using plotting utilities
from plotting import generate_visualizations

plots_to_generate = config.plot_types
generate_visualizations(
    training_history,
    plots_to_generate,
    save_dir=config.plot_dir
)

# Example: Advanced data loading with data_utils
from data_utils import DatasetFactory, AugmentationPipeline

# Create custom dataset with augmentations
dataset = DatasetFactory.create(
    dataset_name=config.dataset,
    **config.dataset_params
)

if config.augmentation:
    augmentation = AugmentationPipeline(config.augmentation_params)
    dataset = augmentation.apply(dataset)

train_loader = dataset.get_dataloader(
    split='train',
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)
```

---

## **🆘 Common Issues & Solutions**

### **Import Errors**
- Ensure you've run `pip install -e ..\..` from the model directory
- Check that virtual environment is activated
- Verify shared packages are in the correct location

### **Training Failures**
- Check dataset loading with a small batch first
- Verify model architecture matches expected input/output shapes
- Ensure proper device placement (CPU vs GPU)

### **Visualization Issues**
- Make sure `--visualize` flag is included
- Check that `outputs\visualizations\` directory exists
- Verify plotting functions are imported correctly

---

This guide should serve as your primary reference for developing models in this project. For detailed specifications, refer to the individual strategy documents linked above. 