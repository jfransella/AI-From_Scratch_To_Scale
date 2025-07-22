# **Quick Reference Cards: AI From Scratch to Scale**

This document provides quick reference cards for common patterns, commands, and configurations used in the "AI From
Scratch to Scale" project. Keep this handy for rapid development and troubleshooting.

## **📁 Directory Structure Reference**

### **Project Root Structure**

```text`nai-from-scratch-to-scale\
├── data_utils\              # SHARED: Dataset loading & transformations
├── engine\                  # SHARED: Training/evaluation engine with wandb
├── plotting\                # SHARED: Visualization generation
├── utils\                   # SHARED: Logging, seeds, general utilities
├── tests\                   # SHARED: Automated tests
├── models\                  # Individual model implementations
├── docs\                    # Project documentation
├── .gitignore
├── README.md
└── requirements-dev.txt     # Development dependencies
```text`n### **Model Directory Structure**

```text`nmodels\XX_modelname\
├── src\
│   ├── __init__.py
│   ├── constants.py         # Fixed values for this model
│   ├── config.py           # Hyperparameters & experiment settings
│   ├── model.py            # Model architecture implementation
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── notebooks\              # Analysis notebooks
│   ├── 01_Theory_and_Intuition.ipynb
│   ├── 02_Code_Walkthrough.ipynb
│   └── 03_Empirical_Analysis.ipynb
├── outputs\                # Generated files
│   ├── logs\               # Training logs
│   ├── models\             # Saved model checkpoints
│   └── visualizations\     # Generated plots
├── .venv\                  # Model-specific virtual environment
├── requirements.txt        # Model-specific dependencies
└── README.md              # Model documentation
```text`n### **Current Implementation Status**

```text`nmodels\
├── 01_perceptron\          # ✅ Engine-based pattern (Advanced)
├── 02_adaline\             # 📋 Next - Simple pattern (Educational)  
└── 03_mlp\                 # ✅ Simple pattern (Educational)
```text`n### **Documentation Structure**

```text`ndocs\
├── README.md                    # Documentation guide
├── AI_Development_Guide.md      # Main entry point
├── Quick_Reference.md           # This file
├── Development_FAQ.md           # Troubleshooting guide
├── strategy\                    # High-level planning
│   ├── Project_Charter.md
│   ├── Dataset_Strategy.md
│   └── Notebook_Strategy.md
├── technical\                   # Implementation details
│   ├── Codebase_Architecture.md
│   └── Coding_Standards.md
└── templates\                   # Code templates
    ├── model.py
    ├── train.py
    ├── config.py
    ├── constants.py
    └── requirements.txt
```text`n---

## **⚡ Command Patterns Reference**

### **Environment Setup Commands**

```powershell
# Create new model directory
New-Item -ItemType Directory -Force -Path "models\XX_modelname\src"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\notebooks"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\logs"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\models"
New-Item -ItemType Directory -Force -Path "models\XX_modelname\outputs\visualizations"

# Setup virtual environment
Set-Location models\XX_modelname
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r ..\..\requirements-dev.txt
pip install -e ..\..

# Verify installation
python -c "from data_utils import load_dataset; print('OK')"
```text`n### **Training Commands**

```powershell
# Basic training
python src\train.py --experiment xor

# Training with visualization
python src\train.py --experiment xor --visualize

# Training with custom parameters
python src\train.py --experiment xor --epochs 100 --batch-size 32 --learning-rate 0.01

# Debug mode training
python src\train.py --experiment debug_small --debug

# Training with wandb logging
python src\train.py --experiment xor --wandb
```text`n### **Evaluation Commands**

```powershell
# Basic evaluation
python src\evaluate.py --checkpoint outputs\models\model.pth --experiment xor

# Evaluation with visualization
python src\evaluate.py --checkpoint outputs\models\model.pth --experiment xor --visualize

# Evaluate all experiments
$experiments = @("xor", "iris_multiclass", "mnist_binary")
foreach ($exp in $experiments) {
    python src\evaluate.py --checkpoint "outputs\models\${exp}_model.pth" --experiment $exp
}
```text`n### **Testing Commands**

```powershell
# Run all tests
pytest ..\..\tests\ -v

# Run specific test category
pytest ..\..\tests\unit\ -v
pytest ..\..\tests\integration\ -v
pytest ..\..\tests\smoke\ -v

# Run model-specific tests
pytest ..\..\tests\unit\test_modelname.py -v

# Run with coverage
pytest ..\..\tests\ -v --cov=src --cov-report=html
```text`n### **Code Quality Commands**

```powershell
# Format code
black src\
black notebooks\

# Check code style
flake8 src\

# Type checking (if using mypy)
mypy src\

# Run all quality checks
black src\ && flake8 src\ && pytest ..\..\tests\smoke\ -v
```text`n---

## **🔧 Configuration Reference**

## **🎯 Implementation Patterns Reference**

### **Pattern 1: Engine-Based (Advanced)**

**Used by**: `01_perceptron`  
**Best for**: Production deployment, advanced research, unified pipelines

```python
# Engine-based model structure
class YourModel(nn.Module, BaseModel):
    def forward(self, x): ...
    def predict(self, x): ...
    def get_model_info(self): ...
    def fit(self, x_data, y_target): ...

# Configuration
def get_training_config(experiment_name: str) -> TrainingConfig:
    return TrainingConfig(experiment_name=experiment_name, ...)

# Training
trainer = Trainer(ModelAdapter(model), config, train_loader, val_loader)
result = trainer.train()
```text`n### **Pattern 2: Simple (Educational)**

**Used by**: `03_mlp`, **Recommended for**: `02_adaline`  
**Best for**: Educational demonstrations, quick prototyping, conceptual studies

```python
# Simple model structure
class YourModel(nn.Module):
    def forward(self, x): ...
    def predict(self, x): ...
    def get_model_info(self): ...

# Configuration
 (2)
@dataclass
class ExperimentConfig:
    name: str
    learning_rate: float = 0.01
    # ... other parameters

# Training
 (2)
for epoch in range(config.epochs):
    # Manual training loop
    optimizer.step()
```text`n### **Model Configuration Hierarchy**

```text`nEngine-Based Models:
├── TrainingConfig + ModelConfig pattern
├── BaseModel interface requirements
└── Engine framework integration

Simple Models:
├── Dataclass configurations
├── Direct parameter management
└── Manual training loops
```text`n### **Configuration Creation Pattern**

```python
# Using factory function
from templates.config import create_config

config = create_config(
    model_name="MLP",
    model_type="mlp",
    experiment_name="xor",
    env="default",
    # Additional overrides
    learning_rate=0.01,
    batch_size=32
)

# Direct instantiation
from templates.config import MLPConfig
config = MLPConfig(
    model_name="CustomMLP",
    learning_rate=0.01,
    batch_size=32
)
```text`n### **Common Configuration Parameters**

| Parameter | Default | Description | Valid Values |
|-----------|---------|-------------|--------------|
| `learning_rate` | 0.01 | Learning rate for optimizer | 1e-6 to 1.0 |
| `batch_size` | 32 | Training batch size | 1 to 1024 |
| `epochs` | 100 | Number of training epochs | 1 to 1000 |
| `optimizer` | "adam" | Optimizer type | adam, sgd, rmsprop |
| `loss_function` | "crossentropy" | Loss function | crossentropy, mse, bce |
| `device` | "auto" | Training device | auto, cpu, cuda, mps |
| `seed` | 42 | Random seed | Any integer |

### **Experiment Name Patterns**

| Pattern | Description | Example |
|---------|-------------|---------|
| `debug_small` | Quick debug with tiny dataset | `debug_small` |
| `debug_overfit` | Overfitting test with minimal data | `debug_overfit` |
| `{dataset}` | Standard dataset experiment | `xor`, `iris_multiclass` |
| `{dataset}_{param}` | Dataset with specific parameter | `mnist_binary`, `cifar10` |

---

## **📊 Dataset Reference**

### **Dataset Categories**

| Category | Examples | Storage Format | Use Cases |
|----------|----------|----------------|-----------|
| **Synthetic** | XOR, circles, moons | NPZ, CSV | Algorithm validation |
| **Classic ML** | Iris, Wine, Boston | CSV + JSON | Benchmarking |
| **Image** | MNIST, CIFAR-10, Fashion-MNIST | HDF5, PNG | Vision models |
| **Text** | Shakespeare, IMDb | TXT, JSON | NLP models |
| **Graph** | Cora, social networks | NPZ, JSON | Graph models |

### **Dataset Loading Pattern**

```python
from data_utils import load_dataset

# Basic loading
train_loader, val_loader = load_dataset(
    dataset='xor',
    batch_size=32,
    train_split=0.7,
    val_split=0.3
)

# With parameters
train_loader, val_loader = load_dataset(
    dataset='mnist',
    dataset_params={'classes': [0, 1], 'flatten': True},
    batch_size=64
)
```text`n### **Common Dataset Parameters**

| Dataset | Common Parameters | Example |
|---------|-------------------|---------|
| `xor` | `n_samples`, `noise` | `{'n_samples': 1000, 'noise': 0.1}` |
| `iris` | `classes`, `binary_target` | `{'classes': ['setosa', 'versicolor']}` |
| `mnist` | `classes`, `flatten` | `{'classes': [0, 1], 'flatten': True}` |
| `cifar10` | `flatten`, `augment` | `{'flatten': False, 'augment': True}` |

---

## **🧪 Testing Reference**

### **Test Categories**

| Category | Purpose | Location | Run Command |
|----------|---------|----------|-------------|
| **Unit** | Test individual functions | `tests\unit\` | `pytest tests\unit\ -v` |
| **Integration** | Test component interactions | `tests\integration\` | `pytest tests\integration\ -v` |
| **Smoke** | Quick end-to-end validation | `tests\smoke\` | `pytest tests\smoke\ -v` |
| **Performance** | Performance benchmarking | `tests\performance\` | `pytest tests\performance\ -v` |

### **Test Naming Conventions**

| Element | Format | Example |
|---------|--------|---------|
| **File** | `test_{component}_{functionality}.py` | `test_model_forward_pass.py` |
| **Function** | `test_{what}_{condition}_{expected}` | `test_model_forward_pass_returns_correct_shape` |
| **Class** | `Test{Component}{Functionality}` | `TestModelForwardPass` |

### **Common Test Patterns**

```python
# Basic test structure
def test_model_forward_pass_returns_correct_shape():
    """Test that model forward pass returns correct output shape."""
    model = create_test_model()
    input_data = torch.randn(32, 10)
    output = model(input_data)
    assert output.shape == (32, 1)

# Parameterized test
@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_model_handles_different_batch_sizes(batch_size):
    """Test model with different batch sizes."""
    model = create_test_model()
    input_data = torch.randn(batch_size, 10)
    output = model(input_data)
    assert output.shape == (batch_size, 1)
```text`n---

## **📈 Visualization Reference**

### **Plot Types**

| Plot Type | Purpose | When to Use |
|-----------|---------|-------------|
| `loss_curve` | Training progress | All experiments |
| `decision_boundary` | Classification boundaries | 2D classification |
| `confusion_matrix` | Classification accuracy | Multi-class problems |
| `feature_importance` | Feature analysis | Tabular data |
| `attention_weights` | Attention visualization | Sequence models |

### **Visualization Generation**

```python
# Enable visualization in training
python src\train.py --experiment xor --visualize

# Generate specific plots
from plotting import generate_loss_curve, generate_decision_boundary

generate_loss_curve(train_losses, val_losses, save_path='loss_curve.png')
generate_decision_boundary(model, X, y, save_path='boundary.png')
```text`n### **Plot File Organization**

```text`noutputs\visualizations\
├── loss_curve.png           # Training/validation loss
├── decision_boundary.png    # Classification boundaries
├── confusion_matrix.png     # Multi-class accuracy
├── feature_importance.png   # Feature analysis
└── attention_weights.png    # Attention visualization
```text`n---

## **🔍 Debugging Reference**

### **Common Issues and Quick Fixes**

| Issue | Symptom | Quick Fix |
|-------|---------|-----------|
| **Import Error** | `ModuleNotFoundError: No module named 'src'` | `pip install -e ..\..` |
| **CUDA Error** | `CUDA out of memory` | Reduce batch size: `--batch-size 16` |
| **Training Stall** | Loss not decreasing | Increase learning rate: `--learning-rate 0.1` |
| **NaN Loss** | `Loss: nan` | Add gradient clipping or reduce LR |
| **Slow Training** | Very slow epochs | Increase `num_workers` in DataLoader |

### **Debug Commands**

```powershell
# Quick environment check
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "from src.model import ModelClass; print('Imports OK')"

# Memory check
python -c "import torch; print(f'CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')"

# Quick training test
python src\train.py --experiment debug_small --epochs 1 --debug
```text`n### **Debugging Workflow**

1. **Check Environment**: Virtual environment activated, packages installed
2. **Test Imports**: All modules can be imported without errors
3. **Test Configuration**: Configuration loads and validates correctly
4. **Test Data**: Data loading works with small batch
5. **Test Model**: Model instantiation and forward pass work
6. **Test Training**: Single training step completes
7. **Scale Up**: Gradually increase complexity

---

## **🚀 Performance Reference**

### **Performance Optimization Checklist**

- [ ] Use appropriate batch size (16-128 for most models)
- [ ] Set `num_workers` in DataLoader (2-8 recommended)
- [ ] Enable `pin_memory=True` for GPU training
- [ ] Use mixed precision training for large models
- [ ] Profile training loop to identify bottlenecks
- [ ] Monitor GPU utilization (`nvidia-smi`)

### **Memory Management**

```python
# Reduce memory usage
config.batch_size = 16  # Smaller batches
config.gradient_accumulation_steps = 4  # Simulate larger batches

# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```text`n### **Performance Monitoring**

```powershell
# GPU monitoring
nvidia-smi -l 1  # Update every second

# CPU monitoring
Get-Process python | Select-Object CPU,WorkingSet

# Training speed benchmark
Measure-Command { python src\train.py --experiment debug_small --epochs 5 }
```text`n---

## **📝 Common Code Patterns**

### **Model Implementation Pattern**

```python
# Standard model structure
class ModelName(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Build layers
        self._build_model()
        self._initialize_weights()
    
    def _build_model(self):
        # Define layers
        pass
    
    def _initialize_weights(self):
        # Initialize parameters
        pass
    
    def forward(self, x):
        # Forward pass
        return output
```text`n### **Training Script Pattern**

```python
# Standard training script structure
def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config(args.experiment)
    
    # Setup environment
    setup_logging(level=config.log_level)
    set_random_seed(config.seed)
    
    # Create model and data
    model = ModelClass(config)
    train_loader, val_loader = load_dataset(config)
    
    # Create trainer and train
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
```text`n### **Configuration Pattern**

```python
# Configuration loading pattern
def get_config(experiment_name: str):
    return create_config(
        model_name="ModelName",
        model_type="model_type",
        experiment_name=experiment_name,
        # Model-specific defaults
        hidden_size=64,
        num_layers=2
    )
```text`n---

## **📚 Documentation Patterns**

### **README Template**

```markdown
# Model Name

Brief description of the model and its key innovation.

## Quick Start
```powershell
# Activate environment
.venv\Scripts\activate

# Train on strength dataset
python src\train.py --experiment strength_dataset --visualize

# Train on weakness dataset  
python src\train.py --experiment weakness_dataset --visualize
```text`n## Key Results

- **Strengths**: What the model does well
- **Weaknesses**: What the model struggles with
- **Next Model**: What model addresses these weaknesses

```text`n### **Docstring Pattern**
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
    pass
```text`n---

## **🎯 Workflow Checklists**

### **New Model Implementation Checklist**

- [ ] Create directory structure
- [ ] Set up virtual environment
- [ ] Implement `constants.py` with model metadata
- [ ] Implement `config.py` with experiments
- [ ] Implement `model.py` with architecture
- [ ] Implement `train.py` with training logic
- [ ] Implement `evaluate.py` with evaluation logic
- [ ] Test on debug dataset
- [ ] Test on strength datasets
- [ ] Test on weakness datasets
- [ ] Generate visualizations
- [ ] Write unit tests
- [ ] Create documentation
- [ ] Run code quality checks

### **Training Experiment Checklist**

- [ ] Configuration loads correctly
- [ ] Data loading works
- [ ] Model instantiation succeeds
- [ ] Forward pass works
- [ ] Loss computation works
- [ ] Backward pass works
- [ ] Training loop runs
- [ ] Checkpointing works
- [ ] Visualization generation works
- [ ] Logs are written
- [ ] Final model saved

### **Code Quality Checklist**

- [ ] Code formatted with black
- [ ] No linting errors (flake8)
- [ ] All imports work
- [ ] All tests pass
- [ ] Documentation complete
- [ ] README updated
- [ ] Git status clean
- [ ] Notebook outputs cleared

---

## **📞 Quick Help**

### **Emergency Commands**

```powershell
# Nuclear option - fresh start
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r ..\..\requirements-dev.txt
pip install -e ..\..

# Quick validation
python -c "from src.model import ModelClass; print('Model OK')"
python -c "from src.config import get_config; print('Config OK')"
python src\train.py --experiment debug_small --epochs 1
```text`n### **Where to Find Help**

1. **This Reference**: Common patterns and commands
2. **Development FAQ**: Detailed troubleshooting (docs/Development_FAQ.md)
3. **AI Development Guide**: Comprehensive development guide (docs/AI_Development_Guide.md)
4. **Architecture Guide**: Technical architecture details (docs/technical/Codebase_Architecture.md)

### **Getting Support**

Include this information when asking for help:

```powershell
# System information
python -c "import sys, platform, torch; print(f'Python: {sys.version}'); print(f'Platform: {platform.system()}
{platform.release()}'); print(f'PyTorch: {torch.__version__}')"

# Working directory
pwd

# Virtual environment
python -c "import sys; print(f'Python executable: {sys.executable}')"

# Package versions
pip list | grep -E "(torch|numpy|matplotlib)"
```text`n---

**💡 Pro Tips**:

- Keep this reference open during development
- Use Ctrl+F to quickly find specific commands
- Bookmark the FAQ for detailed troubleshooting
- Create custom shortcuts for frequently used commands
- Always test with `debug_small` first before running full experiments

This reference is designed to be your go-to resource for rapid development and troubleshooting. Happy coding! 🚀

---

## **🔬 Wandb Integration Reference**

### **Quick Start Commands**

```powershell
# Train with wandb
python src\train.py --experiment xor_breakthrough --wandb

# Train with custom project and tags
python src\train.py --experiment iris_binary --wandb --wandb-project my-research --tags strength comparison

# Train with offline mode
python src\train.py --experiment debug_small --wandb --mode offline

# Run hyperparameter sweep
wandb sweep docs\templates\wandb_sweep.yaml
wandb agent {sweep_id}
```

### **Project Naming Convention**

| Model | Project Name |
|-------|--------------|
| Perceptron | `ai-from-scratch-perceptron` |
| ADALINE | `ai-from-scratch-adaline` |
| MLP | `ai-from-scratch-mlp` |
| LeNet-5 | `ai-from-scratch-lenet5` |
| ResNet | `ai-from-scratch-resnet` |
| Transformer | `ai-from-scratch-transformer` |

### **Standard Tags**

**Base Tags (Automatic)**:

- Model name: `perceptron`, `adaline`, `mlp`
- Module: `module-1`, `module-2`, etc.
- Category: `foundation`, `cnn`, `sequence`, `generative`, `modern`
- Pattern: `engine-based`, `simple`

**Experiment Tags**:

- Type: `strength`, `weakness`, `comparison`, `debug`
- Dataset: `synthetic`, `real`, `vision`, `nlp`
- Scale: `small`, `medium`, `large`

### **Configuration Templates**

**Simple Pattern Integration**:

```python
from docs.templates.wandb_config_example import SimpleWandbIntegration, get_model_info_template

# Setup model info
model_info = get_model_info_template()
model_info.update({"name": "ADALINE", "category": "foundation"})

# Create wandb helper
wandb_helper = SimpleWandbIntegration(model_info)

# Initialize and use
wandb_helper.init_wandb("experiment_name")
wandb_helper.log_metrics({"loss": 0.1}, step=1)
wandb_helper.finish()
```

**Engine Pattern Integration**:

```python
# Automatic integration through TrainingConfig
config = get_training_config("experiment_name", 
                           use_wandb=True,
                           wandb_tags=["strength", "comparison"])
```

### **Artifact Types**

| Type | Description | Examples |
|------|-------------|----------|
| `model` | Model checkpoints and weights | `.pth`, `.pkl` files |
| `dataset` | Dataset versions and metadata | Processed data, statistics |
| `visualization` | Generated plots and images | Learning curves, decision boundaries |
| `analysis` | Analysis reports and comparisons | Comparison studies, ablations |

### **Common Wandb Commands**

```powershell
# Login to wandb
wandb login

# Check wandb status
wandb status

# Sync offline runs
wandb sync wandb\offline-run-*

# View local runs
wandb runs

# Download artifacts
wandb artifact get project/artifact:version
```

### **Dashboard Organization**

**Module-Based Organization**:

```text
ai-from-scratch-{model}/
├── module-1-foundations/
├── module-2-cnn-revolution/
├── module-3-cnn-applications/
├── module-4-sequence-models/
├── module-5-generative-era/
└── module-6-modern-paradigm/
```

**Experiment Grouping**:

- `{model-name}-strengths`
- `{model-name}-weaknesses`
- `{model-name}-vs-{other-model}`
- `{model-name}-sweep-{date}`

---
