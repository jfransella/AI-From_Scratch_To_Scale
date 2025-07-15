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

- **[Project Charter](Project%20Charter_%20AI%20From%20Scratch%20to%20Scale_%20A%20Hands-On%20Journey%20Through%20the%20History%20of%20Neural%20Networks%20(1).md)**: Full project scope and roadmap
- **[Codebase Architecture](Codebase%20Architecture%20&%20Strategy.md)**: Detailed technical architecture
- **[Coding Standards](Coding%20Standards%20Guide%20for%20'AI%20From%20Scratch%20to%20Scale'%20(1).md)**: Complete style guide
- **[Notebook Strategy](Notebook%20Implementation%20Strategy.md)**: Jupyter notebook guidelines
- **[Dataset Strategy](Project%20Dataset%20Strategy.md)**: Complete dataset specifications

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