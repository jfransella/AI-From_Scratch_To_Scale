# **Codebase Architecture & Strategy**

## **1\. Guiding Philosophy: A Scalable & Focused Architecture**

This architecture is designed to support the project's core mission: "learning by building." It achieves this by separating **shared, reusable infrastructure** from **unique, model-specific code**. The primary goals are to focus on learning, maximize consistency, and minimize repetition.

## **2\. Top-Level Directory Structure**

The repository will be organized with shared packages and configuration at the root level.

ai-from-scratch-to-scale\  
|  
|-- .github\             # Contains GitHub-specific files for community management.  
|-- data_utils\          # SHARED: Handles all dataset loading & transformations.  
|-- docs\                # SHARED: Project-wide documentation (e.g., this file).  
|-- engine\              # SHARED: Reusable training/evaluation engine with wandb.  
|-- plotting\            # SHARED: Generates and saves visualization files.  
|-- tests\               # SHARED: Automated tests for the project.  
|-- utils\               # SHARED: General-purpose utilities (logging, seeds, etc.).  
|  
|-- models\              # Contains all individual model projects.  
|   |-- 01_perceptron\  
|   |-- ...  
|  
|-- .gitignore            # Defines ignored files for the entire project.  
|-- LICENSE  
|-- README.md             # The main project README, links to docs\.  
|-- requirements-dev.txt  # Development-only dependencies (pytest, black, flake8).

## **3\. The Shared Packages & Configuration**

### **/.github**

* **Purpose:** To improve project management and standardize community interaction.  
* **Contents:**  
  * ISSUE\_TEMPLATE/: Templates for bug reports and feature requests.  
  * PULL\_REQUEST\_TEMPLATE.md: A checklist and guide for new pull requests.  
  * workflows/: GitHub Actions for Continuous Integration (e.g., running black for formatting checks, flake8 for linting, and pytest for testing).  
  * CONTRIBUTING.md: Guidelines for how others can contribute.  
  * dependabot.yml: Configuration for automated dependency updates.  
  * copilot-instructions.md: Context for GitHub Copilot to improve its suggestions within the repo.

### **/data\_utils, /engine, /plotting, /utils**

These packages contain the shared, reusable code for data handling, training, visualization, and general utilities, respectively.

### **3.1. Dependency & Environment Management**

To ensure historical accuracy and prevent dependency conflicts, we will adopt a **per-model virtual environment** strategy.

* **Model Dependencies (requirements.txt):** Each model project inside models\ will have its own requirements.txt file, listing only the specific libraries and versions needed for that model.  
* **Development Dependencies (requirements-dev.txt):** A single requirements-dev.txt file at the project root will define dependencies needed only for development and testing: pytest for testing, black for code formatting, and flake8 for linting.  
* **Workflow for Shared Code:** To work on or test the shared packages, you will use a model's virtual environment as the "host." The process is:  
  1. Activate a chosen model's virtual environment.  
  2. Install the development dependencies using `pip install -r ..\..\requirements-dev.txt` from the model directory.  
  3. Install the project in editable mode using `pip install -e ..\..` from the model directory. This makes the shared packages available within the active environment.

### **3.3. Shared Package Dependency Matrix**

Each model type uses different combinations of shared packages. This matrix shows which packages are required for different model categories:

| Model Type | data_utils | engine | plotting | utils | Primary Framework |
|------------|------------|---------|----------|-------|------------------|
| **NumPy-based** (Perceptron, ADALINE) | ✓ | ✓ | ✓ | ✓ | NumPy only |
| **Simple PyTorch** (MLP, Basic CNN) | ✓ | ✓ | ✓ | ✓ | PyTorch |
| **Advanced PyTorch** (ResNet, Transformer) | ✓ | ✓ | ✓ | ✓ | PyTorch + extras |
| **Specialized Models** (GAN, VAE, etc.) | ✓ | ✓ | ✓ | ✓ | PyTorch + domain libs |

**Package Responsibilities:**

* **data_utils**: Dataset loading, preprocessing, DataLoader creation
* **engine**: Training loops, evaluation, checkpointing, wandb integration
* **plotting**: Visualization generation, plot saving, wandb image logging
* **utils**: Logging setup, random seed setting, device management

**Import Patterns by Model Type:**

```python
# NumPy-based models
from data_utils import load_dataset, create_data_loaders
from engine import Trainer, Evaluator  # May use simpler versions
from plotting import generate_visualizations
from utils import setup_logging, set_random_seed

# PyTorch-based models  
from data_utils import load_dataset, create_data_loaders
from engine import Trainer, Evaluator
from plotting import generate_visualizations  
from utils import setup_logging, set_random_seed
import torch
import torch.nn as nn
import torch.optim as optim
```

### **3.2. Testing Strategy**

To ensure the reliability and correctness of the shared infrastructure, we will adopt a formal testing strategy using the **pytest** framework.

* **Unit Tests:** These will test individual functions in the shared packages (/utils, /plotting, /data\_utils) in isolation.  
* **Integration Tests:** These will test that the core components work together. The most important will be a "smoke test" for the /engine that runs a dummy model for a single epoch to ensure the entire training pipeline is functional.  
* **Automation (CI):** The GitHub Actions workflow (/.github/workflows/ci.yml) will be configured to run all tests automatically on every pull request, acting as a quality gate to prevent regressions.

## **4\. Dual-System Logging Strategy**

To capture both a human-readable narrative and structured metrics, we will use two systems in parallel, orchestrated by the /engine.

* **System 1: Python logging (The Narrative Log)**  
* **System 2: Weights & Biases (wandb) (The Metrics Database)**

## 5.1 Visualization as a Core Architectural Component

Visualization is a first-class citizen in this project, serving as both a diagnostic and interpretive tool for all models. The following guidelines ensure that visualization is systematically integrated into the workflow:

* **Centralized Utilities:** All reusable plotting and visualization utilities must reside in the shared `/plotting` package. This includes functions for generating learning curves, confusion matrices, decision boundaries, feature maps, Grad-CAM, t-SNE/UMAP projections, and more. Model-specific visualizations should be implemented in the model’s notebook or in a dedicated script within the model’s directory, but should leverage shared utilities whenever possible.
* **Activation:** Visualization generation is triggered by the `--visualize` command-line flag in training and evaluation scripts. This flag should be supported in all model scripts, and its logic should be handled in accordance with the selection logic described above.
* **Outputs:** All generated figures must be saved to the standardized `outputs/visualizations/` directory within each model’s project. This ensures reproducibility and easy access for analysis and reporting.
* **Documentation:** Every model’s README and analysis notebook must include a section on visualizations, listing required plots and referencing the Visualization Playbooks (see `docs/visualization/Playbooks.md`).
* **Extensibility:** When new visualization types are needed, they should be added to the `/plotting` package with clear, reusable APIs and documented in the Implementation Guide (`docs/visualization/Implementation_Guide.md`).
* **Best Practices:** Visualization code should follow the standards outlined in `Coding_Standards.md`, including naming conventions, docstrings, and reproducibility (e.g., setting random seeds for t-SNE/UMAP).

This approach ensures that visualization is not an afterthought, but a systematic, reproducible, and extensible part of the project’s architecture, supporting both automated and interactive analysis workflows.

## **6\. Model Checkpointing & Loading**

Saving a model checkpoint after a training run is the default, mandatory behavior to ensure results are preserved. Loading a checkpoint is an optional input for fine-tuning or evaluation.

* **Primary Method: wandb Artifacts**: When wandb is enabled, the trained model's state_dict will be saved as a versioned wandb Artifact.  
* **Local Fallback**: If wandb is disabled, the model's state_dict will be saved to the local `outputs\models\` directory.  
* **Error Recovery**: Checkpoint loading uses the recovery patterns from section 9.4, gracefully handling corrupted or missing checkpoints.  
* **Template Integration**: The model.py template (section 8.1) includes built-in methods for saving and loading checkpoints with proper error handling.

## **7\. Dedicated Evaluation Workflow**

To cleanly separate the act of training from analysis, each model will have a dedicated evaluate.py script following the template patterns.

* **Purpose**: To load a pre-trained model checkpoint and evaluate its performance on a specified test dataset.  
* **Core Logic**: The script will instantiate the model, load the checkpoint using recovery patterns from section 9.4, fetch the test data, and use the Evaluator class from the shared `engine\` to compute and log final metrics.  
* **Key Arguments**: The script will be driven by arguments like `--checkpoint` (required, path or wandb ID), `--experiment` (required, to select the dataset), and `--visualize` (optional).  
* **Error Handling**: Uses the same error handling patterns as training (section 9) to gracefully handle missing checkpoints, data loading failures, and evaluation errors.

## **8\. The Model-Specific Project Structure**

Each model's directory is a self-contained project with its own environment and dependencies.

```text
models\01_perceptron\  
|-- docs\  
|-- notebooks\  
|-- outputs\  
|   |-- logs\  
|   |-- models\  
|   |-- visualizations\  
|-- src\  
|   |-- __init__.py  
|   |-- constants.py  
|   |-- config.py  
|   |-- model.py  
|   |-- train.py  
|   |-- evaluate.py       # New evaluation script  
|-- .venv\                # Model-specific virtual environment (in .gitignore).  
|-- requirements.txt      # Model-specific dependencies.  
|-- README.md
```

### **8.1. File Templates and Expected Structure**

To ensure consistency across all model implementations, comprehensive templates are provided in `docs\templates\`. These templates should be copied to each new model directory and customized.

**Template Files Available:**

* **[model.py](../templates/model.py)**: Complete neural network model template with PyTorch structure
* **[train.py](../templates/train.py)**: Training script with argument parsing and shared infrastructure integration
* **[config.py](../templates/config.py)**: Configuration management with experiment-specific overrides
* **[constants.py](../templates/constants.py)**: Model metadata, file paths, and validation constants
* **[requirements.txt](../templates/requirements.txt)**: Dependency template with examples for different model types

**Expected File Structure and Imports:**

**constants.py** - Model metadata and paths:

```python
# Model metadata
MODEL_NAME = "YourModelName"
YEAR_INTRODUCED = 1957
ORIGINAL_AUTHOR = "Author Name"

# File paths (Windows-style)
OUTPUT_DIR = MODEL_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
```

**config.py** - Experiment configuration:

```python
def get_config(experiment_name: str) -> dict:
    base_config = {
        'model_name': MODEL_NAME,
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        # ... other base parameters
    }
    
    experiments = {
        'xor': {
            'dataset': 'xor',
            'learning_rate': 0.1,
            # ... experiment overrides
        }
    }
    
    return {**base_config, **experiments[experiment_name]}
```

**model.py** - Neural network implementation:

```python
import torch
import torch.nn as nn
from utils import setup_logging
from constants import MODEL_NAME

class YourModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Model architecture
        
    def forward(self, x):
        # Forward pass
        pass
        
    def save_checkpoint(self, filepath, epoch=None):
        # Checkpoint saving logic
        pass
```

**train.py** - Training script:

```python
import argparse
from engine import Trainer
from data_utils import load_dataset, create_data_loaders
from plotting import generate_visualizations
from utils import setup_logging, set_random_seed
from model import YourModel, create_model
from config import get_config

def main():
    args = parse_arguments()
    config = get_config(args.experiment)
    
    # Training workflow
    train_loader, val_loader, test_loader = load_data(config)
    model = create_model(config)
    trainer = Trainer(model, optimizer, criterion, device, config)
    trainer.train(train_loader, val_loader, config['epochs'])
```

## **9\. Error Handling Patterns**

To ensure robust and maintainable code across all models, standardized error handling patterns are enforced throughout the codebase.

### **9.1. Exception Hierarchy**

**Custom Exceptions:**

```python
# In utils/exceptions.py
class AIFromScratchError(Exception):
    """Base exception for all project-specific errors"""
    pass

class ModelError(AIFromScratchError):
    """Model-specific errors (architecture, forward pass, etc.)"""
    pass

class DataError(AIFromScratchError):
    """Data loading and preprocessing errors"""
    pass

class ConfigError(AIFromScratchError):
    """Configuration validation errors"""
    pass

class TrainingError(AIFromScratchError):
    """Training loop and optimization errors"""
    pass
```

### **9.2. Error Handling by Component**

**Model Implementation (model.py):**

```python
def forward(self, x):
    try:
        # Validate input
        if x.dim() != 2:
            raise ModelError(f"Expected 2D input, got {x.dim()}D")
        
        # Forward pass
        output = self._forward_impl(x)
        
        # Validate output
        if torch.isnan(output).any():
            raise ModelError("Model produced NaN values")
            
        return output
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        raise ModelError(f"Forward pass failed: {e}") from e
```

**Configuration Management (config.py):**

```python
def get_config(experiment_name: str) -> dict:
    try:
        if experiment_name not in experiments:
            available = list(experiments.keys())
            raise ConfigError(
                f"Unknown experiment: {experiment_name}. "
                f"Available: {available}"
            )
        
        config = {**base_config, **experiments[experiment_name]}
        _validate_config(config)
        return config
        
    except ConfigError:
        raise  # Re-raise config errors
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise ConfigError(f"Configuration failed: {e}") from e
```

**Training Script (train.py):**

```python
def main():
    try:
        # Parse and validate arguments
        args = parse_arguments()
        config = get_config(args.experiment)
        
        # Setup with error handling
        setup_logging_with_error_handling()
        set_random_seed(config['seed'])
        
        # Training workflow
        train_loader, val_loader, test_loader = load_data(config)
        model = create_model(config)
        trainer = Trainer(model, optimizer, criterion, device, config)
        
        # Training with checkpointing on interruption
        training_history = trainer.train(train_loader, val_loader, config['epochs'])
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except DataError as e:
        logger.error(f"Data loading error: {e}")
        sys.exit(1)
    except ModelError as e:
        logger.error(f"Model error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint before exiting
        if 'model' in locals():
            save_interrupt_checkpoint(model, config)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Save debug information
        save_debug_info(locals(), config)
        sys.exit(1)
```

### **9.3. Graceful Degradation Patterns**

**Data Loading Fallbacks:**

```python
def load_dataset(dataset_name, **kwargs):
    try:
        # Try primary data source
        return _load_from_primary(dataset_name, **kwargs)
    except DataError:
        logger.warning(f"Primary data source failed for {dataset_name}")
        try:
            # Try alternative source
            return _load_from_alternative(dataset_name, **kwargs)
        except DataError:
            logger.error(f"All data sources failed for {dataset_name}")
            raise
```

**Device Fallbacks:**

```python
def setup_device(device_arg='auto'):
    try:
        if device_arg == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using MPS device")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU device")
        else:
            device = torch.device(device_arg)
            
        # Test device
        test_tensor = torch.tensor([1.0]).to(device)
        return device
        
    except Exception as e:
        logger.warning(f"Device setup failed: {e}, falling back to CPU")
        return torch.device('cpu')
```

### **9.4. Recovery and Cleanup Patterns**

**Checkpoint Recovery:**

```python
def resume_training(checkpoint_path, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed training from epoch {start_epoch}")
        return start_epoch
    except Exception as e:
        logger.error(f"Failed to resume from checkpoint: {e}")
        logger.info("Starting training from scratch")
        return 0
```

**Resource Cleanup:**

```python
def cleanup_resources():
    """Cleanup function for graceful shutdown"""
    try:
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Close any open files
        for handler in logger.handlers[:]:
            if hasattr(handler, 'close'):
                handler.close()
                
    except Exception as e:
        print(f"Cleanup failed: {e}")  # Use print since logger might be closed
```

## **10\. Parameterized Experiment Execution**

The train.py script is the "Execution Interface" for a run, configured via command-line arguments following the template patterns defined in section 8.1.

| Argument | Description | Example |
| :---- | :---- | :---- |
| `--experiment` | **(Required)** The name of the experiment (e.g., xor, iris-hard). | `--experiment xor` |
| `--epochs` | Overrides the default number of epochs. | `--epochs 200` |
| `--batch-size` | Overrides the default batch size. | `--batch-size 64` |
| `--learning-rate` | Overrides the default learning rate. | `--learning-rate 0.001` |
| `--load-checkpoint` | Path or wandb artifact ID to load model weights from before training. | `--load-checkpoint outputs\models\model.pth` |
| `--no-save-checkpoint` | A flag to prevent saving the final model checkpoint, useful for debugging. | `--no-save-checkpoint` |
| `--no-wandb` | A flag to disable wandb logging for the run. | `--no-wandb` |
| `--seed` | Sets the random seed for reproducibility. | `--seed 42` |
| `--tags` | Attaches tags to the wandb run for easy filtering. | `--tags experiment baseline` |
| `--visualize` | A flag to enable the generation and saving of standard visualizations. | `--visualize` |
| `--device` | Specifies the device to use (auto, cpu, cuda, mps). | `--device cuda` |

**Example Training Commands:**

```powershell
# Basic training
python src\train.py --experiment xor

# Training with custom parameters and visualization
python src\train.py --experiment iris-hard --epochs 300 --batch-size 16 --visualize

# Resume training from checkpoint
python src\train.py --experiment xor --load-checkpoint outputs\models\xor_checkpoint.pth
```

## **11\. The End-to-End Workflow in Practice**

### **11.1. Initial Setup**

1. **Copy Templates**: Copy files from `docs\templates\` to your model directory
2. **Customize Templates**: Replace placeholders (e.g., `[MODEL_NAME]`, `[YEAR]`) with actual values
3. **Create Environment**: Navigate to model directory, create virtual environment, activate it
4. **Install Dependencies**: Run `pip install -r requirements.txt` for model dependencies
5. **Setup Development**: Run `pip install -r ..\..\requirements-dev.txt` and `pip install -e ..\..` for shared packages

### **11.2. Development Workflow**

1. **Implement Model**: Follow the template structure in `model.py` with proper error handling
2. **Configure Experiments**: Define experiments in `config.py` following the template patterns
3. **Test Implementation**: Use the provided validation functions in `constants.py`
4. **Validate Setup**: Run basic import tests to ensure shared packages work correctly

### **11.3. Training Workflow**

1. **Training**: Run `python src\train.py --experiment experiment_name --visualize`
   * The script uses error handling patterns from section 9
   * Calls helpers from `utils\`, fetches data from `data_utils\`
   * Instantiates the model and passes everything to the Trainer in `engine\`
   * The Trainer runs the training loop with automatic checkpointing
2. **Monitoring**: Watch logs and wandb dashboard for training progress
3. **Error Recovery**: If training fails, use checkpoint recovery patterns from section 9.4

### **11.4. Evaluation Workflow**

1. **Evaluation**: Run `python src\evaluate.py --checkpoint path\to\checkpoint.pth --experiment experiment_name`
   * Loads the model using the checkpoint recovery patterns
   * Uses the Evaluator from `engine\` to compute test metrics
   * Applies the same error handling patterns as training
2. **Analysis**: Generated logs, visualizations, and metrics are available:
   * Local files: `outputs\logs\`, `outputs\visualizations\`, `outputs\models\`
   * Wandb dashboard: For structured metrics and experiment comparison
   * Notebooks: For detailed analysis using the three-notebook approach

### **11.5. Troubleshooting**

* **Import Errors**: Ensure `pip install -e ..\..` was run from model directory
* **Configuration Errors**: Check experiment name against available options in `config.py`
* **Training Failures**: Review error logs and use checkpoint recovery if needed
* **Device Issues**: The code automatically falls back to CPU if GPU setup fails
* **Data Issues**: Check dataset loading with fallback mechanisms in place
