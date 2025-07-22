# Comprehensive Wandb Integration Plan for Base Classes

## 🎯 Overview

This plan implements Weights & Biases (wandb) as a **first-class citizen** in the base architecture while maintaining
educational value and graceful degradation. The design supports the full 25-model roadmap across 6 modules, from
simple NumPy implementations to complex modern architectures.

### 📋 Scope & Requirements

**Target Models**: 25 models across 6 modules:

- **Module 1: Foundations** (4 models) - Perceptron, ADALINE, MLP, Hopfield
- **Module 2: CNN Revolution** (5 models) - LeNet-5, AlexNet, VGGNet, GoogLeNet, ResNet  
- **Module 3: Applying CNNs** (5 models) - R-CNN, Faster R-CNN, YOLO, U-Net, Mask R-CNN
- **Module 4: Sequence Models** (5 models) - RNN, LSTM, GRU, LSTM+Attention, Transformer
- **Module 5: Generative Era** (4 models) - VAE, GAN, DCGAN, DDPM
- **Module 6: Modern Paradigm** (3 models) - GCN, BERT, BitNet 1.58b

#### Implementation Patterns

- **Engine-Based**: Advanced models with BaseModel inheritance
- **Simple Pattern**: Educational implementations with manual wandb integration

---

## 🏗️ Phase 1: Enhanced Base Class Architecture

### 1.1 BaseModel Wandb Integration

**File**: `engine/base.py`

#### New Methods Added to BaseModel

```python
class BaseModel(ABC):
    """Enhanced BaseModel with comprehensive wandb integration."""
    
    def __init__(self):
        """Initialize with wandb tracking capabilities."""
        self.wandb_run = None
        self.wandb_config = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # Core wandb methods
    def init_wandb(self, project=None, name=None, tags=None, config=None, 
                   notes=None, mode="online") -> bool
    def log_metrics(self, metrics: Dict, step=None, commit=True)
    def log_artifact(self, filepath: str, artifact_type="model", name=None, description=None)
    def log_image(self, image_path: str, caption=None, step=None)
    def watch_model(self, log="gradients", log_freq=100)
    def finish_wandb(self)
    def get_wandb_url(self) -> Optional[str]
    def get_wandb_id(self) -> Optional[str]
```

#### Key Features

- ✅ **Automatic Project Naming**: Based on model info (`ai-from-scratch-{model_name}`)
- ✅ **Smart Tagging**: Auto-generates tags from model metadata
- ✅ **Graceful Degradation**: Works without wandb installed
- ✅ **Error Handling**: Comprehensive try/catch with logging
- ✅ **Artifact Management**: Unified interface for checkpoints, visualizations

### 1.2 Enhanced TrainingConfig

**File**: `engine/trainer.py`

```python
@dataclass
class TrainingConfig:
    """Enhanced with comprehensive wandb configuration."""
    
    # Existing fields...
    
    # Enhanced wandb configuration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    
    # Advanced wandb features
    wandb_watch_model: bool = False
    wandb_watch_log: str = "gradients"  # "gradients", "parameters", "all"
    wandb_watch_freq: int = 100
    
    # Artifact configuration
    wandb_log_checkpoints: bool = True
    wandb_log_visualizations: bool = True
    wandb_log_datasets: bool = False
    
    # Group and sweep support
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_sweep_id: Optional[str] = None
```

---

## 🎯 Phase 2: Model-Specific Configuration Standards

### 2.1 Project Naming Convention

**Base Pattern**: `ai-from-scratch-{model-name}`

| Model Category | Project Name | Examples |
|---------------|--------------|----------|
| **Foundations** | `ai-from-scratch-{model}` | `ai-from-scratch-perceptron`, `ai-from-scratch-adaline` |
| **CNNs** | `ai-from-scratch-{model}` | `ai-from-scratch-lenet5`, `ai-from-scratch-resnet` |
| **Sequence** | `ai-from-scratch-{model}` | `ai-from-scratch-lstm`, `ai-from-scratch-transformer` |
| **Generative** | `ai-from-scratch-{model}` | `ai-from-scratch-vae`, `ai-from-scratch-gan` |
| **Modern** | `ai-from-scratch-{model}` | `ai-from-scratch-bert`, `ai-from-scratch-gcn` |

### 2.2 Tagging Strategy

#### Hierarchical Tagging System

```python
# Base tags (automatic)
[
    model_name,          # "perceptron", "lstm", "resnet"
    f"module-{number}",  # "module-1", "module-2", etc.
    category,            # "foundation", "cnn", "sequence", "generative", "modern"
    pattern,             # "engine-based", "simple"
]

# Experiment-specific tags
[
    experiment_name,     # "xor_breakthrough", "iris_binary", "mnist_classification"
    "strength",          # or "weakness" 
    dataset_type,        # "synthetic", "real", "vision", "nlp"
]

# Advanced tags (for later modules)
[
    framework,           # "pytorch", "numpy", "transformers"
    scale,              # "small", "medium", "large"
    domain,             # "vision", "nlp", "graph", "multimodal"
]
```

### 2.3 Model Information Standards

#### Required `get_model_info()` Structure

```python
def get_model_info(self) -> Dict[str, Any]:
    """Standardized model information for wandb integration."""
    return {
        # Core identification
        "name": "ModelName",
        "full_name": "Full Model Name",
        "category": "foundation|cnn|sequence|generative|modern",
        "module": 1,  # Module number (1-6)
        "pattern": "engine-based|simple",
        
        # Historical context
        "year_introduced": 1957,
        "authors": ["Author1", "Author2"],
        "paper_title": "Original Paper Title",
        "key_innovations": ["Innovation1", "Innovation2"],
        
        # Architecture details
        "architecture_type": "single-layer|multi-layer|cnn|rnn|transformer|gan|etc",
        "input_size": self.input_size,
        "output_size": self.output_size,
        "parameter_count": self.get_parameter_count(),
        
        # Training characteristics
        "learning_algorithm": "perceptron-rule|delta-rule|backprop|etc",
        "loss_function": "binary-cross-entropy|mse|cross-entropy|etc",
        "optimizer": "sgd|adam|custom",
        
        # Implementation details
        "framework": "numpy|pytorch|transformers",
        "precision": "float32|float16|int8",
        "device_support": ["cpu", "gpu", "mps"],
        
        # Educational metadata
        "difficulty_level": "beginner|intermediate|advanced",
        "estimated_training_time": "seconds|minutes|hours",
        "key_learning_objectives": ["Objective1", "Objective2"]
    }
```

---

## 🔧 Phase 3: Training Engine Integration

### 3.1 Enhanced Trainer Class

**File**: `engine/trainer.py`

#### New Wandb Integration Methods

```python
class Trainer:
    """Enhanced trainer with comprehensive wandb integration."""
    
    def __init__(self, config: TrainingConfig):
        # ... existing initialization ...
        
        # Initialize wandb for trainer
        if config.use_wandb:
            self._init_trainer_wandb()
    
    def _init_trainer_wandb(self):
        """Initialize wandb at trainer level."""
        # Set up trainer-level wandb configuration
        # Handle model watching, artifact logging, etc.
    
    def train(self, model: BaseModel, data: DataSplit) -> TrainingResult:
        """Enhanced training with comprehensive wandb logging."""
        
        # Initialize model wandb if needed
        if self.config.use_wandb and hasattr(model, 'init_wandb'):
            model.init_wandb(
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                tags=self.config.wandb_tags,
                config=self.config.__dict__,
                notes=self.config.wandb_notes,
                mode=self.config.wandb_mode
            )
            
            # Set up model watching
            if self.config.wandb_watch_model:
                model.watch_model(
                    log=self.config.wandb_watch_log,
                    log_freq=self.config.wandb_watch_freq
                )
        
        # Training loop with wandb logging
        for epoch in range(self.config.max_epochs):
            # ... training logic ...
            
            # Log metrics every epoch
            if self.config.use_wandb and hasattr(model, 'log_metrics'):
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "learning_rate": current_lr,
                }
                
                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                    metrics["val_accuracy"] = val_accuracy
                
                model.log_metrics(metrics, step=epoch)
            
            # Log artifacts at checkpoints
            if (self.config.wandb_log_checkpoints and 
                epoch % self.config.checkpoint_freq == 0):
                checkpoint_path = f"outputs/models/checkpoint_epoch_{epoch}.pth"
                model.save_model(checkpoint_path)
                model.log_artifact(
                    checkpoint_path, 
                    artifact_type="model",
                    description=f"Model checkpoint at epoch {epoch}"
                )
        
        # Final cleanup
        if hasattr(model, 'finish_wandb'):
            model.finish_wandb()
```

### 3.2 Visualization Integration

**File**: `plotting/` package integration

```python
# Enhanced plotting functions with wandb logging
def plot_learning_curve(history, save_path=None, wandb_model=None, 
                       wandb_step=None, wandb_caption=None):
    """Generate learning curve with automatic wandb logging."""
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting logic ...
    
    # Save locally
    if save_path:
        plt.savefig(save_path)
    
    # Log to wandb
    if wandb_model and hasattr(wandb_model, 'log_image'):
        wandb_model.log_image(
            save_path, 
            caption=wandb_caption or "Learning Curve",
            step=wandb_step
        )
    
    return fig

# Similar integration for all plotting functions:
# - plot_confusion_matrix()
# - plot_decision_boundary()
# - plot_gradient_flow()
# - plot_attention_weights()
# - etc.
```

---

## 📊 Phase 4: Experiment Organization & Management

### 4.1 Experiment Grouping Strategy

#### Hierarchical Organization

```text
ai-from-scratch-{model}/
├── module-1-foundations/
│   ├── perceptron-strengths/
│   ├── perceptron-weaknesses/
│   ├── adaline-vs-perceptron/
│   └── mlp-breakthrough/
├── module-2-cnn-revolution/
│   ├── lenet5-introduction/
│   ├── alexnet-scaling/
│   └── resnet-depth/
└── comparative-analysis/
    ├── linear-vs-nonlinear/
    ├── cnn-vs-mlp/
    └── historical-progression/
```

#### Group Configuration

```python
# In model config files
def get_wandb_groups():
    """Define wandb group organization."""
    return {
        "module_group": f"module-{MODULE_NUMBER}-{MODULE_NAME}",
        "model_group": f"{MODEL_NAME}-experiments",
        "comparison_group": f"{MODEL_NAME}-vs-{PREVIOUS_MODEL}",
        "strength_group": f"{MODEL_NAME}-strengths",
        "weakness_group": f"{MODEL_NAME}-weaknesses"
    }
```

### 4.2 Sweep Configuration Templates

**File**: `docs/templates/wandb_sweep.yaml`

```yaml
# Template for hyperparameter sweeps
program: src/train.py
method: bayes
project: ai-from-scratch-{model-name}
name: {model-name}-hyperparameter-sweep

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128]
  epochs:
    distribution: int_uniform
    min: 100
    max: 1000
  experiment:
    value: {experiment_name}
  wandb:
    value: true

metric:
  name: val_accuracy
  goal: maximize

early_terminate:
  type: hyperband
  min_iter: 50
```

### 4.3 Dashboard Templates

#### Custom Wandb Dashboards for Each Module

```python
# Dashboard configuration for each module
MODULE_DASHBOARDS = {
    1: {  # Foundations
        "name": "Module 1: Foundations Dashboard",
        "sections": [
            "Model Comparison (Perceptron vs ADALINE vs MLP)",
            "Learning Algorithm Analysis",
            "Linear vs Non-linear Capability",
            "Historical Timeline"
        ]
    },
    2: {  # CNN Revolution
        "name": "Module 2: CNN Revolution Dashboard", 
        "sections": [
            "Architecture Evolution (LeNet → AlexNet → ResNet)",
            "Feature Visualization",
            "Scale and Performance Analysis",
            "Computational Efficiency"
        ]
    },
    # ... other modules
}
```

---

## 🔍 Phase 5: Advanced Features & Integrations

### 5.1 Artifact Management System

#### Comprehensive Artifact Strategy

```python
class WandbArtifactManager:
    """Centralized artifact management for all models."""
    
    ARTIFACT_TYPES = {
        "model": {
            "checkpoints": "Model checkpoints (.pth files)",
            "final_models": "Final trained models",
            "architecture": "Model architecture definitions"
        },
        "dataset": {
            "processed": "Processed training datasets",
            "synthetic": "Generated synthetic datasets",
            "metadata": "Dataset metadata and statistics"
        },
        "visualization": {
            "learning_curves": "Training progress visualizations",
            "decision_boundaries": "Model decision boundary plots",
            "feature_maps": "CNN feature map visualizations",
            "attention_maps": "Attention weight visualizations"
        },
        "analysis": {
            "comparison_reports": "Model comparison analyses", 
            "ablation_studies": "Ablation study results",
            "error_analysis": "Error analysis reports"
        }
    }
    
    def log_model_checkpoint(self, model, epoch, metrics):
        """Log model checkpoint with comprehensive metadata."""
        
    def log_visualization_batch(self, plots_dir, experiment_name):
        """Log all visualizations from an experiment."""
        
    def log_dataset_version(self, dataset_name, version, metadata):
        """Log dataset with version control."""
```

### 5.2 Model Comparison Framework

#### Cross-Model Analysis Tools

```python
class ModelComparisonTracker:
    """Track and compare models across experiments."""
    
    def __init__(self, comparison_project="ai-from-scratch-comparison"):
        self.project = comparison_project
        
    def compare_models(self, model_runs):
        """Generate comparison analysis across multiple model runs."""
        
    def track_historical_progression(self, chronological_models):
        """Track the historical progression of model capabilities."""
        
    def generate_capability_matrix(self, models, datasets):
        """Generate a capability matrix showing model performance."""
```

### 5.3 Educational Analytics

#### Learning Progress Tracking

```python
class EducationalAnalytics:
    """Track educational progress and insights."""
    
    def track_learning_objectives(self, model_name, objectives_met):
        """Track which learning objectives have been achieved."""
        
    def measure_implementation_complexity(self, model_info):
        """Measure and track implementation complexity metrics."""
        
    def generate_insight_reports(self, module_number):
        """Generate educational insight reports for each module."""
```

---

## 📋 Phase 6: Implementation Roadmap

### 6.1 Priority Implementation Order

#### Phase 6.1: Foundation (Weeks 1-2)

- ✅ Enhanced BaseModel with wandb methods
- ✅ Updated TrainingConfig with wandb fields  
- ✅ Basic trainer integration
- ✅ Template updates for all patterns

#### Phase 6.2: Model Integration (Weeks 3-4)

- 🔄 Update existing Perceptron model (already mostly done)
- 📋 Implement ADALINE with wandb integration
- 📋 Enhance MLP wandb integration
- 📋 Create standardized model info templates

#### Phase 6.3: Visualization Integration (Weeks 5-6)

- 📋 Enhance plotting package with wandb logging
- 📋 Create visualization artifact management
- 📋 Implement dashboard templates
- 📋 Create comparison visualization tools

#### Phase 6.4: Advanced Features (Weeks 7-8)

- 📋 Implement artifact management system
- 📋 Create sweep configuration templates
- 📋 Build model comparison framework
- 📋 Develop educational analytics

#### Phase 6.5: Documentation & Validation (Weeks 9-10)

- 📋 Complete documentation updates
- 📋 Create user guides and tutorials
- 📋 Validate all 25 model compatibility
- 📋 Performance testing and optimization

### 6.2 Success Metrics

#### Technical Metrics

- ✅ All 25 models support wandb integration
- ✅ Zero breaking changes to existing code
- ✅ <5% performance overhead from wandb
- ✅ 100% graceful degradation without wandb

#### Educational Metrics

- ✅ Comprehensive experiment tracking
- ✅ Clear learning progression visualization
- ✅ Easy model comparison capabilities
- ✅ Rich educational insights generation

#### Usability Metrics

- ✅ Single command wandb activation (`--wandb`)
- ✅ Automatic project/tag organization
- ✅ Zero manual configuration required
- ✅ Clear error messages and fallbacks

---

## 🎯 Conclusion

This comprehensive wandb integration plan transforms the "AI From Scratch to Scale" project into a modern,
professional ML platform while maintaining its educational focus. The integration provides:

1. **Unified Tracking**: Consistent experiment tracking across all 25 models
2. **Educational Value**: Clear learning progression and model comparison
3. **Professional Standards**: Industry-standard MLOps practices
4. **Scalable Architecture**: Supports growth from simple to complex models
5. **Graceful Degradation**: Works with or without wandb installed

The implementation follows a phased approach that minimizes disruption while maximizing value, ensuring the project
remains both educationally valuable and professionally relevant.
