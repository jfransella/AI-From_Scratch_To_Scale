"""
Example of standardized wandb configuration for model implementations.

This template shows how to implement wandb integration following the 
comprehensive integration plan for all patterns (engine-based and simple).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

# Optional wandb integration
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# =============================================================================
# STANDARDIZED MODEL INFO STRUCTURE
# =============================================================================

def get_model_info_template() -> Dict[str, Any]:
    """
    Template for standardized model information.
    
    This structure ensures consistent wandb integration across all models.
    Customize for each specific model implementation.
    """
    return {
        # Core identification - REQUIRED
        "name": "ModelName",  # e.g., "Perceptron", "ADALINE", "MLP"
        "full_name": "Full Model Name",  # e.g., "Multi-Layer Perceptron"
        "category": "foundation",  # foundation|cnn|sequence|generative|modern
        "module": 1,  # Module number (1-6)
        "pattern": "simple",  # engine-based|simple
        
        # Historical context - REQUIRED
        "year_introduced": 1957,
        "authors": ["Author1", "Author2"],
        "paper_title": "Original Paper Title",
        "key_innovations": ["Innovation1", "Innovation2"],
        
        # Architecture details - REQUIRED
        "architecture_type": "single-layer",  # single-layer|multi-layer|cnn|rnn|etc
        "input_size": None,  # Will be filled at runtime
        "output_size": None,  # Will be filled at runtime
        "parameter_count": None,  # Will be calculated
        
        # Training characteristics - REQUIRED
        "learning_algorithm": "perceptron-rule",  # perceptron-rule|delta-rule|backprop|etc
        "loss_function": "binary-cross-entropy",  # binary-cross-entropy|mse|cross-entropy|etc
        "optimizer": "sgd",  # sgd|adam|custom
        
        # Implementation details - REQUIRED
        "framework": "pytorch",  # numpy|pytorch|transformers
        "precision": "float32",  # float32|float16|int8
        "device_support": ["cpu", "gpu"],  # List of supported devices
        
        # Educational metadata - REQUIRED
        "difficulty_level": "beginner",  # beginner|intermediate|advanced
        "estimated_training_time": "seconds",  # seconds|minutes|hours
        "key_learning_objectives": [
            "Understand linear decision boundaries",
            "Learn basic neural network concepts"
        ]
    }


# =============================================================================
# WANDB INTEGRATION FOR SIMPLE PATTERN MODELS
# =============================================================================

class SimpleWandbIntegration:
    """
    Wandb integration helper for simple pattern models.
    
    Use this for models that don't inherit from BaseModel but want
    wandb functionality.
    """
    
    def __init__(self, model_info: Dict[str, Any]):
        """
        Initialize wandb integration.
        
        Args:
            model_info: Model information dictionary (use get_model_info_template)
        """
        self.model_info = model_info
        self.wandb_run = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def init_wandb(self, 
                   experiment_name: str,
                   project: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   config: Optional[Dict[str, Any]] = None,
                   notes: Optional[str] = None,
                   mode: str = "online") -> bool:
        """Initialize wandb tracking."""
        if not _WANDB_AVAILABLE:
            self.logger.warning("wandb not available, skipping initialization")
            return False
        
        try:
            # Generate project name
            if project is None:
                model_name = self.model_info.get('name', 'unknown').lower()
                project = f"ai-from-scratch-{model_name}"
            
            # Generate tags
            if tags is None:
                tags = []
            
            auto_tags = [
                self.model_info.get('name', 'unknown').lower(),
                self.model_info.get('category', 'unknown'),
                f"module-{self.model_info.get('module', 'unknown')}",
                self.model_info.get('pattern', 'unknown'),
                experiment_name
            ]
            tags.extend(auto_tags)
            
            # Prepare config
            wandb_config = {}
            if config:
                wandb_config.update(config)
            wandb_config.update(self.model_info)
            wandb_config["experiment_name"] = experiment_name
            
            # Initialize wandb
            self.wandb_run = wandb.init(
                project=project,
                name=f"{experiment_name}-{self.model_info.get('name', 'unknown')}",
                tags=list(set(tags)),
                config=wandb_config,
                notes=notes,
                mode=mode,
                reinit=True
            )
            
            self.logger.info(f"Wandb initialized for project: {project}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_artifact(self, filepath: str, artifact_type: str = "model", 
                     name: Optional[str] = None, description: Optional[str] = None):
        """Log artifact to wandb."""
        if self.wandb_run is not None:
            try:
                from pathlib import Path
                
                if name is None:
                    name = Path(filepath).stem
                
                artifact = wandb.Artifact(
                    name=name,
                    type=artifact_type,
                    description=description
                )
                artifact.add_file(filepath)
                self.wandb_run.log_artifact(artifact)
                
                self.logger.info(f"Logged artifact: {name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to log artifact: {e}")
    
    def log_image(self, image_path: str, caption: Optional[str] = None, 
                  step: Optional[int] = None):
        """Log image to wandb."""
        if self.wandb_run is not None:
            try:
                image = wandb.Image(image_path, caption=caption)
                self.wandb_run.log({"visualization": image}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log image: {e}")
    
    def finish(self):
        """Finish wandb run."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                self.wandb_run = None
                self.logger.info("Wandb run finished")
            except Exception as e:
                self.logger.warning(f"Failed to finish wandb: {e}")


# =============================================================================
# EXPERIMENT CONFIGURATION TEMPLATES
# =============================================================================

@dataclass
class WandbExperimentConfig:
    """
    Standardized wandb experiment configuration.
    
    Use this for consistent wandb setup across all experiments.
    """
    
    # Core experiment info
    experiment_name: str
    model_name: str
    
    # Wandb configuration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    
    # Advanced features
    wandb_watch_model: bool = False
    wandb_watch_log: str = "gradients"
    wandb_watch_freq: int = 100
    
    # Artifact logging
    log_checkpoints: bool = True
    log_visualizations: bool = True
    log_final_model: bool = True
    
    # Group and organization
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    
    def get_wandb_config(self, model_info: Dict[str, Any], 
                         training_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate complete wandb configuration.
        
        Args:
            model_info: Model information dictionary
            training_config: Training configuration dictionary
            
        Returns:
            Complete wandb configuration
        """
        config = {}
        
        # Add model info
        config.update(model_info)
        
        # Add training config
        if training_config:
            config.update(training_config)
        
        # Add experiment info
        config.update({
            "experiment_name": self.experiment_name,
            "wandb_mode": self.wandb_mode,
            "wandb_group": self.wandb_group,
            "wandb_job_type": self.wandb_job_type
        })
        
        return config
    
    def get_project_name(self) -> str:
        """Generate standardized project name."""
        if self.wandb_project:
            return self.wandb_project
        return f"ai-from-scratch-{self.model_name.lower()}"
    
    def get_run_name(self) -> str:
        """Generate standardized run name."""
        if self.wandb_name:
            return self.wandb_name
        return f"{self.experiment_name}-{self.model_name}"
    
    def get_tags(self, model_info: Dict[str, Any]) -> List[str]:
        """Generate complete tag list."""
        tags = self.wandb_tags.copy()
        
        # Add automatic tags
        auto_tags = [
            model_info.get('name', 'unknown').lower(),
            model_info.get('category', 'unknown'),
            f"module-{model_info.get('module', 'unknown')}",
            model_info.get('pattern', 'unknown'),
            self.experiment_name
        ]
        
        # Determine experiment type
        if 'strength' in self.experiment_name.lower():
            auto_tags.append('strength')
        elif 'weakness' in self.experiment_name.lower():
            auto_tags.append('weakness')
        elif 'comparison' in self.experiment_name.lower():
            auto_tags.append('comparison')
        elif 'debug' in self.experiment_name.lower():
            auto_tags.append('debug')
        
        tags.extend(auto_tags)
        return list(set(tags))  # Remove duplicates


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_simple_pattern_usage():
    """Example of how to use wandb with simple pattern models."""
    
    # 1. Define model info
    model_info = get_model_info_template()
    model_info.update({
        "name": "ADALINE",
        "full_name": "Adaptive Linear Neuron",
        "year_introduced": 1960,
        "authors": ["Bernard Widrow", "Ted Hoff"],
        "learning_algorithm": "delta-rule",
        "loss_function": "mse"
    })
    
    # 2. Create wandb integration
    wandb_helper = SimpleWandbIntegration(model_info)
    
    # 3. Initialize for experiment
    success = wandb_helper.init_wandb(
        experiment_name="delta_rule_demo",
        tags=["educational", "comparison"],
        config={"learning_rate": 0.01, "epochs": 100},
        notes="Demonstrating Delta Rule vs Perceptron Learning Rule"
    )
    
    if success:
        # 4. Log training metrics
        for epoch in range(100):
            wandb_helper.log_metrics({
                "epoch": epoch,
                "train_loss": 0.1 * (100 - epoch) / 100,
                "train_accuracy": epoch / 100
            }, step=epoch)
        
        # 5. Log artifacts
        wandb_helper.log_artifact("outputs/models/adaline_model.pth", "model")
        wandb_helper.log_image("outputs/visualizations/learning_curve.png", 
                             "Learning Curve")
        
        # 6. Finish run
        wandb_helper.finish()


def example_engine_pattern_usage():
    """Example of how to use wandb with engine pattern models."""
    
    # For engine-based models, wandb integration is handled automatically
    # through the enhanced BaseModel and TrainingConfig
    
    config = WandbExperimentConfig(
        experiment_name="iris_binary",
        model_name="Perceptron",
        use_wandb=True,
        wandb_tags=["strength", "binary-classification"],
        wandb_notes="Demonstrating Perceptron strength on linearly separable data",
        wandb_watch_model=True,
        log_visualizations=True
    )
    
    # The trainer will automatically handle wandb initialization and logging
    # based on the config
    
    return config


if __name__ == "__main__":
    # Run examples
    print("Simple pattern example:")
    example_simple_pattern_usage()
    
    print("\nEngine pattern example:")
    config = example_engine_pattern_usage()
    print(f"Project: {config.get_project_name()}")
    print(f"Run name: {config.get_run_name()}") 