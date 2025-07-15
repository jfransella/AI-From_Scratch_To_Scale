#!/usr/bin/env python3
"""
Quick Validation Script for "AI From Scratch to Scale"

This script provides simple, quick validation commands for common tasks.
Perfect for rapid development and CI/CD integration.

Usage:
    python quick_validate.py check-model XX_modelname
    python quick_validate.py check-structure
    python quick_validate.py check-all
    python quick_validate.py fix-structure XX_modelname
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import shutil

# Import from main validator
from validate_project import ProjectValidator, ValidationLevel


class QuickValidator:
    """Quick validation commands."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.validator = ProjectValidator(project_root)
    
    def check_model(self, model_name: str) -> bool:
        """Quick check for a specific model."""
        model_dir = self.project_root / "models" / model_name
        
        if not model_dir.exists():
            print(f"❌ Model directory does not exist: {model_dir}")
            return False
        
        print(f"🔍 Validating {model_name}...")
        
        # Run validation
        self.validator.validate_model(model_dir)
        
        # Get results
        errors = [r for r in self.validator.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.validator.results if r.level == ValidationLevel.WARNING]
        
        # Print quick summary
        if errors:
            print(f"❌ {model_name} failed validation ({len(errors)} errors)")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  • {error.message}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            return False
        elif warnings:
            print(f"⚠️  {model_name} passed with warnings ({len(warnings)} warnings)")
            return True
        else:
            print(f"✅ {model_name} passed validation")
            return True
    
    def check_structure(self) -> bool:
        """Quick project structure check."""
        print("🔍 Validating project structure...")
        
        self.validator.validate_project_structure()
        
        # Get results
        errors = [r for r in self.validator.results if r.level == ValidationLevel.ERROR]
        
        if errors:
            print(f"❌ Project structure invalid ({len(errors)} errors)")
            for error in errors:
                print(f"  • {error.message}")
            return False
        else:
            print("✅ Project structure valid")
            return True
    
    def check_all(self) -> bool:
        """Quick check of everything."""
        print("🔍 Running complete validation...")
        
        # Structure check
        structure_ok = self.check_structure()
        
        # Find models
        models_dir = self.project_root / "models"
        model_results = []
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    # Reset results for each model
                    self.validator.results = []
                    result = self.check_model(model_dir.name)
                    model_results.append((model_dir.name, result))
        
        # Summary
        print("\n" + "="*50)
        print("QUICK VALIDATION SUMMARY")
        print("="*50)
        print(f"Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
        
        if model_results:
            passed = sum(1 for _, result in model_results if result)
            print(f"Models: {passed}/{len(model_results)} passed")
            
            for model_name, result in model_results:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {model_name}: {status}")
        
        overall_success = structure_ok and all(result for _, result in model_results)
        print(f"\nOverall: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        return overall_success
    
    def fix_structure(self, model_name: str) -> bool:
        """Create missing directories and files for a model."""
        model_dir = self.project_root / "models" / model_name
        
        print(f"🔧 Fixing structure for {model_name}...")
        
        # Create directories
        directories = [
            model_dir / "src",
            model_dir / "notebooks", 
            model_dir / "outputs" / "logs",
            model_dir / "outputs" / "models",
            model_dir / "outputs" / "visualizations",
            model_dir / "tests"
        ]
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created directory: {directory.relative_to(self.project_root)}")
        
        # Create essential files
        files_to_create = [
            (model_dir / "src" / "__init__.py", ""),
            (model_dir / "README.md", self.generate_readme_template(model_name)),
            (model_dir / "requirements.txt", self.generate_requirements_template()),
            (model_dir / ".gitignore", self.generate_gitignore_template())
        ]
        
        for file_path, content in files_to_create:
            if not file_path.exists():
                file_path.write_text(content)
                print(f"  ✅ Created file: {file_path.relative_to(self.project_root)}")
        
        # Copy templates if they exist
        templates_dir = self.project_root / "docs" / "templates"
        if templates_dir.exists():
            template_files = ["model.py", "config.py", "constants.py", "train.py"]
            
            for template_file in template_files:
                template_path = templates_dir / template_file
                target_path = model_dir / "src" / template_file
                
                if template_path.exists() and not target_path.exists():
                    shutil.copy2(template_path, target_path)
                    print(f"  ✅ Copied template: {template_file}")
        
        print(f"✅ Structure fixed for {model_name}")
        return True
    
    def generate_readme_template(self, model_name: str) -> str:
        """Generate README template for a model."""
        return f"""# {model_name}

## Overview
Brief description of the {model_name} model and its key innovation.

## Quick Start

### Setup
```powershell
# Navigate to model directory
Set-Location models\\{model_name}

# Create virtual environment
python -m venv .venv
.venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
pip install -r ..\\..\\requirements-dev.txt
pip install -e ..\\..\n```

### Training
```powershell
# Train on strength dataset
python src\\train.py --experiment strength_dataset --visualize

# Train on weakness dataset  
python src\\train.py --experiment weakness_dataset --visualize
```

### Evaluation
```powershell
# Evaluate trained model
python src\\evaluate.py --checkpoint outputs\\models\\model.pth --experiment test_dataset --visualize
```

## Key Results

### Strengths
- What the model does well
- Key capabilities

### Weaknesses
- What the model struggles with
- Fundamental limitations

### Next Model
- What model addresses these weaknesses
- Key innovations needed

## Implementation Details

### Architecture
- Model architecture description
- Key parameters
- Historical context

### Training
- Training algorithm
- Learning rate scheduling
- Convergence criteria

### Evaluation
- Metrics used
- Benchmarks
- Performance analysis

## References
- Original paper citations
- Historical context
- Related work
"""
    
    def generate_requirements_template(self) -> str:
        """Generate requirements.txt template."""
        return """# Core dependencies
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Data handling
pandas>=1.3.0
scikit-learn>=1.0.0

# Utilities
tqdm>=4.62.0
wandb>=0.12.0

# Development
pytest>=6.2.0
black>=21.0.0
flake8>=4.0.0
"""
    
    def generate_gitignore_template(self) -> str:
        """Generate .gitignore template."""
        return """.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
.mypy_cache/
.idea/
.vscode/
*.swp
*.swo
*~

# Model outputs
outputs/models/*.pth
outputs/models/*.pkl
outputs/logs/*.log
outputs/visualizations/*.png
outputs/visualizations/*.jpg

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick validation for AI From Scratch to Scale project"
    )
    parser.add_argument(
        "command",
        choices=["check-model", "check-structure", "check-all", "fix-structure"],
        help="Validation command to run"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (for model-specific commands)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(args.project_root).resolve()
    
    # Validate project root
    if not project_root.exists():
        print(f"❌ Project root does not exist: {project_root}")
        sys.exit(1)
    
    # Create validator
    validator = QuickValidator(project_root)
    
    # Run command
    try:
        if args.command == "check-model":
            if not args.model:
                print("❌ Model name required for check-model command")
                sys.exit(1)
            success = validator.check_model(args.model)
            
        elif args.command == "check-structure":
            success = validator.check_structure()
            
        elif args.command == "check-all":
            success = validator.check_all()
            
        elif args.command == "fix-structure":
            if not args.model:
                print("❌ Model name required for fix-structure command")
                sys.exit(1)
            success = validator.fix_structure(args.model)
            
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 