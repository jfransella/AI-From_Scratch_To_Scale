#!/usr/bin/env python3
"""
Project Validation Script for "AI From Scratch to Scale"

This script validates project structure, code quality, and implementation
standards across all model implementations.

Usage:
    python validate_project.py --model models/XX_modelname
    python validate_project.py --all
    python validate_project.py --check structure
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import importlib.util
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    file_path: Optional[Path] = None


class ProjectValidator:
    """Main validation class for the project."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.models_dir = project_root / "models"
        self.docs_dir = project_root / "docs"
        self.templates_dir = project_root / "docs" / "templates"

    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        logger.info("Starting comprehensive project validation")

        # Structure validation
        self.validate_project_structure()

        # Find all model directories
        model_dirs = []
        if self.models_dir.exists():
            model_dirs = [
                d
                for d in self.models_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

        # Validate each model
        for model_dir in model_dirs:
            self.validate_model(model_dir)

        # Documentation validation
        self.validate_documentation()

        # Template validation
        self.validate_templates()

        # Summary
        self.print_summary()

        return self.results

    def validate_project_structure(self):
        """Validate overall project structure."""
        logger.info("Validating project structure")

        # Required directories
        required_dirs = [
            "docs",
            "docs/templates",
            "docs/strategy",
            "docs/technical",
            "docs/examples",
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.add_result(
                    "structure_check",
                    ValidationLevel.SUCCESS,
                    f"Required directory exists: {dir_path}",
                )
            else:
                self.add_result(
                    "structure_check",
                    ValidationLevel.ERROR,
                    f"Missing required directory: {dir_path}",
                    file_path=full_path,
                )

        # Required files
        required_files = [
            "docs/AI_Development_Guide.md",
            "docs/Quick_Reference.md",
            "docs/Development_FAQ.md",
            "docs/README.md",
            "docs/templates/config.py",
            "docs/templates/model.py",
            "docs/templates/constants.py",
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.add_result(
                    "structure_check",
                    ValidationLevel.SUCCESS,
                    f"Required file exists: {file_path}",
                )
            else:
                self.add_result(
                    "structure_check",
                    ValidationLevel.ERROR,
                    f"Missing required file: {file_path}",
                    file_path=full_path,
                )

    def validate_model(self, model_dir: Path):
        """Validate a specific model implementation."""
        model_name = model_dir.name
        logger.info("Validating model: %s", model_name)

        # Check directory structure
        self.validate_model_structure(model_dir)

        # Check required files
        self.validate_model_files(model_dir)

        # Check code quality
        self.validate_model_code(model_dir)

        # Check configuration
        self.validate_model_config(model_dir)

        # Check documentation
        self.validate_model_documentation(model_dir)

    def validate_model_structure(self, model_dir: Path):
        """Validate model directory structure."""
        required_structure = {
            "src": ["__init__.py", "model.py", "config.py", "constants.py", "train.py"],
            "notebooks": [
                "01_Theory_and_Intuition.ipynb",
                "02_Code_Walkthrough.ipynb",
                "03_Empirical_Analysis.ipynb",
            ],
            "outputs": ["logs", "models", "visualizations"],
            "tests": [],
        }

        for dir_name, files in required_structure.items():
            dir_path = model_dir / dir_name

            if dir_path.exists():
                self.add_result(
                    "model_structure",
                    ValidationLevel.SUCCESS,
                    f"Directory exists: {model_dir.name}/{dir_name}",
                )

                # Check required files in directory
                for file_name in files:
                    file_path = dir_path / file_name
                    if file_path.exists():
                        self.add_result(
                            "model_structure",
                            ValidationLevel.SUCCESS,
                            f"Required file exists: {model_dir.name}/{dir_name}/{file_name}",
                        )
                    else:
                        self.add_result(
                            "model_structure",
                            ValidationLevel.ERROR,
                            f"Missing required file: {model_dir.name}/{dir_name}/{file_name}",
                            file_path=file_path,
                        )
            else:
                self.add_result(
                    "model_structure",
                    ValidationLevel.ERROR,
                    f"Missing required directory: {model_dir.name}/{dir_name}",
                    file_path=dir_path,
                )

    def validate_model_files(self, model_dir: Path):
        """Validate model files exist and are properly structured."""
        src_dir = model_dir / "src"

        if not src_dir.exists():
            return

        # Check Python files
        python_files = ["model.py", "config.py", "constants.py", "train.py"]

        for file_name in python_files:
            file_path = src_dir / file_name
            if file_path.exists():
                self.validate_python_file(file_path, model_dir.name)
            else:
                self.add_result(
                    "model_files",
                    ValidationLevel.ERROR,
                    f"Missing Python file: {model_dir.name}/src/{file_name}",
                    file_path=file_path,
                )

    def validate_python_file(self, file_path: Path, model_name: str):
        """Validate a Python file for basic requirements."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for docstring
            if '"""' in content or "'''" in content:
                self.add_result(
                    "code_quality",
                    ValidationLevel.SUCCESS,
                    f"File has docstring: {model_name}/{file_path.name}",
                )
            else:
                self.add_result(
                    "code_quality",
                    ValidationLevel.WARNING,
                    f"File missing docstring: {model_name}/{file_path.name}",
                    file_path=file_path,
                )

            # Check for imports
            if "import" in content:
                self.add_result(
                    "code_quality",
                    ValidationLevel.SUCCESS,
                    f"File has imports: {model_name}/{file_path.name}",
                )

            # File-specific checks
            if file_path.name == "model.py":
                self.validate_model_py(file_path, content, model_name)
            elif file_path.name == "config.py":
                self.validate_config_py(file_path, content, model_name)
            elif file_path.name == "constants.py":
                self.validate_constants_py(file_path, content, model_name)
            elif file_path.name == "train.py":
                self.validate_train_py(file_path, content, model_name)

        except Exception as e:
            self.add_result(
                "code_quality",
                ValidationLevel.ERROR,
                f"Error reading file: {model_name}/{file_path.name}: {str(e)}",
                file_path=file_path,
            )

    def validate_model_py(self, file_path: Path, content: str, model_name: str):
        """Validate model.py file specifically."""
        checks = [
            ("class.*nn.Module", "Model class inherits from nn.Module"),
            ("def forward", "Forward method defined"),
            ("def __init__", "Constructor defined"),
            ("torch", "PyTorch imported"),
        ]

        for pattern, description in checks:
            if re.search(pattern, content):
                self.add_result(
                    "model_implementation",
                    ValidationLevel.SUCCESS,
                    f"Model check passed: {description} ({model_name})",
                )
            else:
                self.add_result(
                    "model_implementation",
                    ValidationLevel.WARNING,
                    f"Model check failed: {description} ({model_name})",
                    file_path=file_path,
                )

    def validate_config_py(self, file_path: Path, content: str, model_name: str):
        """Validate config.py file specifically."""
        checks = [
            ("def get_config", "get_config function defined"),
            ("experiment", "Experiment handling present"),
            ("learning_rate", "Learning rate configuration"),
            ("epochs", "Epochs configuration"),
        ]

        for pattern, description in checks:
            if pattern in content:
                self.add_result(
                    "config_validation",
                    ValidationLevel.SUCCESS,
                    f"Config check passed: {description} ({model_name})",
                )
            else:
                self.add_result(
                    "config_validation",
                    ValidationLevel.WARNING,
                    f"Config check failed: {description} ({model_name})",
                    file_path=file_path,
                )

    def validate_constants_py(self, file_path: Path, content: str, model_name: str):
        """Validate constants.py file specifically."""
        checks = [
            ("MODEL_NAME", "MODEL_NAME defined"),
            ("YEAR_INTRODUCED", "Historical year defined"),
            ("AUTHORS", "Authors defined"),
            ("EXPERIMENTS", "Experiments defined"),
        ]

        for pattern, description in checks:
            if pattern in content:
                self.add_result(
                    "constants_validation",
                    ValidationLevel.SUCCESS,
                    f"Constants check passed: {description} ({model_name})",
                )
            else:
                self.add_result(
                    "constants_validation",
                    ValidationLevel.WARNING,
                    f"Constants check failed: {description} ({model_name})",
                    file_path=file_path,
                )

    def validate_train_py(self, file_path: Path, content: str, model_name: str):
        """Validate train.py file specifically."""
        checks = [
            ("argparse", "Argument parsing"),
            ("def main", "Main function defined"),
            ("if __name__", "Main guard present"),
            ("logging", "Logging configured"),
        ]

        for pattern, description in checks:
            if pattern in content:
                self.add_result(
                    "training_validation",
                    ValidationLevel.SUCCESS,
                    f"Training check passed: {description} ({model_name})",
                )
            else:
                self.add_result(
                    "training_validation",
                    ValidationLevel.WARNING,
                    f"Training check failed: {description} ({model_name})",
                    file_path=file_path,
                )

    def validate_model_code(self, model_dir: Path):
        """Validate code quality using external tools."""
        src_dir = model_dir / "src"

        if not src_dir.exists():
            return

        # Run flake8 if available
        try:
            result = subprocess.run(
                ["flake8", str(src_dir)],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

            if result.returncode == 0:
                self.add_result(
                    "code_quality",
                    ValidationLevel.SUCCESS,
                    f"Code style check passed: {model_dir.name}",
                )
            else:
                self.add_result(
                    "code_quality",
                    ValidationLevel.WARNING,
                    f"Code style issues found: {model_dir.name}",
                    details={"flake8_output": result.stdout},
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.add_result(
                "code_quality",
                ValidationLevel.INFO,
                f"Code style check skipped (flake8 not available): {model_dir.name}",
            )

    def validate_model_config(self, model_dir: Path):
        """Validate model configuration."""
        config_path = model_dir / "src" / "config.py"

        if not config_path.exists():
            return

        try:
            # Try to import and validate config
            spec = importlib.util.spec_from_file_location(
                f"{model_dir.name}.config", config_path
            )
            if spec is None or spec.loader is None:
                self.add_result(
                    "config_validation",
                    ValidationLevel.ERROR,
                    f"Could not load config module: {model_dir.name}",
                    file_path=config_path,
                )
                return
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            # Check if get_config function exists
            if hasattr(config_module, "get_config"):
                self.add_result(
                    "config_validation",
                    ValidationLevel.SUCCESS,
                    f"Config module loads successfully: {model_dir.name}",
                )

                # Try to get a config
                try:
                    _ = config_module.get_config("debug_small")
                    self.add_result(
                        "config_validation",
                        ValidationLevel.SUCCESS,
                        f"Config creation successful: {model_dir.name}",
                    )
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    self.add_result(
                        "config_validation",
                        ValidationLevel.WARNING,
                        f"Config creation failed: {model_dir.name}: {str(e)}",
                    )
            else:
                self.add_result(
                    "config_validation",
                    ValidationLevel.ERROR,
                    f"get_config function missing: {model_dir.name}",
                    file_path=config_path,
                )

        except Exception as e:
            self.add_result(
                "config_validation",
                ValidationLevel.ERROR,
                f"Config import failed: {model_dir.name}: {str(e)}",
                file_path=config_path,
            )

    def validate_model_documentation(self, model_dir: Path):
        """Validate model documentation."""
        # Check README
        readme_path = model_dir / "README.md"
        if readme_path.exists():
            self.add_result(
                "documentation",
                ValidationLevel.SUCCESS,
                f"README exists: {model_dir.name}",
            )

            # Check README content
            content = readme_path.read_text(encoding="utf-8")
            if len(content) < 100:
                self.add_result(
                    "documentation",
                    ValidationLevel.WARNING,
                    f"README is very short: {model_dir.name}",
                    file_path=readme_path,
                )
        else:
            self.add_result(
                "documentation",
                ValidationLevel.WARNING,
                f"README missing: {model_dir.name}",
                file_path=readme_path,
            )

        # Check notebooks
        notebooks_dir = model_dir / "notebooks"
        if notebooks_dir.exists():
            notebook_files = list(notebooks_dir.glob("*.ipynb"))
            if len(notebook_files) >= 3:
                self.add_result(
                    "documentation",
                    ValidationLevel.SUCCESS,
                    f"Notebooks present: {model_dir.name} ({len(notebook_files)} files)",
                )
            else:
                self.add_result(
                    "documentation",
                    ValidationLevel.WARNING,
                    f"Incomplete notebooks: {model_dir.name} ({len(notebook_files)}/3 files)",
                )

    def validate_documentation(self):
        """Validate project documentation."""
        logger.info("Validating documentation")

        # Check main documentation files
        doc_files = [
            "AI_Development_Guide.md",
            "Quick_Reference.md",
            "Development_FAQ.md",
            "README.md",
        ]

        for doc_file in doc_files:
            file_path = self.docs_dir / doc_file
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")

                # Check content length
                if len(content) > 1000:
                    self.add_result(
                        "documentation",
                        ValidationLevel.SUCCESS,
                        f"Documentation complete: {doc_file}",
                    )
                else:
                    self.add_result(
                        "documentation",
                        ValidationLevel.WARNING,
                        f"Documentation may be incomplete: {doc_file}",
                        file_path=file_path,
                    )
            else:
                self.add_result(
                    "documentation",
                    ValidationLevel.ERROR,
                    f"Missing documentation: {doc_file}",
                    file_path=file_path,
                )

    def validate_templates(self):
        """Validate template files."""
        logger.info("Validating templates")

        template_files = [
            "model.py",
            "config.py",
            "constants.py",
            "train.py",
            "requirements.txt",
        ]

        for template_file in template_files:
            file_path = self.templates_dir / template_file
            if file_path.exists():
                self.add_result(
                    "templates",
                    ValidationLevel.SUCCESS,
                    f"Template exists: {template_file}",
                )

                # Check template content
                content = file_path.read_text(encoding="utf-8")
                if len(content) > 500:
                    self.add_result(
                        "templates",
                        ValidationLevel.SUCCESS,
                        f"Template has content: {template_file}",
                    )
                else:
                    self.add_result(
                        "templates",
                        ValidationLevel.WARNING,
                        f"Template may be incomplete: {template_file}",
                        file_path=file_path,
                    )
            else:
                self.add_result(
                    "templates",
                    ValidationLevel.ERROR,
                    f"Missing template: {template_file}",
                    file_path=file_path,
                )

    def add_result(
        self,
        check_name: str,
        level: ValidationLevel,
        message: str,
        details: Optional[Dict] = None,
        file_path: Optional[Path] = None,
    ):
        """Add a validation result."""
        result = ValidationResult(
            check_name=check_name,
            level=level,
            message=message,
            details=details,
            file_path=file_path,
        )
        self.results.append(result)

        # Log based on level
        if level == ValidationLevel.ERROR:
            logger.error(message)
        elif level == ValidationLevel.WARNING:
            logger.warning(message)
        elif level == ValidationLevel.INFO:
            logger.info(message)
        else:
            logger.debug(message)

    def print_summary(self):
        """Print validation summary."""
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        successes = [r for r in self.results if r.level == ValidationLevel.SUCCESS]

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Successes: {len(successes)}")
        print(f"⚠️  Warnings:  {len(warnings)}")
        print(f"❌ Errors:    {len(errors)}")
        print("=" * 60)

        if errors:
            print("\n❌ ERRORS:")
            for error in errors:
                print(f"  • {error.message}")
                if error.file_path:
                    print(f"    File: {error.file_path}")

        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  • {warning.message}")
                if warning.file_path:
                    print(f"    File: {warning.file_path}")

        # Overall status
        print("\n" + "=" * 60)
        if errors:
            print("❌ VALIDATION FAILED - Please fix errors above")
            return False
        elif warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            return True
        else:
            print("✅ VALIDATION PASSED - All checks successful!")
            return True

    def export_results(self, output_file: Path):
        """Export results to JSON file."""
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "check_name": result.check_name,
                    "level": result.level.value,
                    "message": result.message,
                    "details": result.details,
                    "file_path": str(result.file_path) if result.file_path else None,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        logger.info("Results exported to %s", output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate AI From Scratch to Scale project"
    )
    parser.add_argument("--model", type=str, help="Validate specific model directory")
    parser.add_argument(
        "--all", action="store_true", help="Validate all models and project structure"
    )
    parser.add_argument(
        "--check",
        choices=["structure", "models", "docs", "templates"],
        help="Run specific check type",
    )
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument(
        "--project-root", type=str, default=".", help="Project root directory"
    )

    args = parser.parse_args()

    # Find project root
    project_root = Path(args.project_root).resolve()

    # Validate project root
    if not project_root.exists():
        logger.error("Project root does not exist: %s", project_root)
        sys.exit(1)

    # Create validator
    validator = ProjectValidator(project_root)

    # Run validation
    if args.all:
        validator.validate_all()
    elif args.model:
        model_path = project_root / args.model
        if model_path.exists():
            validator.validate_model(model_path)
        else:
            logger.error("Model directory does not exist: %s", model_path)
            sys.exit(1)
    elif args.check:
        if args.check == "structure":
            validator.validate_project_structure()
        elif args.check == "docs":
            validator.validate_documentation()
        elif args.check == "templates":
            validator.validate_templates()
        else:
            logger.error("Unknown check type: %s", args.check)
            sys.exit(1)
    else:
        # Default to full validation
        validator.validate_all()

    # Export results if requested
    if args.export:
        validator.export_results(Path(args.export))

    # Print summary and exit with appropriate code
    success = validator.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
