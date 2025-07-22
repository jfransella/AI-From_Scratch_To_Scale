"""
Test isolation utilities to prevent import collisions between models.

This module provides utilities to safely import model-specific modules
without causing conflicts when multiple models have files with the same names.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager


class ModelTestIsolation:
    """Context manager for isolated model imports."""
    
    def __init__(self, model_path: Path, model_name: str):
        """
        Initialize test isolation for a specific model.
        
        Args:
            model_path: Path to the model's src directory
            model_name: Name of the model (for unique module naming)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.original_modules = {}
        self.temporary_modules = []
        
    def __enter__(self):
        """Enter the isolation context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the isolation context and clean up."""
        # Remove temporary modules
        for module_name in self.temporary_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Restore original modules
        for module_name, original_module in self.original_modules.items():
            if original_module is None:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            else:
                sys.modules[module_name] = original_module
    
    def import_module(self, module_name: str, symbol: Optional[str] = None) -> Any:
        """
        Import a module from the isolated model directory.
        
        Args:
            module_name: Name of the module file (without .py)
            symbol: Specific symbol to import from the module
            
        Returns:
            The imported module or symbol
        """
        module_path = self.model_path / f"{module_name}.py"
        if not module_path.exists():
            raise ImportError(f"Module {module_name} not found in {self.model_path}")
        
        # If this module is already loaded, return it
        if module_name in sys.modules:
            mod = sys.modules[module_name]
            if symbol:
                return getattr(mod, symbol)
            return mod
        
        # Pre-load commonly needed modules to handle cross-imports
        self._preload_common_modules()
        
        # Create unique module name to avoid conflicts
        unique_name = f"{self.model_name}_{module_name}"
        
        # Load the module with a unique name first
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None:
            raise ImportError(f"Could not create spec for {module_name}")
        
        mod = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules with both names for cross-imports to work
        sys.modules[unique_name] = mod
        sys.modules[module_name] = mod
        
        self.temporary_modules.extend([unique_name, module_name])
        
        # Execute the module
        spec.loader.exec_module(mod)
        
        if symbol:
            return getattr(mod, symbol)
        return mod
    
    def _preload_common_modules(self):
        """Pre-load common modules that are often cross-imported."""
        common_modules = ['constants', 'config']
        
        for module_name in common_modules:
            if module_name not in sys.modules:
                module_path = self.model_path / f"{module_name}.py"
                if module_path.exists():
                    try:
                        self._load_single_module(module_name)
                    except (ImportError, FileNotFoundError, AttributeError):
                        # If pre-loading fails, continue - it will be handled later
                        pass
    
    def _load_single_module(self, module_name: str):
        """Load a single module without returning it."""
        module_path = self.model_path / f"{module_name}.py"
        unique_name = f"{self.model_name}_{module_name}"
        
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None:
            return
        
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        sys.modules[module_name] = mod
        self.temporary_modules.extend([unique_name, module_name])
        
        spec.loader.exec_module(mod)


@contextmanager
def isolated_model_import(model_name: str, model_src_path: str):
    """
    Context manager for isolated model imports.
    
    Args:
        model_name: Name of the model for unique naming
        model_src_path: Path to the model's src directory
        
    Yields:
        ModelTestIsolation instance for importing modules
    """
    model_path = Path(model_src_path)
    isolation = ModelTestIsolation(model_path, model_name)
    
    with isolation:
        yield isolation


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_model_src_path(model_dir: str) -> Path:
    """Get the src path for a specific model."""
    return get_project_root() / "models" / model_dir / "src"
