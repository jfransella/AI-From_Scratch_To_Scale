# **Testing Strategy for AI From Scratch to Scale**

This document defines the comprehensive testing approach for the "AI From Scratch to Scale" project, ensuring reliability, maintainability, and educational value across all 25 neural network implementations.

## **Table of Contents**

1. [Testing Philosophy](#testing-philosophy)
2. [Test Categories & Organization](#test-categories--organization)
3. [Naming Conventions](#naming-conventions)
4. [Template Test Files](#template-test-files)
5. [Performance Benchmarking](#performance-benchmarking)
6. [Automated Testing Pipeline](#automated-testing-pipeline)
7. [Testing Infrastructure](#testing-infrastructure)
8. [Quality Assurance Standards](#quality-assurance-standards)

---

## **Testing Philosophy**

### **Educational Testing Approach**

Our testing strategy serves dual purposes:

1. **Reliability**: Ensure code works correctly across different environments
2. **Learning**: Tests serve as executable documentation and examples

### **Testing Priorities**

1. **Shared Infrastructure**: Most critical - affects all models
2. **Model Integration**: Ensure models work with shared systems
3. **Model Correctness**: Verify mathematical implementations
4. **Performance**: Ensure acceptable training/inference speeds
5. **Robustness**: Handle edge cases and error conditions

### **Test-Driven Development**

- Write tests before implementing complex functionality
- Use tests to validate mathematical correctness
- Tests serve as regression prevention during refactoring

---

## **Test Categories & Organization**

### **1. Unit Tests** (`tests/unit/`)

Test individual functions and classes in isolation.

#### **Categories:**

- **Mathematical Functions**: Activation functions, loss functions, optimizers
- **Data Processing**: Transformations, normalization, augmentation
- **Model Components**: Individual layers, forward passes, parameter updates
- **Utilities**: Logging, seeding, device management, configuration loading

#### **Structure:**

```
tests/unit/
├── test_data_utils.py       # Data loading and preprocessing
├── test_engine.py           # Training and evaluation components
├── test_plotting.py         # Visualization functions
├── test_utils.py            # General utilities
├── models/                  # Model-specific unit tests
│   ├── test_perceptron.py
│   ├── test_mlp.py
│   └── test_cnn.py
└── fixtures/                # Test data and configurations
    ├── sample_data.npz
    ├── test_configs.json
    └── mock_models.py
```

### **2. Integration Tests** (`tests/integration/`)

Test interactions between multiple components.

#### **Categories:**

- **Data Pipeline**: End-to-end data loading → preprocessing → model input
- **Training Pipeline**: Model + optimizer + loss + data → training step
- **Evaluation Pipeline**: Trained model + test data → metrics
- **Logging Pipeline**: Training events → logs + wandb + visualizations

#### **Structure:**

```
tests/integration/
├── test_training_pipeline.py    # Full training workflow
├── test_evaluation_pipeline.py  # Full evaluation workflow
├── test_data_pipeline.py        # Data loading to model input
├── test_logging_pipeline.py     # Logging and visualization
└── test_model_lifecycle.py      # Save/load/checkpoint workflows
```

### **3. Smoke Tests** (`tests/smoke/`)

Fast end-to-end tests that verify basic functionality.

#### **Categories:**

- **Quick Training**: Single epoch training on tiny datasets
- **Model Instantiation**: All models can be created without errors
- **Data Loading**: All datasets can be loaded successfully
- **Command Line**: All CLI commands execute without crashing

#### **Structure:**

```
tests/smoke/
├── test_quick_training.py       # 1-epoch training tests
├── test_model_creation.py       # Model instantiation
├── test_data_loading.py         # Dataset loading
├── test_cli_commands.py         # Command-line interface
└── test_shared_imports.py       # Import all shared modules
```

### **4. Performance Tests** (`tests/performance/`)

Measure and validate performance characteristics.

#### **Categories:**

- **Training Speed**: Iterations per second, memory usage
- **Inference Speed**: Predictions per second, latency
- **Memory Usage**: Peak memory, memory leaks
- **Scalability**: Performance with different batch sizes

#### **Structure:**

```
tests/performance/
├── test_training_speed.py       # Training performance benchmarks
├── test_inference_speed.py      # Inference performance benchmarks
├── test_memory_usage.py         # Memory consumption tests
├── test_scalability.py          # Batch size scaling tests
└── benchmarks/                  # Reference performance data
    ├── training_benchmarks.json
    ├── inference_benchmarks.json
    └── memory_benchmarks.json
```

### **5. Regression Tests** (`tests/regression/`)

Ensure model outputs remain consistent across code changes.

#### **Categories:**

- **Model Outputs**: Verify identical outputs for same inputs
- **Training Curves**: Compare loss curves across versions
- **Final Metrics**: Ensure final accuracy doesn't degrade
- **Visualization**: Compare generated plots across versions

#### **Structure:**

```
tests/regression/
├── test_model_outputs.py        # Output consistency tests
├── test_training_curves.py      # Training curve comparisons
├── test_final_metrics.py        # Final performance tests
├── test_visualization.py        # Plot comparison tests
└── reference_data/              # Baseline data for comparison
    ├── model_outputs/
    ├── training_curves/
    ├── final_metrics/
    └── visualizations/
```

---

## **Naming Conventions**

### **Test File Names**

- **Format**: `test_{component}_{functionality}.py`
- **Examples**:
  - `test_perceptron_forward_pass.py`
  - `test_data_utils_loading.py`
  - `test_engine_training.py`
  - `test_plotting_visualization.py`

### **Test Function Names**

- **Format**: `test_{what}_{condition}_{expected_result}`
- **Examples**:
  - `test_perceptron_forward_pass_returns_correct_output()`
  - `test_data_loader_invalid_path_raises_error()`
  - `test_trainer_early_stopping_triggers_correctly()`
  - `test_optimizer_zero_lr_no_parameter_update()`

### **Test Class Names**

- **Format**: `Test{Component}{Functionality}`
- **Examples**:
  - `TestPerceptronForwardPass`
  - `TestDataUtilsLoading`
  - `TestEngineTraining`
  - `TestPlottingVisualization`

### **Test Data Names**

- **Format**: `{test_type}_{size}_{description}`
- **Examples**:
  - `unit_small_xor_data.npz`
  - `integration_medium_mnist_batch.npz`
  - `smoke_tiny_synthetic_data.npz`
  - `performance_large_benchmark_data.npz`

---

## **Template Test Files**

### **Unit Test Template**

```python
"""
Unit tests for {component_name} module.

This module tests individual functions and classes in the {component_name}
module to ensure they work correctly in isolation.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from {module_path} import {component_class}
from tests.fixtures import {fixture_imports}


class Test{ComponentName}:
    """Test suite for {ComponentName} class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_data = self._create_test_data()
        self.mock_config = self._create_mock_config()
        self.component = {ComponentName}(self.mock_config)
    
    def _create_test_data(self):
        """Create test data for the test suite."""
        return {
            'input': np.random.rand(32, 10),
            'target': np.random.randint(0, 2, (32, 1)),
            'expected_output_shape': (32, 1)
        }
    
    def _create_mock_config(self):
        """Create mock configuration for testing."""
        return {
            'input_size': 10,
            'output_size': 1,
            'learning_rate': 0.01,
            'device': 'cpu'
        }
    
    def test_initialization_valid_config_creates_instance(self):
        """Test that valid configuration creates instance correctly."""
        component = {ComponentName}(self.mock_config)
        assert component.input_size == 10
        assert component.output_size == 1
        assert component.learning_rate == 0.01
    
    def test_initialization_invalid_config_raises_error(self):
        """Test that invalid configuration raises appropriate error."""
        invalid_config = {'invalid': 'config'}
        with pytest.raises(ValueError, match="Missing required configuration"):
            {ComponentName}(invalid_config)
    
    def test_forward_pass_correct_input_returns_expected_shape(self):
        """Test that forward pass returns correct output shape."""
        output = self.component.forward(self.test_data['input'])
        assert output.shape == self.test_data['expected_output_shape']
    
    def test_forward_pass_invalid_input_raises_error(self):
        """Test that invalid input raises appropriate error."""
        invalid_input = np.random.rand(32, 5)  # Wrong input size
        with pytest.raises(ValueError, match="Input size mismatch"):
            self.component.forward(invalid_input)
    
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    def test_forward_pass_different_batch_sizes_works(self, batch_size):
        """Test that forward pass works with different batch sizes."""
        input_data = np.random.rand(batch_size, 10)
        output = self.component.forward(input_data)
        assert output.shape == (batch_size, 1)
    
    def test_parameter_count_returns_correct_number(self):
        """Test that parameter count is calculated correctly."""
        expected_params = (10 * 1) + 1  # weights + bias
        assert self.component.parameter_count() == expected_params
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up any resources, close files, etc.
        pass


class Test{ComponentName}EdgeCases:
    """Test edge cases and error conditions for {ComponentName}."""
    
    def test_zero_learning_rate_no_parameter_update(self):
        """Test that zero learning rate prevents parameter updates."""
        config = {'learning_rate': 0.0, 'input_size': 10, 'output_size': 1}
        component = {ComponentName}(config)
        
        initial_params = component.get_parameters().copy()
        # Simulate training step
        component.training_step(np.random.rand(32, 10), np.random.rand(32, 1))
        updated_params = component.get_parameters()
        
        np.testing.assert_array_equal(initial_params, updated_params)
    
    def test_extreme_learning_rate_stability(self):
        """Test behavior with extremely high learning rate."""
        config = {'learning_rate': 1e6, 'input_size': 10, 'output_size': 1}
        component = {ComponentName}(config)
        
        # Should not crash or produce NaN values
        output = component.forward(np.random.rand(32, 10))
        assert not np.isnan(output).any()
        assert np.isfinite(output).all()


# Fixture definitions
@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        'X': np.random.rand(100, 10),
        'y': np.random.randint(0, 2, (100, 1))
    }


@pytest.fixture
def mock_config():
    """Provide mock configuration for testing."""
    return {
        'input_size': 10,
        'output_size': 1,
        'learning_rate': 0.01,
        'device': 'cpu'
    }
```

### **Integration Test Template**

```python
"""
Integration tests for {component_name} training pipeline.

This module tests the interaction between multiple components to ensure
they work together correctly in realistic scenarios.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from data_utils import load_dataset
from engine import Trainer, Evaluator
from plotting import generate_visualizations
from utils import setup_logging, set_random_seed


class TestTrainingPipeline:
    """Test complete training pipeline integration."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = self._create_test_config()
        set_random_seed(42)
        setup_logging(level='DEBUG')
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_config(self):
        """Create test configuration."""
        return {
            'model_name': 'TestModel',
            'dataset': 'synthetic',
            'batch_size': 16,
            'epochs': 2,
            'learning_rate': 0.01,
            'device': 'cpu',
            'checkpoint_dir': self.temp_dir,
            'plot_dir': self.temp_dir,
            'seed': 42
        }
    
    def test_end_to_end_training_completes_successfully(self):
        """Test that complete training pipeline runs without errors."""
        # Load data
        train_loader, val_loader = load_dataset(
            self.config['dataset'],
            batch_size=self.config['batch_size'],
            train_split=0.8,
            val_split=0.2
        )
        
        # Create model (using a simple test model)
        model = self._create_test_model()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Train model
        training_history = trainer.train()
        
        # Verify training completed
        assert len(training_history['train_loss']) == self.config['epochs']
        assert len(training_history['val_loss']) == self.config['epochs']
        assert all(isinstance(loss, float) for loss in training_history['train_loss'])
        
        # Verify model was saved
        checkpoint_path = Path(self.temp_dir) / 'model_checkpoint.pth'
        assert checkpoint_path.exists()
    
    def test_training_with_early_stopping_triggers_correctly(self):
        """Test that early stopping mechanism works correctly."""
        config = self.config.copy()
        config['early_stopping_patience'] = 1
        config['epochs'] = 10
        
        # Create data that will cause early stopping
        train_loader, val_loader = self._create_early_stopping_data()
        model = self._create_test_model()
        
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        training_history = trainer.train()
        
        # Should stop before reaching max epochs
        assert len(training_history['train_loss']) < config['epochs']
    
    def test_training_with_visualization_generates_plots(self):
        """Test that training with visualization generates expected plots."""
        config = self.config.copy()
        config['plot_types'] = ['loss_curve', 'accuracy_curve']
        
        train_loader, val_loader = load_dataset(
            self.config['dataset'],
            batch_size=self.config['batch_size']
        )
        
        model = self._create_test_model()
        trainer = Trainer(model=model, config=config, 
                         train_loader=train_loader, val_loader=val_loader)
        
        trainer.train()
        
        # Check that plots were generated
        plot_dir = Path(self.temp_dir)
        assert (plot_dir / 'loss_curve.png').exists()
        assert (plot_dir / 'accuracy_curve.png').exists()
    
    def test_model_checkpoint_loading_works_correctly(self):
        """Test that saved model can be loaded and used for inference."""
        # Train a model
        train_loader, val_loader = load_dataset(self.config['dataset'])
        model = self._create_test_model()
        trainer = Trainer(model=model, config=self.config, 
                         train_loader=train_loader, val_loader=val_loader)
        trainer.train()
        
        # Load the saved model
        checkpoint_path = Path(self.temp_dir) / 'model_checkpoint.pth'
        loaded_model = self._create_test_model()
        loaded_model.load_state_dict(torch.load(checkpoint_path))
        
        # Test inference
        test_input = torch.randn(1, 10)
        output = loaded_model(test_input)
        
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()
    
    def _create_test_model(self):
        """Create a simple test model for integration testing."""
        import torch.nn as nn
        
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return torch.sigmoid(self.linear(x))
        
        return SimpleTestModel()
    
    def _create_early_stopping_data(self):
        """Create data that will trigger early stopping."""
        # Implementation depends on your data loading infrastructure
        pass
```

### **Performance Test Template**

```python
"""
Performance tests for {component_name}.

This module tests performance characteristics including training speed,
inference speed, and memory usage to ensure acceptable performance.
"""

import pytest
import time
import psutil
import numpy as np
import torch
from typing import Dict, List

from {module_path} import {component_class}
from tests.fixtures import performance_fixtures


class TestPerformanceBenchmarks:
    """Performance benchmarks for {ComponentName}."""
    
    @pytest.fixture(autouse=True)
    def setup_performance_test(self):
        """Set up performance testing environment."""
        self.performance_data = {}
        self.baseline_metrics = self._load_baseline_metrics()
        
        # Ensure consistent performance testing conditions
        torch.set_num_threads(1)  # Consistent CPU usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics for comparison."""
        return {
            'training_speed': 100.0,  # samples/second
            'inference_speed': 1000.0,  # samples/second
            'memory_usage': 512.0,  # MB
            'convergence_time': 30.0  # seconds
        }
    
    def test_training_speed_benchmark(self):
        """Benchmark training speed and compare to baseline."""
        model = self._create_benchmark_model()
        train_data = self._create_benchmark_data(size='medium')
        
        # Warmup
        self._run_training_steps(model, train_data, steps=5)
        
        # Benchmark
        start_time = time.time()
        steps_run = self._run_training_steps(model, train_data, steps=100)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        samples_per_second = (steps_run * train_data['batch_size']) / elapsed_time
        
        # Record performance
        self.performance_data['training_speed'] = samples_per_second
        
        # Compare to baseline (allow 10% degradation)
        baseline_speed = self.baseline_metrics['training_speed']
        assert samples_per_second >= baseline_speed * 0.9, \
            f"Training speed {samples_per_second:.2f} samples/s is below baseline {baseline_speed:.2f} samples/s"
    
    def test_inference_speed_benchmark(self):
        """Benchmark inference speed and compare to baseline."""
        model = self._create_benchmark_model()
        test_data = self._create_benchmark_data(size='large')
        
        # Warmup
        self._run_inference_steps(model, test_data, steps=10)
        
        # Benchmark
        start_time = time.time()
        steps_run = self._run_inference_steps(model, test_data, steps=1000)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        samples_per_second = (steps_run * test_data['batch_size']) / elapsed_time
        
        # Record performance
        self.performance_data['inference_speed'] = samples_per_second
        
        # Compare to baseline
        baseline_speed = self.baseline_metrics['inference_speed']
        assert samples_per_second >= baseline_speed * 0.9, \
            f"Inference speed {samples_per_second:.2f} samples/s is below baseline {baseline_speed:.2f} samples/s"
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage and compare to baseline."""
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model and data
        model = self._create_benchmark_model()
        train_data = self._create_benchmark_data(size='large')
        
        # Measure memory after model creation
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training to measure peak memory
        self._run_training_steps(model, train_data, steps=10)
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_usage = peak_memory - baseline_memory
        
        # Record performance
        self.performance_data['memory_usage'] = memory_usage
        
        # Compare to baseline (allow 20% increase)
        baseline_memory_usage = self.baseline_metrics['memory_usage']
        assert memory_usage <= baseline_memory_usage * 1.2, \
            f"Memory usage {memory_usage:.2f} MB exceeds baseline {baseline_memory_usage:.2f} MB"
    
    def test_convergence_time_benchmark(self):
        """Benchmark time to convergence and compare to baseline."""
        model = self._create_benchmark_model()
        train_data = self._create_benchmark_data(size='medium')
        
        start_time = time.time()
        convergence_step = self._train_to_convergence(model, train_data)
        end_time = time.time()
        
        convergence_time = end_time - start_time
        
        # Record performance
        self.performance_data['convergence_time'] = convergence_time
        
        # Compare to baseline (allow 50% increase)
        baseline_convergence = self.baseline_metrics['convergence_time']
        assert convergence_time <= baseline_convergence * 1.5, \
            f"Convergence time {convergence_time:.2f}s exceeds baseline {baseline_convergence:.2f}s"
    
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 128])
    def test_batch_size_scaling(self, batch_size):
        """Test performance scaling with different batch sizes."""
        model = self._create_benchmark_model()
        train_data = self._create_benchmark_data(size='medium', batch_size=batch_size)
        
        # Measure performance
        start_time = time.time()
        steps_run = self._run_training_steps(model, train_data, steps=20)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        samples_per_second = (steps_run * batch_size) / elapsed_time
        
        # Record performance by batch size
        if 'batch_scaling' not in self.performance_data:
            self.performance_data['batch_scaling'] = {}
        self.performance_data['batch_scaling'][batch_size] = samples_per_second
        
        # Ensure performance increases with batch size (up to a point)
        if batch_size <= 64:  # Expect scaling up to reasonable batch size
            assert samples_per_second >= 50.0, \
                f"Performance {samples_per_second:.2f} samples/s too low for batch size {batch_size}"
    
    def _create_benchmark_model(self):
        """Create a model for benchmarking."""
        # Implementation depends on your model architecture
        pass
    
    def _create_benchmark_data(self, size: str, batch_size: int = 32) -> Dict:
        """Create benchmark data of specified size."""
        size_map = {
            'small': 1000,
            'medium': 10000,
            'large': 100000
        }
        
        num_samples = size_map[size]
        return {
            'X': np.random.rand(num_samples, 10),
            'y': np.random.randint(0, 2, (num_samples, 1)),
            'batch_size': batch_size
        }
    
    def _run_training_steps(self, model, data, steps: int) -> int:
        """Run training steps for benchmarking."""
        # Implementation depends on your training setup
        pass
    
    def _run_inference_steps(self, model, data, steps: int) -> int:
        """Run inference steps for benchmarking."""
        # Implementation depends on your inference setup
        pass
    
    def _train_to_convergence(self, model, data, max_steps: int = 1000) -> int:
        """Train model until convergence."""
        # Implementation depends on your convergence criteria
        pass
    
    def teardown_method(self):
        """Save performance data after each test."""
        self._save_performance_data()
    
    def _save_performance_data(self):
        """Save performance data for future comparison."""
        # Save to JSON file for comparison in future runs
        pass
```

---

## **Performance Benchmarking**

### **Benchmark Categories**

#### **1. Training Performance**

- **Metrics**: Samples/second, epochs/hour, GPU utilization
- **Baselines**: Historical performance data for each model
- **Thresholds**: Alert if performance drops >10% from baseline

#### **2. Inference Performance**

- **Metrics**: Predictions/second, latency percentiles, memory usage
- **Baselines**: Target inference speeds for different model sizes
- **Thresholds**: Real-time requirements for interactive applications

#### **3. Memory Efficiency**

- **Metrics**: Peak memory usage, memory growth over time
- **Baselines**: Expected memory usage for different batch sizes
- **Thresholds**: Alert if memory usage exceeds 80% of available RAM

#### **4. Convergence Speed**

- **Metrics**: Time to reach target accuracy, training stability
- **Baselines**: Historical convergence times for each dataset/model
- **Thresholds**: Alert if convergence time increases >50%

### **Benchmarking Infrastructure**

#### **Performance Test Runner**

```python
# tests/performance/benchmark_runner.py
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List

class BenchmarkRunner:
    """Run and collect performance benchmarks."""
    
    def __init__(self, output_dir: str = "tests/performance/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_benchmark(self, benchmark_name: str, benchmark_func, *args, **kwargs):
        """Run a single benchmark and collect metrics."""
        process = psutil.Process()
        
        # Collect baseline metrics
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Run benchmark
        start_time = time.time()
        result = benchmark_func(*args, **kwargs)
        end_time = time.time()
        
        # Collect final metrics
        peak_memory = process.memory_info().rss / 1024 / 1024
        
        benchmark_data = {
            'name': benchmark_name,
            'timestamp': time.time(),
            'duration': end_time - start_time,
            'memory_usage': peak_memory - baseline_memory,
            'result': result
        }
        
        # Save benchmark data
        self._save_benchmark_data(benchmark_name, benchmark_data)
        
        return benchmark_data
    
    def _save_benchmark_data(self, name: str, data: Dict):
        """Save benchmark data to JSON file."""
        output_file = self.output_dir / f"{name}_benchmark.json"
        
        # Load existing data
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
```

---

## **Automated Testing Pipeline**

### **GitHub Actions Workflow**

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run smoke tests
      run: |
        pytest tests/smoke/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-only
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### **Pre-commit Hooks**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: local
    hooks:
      - id: tests
        name: run tests
        entry: pytest tests/smoke/ -v
        language: python
        pass_filenames: false
```

---

## **Testing Infrastructure**

### **Test Configuration**

```python
# tests/conftest.py
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        'model_name': 'TestModel',
        'input_size': 10,
        'output_size': 1,
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 5,
        'device': 'cpu',
        'seed': 42
    }

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    np.random.seed(42)
    return {
        'X': np.random.rand(1000, 10),
        'y': np.random.randint(0, 2, (1000, 1))
    }

@pytest.fixture
def tiny_data():
    """Provide tiny dataset for smoke tests."""
    return {
        'X': np.random.rand(10, 10),
        'y': np.random.randint(0, 2, (10, 1))
    }

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
```

### **Test Data Management**

```python
# tests/fixtures/data_fixtures.py
import numpy as np
import torch
from pathlib import Path

class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_classification_data(n_samples=1000, n_features=10, n_classes=2):
        """Generate synthetic classification data."""
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, (n_samples, 1))
        return X, y
    
    @staticmethod
    def generate_regression_data(n_samples=1000, n_features=10, noise=0.1):
        """Generate synthetic regression data."""
        X = np.random.rand(n_samples, n_features)
        true_weights = np.random.rand(n_features, 1)
        y = X @ true_weights + noise * np.random.randn(n_samples, 1)
        return X, y
    
    @staticmethod
    def generate_sequence_data(n_samples=1000, seq_length=50, vocab_size=100):
        """Generate synthetic sequence data."""
        X = np.random.randint(0, vocab_size, (n_samples, seq_length))
        y = np.random.randint(0, vocab_size, (n_samples, seq_length))
        return X, y
    
    @staticmethod
    def generate_image_data(n_samples=1000, height=28, width=28, channels=1):
        """Generate synthetic image data."""
        X = np.random.rand(n_samples, channels, height, width)
        y = np.random.randint(0, 10, (n_samples, 1))
        return X, y

class TestDataLoader:
    """Load and cache test data."""
    
    def __init__(self, cache_dir="tests/fixtures/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_or_generate(self, data_type, **kwargs):
        """Load cached data or generate new data."""
        cache_file = self.cache_dir / f"{data_type}_data.npz"
        
        if cache_file.exists():
            data = np.load(cache_file)
            return data['X'], data['y']
        else:
            generator = TestDataGenerator()
            X, y = getattr(generator, f'generate_{data_type}_data')(**kwargs)
            np.savez(cache_file, X=X, y=y)
            return X, y
```

---

## **Quality Assurance Standards**

### **Code Coverage Requirements**

- **Minimum Coverage**: 80% for shared infrastructure
- **Target Coverage**: 90% for core functionality
- **Critical Components**: 95% coverage for training and evaluation

### **Test Quality Metrics**

- **Test-to-Code Ratio**: At least 1:2 (one test line per two code lines)
- **Test Execution Time**: Unit tests <5 minutes, Integration tests <15 minutes
- **Test Reliability**: <1% flaky test rate

### **Documentation Standards**

- **Test Documentation**: Every test file must have module-level docstring
- **Test Case Documentation**: Complex test cases must have detailed docstrings
- **Failure Messages**: All assertions must have descriptive failure messages

### **Continuous Integration Requirements**

- **All Tests Pass**: 100% test pass rate required for merge
- **Performance Regression**: <10% performance degradation allowed
- **Code Quality**: Flake8 and black compliance required

---

## **Testing Best Practices**

### **Test Organization**

1. **Group Related Tests**: Use test classes to group related functionality
2. **Use Descriptive Names**: Test names should clearly describe what they test
3. **Keep Tests Independent**: Each test should be able to run in isolation
4. **Use Fixtures**: Share common setup code using pytest fixtures

### **Test Data Management**

1. **Use Small Data**: Keep test data small for fast execution
2. **Use Realistic Data**: Test data should represent real-world scenarios
3. **Cache Test Data**: Cache generated test data to improve performance
4. **Clean Up Resources**: Always clean up temporary files and resources

### **Performance Testing**

1. **Establish Baselines**: Create performance baselines for comparison
2. **Monitor Trends**: Track performance over time
3. **Test Different Conditions**: Test with various batch sizes and inputs
4. **Use Consistent Environment**: Ensure consistent testing conditions

### **Error Testing**

1. **Test Edge Cases**: Test boundary conditions and edge cases
2. **Test Error Conditions**: Verify proper error handling
3. **Test Recovery**: Test system behavior after errors
4. **Use Mocking**: Mock external dependencies for reliable tests

---

## **Implementation Checklist**

### **Setting Up Testing**

- [ ] Create test directory structure
- [ ] Set up pytest configuration
- [ ] Create test fixtures and utilities
- [ ] Set up CI/CD pipeline
- [ ] Create baseline performance metrics

### **Writing Tests**

- [ ] Write unit tests for all functions
- [ ] Write integration tests for workflows
- [ ] Write smoke tests for basic functionality
- [ ] Write performance tests for critical paths
- [ ] Write regression tests for bug fixes

### **Test Maintenance**

- [ ] Update tests when code changes
- [ ] Monitor test execution times
- [ ] Update performance baselines
- [ ] Review and refactor old tests
- [ ] Document test failures and fixes

### **Quality Assurance**

- [ ] Achieve target code coverage
- [ ] Monitor test reliability
- [ ] Review test quality metrics
- [ ] Ensure tests are maintainable
- [ ] Train team on testing practices

---

This comprehensive testing strategy ensures that the "AI From Scratch to Scale" project maintains high quality, reliability, and performance across all 25 neural network implementations while serving as an educational resource for the AI community.