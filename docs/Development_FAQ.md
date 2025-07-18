# **Development FAQ: AI From Scratch to Scale**

This document provides solutions to common issues encountered when developing models for the "AI From Scratch to Scale" project. It's designed to help AI assistants quickly troubleshoot problems and maintain development momentum.

## **Table of Contents**

1. [Environment Setup Issues](#environment-setup-issues)
2. [Import and Dependency Problems](#import-and-dependency-problems)
3. [Training Issues](#training-issues)
4. [Data Loading Problems](#data-loading-problems)
5. [Model Implementation Issues](#model-implementation-issues)
6. [Visualization and Logging Problems](#visualization-and-logging-problems)
7. [Performance Issues](#performance-issues)
8. [Testing and Validation Problems](#testing-and-validation-problems)
9. [Windows-Specific Issues](#windows-specific-issues)
10. [Debugging Workflows](#debugging-workflows)

---

## **Environment Setup Issues**

### **Q: Virtual environment creation fails**

**Problem**: `python -m venv .venv` fails with permission errors or module not found.

**Solutions**:

```powershell
# Ensure Python is properly installed
python --version

# If python command not found, try:
py -m venv .venv

# If venv module not found, install it:
pip install virtualenv
virtualenv .venv

# For permission issues on Windows:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Q: Virtual environment activation fails**

**Problem**: `.venv\Scripts\activate` gives execution policy errors.

**Solutions**:

```powershell
# Option 1: Change execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Option 2: Use PowerShell directly
.venv\Scripts\Activate.ps1

# Option 3: Use batch file
.venv\Scripts\activate.bat

# Verify activation worked
where python
# Should show path to .venv\Scripts\python.exe
```

### **Q: Shared packages installation fails**

**Problem**: `pip install -e ..\..` fails with import errors.

**Solutions**:

```powershell
# Ensure you're in the model directory
cd models\XX_modelname

# Activate virtual environment first
.venv\Scripts\activate

# Install development dependencies first
pip install -r ..\..\requirements-dev.txt

# Then install shared packages
pip install -e ..\..

# Verify installation
python -c "from data_utils import load_dataset; print('Success')"
```

### **Q: Package version conflicts**

**Problem**: Conflicting package versions between model requirements and shared packages.

**Solutions**:

```powershell
# Check for conflicts
pip check

# Create fresh environment
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\activate

# Install in correct order
pip install -r requirements.txt
pip install -r ..\..\requirements-dev.txt
pip install -e ..\..

# If still conflicts, pin versions in requirements.txt
```

---

## **Import and Dependency Problems**

### **Q: Cannot import shared modules**

**Problem**: `ImportError: No module named 'data_utils'` when trying to import shared packages.

**Solutions**:

```python
# 1. Verify virtual environment is activated
import sys
print(sys.executable)  # Should show .venv path

# 2. Check if shared packages are installed
pip list | grep -i "ai-from-scratch"  # or use findstr on Windows

# 3. Reinstall shared packages
pip install -e ..\..

# 4. Check sys.path
import sys
print(sys.path)  # Should include project root

# 5. If still failing, add to sys.path temporarily
import sys
sys.path.append(r'C:\path\to\ai-from-scratch-to-scale')
```

### **Q: CUDA/PyTorch installation issues**

**Problem**: PyTorch not using GPU or CUDA version mismatch.

**Solutions**:

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA version: {torch.version.cuda}")

# If CUDA not available, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only development
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **Q: Missing dependencies for specific models**

**Problem**: Model-specific dependencies not installed.

**Solutions**:

```powershell
# Check model's requirements.txt
cat requirements.txt

# Install missing dependencies
pip install -r requirements.txt

# For transformers models
pip install transformers tokenizers

# For vision models
pip install opencv-python pillow

# For graph models
pip install torch-geometric
```

---

## **Training Issues**

### **Q: Training crashes immediately**

**Problem**: Training script exits with errors before starting.

**Debugging Steps**:

```python
# 1. Check configuration
python -c "from src.config import get_config; print(get_config('xor'))"

# 2. Test model instantiation
python -c "from src.model import ModelClass; m = ModelClass(); print(m)"

# 3. Test data loading
python -c "from data_utils import load_dataset; d = load_dataset('xor'); print(next(iter(d)))"

# 4. Run in debug mode
python src\train.py --experiment debug_small --epochs 1
```

### **Q: Loss becomes NaN during training**

**Problem**: Training loss becomes NaN after a few iterations.

**Solutions**:

```python
# 1. Check learning rate
config = get_config('experiment_name')
if config['learning_rate'] > 0.1:
    print("Learning rate too high, try 0.01 or lower")

# 2. Add gradient clipping
import torch.nn.utils as nn_utils
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check for unstable operations
# Add debug prints in forward pass
def forward(self, x):
    x = self.layer1(x)
    print(f"After layer1: min={x.min()}, max={x.max()}, has_nan={torch.isnan(x).any()}")
    return x

# 4. Use more stable loss function
# Replace CrossEntropyLoss with more stable version
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
```

### **Q: Training is extremely slow**

**Problem**: Training takes much longer than expected.

**Solutions**:

```python
# 1. Check device usage
print(f"Using device: {torch.cuda.is_available()}")
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Optimize data loading
train_loader = DataLoader(
    dataset, 
    batch_size=config['batch_size'],
    num_workers=4,  # Increase for faster data loading
    pin_memory=True,
    persistent_workers=True
)

# 3. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 4. Profile the training loop
import torch.profiler
with torch.profiler.profile() as prof:
    # Run training step
    prof.export_chrome_trace("trace.json")
```

### **Q: Model not learning (loss plateaus)**

**Problem**: Training loss doesn't decrease after initial epochs.

**Solutions**:

```python
# 1. Check learning rate
if config['learning_rate'] < 1e-5:
    print("Learning rate too small, try 0.001 or higher")

# 2. Verify data preprocessing
# Check if data is normalized
print(f"Data range: {X.min()} to {X.max()}")
print(f"Data mean: {X.mean()}, std: {X.std()}")

# 3. Check model capacity
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
# Too few parameters = underfitting
# Too many parameters = overfitting

# 4. Verify loss function matches problem
# Binary classification: BCELoss or BCEWithLogitsLoss
# Multi-class: CrossEntropyLoss
# Regression: MSELoss or L1Loss
```

### **Q: Early stopping triggers too early**

**Problem**: Training stops before model has converged.

**Solutions**:

```python
# 1. Increase patience
config['early_stopping_patience'] = 50  # Default was 20

# 2. Reduce minimum delta
config['early_stopping_min_delta'] = 1e-5  # Default was 1e-4

# 3. Monitor validation loss trend
# Add logging to see if loss is actually improving
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.legend()
plt.show()

# 4. Disable early stopping for debugging
config['early_stopping_patience'] = 999999
```

---

## **Data Loading Problems**

### **Q: Dataset not found error**

**Problem**: `FileNotFoundError` when loading datasets.

**Solutions**:

```python
# 1. Check data directory structure
import os
print(os.listdir('data'))  # Should show raw, processed, generated folders

# 2. Create missing directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/generated', exist_ok=True)

# 3. Download datasets manually
# For MNIST, CIFAR-10, etc.
import torchvision.datasets as datasets
datasets.MNIST('data/raw', download=True)
datasets.CIFAR10('data/raw', download=True)

# 4. Use absolute paths for debugging
data_path = os.path.abspath('data/raw/mnist')
```

### **Q: Data loading is very slow**

**Problem**: DataLoader takes too long to iterate through data.

**Solutions**:

```python
# 1. Increase num_workers
train_loader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=min(8, os.cpu_count()),  # Don't exceed CPU count
    pin_memory=True,
    persistent_workers=True
)

# 2. Use faster data format
# Convert to HDF5 for large datasets
import h5py
with h5py.File('data/processed/mnist.h5', 'w') as f:
    f.create_dataset('X', data=X)
    f.create_dataset('y', data=y)

# 3. Preprocess data once
# Save preprocessed data to disk
import joblib
joblib.dump(preprocessed_data, 'data/processed/preprocessed.pkl')

# 4. Use caching
from functools import lru_cache
@lru_cache(maxsize=1000)
def load_sample(idx):
    return expensive_preprocessing(raw_data[idx])
```

### **Q: Out of memory during data loading**

**Problem**: System runs out of memory when loading large datasets.

**Solutions**:

```python
# 1. Reduce batch size
config['batch_size'] = 16  # or smaller

# 2. Use lazy loading
class LazyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # Don't load all data at once
    
    def __getitem__(self, idx):
        # Load only when needed
        return load_single_sample(self.data_path, idx)

# 3. Use memory mapping
data = np.memmap('data/large_dataset.dat', dtype='float32', mode='r')

# 4. Clear cache periodically
import gc
gc.collect()
torch.cuda.empty_cache()  # For GPU memory
```

---

## **Model Implementation Issues**

### **Q: Model architecture errors**

**Problem**: Shape mismatches or layer incompatibilities.

**Solutions**:

```python
# 1. Print tensor shapes during forward pass
def forward(self, x):
    print(f"Input shape: {x.shape}")
    x = self.layer1(x)
    print(f"After layer1: {x.shape}")
    x = self.layer2(x)
    print(f"After layer2: {x.shape}")
    return x

# 2. Use torchsummary for model inspection
from torchsummary import summary
summary(model, input_size=(1, 28, 28))

# 3. Check layer parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 4. Use dummy input for debugging
dummy_input = torch.randn(1, *input_shape)
try:
    output = model(dummy_input)
    print(f"Model works! Output shape: {output.shape}")
except Exception as e:
    print(f"Model error: {e}")
```

### **Q: Parameter initialization issues**

**Problem**: Poor initial weights causing training problems.

**Solutions**:

```python
# 1. Use proper initialization
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')

model.apply(init_weights)

# 2. Check initial parameter values
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")

# 3. Use PyTorch's default initialization
# Most layers have good defaults, avoid manual initialization unless needed
```

### **Q: Model saving/loading issues**

**Problem**: Errors when saving or loading model checkpoints.

**Solutions**:

```python
# 1. Save model state dict, not entire model
torch.save(model.state_dict(), 'model_checkpoint.pth')

# 2. Load with proper error handling
try:
    model.load_state_dict(torch.load('model_checkpoint.pth'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Loading failed: {e}")

# 3. Save additional information
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'config': config
}
torch.save(checkpoint, 'full_checkpoint.pth')

# 4. Handle device mismatches
# Load on CPU first, then move to target device
checkpoint = torch.load('model.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.to(device)
```

---

## **Visualization and Logging Problems**

### **Q: Plots not generating**

**Problem**: `--visualize` flag doesn't create expected plots.

**Solutions**:

```python
# 1. Check plot directory exists
import os
plot_dir = 'outputs/visualizations'
os.makedirs(plot_dir, exist_ok=True)

# 2. Test plotting function directly
from plotting import generate_loss_curve
generate_loss_curve([1, 0.5, 0.2], [1.2, 0.6, 0.3], save_path='test_plot.png')

# 3. Check matplotlib backend
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt

# 4. Add debug prints
def generate_visualization(data, plot_type):
    print(f"Generating {plot_type} with {len(data)} points")
    # plotting code
    plt.savefig(f'{plot_type}.png')
    print(f"Saved {plot_type}.png")
```

### **Q: Wandb logging not working**

**Problem**: Training metrics not appearing in wandb dashboard.

**Solutions**:

```python
# 1. Check wandb initialization
import wandb
wandb.login()  # Enter API key
wandb.init(project='ai-from-scratch', entity='your-username')

# 2. Test logging manually
wandb.log({'test_metric': 0.5})

# 3. Check wandb configuration
print(wandb.config)
print(wandb.run.name)

# 4. Handle offline mode
# If no internet connection
wandb.init(mode='offline')
# Later sync with: wandb sync wandb/offline-run-xxx
```

### **Q: Log files empty or missing**

**Problem**: Training logs not being written to files.

**Solutions**:

```python
# 1. Check logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/training.log'),
        logging.StreamHandler()
    ]
)

# 2. Ensure log directory exists
import os
os.makedirs('outputs/logs', exist_ok=True)

# 3. Force flush logs
import sys
sys.stdout.flush()
sys.stderr.flush()

# 4. Test logging
logger = logging.getLogger(__name__)
logger.info("Test log message")
```

---

## **Performance Issues**

### **Q: High memory usage during training**

**Problem**: System runs out of memory during training.

**Solutions**:

```python
# 1. Reduce batch size
config['batch_size'] = 16  # or smaller

# 2. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Clear GPU cache
torch.cuda.empty_cache()

# 4. Use CPU for some operations
# Move less critical operations to CPU
model.eval()
with torch.no_grad():
    predictions = model(data.cpu()).cpu()
```

### **Q: Slow GPU utilization**

**Problem**: GPU usage is low during training.

**Solutions**:

```python
# 1. Check data transfer bottlenecks
# Use pin_memory and non_blocking transfer
data = data.to(device, non_blocking=True)

# 2. Increase batch size
config['batch_size'] = 128  # or larger if memory allows

# 3. Use multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 4. Profile GPU usage
import torch.profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # training code
    prof.export_chrome_trace("gpu_trace.json")
```

### **Q: Training time much longer than expected**

**Problem**: Training takes significantly longer than benchmarks.

**Solutions**:

```python
# 1. Profile the training loop
import time
import cProfile

def profile_training():
    start_time = time.time()
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        # training code
        epoch_end = time.time()
        print(f"Epoch {epoch}: {epoch_end - epoch_start:.2f}s")

# 2. Identify bottlenecks
# Data loading: increase num_workers
# Model forward: use mixed precision
# Loss computation: use efficient loss functions
# Backward pass: use gradient checkpointing for large models

# 3. Use optimized implementations
# Replace custom operations with PyTorch built-ins
# Use torchvision transforms instead of custom preprocessing
```

---

## **Testing and Validation Problems**

### **Q: Tests failing after code changes**

**Problem**: Previously passing tests now fail.

**Solutions**:

```powershell
# 1. Run specific failing test
pytest tests/unit/test_model.py::TestModel::test_forward_pass -v

# 2. Check for import errors
python -c "from src.model import ModelClass; print('Import successful')"

# 3. Update test fixtures
# Check if test data format changed
pytest tests/unit/test_model.py -v --tb=short

# 4. Run tests in isolation
pytest tests/unit/test_model.py::TestModel::test_forward_pass --forked
```

### **Q: Tests pass locally but fail in CI**

**Problem**: Tests work on local machine but fail in GitHub Actions.

**Solutions**:

```yaml
# 1. Check Python version in CI
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.9'  # Match your local version

# 2. Add debugging to CI
- name: Debug environment
  run: |
    python --version
    pip list
    python -c "import torch; print(torch.__version__)"

# 3. Install exact dependencies
- name: Install dependencies
  run: |
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    pip install -e .

# 4. Set environment variables
env:
  PYTHONPATH: ${{ github.workspace }}
```

---

## **Windows-Specific Issues**

### **Q: Path separator issues**

**Problem**: Linux-style paths don't work on Windows.

**Solutions**:

```python
# 1. Use pathlib for cross-platform paths
from pathlib import Path
data_path = Path('data') / 'raw' / 'mnist'

# 2. Use os.path.join
import os
data_path = os.path.join('data', 'raw', 'mnist')

# 3. Convert forward slashes
path = 'data/raw/mnist'.replace('/', os.sep)

# 4. Use raw strings for Windows paths
windows_path = r'C:\Users\user\data\mnist'
```

### **Q: PowerShell execution policy errors**

**Problem**: Cannot run PowerShell scripts due to execution policy.

**Solutions**:

```powershell
# 1. Check current policy
Get-ExecutionPolicy

# 2. Set policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Run specific script
PowerShell -ExecutionPolicy Bypass -File script.ps1

# 4. Use batch file instead
# Create activate.bat:
@echo off
call .venv\Scripts\activate.bat
```

### **Q: Long path issues**

**Problem**: Windows path length limits causing errors.

**Solutions**:

```python
# 1. Use shorter path names
# Instead of: very_long_descriptive_model_name
# Use: vlmn or short_model

# 2. Move project closer to root
# Instead of: C:\Users\username\Documents\Projects\ai-from-scratch-to-scale
# Use: C:\ai-scratch

# 3. Enable long path support
# Run as administrator in PowerShell:
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## **Debugging Workflows**

### **Systematic Debugging Process**

#### **1. Initial Triage**

```python
# Step 1: Reproduce the issue
python src\train.py --experiment debug_small --epochs 1

# Step 2: Check basic setup
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.config import get_config; print('Config OK')"
python -c "from src.model import ModelClass; print('Model OK')"

# Step 3: Isolate the component
# Test each component separately
```

#### **2. Training Issues Debug Flow**

```python
# Debug training problems systematically
def debug_training():
    print("1. Testing configuration...")
    config = get_config('debug_small')
    print(f"   Config loaded: {config['experiment']}")
    
    print("2. Testing model creation...")
    model = ModelClass(config)
    print(f"   Model created: {model.__class__.__name__}")
    
    print("3. Testing data loading...")
    train_loader = load_dataset(config)
    batch = next(iter(train_loader))
    print(f"   Data loaded: {batch[0].shape}, {batch[1].shape}")
    
    print("4. Testing forward pass...")
    output = model(batch[0])
    print(f"   Forward pass: {output.shape}")
    
    print("5. Testing loss computation...")
    loss = criterion(output, batch[1])
    print(f"   Loss computed: {loss.item()}")
    
    print("6. Testing backward pass...")
    loss.backward()
    print("   Backward pass completed")
    
    print("All components working!")

debug_training()
```

#### **3. Data Issues Debug Flow**

```python
def debug_data():
    print("1. Checking data directory...")
    import os
    print(f"   Data dir exists: {os.path.exists('data')}")
    print(f"   Contents: {os.listdir('data') if os.path.exists('data') else 'None'}")
    
    print("2. Testing data loading...")
    try:
        dataset = load_dataset('synthetic')
        print(f"   Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    print("3. Testing sample access...")
    sample = dataset[0]
    print(f"   Sample shape: X={sample[0].shape}, y={sample[1].shape}")
    print(f"   Sample range: X=[{sample[0].min():.3f}, {sample[0].max():.3f}]")
    
    print("4. Testing data loader...")
    dataloader = DataLoader(dataset, batch_size=4)
    batch = next(iter(dataloader))
    print(f"   Batch shape: X={batch[0].shape}, y={batch[1].shape}")
    
    print("Data pipeline working!")

debug_data()
```

#### **4. Model Issues Debug Flow**

```python
def debug_model():
    print("1. Testing model instantiation...")
    config = get_config('debug_small')
    model = ModelClass(config)
    print(f"   Model: {model}")
    
    print("2. Testing parameter initialization...")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.shape}, mean={param.mean():.4f}")
    
    print("3. Testing forward pass with dummy data...")
    dummy_input = torch.randn(1, config['input_size'])
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("4. Testing gradient computation...")
    loss = output.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"   {name} grad: mean={param.grad.mean():.4f}")
        else:
            print(f"   {name}: No gradient!")
    
    print("Model working!")

debug_model()
```

### **Common Debug Commands**

```powershell
# Quick environment check
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
python -c "from src import *; print('All imports OK')"

# Quick training test
python src\train.py --experiment debug_small --epochs 1 --batch-size 4

# Memory usage monitoring
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# GPU monitoring
python -c "import torch; print(f'GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')"

# Check file permissions
Get-Acl outputs\logs\training.log

# Network connectivity test (for wandb)
Test-NetConnection api.wandb.ai -Port 443
```

### **Error Message Patterns**

#### **Common Error Types and Solutions**

**ImportError patterns**:

```
"No module named 'src'" → Check PYTHONPATH and virtual environment
"cannot import name 'X' from 'Y'" → Check for circular imports
"DLL load failed" → PyTorch/CUDA version mismatch
```

**RuntimeError patterns**:

```
"CUDA out of memory" → Reduce batch size or clear cache
"Expected tensor to be on device X" → Check device placement
"dimension mismatch" → Check tensor shapes
```

**ValueError patterns**:

```
"Target X is out of bounds" → Check label encoding
"Expected input batch_size (X) to match target batch_size (Y)" → Check data loading
```

---

## **Emergency Recovery Procedures**

### **If Everything Breaks**

1. **Nuclear Option - Fresh Start**:

```powershell
# Save your work first!
git add .
git commit -m "Save work before reset"

# Delete virtual environment
Remove-Item -Recurse -Force .venv

# Create fresh environment
python -m venv .venv
.venv\Scripts\activate

# Reinstall everything
pip install -r requirements.txt
pip install -r ..\..\requirements-dev.txt
pip install -e ..\..
```

2. **Minimum Viable Test**:

```python
# Test absolute minimum functionality
import torch
print(f"PyTorch: {torch.__version__}")

x = torch.randn(2, 3)
print(f"Tensor created: {x.shape}")

if torch.cuda.is_available():
    x = x.cuda()
    print("CUDA working")

print("Basic PyTorch working!")
```

3. **Incremental Recovery**:

```python
# Test components one by one
# 1. Basic imports
# 2. Configuration loading
# 3. Model creation
# 4. Data loading
# 5. Training step
# 6. Logging/visualization
```

---

## **Getting Help**

### **When to Ask for Help**

- Spent >30 minutes on environment setup issues
- Reproducible errors with clear error messages
- Performance issues with specific metrics
- Test failures with detailed error logs

### **How to Ask for Help**

```python
# Include this information:
print("=== Debug Information ===")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
print(f"Working directory: {os.getcwd()}")
print(f"Virtual env: {sys.executable}")
print("=== End Debug Info ===")

# Include:
# - Exact error message
# - Steps to reproduce
# - Expected vs actual behavior
# - Recent changes made
```

---

This FAQ should help you quickly resolve most common issues. When in doubt, start with the debugging workflows to systematically identify the problem area.