"""
Device management utilities for AI From Scratch to Scale project.

Provides automatic device detection and management with graceful fallbacks
for CPU, CUDA (NVIDIA), and MPS (Apple Silicon) devices.
"""

import logging
from typing import Optional, List, Dict, Any
import platform

try:
    import torch
    # Verify torch is properly loaded by checking for essential attributes
    if hasattr(torch, '__version__') and hasattr(torch, 'device') and hasattr(torch, 'tensor'):
        _TORCH_AVAILABLE = True
        TorchDevice = torch.device
    else:
        # torch module exists but is broken/incomplete
        _TORCH_AVAILABLE = False
        torch = None
        TorchDevice = Any
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False
    TorchDevice = Any

from .exceptions import DeviceError


def setup_device(device_arg: str = 'auto') -> str:
    """
    Set up compute device with automatic fallback.
    
    Attempts to use the best available device with graceful fallbacks:
    1. CUDA (NVIDIA GPU) if available and requested
    2. MPS (Apple Silicon) if available and requested  
    3. CPU as final fallback
    
    Args:
        device_arg: Device specification ('auto', 'cpu', 'cuda', 'mps', or specific like 'cuda:0')
    
    Returns:
        Device string that was successfully set up
    
    Raises:
        DeviceError: If device setup fails completely
    
    Example:
        device = setup_device('auto')  # Best available device
        device = setup_device('cuda')  # Force CUDA if available
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU")
            return 'cpu'
        
        if device_arg == 'auto':
            device = _auto_select_device()
        else:
            device = _validate_and_setup_device(device_arg)
        
        # Test device functionality
        _test_device(device)
        
        logger.info(f"Successfully set up device: {device}")
        return device
        
    except Exception as e:
        logger.warning(f"Device setup failed ({e}), falling back to CPU")
        return 'cpu'


def get_device_info(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive device information.
    
    Args:
        device: Device to get info for (current device if None)
    
    Returns:
        Dictionary with device information
    
    Example:
        info = get_device_info('cuda:0')
        print(f"GPU Memory: {info['memory_total']} MB")
    """
    if not _TORCH_AVAILABLE:
        return {
            'device': 'cpu',
            'type': 'cpu',
            'available': True,
            'torch_available': False
        }
    
    if device is None:
        device = 'cpu'
    
    info = {
        'device': device,
        'torch_available': True,
        'platform': platform.system(),
        'architecture': platform.machine()
    }
    
    try:
        torch_device = torch.device(device)
        info['type'] = torch_device.type
        info['available'] = True
        
        if torch_device.type == 'cuda':
            info.update(_get_cuda_info(torch_device))
        elif torch_device.type == 'mps':
            info.update(_get_mps_info())
        else:
            info.update(_get_cpu_info())
            
    except Exception as e:
        info['available'] = False
        info['error'] = str(e)
    
    return info


def list_available_devices() -> List[str]:
    """
    List all available compute devices.
    
    Returns:
        List of available device strings
    
    Example:
        devices = list_available_devices()
        print(f"Available devices: {devices}")
    """
    devices = ['cpu']
    
    if not _TORCH_AVAILABLE:
        return devices
    
    # Check for CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f'cuda:{i}')
    
    # Check for MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    
    return devices


def get_device_memory_info(device: str) -> Dict[str, float]:
    """
    Get memory information for a device.
    
    Args:
        device: Device string
    
    Returns:
        Dictionary with memory info in MB
    
    Example:
        memory = get_device_memory_info('cuda:0')
        print(f"Free memory: {memory['free']:.1f} MB")
    """
    if not _TORCH_AVAILABLE:
        return {'total': 0, 'free': 0, 'used': 0}
    
    try:
        torch_device = torch.device(device)
        
        if torch_device.type == 'cuda':
            # CUDA memory info
            memory_stats = torch.cuda.memory_stats(torch_device)
            allocated = memory_stats.get('allocated_bytes.all.current', 0)
            reserved = memory_stats.get('reserved_bytes.all.current', 0)
            
            total_memory = torch.cuda.get_device_properties(torch_device).total_memory
            free_memory = total_memory - reserved
            
            return {
                'total': total_memory / (1024**2),  # Convert to MB
                'allocated': allocated / (1024**2),
                'reserved': reserved / (1024**2),
                'free': free_memory / (1024**2),
                'used': reserved / (1024**2)
            }
        
        elif torch_device.type == 'mps':
            # MPS memory info (limited)
            return {
                'total': -1,  # Not available
                'free': -1,   # Not available
                'used': -1    # Not available
            }
        
        else:
            # CPU memory info
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'total': memory.total / (1024**2),
                    'free': memory.available / (1024**2),
                    'used': memory.used / (1024**2)
                }
            except ImportError:
                return {'total': -1, 'free': -1, 'used': -1}
                
    except Exception:
        return {'total': 0, 'free': 0, 'used': 0}


def clear_device_cache(device: str):
    """
    Clear device cache to free up memory.
    
    Args:
        device: Device to clear cache for
    
    Example:
        clear_device_cache('cuda:0')
    """
    if not _TORCH_AVAILABLE:
        return
    
    logger = logging.getLogger(__name__)
    
    try:
        torch_device = torch.device(device)
        
        if torch_device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug(f"Cleared CUDA cache for {device}")
        elif torch_device.type == 'mps':
            # MPS doesn't have explicit cache clearing
            logger.debug("MPS cache clearing not available")
        else:
            logger.debug("CPU doesn't require cache clearing")
            
    except Exception as e:
        logger.warning(f"Failed to clear cache for {device}: {e}")


def move_to_device(tensor_or_model, device: str, non_blocking: bool = False):
    """
    Move tensor or model to specified device.
    
    Args:
        tensor_or_model: PyTorch tensor or model
        device: Target device
        non_blocking: Whether to use non-blocking transfer
    
    Returns:
        Tensor or model on the target device
    
    Example:
        model = move_to_device(model, 'cuda:0')
        tensor = move_to_device(tensor, device)
    """
    if not _TORCH_AVAILABLE:
        return tensor_or_model
    
    try:
        return tensor_or_model.to(device, non_blocking=non_blocking)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to move to device {device}: {e}")
        return tensor_or_model


# Private helper functions

def _auto_select_device() -> str:
    """Automatically select the best available device."""
    logger = logging.getLogger(__name__)
    
    # Prefer CUDA if available
    if torch.cuda.is_available():
        device = 'cuda'
        logger.debug("Auto-selected CUDA device")
        return device
    
    # Fallback to MPS on Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.debug("Auto-selected MPS device")
        return device
    
    # Final fallback to CPU
    logger.debug("Auto-selected CPU device")
    return 'cpu'


def _validate_and_setup_device(device_arg: str) -> str:
    """Validate and setup a specific device."""
    logger = logging.getLogger(__name__)
    
    if device_arg == 'cpu':
        return 'cpu'
    
    if device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            raise DeviceError(f"CUDA requested but not available")
        
        # Parse device index if specified
        if ':' in device_arg:
            device_idx = int(device_arg.split(':')[1])
            if device_idx >= torch.cuda.device_count():
                raise DeviceError(f"CUDA device {device_idx} not available")
        
        return device_arg
    
    if device_arg == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise DeviceError("MPS requested but not available")
        
        return 'mps'
    
    raise DeviceError(f"Unknown device: {device_arg}")


def _test_device(device: str):
    """Test device functionality with a simple operation."""
    if not _TORCH_AVAILABLE or device == 'cpu':
        return
    
    try:
        # Create a small tensor and move it to device
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        test_tensor = test_tensor.to(device)
        
        # Perform a simple operation
        result = test_tensor * 2
        
        # Move back to CPU to verify
        result_cpu = result.cpu()
        
        expected = torch.tensor([2.0, 4.0, 6.0])
        if not torch.allclose(result_cpu, expected):
            raise DeviceError(f"Device {device} failed functionality test")
            
    except Exception as e:
        raise DeviceError(f"Device {device} functionality test failed: {e}")


def _get_cuda_info(device: TorchDevice) -> Dict[str, Any]:
    """Get CUDA-specific device information."""
    info = {}
    
    try:
        props = torch.cuda.get_device_properties(device)
        info.update({
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_mb': props.total_memory / (1024**2),
            'multiprocessor_count': props.multi_processor_count,
            'cuda_version': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device()
        })
    except Exception as e:
        info['error'] = f"Failed to get CUDA info: {e}"
    
    return info


def _get_mps_info() -> Dict[str, Any]:
    """Get MPS-specific device information."""
    info = {
        'name': 'Apple Metal Performance Shaders',
        'backend': 'MPS'
    }
    
    try:
        # MPS-specific info is limited
        info['torch_mps_available'] = torch.backends.mps.is_available()
        if hasattr(torch.backends.mps, 'is_built'):
            info['torch_mps_built'] = torch.backends.mps.is_built()
    except Exception as e:
        info['error'] = f"Failed to get MPS info: {e}"
    
    return info


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU-specific device information."""
    info = {
        'name': 'CPU',
        'backend': 'CPU'
    }
    
    try:
        import multiprocessing
        info['cpu_count'] = multiprocessing.cpu_count()
        info['platform'] = platform.processor()
        
        # Check for CPU optimizations
        if _TORCH_AVAILABLE:
            info['torch_threads'] = torch.get_num_threads()
            info['mkl_available'] = torch.backends.mkl.is_available()
            info['openmp_available'] = torch.backends.openmp.is_available()
            
    except Exception as e:
        info['error'] = f"Failed to get CPU info: {e}"
    
    return info 