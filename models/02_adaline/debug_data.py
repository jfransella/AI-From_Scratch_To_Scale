#!/usr/bin/env python3
"""
Debug script to examine data generation.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def generate_fallback_data(dataset_type: str, n_samples: int = 100) -> tuple:
    """Generate fallback synthetic data for training."""
    if dataset_type == "simple_linear":
        # Simple linearly separable data
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        return x, y
    
    elif dataset_type == "noisy_linear":
        # Linearly separable data with noise
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        # Add some noise
        y = y + 0.1 * torch.randn_like(y)
        y = torch.clamp(y, 0, 1)  # Keep in [0,1] range
        return x, y

def analyze_data(dataset_type: str):
    """Analyze the generated data."""
    print(f"\n=== Analyzing {dataset_type} dataset ===")
    
    x, y = generate_fallback_data(dataset_type, 100)
    
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Y min: {y.min().item():.4f}")
    print(f"Y max: {y.max().item():.4f}")
    print(f"Y mean: {y.mean().item():.4f}")
    print(f"Y std: {y.std().item():.4f}")
    
    # Count unique values
    unique_values = torch.unique(y)
    print(f"Unique Y values: {unique_values.tolist()}")
    
    # Count values around 0 and 1
    near_zero = torch.sum(y < 0.1)
    near_one = torch.sum(y > 0.9)
    middle = torch.sum((y >= 0.1) & (y <= 0.9))
    
    print(f"Values < 0.1: {near_zero.item()}")
    print(f"Values > 0.9: {near_one.item()}")
    print(f"Values in [0.1, 0.9]: {middle.item()}")
    
    # Show some examples
    print("\nFirst 10 examples:")
    for i in range(min(10, len(x))):
        print(f"  X[{i}]: {x[i].tolist()}, Y[{i}]: {y[i].item():.4f}")

if __name__ == "__main__":
    analyze_data("simple_linear")
    analyze_data("noisy_linear") 