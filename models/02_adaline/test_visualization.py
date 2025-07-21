#!/usr/bin/env python3
"""
Test script to isolate visualization errors.
"""

import sys
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from model import create_adaline
from config import get_experiment_config
from visualize import (
    plot_adaline_training,
    plot_adaline_decision_boundary,
    plot_adaline_confusion_matrix,
    create_adaline_summary_plot
)

def test_visualizations():
    """Test each visualization function individually."""
    
    # Create a simple model and data
    config = get_experiment_config("delta_rule_demo")
    model = create_adaline(config)
    
    # Generate simple data
    x_data = torch.randn(50, 2)
    y_data = (x_data[:, 0] + x_data[:, 1] > 0).float().unsqueeze(1)
    
    # Train the model briefly
    model.fit(x_data, y_data)
    
    print("Testing individual visualization functions...")
    
    try:
        print("1. Testing training plot...")
        fig1 = plot_adaline_training(model)
        print("✓ Training plot successful")
        plt.close(fig1)
    except Exception as e:
        print(f"✗ Training plot failed: {e}")
    
    try:
        print("2. Testing decision boundary plot...")
        fig2 = plot_adaline_decision_boundary(model, x_data, y_data)
        print("✓ Decision boundary plot successful")
        plt.close(fig2)
    except Exception as e:
        print(f"✗ Decision boundary plot failed: {e}")
    
    try:
        print("3. Testing confusion matrix plot...")
        fig3 = plot_adaline_confusion_matrix(model, x_data, y_data)
        print("✓ Confusion matrix plot successful")
        plt.close(fig3)
    except Exception as e:
        print(f"✗ Confusion matrix plot failed: {e}")
    
    try:
        print("4. Testing summary plot...")
        print(f"x_data shape: {x_data.shape}")
        print(f"y_data shape: {y_data.shape}")
        print(f"y_data type: {type(y_data)}")
        fig4 = create_adaline_summary_plot(model, x_data, y_data)
        print("✓ Summary plot successful")
        plt.close(fig4)
    except Exception as e:
        print(f"✗ Summary plot failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualizations() 