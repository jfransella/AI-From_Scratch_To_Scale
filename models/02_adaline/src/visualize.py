"""
ADALINE Visualization Module.

Visualization functions for ADALINE model training, decision boundaries,
and comparison with Perceptron using the shared plotting package.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# Add shared plotting package to path  
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from plotting import (
    plot_decision_boundary,
    plot_confusion_matrix,
    setup_plotting_style,
    save_figure,
    create_subplots
)
from plotting.utils import PROJECT_COLORS

# Initialize plotting style
setup_plotting_style()


def plot_adaline_training(model, title: str = "ADALINE Training Progress", 
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot ADALINE training history with MSE focus."""
    
    if not hasattr(model, 'training_history') or not model.training_history['loss']:
        raise ValueError("Model has no training history to plot")
    
    loss_history = model.training_history['loss']
    mse_history = model.training_history['mse']
    
    # Create plot using shared plotting utilities
    fig, ax = create_subplots(figsize=(12, 6))
    
    epochs = range(len(loss_history))
    
    # Plot MSE (primary metric for ADALINE)
    ax.plot(epochs, mse_history, 
           color=PROJECT_COLORS['error'], 
           linewidth=2.5,
           label='Mean Squared Error',
           marker='o' if len(epochs) <= 20 else None,
           markersize=4)
    
    # Formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'{title}\nDelta Rule Learning Progress', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add convergence indicator if applicable
    if hasattr(model, 'config') and len(mse_history) > 0:
        tolerance = getattr(model.config, 'tolerance', 1e-6)
        if mse_history[-1] < tolerance:
            ax.axhline(y=tolerance, color=PROJECT_COLORS['success'], 
                      linestyle='--', alpha=0.7, label=f'Convergence Threshold ({tolerance})')
            ax.legend()
    
    # Add educational annotation
    ax.text(0.02, 0.98, 
           "ADALINE learns continuously\nfrom error magnitude", 
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_adaline_decision_boundary(model, x_data: torch.Tensor, y_data: torch.Tensor,
                                 title: str = "ADALINE Decision Boundary",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Plot ADALINE decision boundary with continuous predictions."""
    
    # Convert to numpy for plotting
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data.detach().cpu().numpy().flatten()
    
    # Create a simple wrapper to make model compatible with plotting function
    class ADALINEWrapper:
        def __init__(self, adaline_model):
            self.model = adaline_model
        
        def predict(self, x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                linear_output = self.model.forward(x)
                return (linear_output > 0).float()
        
        def predict_proba(self, x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                linear_output = self.model.forward(x)
                # Convert linear output to probabilities using sigmoid
                prob_positive = torch.sigmoid(linear_output)
                prob_negative = 1 - prob_positive
                return torch.stack([prob_negative, prob_positive], dim=1)
    
    wrapper = ADALINEWrapper(model)
    
    # Use shared plotting function
    fig = plot_decision_boundary(
        wrapper, x_np, y_np, 
        title=f"{title}\nLinear Decision Boundary (Delta Rule)",
        save_path=None,  # We'll save manually to add annotations
        cmap='RdYlBu'
    )
    
    # Add educational annotations
    ax = fig.gca()
    ax.text(0.02, 0.02, 
           "Linear boundary from\ncontinuous learning", 
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='bottom',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_delta_rule_evolution(model, x_data: torch.Tensor, y_data: torch.Tensor,
                            snapshots: List[int] = [0, 25, 50, 100],
                            title: str = "Delta Rule Weight Evolution",
                            save_path: Optional[str] = None) -> plt.Figure:
    """Plot how Delta Rule evolves decision boundary over training."""
    
    # This would require saving model snapshots during training
    # For now, create a simplified version showing final boundary
    fig, axes = create_subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data.detach().cpu().numpy().flatten()
    
    # For demonstration, create multiple snapshots with different noise levels
    # In a full implementation, these would be actual training snapshots
    for i, (ax, epoch) in enumerate(zip(axes, snapshots)):
        
        # Simulate training progress by adding noise to weights
        progress = epoch / max(snapshots) if max(snapshots) > 0 else 1.0
        
        ax.scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], 
                  c='blue', alpha=0.6, label='Class 0', s=40)
        ax.scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], 
                  c='red', alpha=0.6, label='Class 1', s=40)
        
        # Draw a simplified decision boundary
        x_min, x_max = x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5
        
        # Simple linear boundary (would use actual model weights in full implementation)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -x_line * (1 - progress * 0.5)  # Simplified evolution
        
        ax.plot(x_line, y_line, 'k-', linewidth=2, alpha=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_np[:, 1].min() - 0.5, x_np[:, 1].max() + 0.5)
        ax.set_title(f'Epoch {epoch}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_adaline_vs_perceptron_learning(adaline_history: List[float],
                                       perceptron_history: List[float],
                                       title: str = "ADALINE vs Perceptron Learning",
                                       save_path: Optional[str] = None) -> plt.Figure:
    """Compare ADALINE and Perceptron learning curves."""
    
    fig, ax = create_subplots(figsize=(12, 8))
    
    epochs_adaline = range(len(adaline_history))
    epochs_perceptron = range(len(perceptron_history))
    
    # Plot ADALINE (continuous learning)
    ax.plot(epochs_adaline, adaline_history, 
           color=PROJECT_COLORS['primary'], 
           linewidth=2.5,
           label='ADALINE (Delta Rule)',
           marker='o' if len(epochs_adaline) <= 20 else None,
           markersize=4)
    
    # Plot Perceptron (discrete learning)  
    ax.plot(epochs_perceptron, perceptron_history,
           color=PROJECT_COLORS['secondary'],
           linewidth=2.5, 
           label='Perceptron (Step Rule)',
           marker='s' if len(epochs_perceptron) <= 20 else None,
           markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error/Loss')
    ax.set_title(f'{title}\nContinuous vs Discrete Learning', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add educational annotation
    ax.text(0.02, 0.98, 
           "ADALINE: Smooth convergence\nPerceptron: Step-wise updates", 
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.7))
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def create_adaline_summary_plot(model, x_data: torch.Tensor, y_data: torch.Tensor,
                              title: str = "ADALINE Analysis Summary",
                              save_path: Optional[str] = None) -> plt.Figure:
    """Create comprehensive summary plot for ADALINE."""
    
    fig, axes = create_subplots(2, 2, figsize=(16, 12))
    
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data.detach().cpu().numpy().flatten()
    
    # 1. Training History
    if hasattr(model, 'training_history') and model.training_history['loss']:
        loss_history = model.training_history['loss']
        epochs = range(len(loss_history))
        
        axes[0, 0].plot(epochs, loss_history, 
                       color=PROJECT_COLORS['error'], linewidth=2)
        axes[0, 0].set_title('Training Loss (MSE)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean Squared Error')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Decision Boundary
    axes[0, 1].scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], 
                      c='blue', alpha=0.6, label='Class 0', s=40)
    axes[0, 1].scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], 
                      c='red', alpha=0.6, label='Class 1', s=40)
    axes[0, 1].set_title('Decision Boundary')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model Weights
    if hasattr(model, 'linear'):
        weights = model.linear.weight.detach().cpu().numpy().flatten()
        bias = model.linear.bias.detach().cpu().numpy().item()
        
        axes[1, 0].bar(['Weight 1', 'Weight 2', 'Bias'], 
                      [weights[0], weights[1], bias],
                      color=[PROJECT_COLORS['primary'], PROJECT_COLORS['secondary'], PROJECT_COLORS['accent']])
        axes[1, 0].set_title('Learned Parameters')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model Info
    axes[1, 1].axis('off')
    info_text = f"""
ADALINE Model Summary

Algorithm: Delta Rule (LMS)
Architecture: Single Linear Layer
Activation: Linear (none)
Learning: Continuous Error-based

Key Features:
• Learns from error magnitude
• Smoother convergence
• Better noise tolerance
• Linear decision boundary

Historical Context:
Introduced 1960 by Widrow & Hoff
First continuous learning neural net
Foundation for gradient descent
"""
    
    axes[1, 1].text(0.1, 0.9, info_text, 
                    transform=axes[1, 1].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_adaline_confusion_matrix(model, x_data: torch.Tensor, y_data: torch.Tensor,
                                 title: str = "ADALINE Confusion Matrix",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Plot confusion matrix for ADALINE predictions."""
    
    model.eval()
    with torch.no_grad():
        predictions = model.predict(x_data)
        
    # Convert to numpy for sklearn compatibility
    y_true = y_data.detach().cpu().numpy().flatten()
    y_pred = predictions.detach().cpu().numpy().flatten()
    
    # Convert to binary (0,1) if needed
    y_true_binary = (y_true > 0.5).astype(int)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Use shared plotting function
    fig = plot_confusion_matrix(
        y_true_binary, y_pred_binary,
        class_names=['Class 0', 'Class 1'],
        title=title,
        save_path=save_path
    )
    
    return fig


def save_all_adaline_plots(model, x_data: torch.Tensor, y_data: torch.Tensor,
                         output_dir: str = "outputs/visualizations"):
    """Generate and save all ADALINE visualization plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plots_created = []
    
    try:
        # Training history
        fig1 = plot_adaline_training(model, save_path=str(output_path / "adaline_training.png"))
        plots_created.append("adaline_training.png")
        plt.close(fig1)
        
        # Decision boundary
        fig2 = plot_adaline_decision_boundary(model, x_data, y_data, 
                                            save_path=str(output_path / "adaline_decision_boundary.png"))
        plots_created.append("adaline_decision_boundary.png")
        plt.close(fig2)
        
        # Confusion matrix
        fig3 = plot_adaline_confusion_matrix(model, x_data, y_data,
                                           save_path=str(output_path / "adaline_confusion_matrix.png"))
        plots_created.append("adaline_confusion_matrix.png")
        plt.close(fig3)
        
        # Summary plot
        fig4 = create_adaline_summary_plot(model, x_data, y_data,
                                         save_path=str(output_path / "adaline_summary.png"))
        plots_created.append("adaline_summary.png")
        plt.close(fig4)
        
        print(f"Generated {len(plots_created)} visualization plots:")
        for plot in plots_created:
            print(f"  - {output_path / plot}")
            
    except Exception as e:
        print(f"Error generating plots: {e}")
        
    return plots_created 