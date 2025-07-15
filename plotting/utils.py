"""
Plotting utilities and common functions for AI From Scratch to Scale project.

This module provides consistent styling, figure management, and utility functions
used across all plotting modules in the project.
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Project color palette
PROJECT_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple  
    'accent': '#F18F01',       # Orange
    'success': '#1B998B',      # Teal
    'warning': '#FFBC42',      # Yellow
    'error': '#E74C3C',        # Red
    'neutral': '#7D8597',      # Gray
    'background': '#F8F9FA',   # Light gray
    'text': '#2C3E50'          # Dark blue-gray
}

# Model-specific colors for consistency
MODEL_COLORS = {
    'perceptron': PROJECT_COLORS['primary'],
    'mlp': PROJECT_COLORS['secondary'], 
    'cnn': PROJECT_COLORS['accent'],
    'rnn': PROJECT_COLORS['success'],
    'transformer': PROJECT_COLORS['warning'],
    'default': PROJECT_COLORS['neutral']
}

# Plot style configuration
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PROJECT_COLORS['neutral'],
    'axes.linewidth': 0.8,
    'xtick.color': PROJECT_COLORS['text'],
    'ytick.color': PROJECT_COLORS['text'],
    'text.color': PROJECT_COLORS['text']
}


def setup_plotting_style():
    """Setup consistent plotting style for all figures."""
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams.update(PLOT_STYLE)
    
    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.color': PROJECT_COLORS['neutral'],
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Set color palette
    sns.set_palette([
        PROJECT_COLORS['primary'],
        PROJECT_COLORS['secondary'], 
        PROJECT_COLORS['accent'],
        PROJECT_COLORS['success'],
        PROJECT_COLORS['warning'],
        PROJECT_COLORS['error']
    ])


def get_model_color(model_name: str) -> str:
    """Get consistent color for a model type."""
    model_key = model_name.lower()
    for key in MODEL_COLORS:
        if key in model_key:
            return MODEL_COLORS[key]
    return MODEL_COLORS['default']


def create_subplots(nrows: int = 1, 
                   ncols: int = 1, 
                   figsize: Optional[Tuple[float, float]] = None,
                   **kwargs) -> Tuple[plt.Figure, Any]:
    """
    Create subplots with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns  
        figsize: Figure size override
        **kwargs: Additional arguments for plt.subplots()
        
    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        base_width, base_height = PLOT_STYLE['figure.figsize']
        figsize = (base_width * ncols, base_height * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # Apply consistent styling
    if nrows * ncols == 1:
        axes = [axes]  # Make it a list for consistent handling
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for ax in axes:
        ax.grid(True, alpha=0.3, color=PROJECT_COLORS['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(PROJECT_COLORS['neutral'])
        ax.spines['bottom'].set_color(PROJECT_COLORS['neutral'])
    
    return fig, axes[0] if len(axes) == 1 else axes


def save_figure(fig: plt.Figure, 
               filepath: str,
               dpi: int = 300,
               bbox_inches: str = 'tight',
               transparent: bool = False,
               **kwargs):
    """
    Save figure with consistent settings.
    
    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: Resolution for raster formats
        bbox_inches: Bounding box behavior
        transparent: Whether to use transparent background
        **kwargs: Additional arguments for fig.savefig()
    """
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with high quality settings
    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        facecolor='white' if not transparent else 'none',
        edgecolor='none',
        **kwargs
    )


def add_watermark(ax: plt.Axes, 
                 text: str = "AI From Scratch to Scale",
                 alpha: float = 0.1,
                 fontsize: int = 8):
    """Add subtle watermark to plots."""
    ax.text(0.02, 0.98, text, 
            transform=ax.transAxes,
            fontsize=fontsize,
            alpha=alpha,
            verticalalignment='top',
            horizontalalignment='left',
            color=PROJECT_COLORS['neutral'])


def format_metric_name(metric: str) -> str:
    """Format metric names for display."""
    replacements = {
        'loss': 'Loss',
        'accuracy': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'learning_rate': 'Learning Rate',
        'train_': 'Training ',
        'val_': 'Validation ',
        'test_': 'Test ',
        '_': ' '
    }
    
    formatted = metric
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted.title()


def create_legend(ax: plt.Axes, 
                 labels: List[str],
                 colors: Optional[List[str]] = None,
                 location: str = 'best',
                 **kwargs):
    """Create consistent legend styling."""
    if colors:
        # Create custom legend with specified colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label) 
                          for label, color in zip(labels, colors)]
        ax.legend(handles=legend_elements, loc=location, **kwargs)
    else:
        ax.legend(labels, loc=location, **kwargs)


def set_axis_labels(ax: plt.Axes, 
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   title: Optional[str] = None):
    """Set axis labels with consistent formatting."""
    if xlabel:
        ax.set_xlabel(format_metric_name(xlabel))
    if ylabel:
        ax.set_ylabel(format_metric_name(ylabel))
    if title:
        ax.set_title(title, fontweight='bold', pad=20)


def create_color_map(values: List[str]) -> Dict[str, str]:
    """Create consistent color mapping for categorical values."""
    colors = [
        PROJECT_COLORS['primary'],
        PROJECT_COLORS['secondary'],
        PROJECT_COLORS['accent'], 
        PROJECT_COLORS['success'],
        PROJECT_COLORS['warning'],
        PROJECT_COLORS['error']
    ]
    
    # Extend colors if needed
    while len(colors) < len(values):
        colors.extend(colors)
    
    return {value: colors[i % len(colors)] for i, value in enumerate(values)}


def apply_theme(dark_mode: bool = False):
    """Apply light or dark theme."""
    if dark_mode:
        # Dark theme
        dark_colors = {
            'background': '#1E1E1E',
            'text': '#FFFFFF', 
            'neutral': '#888888'
        }
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': dark_colors['background'],
            'axes.facecolor': dark_colors['background'],
            'text.color': dark_colors['text'],
            'axes.labelcolor': dark_colors['text'],
            'xtick.color': dark_colors['text'],
            'ytick.color': dark_colors['text'],
            'grid.color': dark_colors['neutral']
        })
    else:
        # Light theme (default)
        setup_plotting_style()


# Commonly used plot configurations
TRAINING_PLOT_CONFIG = {
    'loss': {
        'color': PROJECT_COLORS['error'],
        'label': 'Training Loss',
        'linestyle': '-'
    },
    'val_loss': {
        'color': PROJECT_COLORS['error'], 
        'label': 'Validation Loss',
        'linestyle': '--'
    },
    'accuracy': {
        'color': PROJECT_COLORS['success'],
        'label': 'Training Accuracy',
        'linestyle': '-'
    },
    'val_accuracy': {
        'color': PROJECT_COLORS['success'],
        'label': 'Validation Accuracy', 
        'linestyle': '--'
    }
}

CONFUSION_MATRIX_CONFIG = {
    'cmap': 'Blues',
    'annot': True,
    'fmt': 'd',
    'cbar': True,
    'square': True
} 