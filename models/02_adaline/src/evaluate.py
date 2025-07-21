"""
ADALINE Evaluation Script.

Evaluation script for ADALINE model with comparison features.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import logging
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_experiment_config
from model import create_adaline
from data_loader import load_adaline_eval_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ADALINE model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="debug_small",
        help="Experiment name for data generation"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()





def evaluate_model(model, x_data, y_data):
    """Evaluate ADALINE model."""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        linear_output = model.forward(x_data)
        predictions = model.predict(x_data)
        
        # Calculate metrics
        mse = torch.mean((y_data - linear_output) ** 2)
        accuracy = torch.mean((predictions == y_data).float())
        
        # Calculate confusion matrix
        tp = torch.sum((predictions == 1) & (y_data == 1))
        tn = torch.sum((predictions == 0) & (y_data == 0))
        fp = torch.sum((predictions == 1) & (y_data == 0))
        fn = torch.sum((predictions == 0) & (y_data == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "mse": mse.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "confusion_matrix": {
            "tp": tp.item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item()
        }
    }


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*50)
    print("ADALINE EVALUATION RESULTS")
    print("="*50)
    
    print(f"Mean Squared Error: {results['mse']:.6f}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  True Positives: {cm['tp']}")
    print(f"  True Negatives: {cm['tn']}")
    print(f"  False Positives: {cm['fp']}")
    print(f"  False Negatives: {cm['fn']}")
    
    print("="*50)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    try:
        config = get_experiment_config(args.experiment)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create model
    model = create_adaline(config)
    
    # Load checkpoint
    try:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return
        
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Load evaluation data
    try:
        x_data, y_data = load_adaline_eval_data(config.dataset)
        print(f"Loaded {len(x_data)} evaluation samples for {config.dataset}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    # Evaluate model
    try:
        results = evaluate_model(model, x_data, y_data)
        print_results(results)
        
        # Generate visualizations if requested
        if args.visualize:
            print("Visualization not implemented yet")
            # TODO: Implement visualization using shared plotting package
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.debug:
            raise


if __name__ == "__main__":
    main() 