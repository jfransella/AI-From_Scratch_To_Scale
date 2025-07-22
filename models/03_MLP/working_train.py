#!/usr/bin/env python3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb')
    parser.add_argument('--wandb-name', type=str, help='Wandb run name')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f'Arguments: {args}')
