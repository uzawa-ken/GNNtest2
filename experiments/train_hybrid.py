#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Learning Training Script (Phase 5)

Demonstrates gradual transition from supervised to unsupervised learning
using curriculum scheduling and PUP-HAW-U framework.

Usage:
    python experiments/train_hybrid.py --schedule exponential --epochs 100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sage_model import SimpleSAGE
from losses.integrated_loss import PUPHAWHybridLoss
from utils.data_loader import find_time_list, load_case_with_csr
from utils.scheduling import CurriculumScheduler, get_recommended_schedule
from utils.mesh_analysis import extract_boundary_nodes, classify_boundary_types
from config.base_config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PUP-HAW-U Hybrid Training')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Data directory')
    parser.add_argument('--case_name', type=str, default='cavity',
                        help='Case name')

    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GNN layers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (for time steps)')

    # Curriculum scheduling
    parser.add_argument('--schedule', type=str, default='exponential',
                        choices=['linear', 'exponential', 'step', 'cosine', 'polynomial'],
                        help='Curriculum schedule type')
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='Problem difficulty (affects schedule)')
    parser.add_argument('--lambda_data_init', type=float, default=1.0,
                        help='Initial supervised weight')
    parser.add_argument('--lambda_data_final', type=float, default=0.0,
                        help='Final supervised weight')

    # Boundary conditions
    parser.add_argument('--bc_inlet', type=float, default=0.1,
                        help='Inlet BC value')
    parser.add_argument('--bc_outlet', type=float, default=0.0,
                        help='Outlet BC value')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/hybrid',
                        help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Model save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=1,
                        help='Logging frequency (epochs)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    return parser.parse_args()


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / 'checkpoints').mkdir(exist_ok=True)
    (output_path / 'logs').mkdir(exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)

    return output_path


def load_data(data_dir: str, case_name: str, device: str):
    """
    Load CFD data.

    Returns:
        data: PyG Data object
        A_csr: CSR matrix
        targets: Ground truth solutions
        time_list: List of time steps
    """
    data_path = Path(data_dir) / case_name

    # Find available time steps
    time_list = find_time_list(str(data_path))
    print(f"Found {len(time_list)} time steps: {time_list[:5]}...")

    # Load first time step as example
    feats, x_gt, A_csr = load_case_with_csr(str(data_path), time_list[0])

    # Convert to torch tensors
    feats = torch.from_numpy(feats).float().to(device)
    x_gt = torch.from_numpy(x_gt).float().to(device)

    # Build edge index from CSR (approximate)
    # For simplicity, we'll create edges from CSR non-zero entries
    num_nodes = feats.shape[0]
    edge_index = []

    row_ptr = A_csr['row_ptr']
    col_ind = A_csr['col_ind']

    for i in range(len(row_ptr) - 1):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        for j in range(start, end):
            col = col_ind[j]
            if i != col:  # Exclude self-loops
                edge_index.append([i, col])

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()

    # Create PyG Data object
    data = Data(x=feats, edge_index=edge_index)

    # Load all targets
    targets = []
    for t in time_list:
        _, x_t, _ = load_case_with_csr(str(data_path), t)
        targets.append(torch.from_numpy(x_t).float().to(device))

    return data, A_csr, targets, time_list


def initialize_model_and_loss(args, data, device):
    """Initialize model, loss function, and optimizer."""
    # Model
    in_channels = data.x.shape[1]
    model = SimpleSAGE(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    ).to(device)

    # Extract boundary information
    num_nodes = data.x.shape[0]
    boundary_mask = extract_boundary_nodes(data.edge_index, num_nodes, method='degree')
    boundary_types = classify_boundary_types(data.edge_index, boundary_mask)

    # Move boundary types to device
    for key in boundary_types:
        boundary_types[key] = boundary_types[key].to(device)

    # BC values
    bc_values = {
        'inlet': torch.tensor(args.bc_inlet, device=device),
        'outlet': torch.tensor(args.bc_outlet, device=device)
    }

    # Loss function (hybrid)
    loss_fn = PUPHAWHybridLoss(
        boundary_types=boundary_types,
        bc_values=bc_values,
        conservation_type='mass'
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return model, loss_fn, optimizer


def train_epoch(
    model,
    loss_fn,
    optimizer,
    data,
    A_csr,
    target,
    epoch,
    lambda_data,
    device
):
    """Train one epoch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred = model(data.x, data.edge_index).squeeze()

    # Compute hybrid loss
    loss, info = loss_fn(
        pred=pred,
        target=target,
        feats=data.x,
        A_csr=A_csr,
        b=torch.zeros_like(pred),  # Assuming homogeneous BC for PDE
        edge_index=data.edge_index,
        epoch=epoch,
        lambda_data=lambda_data
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item(), info


def validate(model, data, A_csr, target, device):
    """Validation step."""
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).squeeze()

        # Compute metrics
        mse = torch.mean((pred - target) ** 2).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        relative_error = (torch.norm(pred - target) / torch.norm(target)).item()

    return {'mse': mse, 'mae': mae, 'relative_error': relative_error}


def save_checkpoint(model, optimizer, epoch, output_dir: Path, filename: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, output_dir / 'checkpoints' / filename)


def save_training_history(history: List[Dict], output_dir: Path):
    """Save training history to JSON."""
    with open(output_dir / 'logs' / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history: List[Dict], output_dir: Path):
    """Plot training curves."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not available. Skipping plots.")
        return

    epochs = [h['epoch'] for h in history]

    # Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(epochs, [h['loss_total'] for h in history], linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Individual losses
    axes[0, 1].plot(epochs, [h['loss_data'] for h in history], label='Data', linewidth=2)
    axes[0, 1].plot(epochs, [h['loss_pde'] for h in history], label='PDE', linewidth=2)
    axes[0, 1].plot(epochs, [h['loss_bc'] for h in history], label='BC', linewidth=2)
    axes[0, 1].plot(epochs, [h['loss_conservation'] for h in history], label='Conservation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Individual Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Lambda_data (curriculum weight)
    axes[1, 0].plot(epochs, [h['lambda_data_curriculum'] for h in history], linewidth=2, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('λ_data (Supervised Weight)')
    axes[1, 0].set_title('Curriculum Schedule')
    axes[1, 0].grid(True, alpha=0.3)

    # Validation metrics
    axes[1, 1].plot(epochs, [h['val_mse'] for h in history], label='MSE', linewidth=2)
    axes[1, 1].plot(epochs, [h['val_relative_error'] for h in history], label='Relative Error', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {output_dir / 'plots' / 'training_curves.png'}")


def main():
    """Main training loop."""
    args = parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = setup_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print("Loading data...")
    data, A_csr, targets, time_list = load_data(args.data_dir, args.case_name, device)
    print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges")

    # Initialize model and loss
    print("Initializing model and loss...")
    model, loss_fn, optimizer = initialize_model_and_loss(args, data, device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Curriculum scheduler
    if args.difficulty != 'medium':
        # Use recommended schedule based on difficulty
        schedule_config = get_recommended_schedule(args.epochs, args.difficulty)
        scheduler = CurriculumScheduler(**schedule_config)
    else:
        # Use custom schedule
        scheduler = CurriculumScheduler(
            schedule_type=args.schedule,
            total_epochs=args.epochs,
            lambda_data_init=args.lambda_data_init,
            lambda_data_final=args.lambda_data_final
        )

    print(f"Curriculum schedule: {scheduler.get_schedule_name()}")

    # Plot schedule
    scheduler.plot_schedule(save_path=output_dir / 'plots' / 'curriculum_schedule.png')

    # Training loop
    print("\nStarting training...")
    history = []

    for epoch in range(args.epochs):
        # Get curriculum weight
        lambda_data = scheduler.get_lambda_data(epoch)

        # Use first target for training (can be extended to multiple time steps)
        target = targets[0]

        # Train one epoch
        train_loss, train_info = train_epoch(
            model, loss_fn, optimizer, data, A_csr, target,
            epoch, lambda_data, device
        )

        # Validation
        val_metrics = validate(model, data, A_csr, target, device)

        # Record history
        epoch_history = {
            'epoch': epoch,
            'loss_total': train_info['loss_total'],
            'loss_data': train_info['loss_data'],
            'loss_pde': train_info['loss_pde'],
            'loss_bc': train_info['loss_bc'],
            'loss_ic': train_info['loss_ic'],
            'loss_conservation': train_info['loss_conservation'],
            'lambda_data_curriculum': lambda_data,
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_relative_error': val_metrics['relative_error']
        }
        history.append(epoch_history)

        # Logging
        if (epoch + 1) % args.log_freq == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {train_loss:.6f} | "
                  f"λ_data: {lambda_data:.4f} | "
                  f"Val MSE: {val_metrics['mse']:.6f} | "
                  f"Rel Err: {val_metrics['relative_error']:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, output_dir, f'checkpoint_epoch_{epoch+1}.pt')

    # Final save
    save_checkpoint(model, optimizer, args.epochs - 1, output_dir, 'checkpoint_final.pt')
    save_training_history(history, output_dir)
    plot_training_curves(history, output_dir)

    print("\nTraining completed!")
    print(f"Final validation MSE: {history[-1]['val_mse']:.6f}")
    print(f"Final relative error: {history[-1]['val_relative_error']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
