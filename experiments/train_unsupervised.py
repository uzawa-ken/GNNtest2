#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fully Unsupervised Learning Training Script (Phase 6)

Trains GNN for PDE solving using only physics constraints (no supervised data).
Uses PUP-HAW-U framework with hierarchical adaptive weighting.

Usage:
    python experiments/train_unsupervised.py --epochs 200 --bc_inlet 0.1
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
from losses.integrated_loss import PUPHAWUnsupervisedLoss
from utils.data_loader import find_time_list, load_case_with_csr
from utils.mesh_analysis import extract_boundary_nodes, classify_boundary_types
from utils.monitoring import TrainingMonitor
from config.base_config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PUP-HAW-U Unsupervised Training')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Data directory')
    parser.add_argument('--case_name', type=str, default='cavity',
                        help='Case name')
    parser.add_argument('--use_reference', action='store_true',
                        help='Use ground truth for validation only (not for training)')

    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GNN layers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')

    # Physics constraints
    parser.add_argument('--lambda_pde_init', type=float, default=1.0,
                        help='Initial PDE residual weight')
    parser.add_argument('--lambda_bc_init', type=float, default=1.0,
                        help='Initial BC weight')
    parser.add_argument('--lambda_cons_init', type=float, default=0.1,
                        help='Initial conservation weight')

    # Boundary conditions
    parser.add_argument('--bc_inlet', type=float, default=0.1,
                        help='Inlet BC value')
    parser.add_argument('--bc_outlet', type=float, default=0.0,
                        help='Outlet BC value')
    parser.add_argument('--conservation_type', type=str, default='mass',
                        choices=['mass', 'momentum', 'energy'],
                        help='Conservation law type')

    # Mesh quality weighting
    parser.add_argument('--use_learnable_weights', action='store_true',
                        help='Use learnable mesh quality parameters')
    parser.add_argument('--use_topology_propagation', action='store_true', default=True,
                        help='Use topology-aware weight propagation')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/unsupervised',
                        help='Output directory')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Model save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=5,
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


def load_data(data_dir: str, case_name: str, device: str, use_reference: bool = False):
    """
    Load CFD data.

    Returns:
        data: PyG Data object
        A_csr: CSR matrix
        reference: Ground truth solution (if use_reference=True)
        time_list: List of time steps
    """
    data_path = Path(data_dir) / case_name

    # Find available time steps
    time_list = find_time_list(str(data_path))
    print(f"Found {len(time_list)} time steps: {time_list[:5]}...")

    # Load first time step
    feats, x_gt, A_csr = load_case_with_csr(str(data_path), time_list[0])

    # Convert to torch tensors
    feats = torch.from_numpy(feats).float().to(device)

    if use_reference:
        reference = torch.from_numpy(x_gt).float().to(device)
    else:
        reference = None

    # Build edge index from CSR
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

    return data, A_csr, reference, time_list


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

    # Mesh quality config
    mesh_quality_config = {}
    if args.use_learnable_weights:
        from losses.physics_based_weights import LearnableMeshQualityWeight
        # Use learnable version (Phase 2)
        mesh_quality_config = {'learnable': True}

    # Hierarchical adaptive config
    hierarchical_config = {
        'constraint_types': ['pde', 'bc', 'ic', 'conservation'],
        'lambda_init': {
            'pde': args.lambda_pde_init,
            'bc': args.lambda_bc_init,
            'ic': 1.0,  # Not used for steady-state
            'conservation': args.lambda_cons_init
        }
    }

    # Loss function (fully unsupervised)
    loss_fn = PUPHAWUnsupervisedLoss(
        mesh_quality_config=mesh_quality_config,
        hierarchical_config=hierarchical_config,
        boundary_types=boundary_types,
        bc_values=bc_values,
        conservation_type=args.conservation_type,
        use_topology_propagation=args.use_topology_propagation
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, loss_fn, optimizer


def train_epoch(
    model,
    loss_fn,
    optimizer,
    data,
    A_csr,
    epoch,
    device
):
    """Train one epoch (fully unsupervised)."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred = model(data.x, data.edge_index).squeeze()

    # Compute unsupervised loss (no ground truth used)
    loss, info = loss_fn(
        pred=pred,
        feats=data.x,
        A_csr=A_csr,
        b=torch.zeros_like(pred),  # Assuming homogeneous BC for PDE
        edge_index=data.edge_index,
        epoch=epoch
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item(), info


def evaluate_physics_constraints(
    model,
    data,
    A_csr,
    loss_fn,
    device
):
    """
    Evaluate physics constraint satisfaction.

    Returns metrics for PDE residual, BC error, and conservation.
    """
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).squeeze()

        # Compute all physics losses
        _, info = loss_fn(
            pred=pred,
            feats=data.x,
            A_csr=A_csr,
            b=torch.zeros_like(pred),
            edge_index=data.edge_index,
            epoch=0  # Doesn't matter for evaluation
        )

        metrics = {
            'pde_residual': info['loss_pde'],
            'bc_error': info['loss_bc'],
            'conservation_error': info['loss_conservation'],
            'total_physics_loss': info['loss_total']
        }

        # Add individual BC errors if available
        for key in info:
            if key.startswith('bc_'):
                metrics[key] = info[key]

    return metrics


def evaluate_with_reference(model, data, reference, device):
    """
    Evaluate against ground truth (for validation only, not training).

    Returns metrics comparing prediction to reference solution.
    """
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).squeeze()

        mse = torch.mean((pred - reference) ** 2).item()
        mae = torch.mean(torch.abs(pred - reference)).item()
        relative_error = (torch.norm(pred - reference) / torch.norm(reference)).item()
        max_error = torch.max(torch.abs(pred - reference)).item()

    return {
        'mse': mse,
        'mae': mae,
        'relative_error': relative_error,
        'max_error': max_error
    }


def save_checkpoint(model, optimizer, epoch, output_dir: Path, filename: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, output_dir / 'checkpoints' / filename)


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
    data, A_csr, reference, time_list = load_data(
        args.data_dir, args.case_name, device, args.use_reference
    )
    print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges")

    if reference is not None:
        print("Reference solution loaded (for validation only)")
    else:
        print("Training in fully unsupervised mode (no reference)")

    # Initialize model and loss
    print("Initializing model and loss...")
    model, loss_fn, optimizer = initialize_model_and_loss(args, data, device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Monitor
    monitor = TrainingMonitor(output_dir, log_interval=args.log_freq)

    # Training loop
    print("\nStarting unsupervised training...")
    print("=" * 60)
    print("Training with physics constraints only:")
    print("  - PDE residual minimization")
    print("  - Boundary condition enforcement")
    print("  - Conservation law satisfaction")
    print("  - Hierarchical adaptive weighting (automatic)")
    print("=" * 60 + "\n")

    for epoch in range(args.epochs):
        # Train one epoch (unsupervised)
        train_loss, train_info = train_epoch(
            model, loss_fn, optimizer, data, A_csr, epoch, device
        )

        # Evaluate physics constraints
        physics_metrics = evaluate_physics_constraints(
            model, data, A_csr, loss_fn, device
        )

        # Evaluate against reference (if available)
        val_metrics = {}
        if reference is not None:
            ref_metrics = evaluate_with_reference(model, data, reference, device)
            val_metrics.update(ref_metrics)

        # Combine all metrics
        all_metrics = {**physics_metrics, **val_metrics}

        # Log
        monitor.log(epoch, train_info, lambda_data=0.0, val_metrics=all_metrics)

        # Detailed logging
        if (epoch + 1) % args.log_freq == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Total loss: {train_loss:.6f}")
            print(f"  PDE residual: {physics_metrics['pde_residual']:.6f}")
            print(f"  BC error: {physics_metrics['bc_error']:.6f}")
            print(f"  Conservation: {physics_metrics['conservation_error']:.6f}")

            if reference is not None:
                print(f"  [Ref] MSE: {val_metrics['mse']:.6f}")
                print(f"  [Ref] Relative error: {val_metrics['relative_error']:.4f}")

            # Show adaptive weights
            if 'lambdas' in train_info:
                lambdas = train_info['lambdas']
                print(f"  Adaptive weights: λ_pde={lambdas.get('pde', 0):.3f}, "
                      f"λ_bc={lambdas.get('bc', 0):.3f}, "
                      f"λ_cons={lambdas.get('conservation', 0):.3f}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, output_dir, f'checkpoint_epoch_{epoch+1}.pt')

    # Final save
    save_checkpoint(model, optimizer, args.epochs - 1, output_dir, 'checkpoint_final.pt')
    monitor.save()
    monitor.plot_all()
    monitor.print_summary()

    print("\nUnsupervised training completed!")
    print(f"Results saved to {output_dir}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL PHYSICS CONSTRAINT SATISFACTION")
    print("=" * 60)
    final_physics = evaluate_physics_constraints(model, data, A_csr, loss_fn, device)
    print(f"PDE residual:      {final_physics['pde_residual']:.6e}")
    print(f"BC error:          {final_physics['bc_error']:.6e}")
    print(f"Conservation error: {final_physics['conservation_error']:.6e}")

    if reference is not None:
        print("\n" + "=" * 60)
        print("REFERENCE COMPARISON (Validation Only)")
        print("=" * 60)
        final_ref = evaluate_with_reference(model, data, reference, device)
        print(f"MSE:            {final_ref['mse']:.6e}")
        print(f"MAE:            {final_ref['mae']:.6e}")
        print(f"Relative error: {final_ref['relative_error']:.4f}")
        print(f"Max error:      {final_ref['max_error']:.6e}")

    print("=" * 60)


if __name__ == '__main__':
    main()
