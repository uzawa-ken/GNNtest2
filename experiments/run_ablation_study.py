#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation Study Framework (Phase 7)

Systematic evaluation of component contributions in PUP-HAW-U.

Tests the impact of removing each component:
- Physics-based uncertainty propagation (Phase 2)
- Hierarchical adaptive weighting (Phase 3)
- Topology-aware propagation (Phase 2)
- Multi-physics constraints (Phase 4)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import time

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gnn_pde_solver import PUPHAWUnsupervised
from utils.data_loader import find_time_list, load_case_with_csr
from utils.mesh_analysis import extract_boundary_nodes, classify_boundary_types
from utils.evaluation import evaluate_model_comprehensive, BaselineComparator
from torch_geometric.data import Data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PUP-HAW-U Ablation Study')

    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--case_name', type=str, default='cavity')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='./outputs/ablation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--runs_per_config', type=int, default=3,
                        help='Number of runs per configuration for robustness')

    return parser.parse_args()


def create_ablation_configs():
    """
    Create configurations for ablation study.

    Returns:
        configs: Dictionary of configurations
    """
    configs = {
        'full': {
            'name': 'PUP-HAW-U (Full)',
            'description': 'Complete model with all components',
            'mesh_quality_config': {
                'use_physics_based': True,
                'learnable': True
            },
            'hierarchical_config': {
                'use_hierarchical': True,
                'use_level1': True,
                'use_level2': True
            },
            'use_topology_propagation': True,
            'use_multi_physics': True
        },

        'no_physics_weights': {
            'name': 'w/o Physics-based Weights',
            'description': 'Without Phase 2 physics-based uncertainty propagation',
            'mesh_quality_config': {
                'use_physics_based': False,  # Use baseline linear weights
                'learnable': False
            },
            'hierarchical_config': {
                'use_hierarchical': True,
                'use_level1': True,
                'use_level2': True
            },
            'use_topology_propagation': True,
            'use_multi_physics': True
        },

        'no_hierarchical': {
            'name': 'w/o Hierarchical Adaptive',
            'description': 'Without Phase 3 hierarchical adaptive weighting',
            'mesh_quality_config': {
                'use_physics_based': True,
                'learnable': True
            },
            'hierarchical_config': {
                'use_hierarchical': False,  # Fixed weights
                'use_level1': False,
                'use_level2': False
            },
            'use_topology_propagation': True,
            'use_multi_physics': True
        },

        'no_topology': {
            'name': 'w/o Topology Propagation',
            'description': 'Without topology-aware weight propagation',
            'mesh_quality_config': {
                'use_physics_based': True,
                'learnable': True
            },
            'hierarchical_config': {
                'use_hierarchical': True,
                'use_level1': True,
                'use_level2': True
            },
            'use_topology_propagation': False,  # Local weights only
            'use_multi_physics': True
        },

        'no_multi_physics': {
            'name': 'w/o Multi-physics Constraints',
            'description': 'PDE loss only (no BC, IC, conservation)',
            'mesh_quality_config': {
                'use_physics_based': True,
                'learnable': True
            },
            'hierarchical_config': {
                'use_hierarchical': True,
                'use_level1': True,
                'use_level2': True
            },
            'use_topology_propagation': True,
            'use_multi_physics': False  # PDE only
        },

        'baseline': {
            'name': 'Baseline (Fixed Weights)',
            'description': 'Simple fixed weight baseline',
            'mesh_quality_config': {
                'use_physics_based': False,
                'learnable': False
            },
            'hierarchical_config': {
                'use_hierarchical': False,
                'use_level1': False,
                'use_level2': False
            },
            'use_topology_propagation': False,
            'use_multi_physics': False
        }
    }

    return configs


def load_data(data_dir, case_name, device):
    """Load data for ablation study."""
    data_path = Path(data_dir) / case_name
    time_list = find_time_list(str(data_path))

    feats, x_gt, A_csr = load_case_with_csr(str(data_path), time_list[0])

    feats = torch.from_numpy(feats).float().to(device)
    reference = torch.from_numpy(x_gt).float().to(device)

    # Build edge index
    num_nodes = feats.shape[0]
    edge_index = []

    row_ptr = A_csr['row_ptr']
    col_ind = A_csr['col_ind']

    for i in range(len(row_ptr) - 1):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        for j in range(start, end):
            col = col_ind[j]
            if i != col:
                edge_index.append([i, col])

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
    data = Data(x=feats, edge_index=edge_index)

    # Boundary info
    boundary_mask = extract_boundary_nodes(data.edge_index, num_nodes, method='degree')
    boundary_types = classify_boundary_types(data.edge_index, boundary_mask)
    for key in boundary_types:
        boundary_types[key] = boundary_types[key].to(device)

    bc_values = {
        'inlet': torch.tensor(0.1, device=device),
        'outlet': torch.tensor(0.0, device=device)
    }

    return data, A_csr, reference, boundary_types, bc_values


def train_model(model, loss_fn, optimizer, data, A_csr, epochs, device):
    """Train model for ablation study."""
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(data.x, data.edge_index).squeeze()

        loss, _ = loss_fn(
            pred=pred,
            feats=data.x,
            A_csr=A_csr,
            b=torch.zeros_like(pred),
            edge_index=data.edge_index,
            epoch=epoch
        )

        loss.backward()
        optimizer.step()

    return model


def run_ablation_experiment(
    config_name: str,
    config: Dict,
    data,
    A_csr,
    reference,
    boundary_types,
    bc_values,
    epochs: int,
    lr: float,
    device: str,
    run_idx: int = 0
):
    """
    Run single ablation experiment.

    Args:
        config_name: Configuration name
        config: Configuration dictionary
        data: Data object
        A_csr: System matrix
        reference: Ground truth
        boundary_types: Boundary information
        bc_values: BC values
        epochs: Training epochs
        lr: Learning rate
        device: Device
        run_idx: Run index for multiple runs

    Returns:
        results: Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Running: {config['name']} (Run {run_idx + 1})")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    # Create model based on config
    in_channels = data.x.shape[1]

    # Adjust configs based on ablation settings
    mesh_quality_config = config.get('mesh_quality_config', {})
    hierarchical_config = config.get('hierarchical_config', {})
    use_topology = config.get('use_topology_propagation', True)

    # For multi-physics ablation, modify boundary types
    if not config.get('use_multi_physics', True):
        # Empty boundaries = no BC/conservation losses
        boundary_types_ablation = {k: torch.zeros_like(v) for k, v in boundary_types.items()}
    else:
        boundary_types_ablation = boundary_types

    model = PUPHAWUnsupervised(
        in_channels=in_channels,
        boundary_types=boundary_types_ablation,
        bc_values=bc_values if config.get('use_multi_physics', True) else None,
        conservation_type='mass',
        mesh_quality_config=mesh_quality_config,
        hierarchical_config=hierarchical_config,
        use_topology_propagation=use_topology
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    start_time = time.time()
    model = train_model(model, model.loss_fn, optimizer, data, A_csr, epochs, device)
    training_time = time.time() - start_time

    # Evaluation
    metrics = evaluate_model_comprehensive(
        model.gnn,
        data,
        A_csr,
        boundary_types,
        bc_values,
        reference=reference,
        device=device
    )

    metrics['training_time'] = training_time
    metrics['config_name'] = config_name
    metrics['run_idx'] = run_idx

    print(f"\nResults:")
    print(f"  PDE Residual (L2): {metrics['pde_residual_l2']:.6e}")
    print(f"  BC Error (MAE): {metrics.get('bc_total_mae', 0):.6e}")
    print(f"  Conservation (L2): {metrics['conservation_l2']:.6e}")
    if 'mse' in metrics:
        print(f"  MSE vs Reference: {metrics['mse']:.6e}")
        print(f"  Relative Error: {metrics['relative_error']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")

    return metrics, model


def aggregate_results(results: List[Dict]) -> Dict:
    """
    Aggregate results from multiple runs.

    Args:
        results: List of result dictionaries

    Returns:
        aggregated: Aggregated statistics (mean, std)
    """
    import numpy as np

    # Get all metric names
    metric_names = set()
    for r in results:
        metric_names.update(r.keys())

    metric_names.discard('config_name')
    metric_names.discard('run_idx')

    aggregated = {}
    for metric_name in metric_names:
        values = [r[metric_name] for r in results if metric_name in r]
        if len(values) > 0:
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_min'] = np.min(values)
            aggregated[f'{metric_name}_max'] = np.max(values)

    return aggregated


def main():
    """Main ablation study."""
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("PUP-HAW-U ABLATION STUDY")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Runs per config: {args.runs_per_config}")
    print()

    # Load data
    print("Loading data...")
    data, A_csr, reference, boundary_types, bc_values = load_data(
        args.data_dir, args.case_name, device
    )
    print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges\n")

    # Get ablation configs
    configs = create_ablation_configs()

    # Run all experiments
    all_results = {}
    all_models = {}

    for config_name, config in configs.items():
        run_results = []

        for run_idx in range(args.runs_per_config):
            metrics, model = run_ablation_experiment(
                config_name,
                config,
                data,
                A_csr,
                reference,
                boundary_types,
                bc_values,
                args.epochs,
                args.lr,
                device,
                run_idx
            )

            run_results.append(metrics)

            if run_idx == 0:  # Save first run's model
                all_models[config_name] = model

        # Aggregate results
        aggregated = aggregate_results(run_results)
        all_results[config_name] = {
            'config': config,
            'runs': run_results,
            'aggregated': aggregated
        }

        # Save individual config results
        with open(output_dir / f'{config_name}_results.json', 'w') as f:
            json.dump(all_results[config_name], f, indent=2, default=str)

    # Save complete results
    with open(output_dir / 'ablation_complete.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate comparison report
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)

    comparator = BaselineComparator(str(output_dir))

    for config_name, result_data in all_results.items():
        aggregated = result_data['aggregated']

        # Convert aggregated means to metrics dict
        metrics = {k.replace('_mean', ''): v for k, v in aggregated.items() if k.endswith('_mean')}

        comparator.add_method(
            config_name,
            all_models[config_name].gnn,
            metrics
        )

    comparator.generate_report()

    print("\nAblation study complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
