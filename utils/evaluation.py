"""
Evaluation Utilities for PDE Solving with GNNs

Provides comprehensive metrics for evaluating physics-informed learning,
including PDE residual, boundary condition satisfaction, and conservation laws.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class PhysicsEvaluator:
    """
    Evaluator for physics constraint satisfaction.

    Computes detailed metrics for PDE solving without requiring ground truth.

    Parameters
    ----------
    A_csr : dict
        System matrix in CSR format
    boundary_types : dict
        Boundary condition types and masks
    bc_values : dict, optional
        Expected boundary condition values

    Examples
    --------
    >>> evaluator = PhysicsEvaluator(A_csr, boundary_types, bc_values)
    >>> metrics = evaluator.evaluate(pred, feats, edge_index)
    >>> print(f"PDE residual: {metrics['pde_residual']:.6e}")
    """

    def __init__(
        self,
        A_csr: Dict,
        boundary_types: Dict[str, torch.Tensor],
        bc_values: Optional[Dict[str, float]] = None
    ):
        self.A_csr = A_csr
        self.boundary_types = boundary_types
        self.bc_values = bc_values or {}

    def evaluate(
        self,
        pred: torch.Tensor,
        feats: torch.Tensor,
        edge_index: torch.Tensor,
        b: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate physics constraint satisfaction.

        Args:
            pred: Predicted solution [num_nodes]
            feats: Node features [num_nodes, num_feats]
            edge_index: Edge connectivity [2, num_edges]
            b: RHS vector [num_nodes] (optional, defaults to zero)

        Returns:
            metrics: Dictionary with physics metrics
        """
        device = pred.device
        if b is None:
            b = torch.zeros_like(pred)

        metrics = {}

        # 1. PDE Residual
        pde_metrics = self.compute_pde_residual(pred, b)
        metrics.update(pde_metrics)

        # 2. Boundary Condition Errors
        bc_metrics = self.compute_bc_errors(pred)
        metrics.update(bc_metrics)

        # 3. Conservation Law Satisfaction
        cons_metrics = self.compute_conservation(pred, edge_index)
        metrics.update(cons_metrics)

        # 4. Solution Quality Metrics
        quality_metrics = self.compute_solution_quality(pred)
        metrics.update(quality_metrics)

        return metrics

    def compute_pde_residual(
        self,
        pred: torch.Tensor,
        b: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute PDE residual: ||Ax - b||.

        Args:
            pred: Predicted solution [num_nodes]
            b: RHS vector [num_nodes]

        Returns:
            metrics: PDE residual metrics
        """
        from utils.sparse_ops import matvec_csr_torch

        # Compute Ax
        Ax = matvec_csr_torch(
            self.A_csr['row_ptr'],
            self.A_csr['col_ind'],
            self.A_csr['vals'],
            self.A_csr['row_idx'],
            pred
        )

        # Residual: r = Ax - b
        residual = Ax - b

        metrics = {
            'pde_residual_l2': torch.norm(residual).item(),
            'pde_residual_linf': torch.max(torch.abs(residual)).item(),
            'pde_residual_mean': torch.mean(torch.abs(residual)).item(),
            'pde_residual_std': torch.std(residual).item(),
            'pde_residual_normalized': (torch.norm(residual) / (torch.norm(b) + 1e-10)).item()
        }

        return metrics

    def compute_bc_errors(self, pred: torch.Tensor) -> Dict[str, float]:
        """
        Compute boundary condition errors.

        Args:
            pred: Predicted solution [num_nodes]

        Returns:
            metrics: BC error metrics
        """
        metrics = {}

        for bc_type, mask in self.boundary_types.items():
            if mask.sum() == 0:
                continue  # Skip empty boundaries

            pred_bc = pred[mask]

            # Get expected BC value
            if bc_type in self.bc_values:
                bc_target = torch.tensor(
                    self.bc_values[bc_type],
                    device=pred.device,
                    dtype=pred.dtype
                )
            else:
                # Default values
                if bc_type == 'outlet':
                    bc_target = torch.tensor(0.0, device=pred.device)
                elif bc_type == 'inlet':
                    bc_target = torch.tensor(0.1, device=pred.device)
                else:
                    continue  # Skip if no target

            # Compute errors
            error = pred_bc - bc_target
            metrics[f'bc_{bc_type}_mae'] = torch.mean(torch.abs(error)).item()
            metrics[f'bc_{bc_type}_mse'] = torch.mean(error ** 2).item()
            metrics[f'bc_{bc_type}_max'] = torch.max(torch.abs(error)).item()

        # Overall BC error
        if len(metrics) > 0:
            mae_values = [v for k, v in metrics.items() if 'mae' in k]
            metrics['bc_total_mae'] = np.mean(mae_values)

        return metrics

    def compute_conservation(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute conservation law satisfaction (e.g., mass conservation).

        For incompressible flow: ∇·u ≈ 0
        Approximated by: ∇²p ≈ 0 (Laplacian)

        Args:
            pred: Predicted field [num_nodes]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            metrics: Conservation metrics
        """
        from utils.graph_ops import compute_graph_laplacian

        # Compute Laplacian (approximates divergence for pressure)
        laplacian = compute_graph_laplacian(pred, edge_index, normalized=False)

        metrics = {
            'conservation_l2': torch.norm(laplacian).item(),
            'conservation_linf': torch.max(torch.abs(laplacian)).item(),
            'conservation_mean': torch.mean(torch.abs(laplacian)).item(),
            'conservation_std': torch.std(laplacian).item()
        }

        return metrics

    def compute_solution_quality(self, pred: torch.Tensor) -> Dict[str, float]:
        """
        Compute solution quality metrics (smoothness, range, etc.).

        Args:
            pred: Predicted solution [num_nodes]

        Returns:
            metrics: Quality metrics
        """
        metrics = {
            'solution_mean': torch.mean(pred).item(),
            'solution_std': torch.std(pred).item(),
            'solution_min': torch.min(pred).item(),
            'solution_max': torch.max(pred).item(),
            'solution_range': (torch.max(pred) - torch.min(pred)).item()
        }

        return metrics


class BaselineComparator:
    """
    Compare different training methods.

    Supports comparison between:
    - Supervised learning
    - Fixed-weight unsupervised learning
    - PUP-HAW-U (adaptive weighting)

    Examples
    --------
    >>> comparator = BaselineComparator(output_dir='./comparison')
    >>> comparator.add_method('supervised', model_sup, metrics_sup)
    >>> comparator.add_method('pup-haw-u', model_pup, metrics_pup)
    >>> comparator.generate_report()
    """

    def __init__(self, output_dir: str = './comparison'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.methods = {}
        self.metrics = {}

    def add_method(
        self,
        name: str,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        training_history: Optional[List[Dict]] = None
    ):
        """
        Add a method for comparison.

        Args:
            name: Method name
            model: Trained model
            metrics: Final evaluation metrics
            training_history: Training history (optional)
        """
        self.methods[name] = model
        self.metrics[name] = metrics

        if training_history is not None:
            # Save training history
            with open(self.output_dir / f'{name}_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)

    def compare_metrics(
        self,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare specific metrics across methods.

        Args:
            metric_names: List of metric names to compare (optional)

        Returns:
            comparison: Dictionary with comparison results
        """
        if metric_names is None:
            # Use all available metrics
            all_metrics = set()
            for metrics in self.metrics.values():
                all_metrics.update(metrics.keys())
            metric_names = sorted(all_metrics)

        comparison = {}
        for metric_name in metric_names:
            comparison[metric_name] = {}
            for method_name, metrics in self.metrics.items():
                if metric_name in metrics:
                    comparison[metric_name][method_name] = metrics[metric_name]

        return comparison

    def generate_report(self):
        """Generate comprehensive comparison report."""
        report_path = self.output_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BASELINE COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Methods compared: {len(self.methods)}\n")
            f.write(f"  - {', '.join(self.methods.keys())}\n\n")

            # Compare key metrics
            key_metrics = [
                'pde_residual_l2',
                'bc_total_mae',
                'conservation_l2',
                'mse',
                'relative_error'
            ]

            f.write("Key Metrics Comparison:\n")
            f.write("-" * 80 + "\n")

            comparison = self.compare_metrics(key_metrics)

            for metric_name, values in comparison.items():
                if len(values) == 0:
                    continue

                f.write(f"\n{metric_name}:\n")
                for method_name, value in values.items():
                    f.write(f"  {method_name:20s}: {value:.6e}\n")

                # Find best method (lower is better for most metrics)
                best_method = min(values.items(), key=lambda x: x[1])
                f.write(f"  → Best: {best_method[0]} ({best_method[1]:.6e})\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Comparison report saved to {report_path}")

        # Generate comparison plots
        self.plot_comparison()

    def plot_comparison(self):
        """Generate comparison plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
            return

        # Metrics to compare
        metrics_to_plot = [
            ('pde_residual_l2', 'PDE Residual (L2)'),
            ('bc_total_mae', 'BC Error (MAE)'),
            ('conservation_l2', 'Conservation Error (L2)'),
            ('mse', 'MSE (vs Ground Truth)'),
            ('relative_error', 'Relative Error')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        comparison = self.compare_metrics([m[0] for m in metrics_to_plot])

        for idx, (metric_name, metric_label) in enumerate(metrics_to_plot):
            ax = axes[idx]

            if metric_name not in comparison or len(comparison[metric_name]) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(metric_label)
                continue

            methods = list(comparison[metric_name].keys())
            values = list(comparison[metric_name].values())

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            bars = ax.bar(methods, values, color=colors[:len(methods)])

            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')

            # Rotate x labels
            ax.set_xticklabels(methods, rotation=45, ha='right')

        # Remove unused subplot
        if len(metrics_to_plot) < len(axes):
            for idx in range(len(metrics_to_plot), len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comparison plots saved to {self.output_dir / 'comparison_metrics.png'}")


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    data: 'Data',
    A_csr: Dict,
    boundary_types: Dict,
    bc_values: Dict,
    reference: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Combines physics-based metrics and (optionally) reference-based metrics.

    Args:
        model: Trained model
        data: PyG Data object
        A_csr: System matrix
        boundary_types: Boundary masks
        bc_values: BC values
        reference: Ground truth (optional)
        device: Device

    Returns:
        metrics: Comprehensive evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        pred = model(data.x, data.edge_index).squeeze()

    # Physics-based evaluation
    physics_eval = PhysicsEvaluator(A_csr, boundary_types, bc_values)
    metrics = physics_eval.evaluate(pred, data.x, data.edge_index)

    # Reference-based evaluation (if available)
    if reference is not None:
        mse = torch.mean((pred - reference) ** 2).item()
        mae = torch.mean(torch.abs(pred - reference)).item()
        rel_error = (torch.norm(pred - reference) / torch.norm(reference)).item()
        max_error = torch.max(torch.abs(pred - reference)).item()

        metrics.update({
            'mse': mse,
            'mae': mae,
            'relative_error': rel_error,
            'max_error': max_error
        })

    return metrics
