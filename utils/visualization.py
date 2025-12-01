"""
Visualization Utilities for PUP-HAW-U

Tools for visualizing solutions, weights, and training progress.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_solution_field(
    pred: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    title: str = 'Solution Field',
    save_path: Optional[str] = None,
    reference: Optional[torch.Tensor] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot solution field with optional reference comparison.

    Args:
        pred: Predicted solution [num_nodes]
        coords: Node coordinates [num_nodes, 2 or 3] (optional)
        title: Plot title
        save_path: Path to save plot (optional)
        reference: Ground truth solution (optional)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
    except ImportError:
        print("Matplotlib not available.")
        return

    pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred

    if reference is not None:
        ref_np = reference.detach().cpu().numpy() if torch.is_tensor(reference) else reference
        error_np = np.abs(pred_np - ref_np)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Prediction
        if coords is not None:
            coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else coords
            x, y = coords_np[:, 0], coords_np[:, 1]
            im1 = axes[0].tricontourf(x, y, pred_np, levels=20, cmap='viridis')
        else:
            im1 = axes[0].scatter(range(len(pred_np)), pred_np, c=pred_np, cmap='viridis')

        axes[0].set_title('Prediction')
        plt.colorbar(im1, ax=axes[0])

        # Reference
        if coords is not None:
            im2 = axes[1].tricontourf(x, y, ref_np, levels=20, cmap='viridis')
        else:
            im2 = axes[1].scatter(range(len(ref_np)), ref_np, c=ref_np, cmap='viridis')

        axes[1].set_title('Reference')
        plt.colorbar(im2, ax=axes[1])

        # Error
        if coords is not None:
            im3 = axes[2].tricontourf(x, y, error_np, levels=20, cmap='Reds')
        else:
            im3 = axes[2].scatter(range(len(error_np)), error_np, c=error_np, cmap='Reds')

        axes[2].set_title(f'Absolute Error (Max: {error_np.max():.3e})')
        plt.colorbar(im3, ax=axes[2])

    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        if coords is not None:
            coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else coords
            x, y = coords_np[:, 0], coords_np[:, 1]
            im = ax.tricontourf(x, y, pred_np, levels=20, cmap='viridis')
        else:
            im = ax.scatter(range(len(pred_np)), pred_np, c=pred_np, cmap='viridis', s=5)

        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_weight_distribution(
    weights: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    title: str = 'Weight Distribution',
    save_path: Optional[str] = None
):
    """
    Visualize mesh quality weights distribution.

    Args:
        weights: Weight values [num_nodes]
        coords: Node coordinates (optional)
        title: Plot title
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    weights_np = weights.detach().cpu().numpy() if torch.is_tensor(weights) else weights

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spatial distribution
    if coords is not None:
        coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else coords
        x, y = coords_np[:, 0], coords_np[:, 1]
        im = axes[0].tricontourf(x, y, weights_np, levels=20, cmap='coolwarm')
        axes[0].set_title('Spatial Distribution')
        plt.colorbar(im, ax=axes[0])
    else:
        axes[0].plot(weights_np)
        axes[0].set_title('Weight Values')
        axes[0].set_xlabel('Node Index')
        axes[0].set_ylabel('Weight')

    # Histogram
    axes[1].hist(weights_np, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(weights_np.mean(), color='r', linestyle='--', label=f'Mean: {weights_np.mean():.3f}')
    axes[1].axvline(np.median(weights_np), color='g', linestyle='--', label=f'Median: {np.median(weights_np):.3f}')
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Weight Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ablation_results(
    results: Dict[str, Dict],
    metric_name: str = 'pde_residual_l2',
    save_path: Optional[str] = None
):
    """
    Visualize ablation study results.

    Args:
        results: Ablation results dictionary
        metric_name: Metric to visualize
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = []
    means = []
    stds = []

    for config_name, result_data in results.items():
        if 'aggregated' in result_data:
            agg = result_data['aggregated']
            mean_key = f'{metric_name}_mean'
            std_key = f'{metric_name}_std'

            if mean_key in agg:
                configs.append(result_data['config']['name'])
                means.append(agg[mean_key])
                stds.append(agg.get(std_key, 0))

    if len(configs) == 0:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')

    # Color full model differently
    for i, config in enumerate(configs):
        if 'Full' in config:
            bars[i].set_color('green')
            bars[i].set_alpha(0.8)

    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Ablation Study: {metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def create_training_animation(
    history: List[Dict],
    field_history: Optional[List[torch.Tensor]] = None,
    coords: Optional[torch.Tensor] = None,
    save_path: str = 'training_animation.gif',
    fps: int = 5
):
    """
    Create animation of training progress.

    Args:
        history: Training history
        field_history: Solution field at each epoch (optional)
        coords: Node coordinates
        save_path: Path to save animation
        fps: Frames per second
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Matplotlib not available.")
        return

    if field_history is None:
        print("Field history required for animation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def update(frame):
        axes[0].clear()
        axes[1].clear()

        # Plot solution field
        pred = field_history[frame]
        pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred

        if coords is not None:
            coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else coords
            x, y = coords_np[:, 0], coords_np[:, 1]
            im = axes[0].tricontourf(x, y, pred_np, levels=20, cmap='viridis')
            plt.colorbar(im, ax=axes[0])
        else:
            axes[0].plot(pred_np)

        axes[0].set_title(f'Epoch {frame}: Solution Field')

        # Plot loss curves
        epochs = [h['epoch'] for h in history[:frame+1]]
        axes[1].semilogy(epochs, [h['loss_total'] for h in history[:frame+1]], label='Total', linewidth=2)
        axes[1].semilogy(epochs, [h['loss_pde'] for h in history[:frame+1]], label='PDE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Progress')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    anim = animation.FuncAnimation(fig, update, frames=len(field_history), repeat=True)
    anim.save(save_path, writer='pillow', fps=fps)

    print(f"Animation saved to {save_path}")
    plt.close()


def generate_paper_figures(
    results_dir: str,
    output_dir: str = './paper_figures'
):
    """
    Generate all figures for paper.

    Args:
        results_dir: Directory with experimental results
        output_dir: Output directory for figures
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(results_dir)

    # Load ablation results
    if (results_path / 'ablation_complete.json').exists():
        with open(results_path / 'ablation_complete.json') as f:
            ablation_results = json.load(f)

        # Generate ablation plots for key metrics
        for metric in ['pde_residual_l2', 'bc_total_mae', 'conservation_l2', 'mse', 'relative_error']:
            plot_ablation_results(
                ablation_results,
                metric_name=metric,
                save_path=output_path / f'ablation_{metric}.png'
            )

    print(f"Paper figures generated in {output_dir}")
