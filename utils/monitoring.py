"""
Monitoring and Visualization Utilities for Hybrid Training

Provides tools for tracking and visualizing the transition from
supervised to unsupervised learning.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch


class TrainingMonitor:
    """
    Monitor for hybrid training progress.

    Tracks losses, metrics, and curriculum schedule throughout training.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save monitoring outputs
    log_interval : int, optional
        Logging interval in epochs (default: 1)

    Examples
    --------
    >>> monitor = TrainingMonitor('./outputs/hybrid')
    >>> for epoch in range(100):
    ...     # Train
    ...     loss, info = train_epoch(...)
    ...     lambda_data = scheduler.get_lambda_data(epoch)
    ...     monitor.log(epoch, info, lambda_data)
    ... monitor.save()
    ... monitor.plot_all()
    """

    def __init__(self, output_dir: str, log_interval: int = 1):
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval

        # Create subdirectories
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)

        # History tracking
        self.history = []

        # Best metrics
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

    def log(
        self,
        epoch: int,
        train_info: Dict,
        lambda_data: float,
        val_metrics: Optional[Dict] = None
    ):
        """
        Log training information for current epoch.

        Args:
            epoch: Current epoch
            train_info: Training info dict from loss function
            lambda_data: Curriculum weight
            val_metrics: Validation metrics (optional)
        """
        record = {
            'epoch': epoch,
            'lambda_data_curriculum': lambda_data,
            **train_info
        }

        if val_metrics is not None:
            record.update({f'val_{k}': v for k, v in val_metrics.items()})

            # Track best validation loss
            val_loss = val_metrics.get('mse', float('inf'))
            if val_loss < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = val_loss
                self.best_metrics['best_epoch'] = epoch

        self.history.append(record)

        # Console logging
        if (epoch + 1) % self.log_interval == 0:
            self._print_status(record)

    def _print_status(self, record: Dict):
        """Print training status to console."""
        epoch = record['epoch']
        loss_total = record.get('loss_total', 0.0)
        lambda_data = record.get('lambda_data_curriculum', 1.0)

        msg = f"[Epoch {epoch+1}] Loss: {loss_total:.6f} | λ_data: {lambda_data:.4f}"

        if 'val_mse' in record:
            msg += f" | Val MSE: {record['val_mse']:.6f}"

        print(msg)

    def save(self):
        """Save monitoring history to JSON."""
        output_file = self.output_dir / 'logs' / 'training_history.json'

        with open(output_file, 'w') as f:
            json.dump({
                'history': self.history,
                'best_metrics': self.best_metrics
            }, f, indent=2)

        print(f"Training history saved to {output_file}")

    def plot_all(self):
        """Generate all monitoring plots."""
        if len(self.history) == 0:
            print("No history to plot.")
            return

        self.plot_losses()
        self.plot_curriculum_transition()
        self.plot_physics_constraints()
        self.plot_adaptive_weights()

        print(f"Plots saved to {self.output_dir / 'plots'}")

    def plot_losses(self):
        """Plot loss curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
            return

        epochs = [h['epoch'] for h in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Total loss
        axes[0].plot(epochs, [h['loss_total'] for h in self.history], linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Total Loss', fontsize=12)
        axes[0].set_title('Total Loss', fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Individual losses (log scale)
        axes[1].plot(epochs, [h['loss_data'] for h in self.history],
                    label='Data (Supervised)', linewidth=2)
        axes[1].plot(epochs, [h['loss_pde'] for h in self.history],
                    label='PDE Residual', linewidth=2)
        axes[1].plot(epochs, [h['loss_bc'] for h in self.history],
                    label='Boundary Condition', linewidth=2)
        axes[1].plot(epochs, [h['loss_conservation'] for h in self.history],
                    label='Conservation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Individual Losses (Log Scale)', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'losses.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_curriculum_transition(self):
        """Plot curriculum transition and validation metrics."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs = [h['epoch'] for h in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Curriculum weight
        lambdas = [h['lambda_data_curriculum'] for h in self.history]
        axes[0].plot(epochs, lambdas, linewidth=2, color='red')
        axes[0].axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='50% Supervised')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('λ_data (Supervised Weight)', fontsize=12)
        axes[0].set_title('Curriculum Schedule: Supervised → Unsupervised', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)

        # Add transition phases
        total_epochs = epochs[-1]
        for threshold, label in [(0.75, '75% Supervised'), (0.25, '25% Supervised')]:
            # Find epoch where lambda crosses threshold
            for i, lam in enumerate(lambdas):
                if lam <= threshold:
                    axes[0].axvline(x=i, linestyle=':', color='green', alpha=0.5)
                    axes[0].text(i, threshold + 0.05, label, fontsize=9, rotation=90)
                    break

        # Validation metrics (if available)
        if 'val_mse' in self.history[0]:
            axes[1].plot(epochs, [h['val_mse'] for h in self.history],
                        label='Validation MSE', linewidth=2)
            if 'val_relative_error' in self.history[0]:
                axes[1].plot(epochs, [h['val_relative_error'] for h in self.history],
                            label='Relative Error', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Error', fontsize=12)
            axes[1].set_title('Validation Metrics', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'curriculum_transition.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_physics_constraints(self):
        """Plot physics constraint satisfaction over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs = [h['epoch'] for h in self.history]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot normalized constraint losses
        constraints = ['loss_pde', 'loss_bc', 'loss_conservation']
        labels = ['PDE Residual', 'Boundary Condition', 'Mass Conservation']

        for constraint, label in zip(constraints, labels):
            values = [h[constraint] for h in self.history]
            ax.plot(epochs, values, label=label, linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Constraint Loss', fontsize=12)
        ax.set_title('Physics Constraint Satisfaction', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'physics_constraints.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_adaptive_weights(self):
        """Plot hierarchical adaptive weights."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Check if adaptive weights are available
        if 'lambdas_adaptive' not in self.history[0]:
            return

        epochs = [h['epoch'] for h in self.history]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract lambda values for each constraint
        constraint_names = list(self.history[0]['lambdas_adaptive'].keys())

        for name in constraint_names:
            values = [h['lambdas_adaptive'][name] for h in self.history]
            ax.plot(epochs, values, label=f'λ_{name}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Adaptive Weight', fontsize=12)
        ax.set_title('Hierarchical Adaptive Weights (Level 1)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'adaptive_weights.png', dpi=150, bbox_inches='tight')
        plt.close()

    def get_summary(self) -> Dict:
        """
        Get training summary statistics.

        Returns:
            summary: Dictionary with summary statistics
        """
        if len(self.history) == 0:
            return {}

        final = self.history[-1]

        summary = {
            'total_epochs': len(self.history),
            'best_epoch': self.best_metrics['best_epoch'],
            'best_val_loss': self.best_metrics['best_val_loss'],
            'final_loss_total': final['loss_total'],
            'final_loss_data': final['loss_data'],
            'final_loss_pde': final['loss_pde'],
            'final_loss_bc': final['loss_bc'],
            'final_loss_conservation': final['loss_conservation'],
            'final_lambda_data': final['lambda_data_curriculum']
        }

        if 'val_mse' in final:
            summary['final_val_mse'] = final['val_mse']

        return summary

    def print_summary(self):
        """Print training summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("HYBRID TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Best epoch: {summary['best_epoch']}")
        print(f"Best validation loss: {summary['best_val_loss']:.6f}")
        print(f"\nFinal metrics:")
        print(f"  Total loss: {summary['final_loss_total']:.6f}")
        print(f"  Data loss: {summary['final_loss_data']:.6f}")
        print(f"  PDE loss: {summary['final_loss_pde']:.6f}")
        print(f"  BC loss: {summary['final_loss_bc']:.6f}")
        print(f"  Conservation loss: {summary['final_loss_conservation']:.6f}")
        print(f"  Curriculum λ_data: {summary['final_lambda_data']:.4f}")

        if 'final_val_mse' in summary:
            print(f"  Validation MSE: {summary['final_val_mse']:.6f}")

        print("=" * 60 + "\n")


def compare_schedules(
    schedules: List[str],
    total_epochs: int,
    output_path: Optional[str] = None
):
    """
    Compare different curriculum schedules.

    Args:
        schedules: List of schedule names to compare
        total_epochs: Total training epochs
        output_path: Path to save comparison plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from utils.scheduling import CurriculumScheduler
    except ImportError:
        print("Matplotlib not available.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    epochs = np.arange(total_epochs)

    for schedule_name in schedules:
        scheduler = CurriculumScheduler(
            schedule_type=schedule_name,
            total_epochs=total_epochs
        )

        lambdas = [scheduler.get_lambda_data(e) for e in epochs]

        ax.plot(epochs, lambdas, label=schedule_name.capitalize(), linewidth=2.5)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('λ_data (Supervised Weight)', fontsize=14)
    ax.set_title('Curriculum Schedule Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Add reference lines
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='50%')
    ax.axhline(y=0.25, linestyle='--', color='gray', alpha=0.3)
    ax.axhline(y=0.75, linestyle='--', color='gray', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Schedule comparison saved to {output_path}")
    else:
        plt.show()

    plt.close()
