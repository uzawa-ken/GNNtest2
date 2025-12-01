"""
Curriculum Learning Scheduling Strategies

Implements various scheduling strategies for transitioning from
supervised to unsupervised learning.
"""

import torch
import numpy as np
from typing import Dict, Callable, Optional
from enum import Enum


class ScheduleType(Enum):
    """Schedule types for curriculum learning."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"


class CurriculumScheduler:
    """
    Curriculum learning scheduler for hybrid training.

    Gradually transitions from supervised to unsupervised learning
    by decaying the weight of supervised loss over time.

    Parameters
    ----------
    schedule_type : str or ScheduleType
        Type of schedule ('linear', 'exponential', 'step', 'cosine', 'polynomial')
    total_epochs : int
        Total number of training epochs
    lambda_data_init : float, optional
        Initial weight for supervised loss (default: 1.0)
    lambda_data_final : float, optional
        Final weight for supervised loss (default: 0.0)
    decay_rate : float, optional
        Decay rate for exponential schedule (default: 0.1)
    step_size : int, optional
        Step size for step schedule (default: 10)
    step_gamma : float, optional
        Decay factor for step schedule (default: 0.5)
    polynomial_power : float, optional
        Power for polynomial schedule (default: 2.0)
    warmup_epochs : int, optional
        Number of epochs for warmup (default: 0)
    freeze_epochs : int, optional
        Number of initial epochs to freeze at init value (default: 0)

    Examples
    --------
    >>> scheduler = CurriculumScheduler('exponential', total_epochs=100)
    >>> for epoch in range(100):
    ...     lambda_data = scheduler.get_lambda_data(epoch)
    ...     # Use lambda_data in hybrid loss
    """

    def __init__(
        self,
        schedule_type: str,
        total_epochs: int,
        lambda_data_init: float = 1.0,
        lambda_data_final: float = 0.0,
        decay_rate: float = 0.1,
        step_size: int = 10,
        step_gamma: float = 0.5,
        polynomial_power: float = 2.0,
        warmup_epochs: int = 0,
        freeze_epochs: int = 0
    ):
        self.schedule_type = ScheduleType(schedule_type)
        self.total_epochs = total_epochs
        self.lambda_data_init = lambda_data_init
        self.lambda_data_final = lambda_data_final
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.polynomial_power = polynomial_power
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs

        # History tracking
        self.history = []

    def get_lambda_data(self, epoch: int) -> float:
        """
        Get supervised loss weight for current epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            lambda_data: Weight for supervised loss [0, 1]
        """
        # Freeze period: keep at initial value
        if epoch < self.freeze_epochs:
            lambda_data = self.lambda_data_init

        # Warmup period: linearly increase from 0 to init
        elif epoch < self.freeze_epochs + self.warmup_epochs:
            progress = (epoch - self.freeze_epochs) / self.warmup_epochs
            lambda_data = self.lambda_data_init * progress

        # Main schedule
        else:
            effective_epoch = epoch - self.freeze_epochs - self.warmup_epochs
            effective_total = self.total_epochs - self.freeze_epochs - self.warmup_epochs

            if self.schedule_type == ScheduleType.LINEAR:
                lambda_data = self._linear_schedule(effective_epoch, effective_total)
            elif self.schedule_type == ScheduleType.EXPONENTIAL:
                lambda_data = self._exponential_schedule(effective_epoch)
            elif self.schedule_type == ScheduleType.STEP:
                lambda_data = self._step_schedule(effective_epoch)
            elif self.schedule_type == ScheduleType.COSINE:
                lambda_data = self._cosine_schedule(effective_epoch, effective_total)
            elif self.schedule_type == ScheduleType.POLYNOMIAL:
                lambda_data = self._polynomial_schedule(effective_epoch, effective_total)
            elif self.schedule_type == ScheduleType.CONSTANT:
                lambda_data = self.lambda_data_init
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Clamp to [lambda_data_final, lambda_data_init]
        lambda_data = np.clip(lambda_data, self.lambda_data_final, self.lambda_data_init)

        # Track history
        self.history.append({'epoch': epoch, 'lambda_data': lambda_data})

        return float(lambda_data)

    def _linear_schedule(self, epoch: int, total: int) -> float:
        """Linear decay from init to final."""
        progress = min(epoch / max(total, 1), 1.0)
        return self.lambda_data_init - progress * (self.lambda_data_init - self.lambda_data_final)

    def _exponential_schedule(self, epoch: int) -> float:
        """Exponential decay: lambda = init * exp(-decay_rate * epoch)."""
        return self.lambda_data_init * np.exp(-self.decay_rate * epoch)

    def _step_schedule(self, epoch: int) -> float:
        """Step decay: reduce by gamma every step_size epochs."""
        num_steps = epoch // self.step_size
        return self.lambda_data_init * (self.step_gamma ** num_steps)

    def _cosine_schedule(self, epoch: int, total: int) -> float:
        """Cosine annealing from init to final."""
        progress = min(epoch / max(total, 1), 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return self.lambda_data_final + (self.lambda_data_init - self.lambda_data_final) * cosine_decay

    def _polynomial_schedule(self, epoch: int, total: int) -> float:
        """Polynomial decay: (1 - progress)^power."""
        progress = min(epoch / max(total, 1), 1.0)
        decay = (1 - progress) ** self.polynomial_power
        return self.lambda_data_final + (self.lambda_data_init - self.lambda_data_final) * decay

    def get_schedule_name(self) -> str:
        """Get human-readable schedule name."""
        return f"{self.schedule_type.value}_decay"

    def plot_schedule(self, save_path: Optional[str] = None):
        """
        Plot the schedule curve.

        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return

        epochs = np.arange(self.total_epochs)
        lambdas = [self.get_lambda_data(e) for e in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lambdas, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('λ_data (Supervised Loss Weight)', fontsize=12)
        plt.title(f'Curriculum Schedule: {self.get_schedule_name()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Schedule plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum scheduler that adjusts based on performance.

    Extends CurriculumScheduler to allow dynamic adjustment based on
    training metrics (e.g., validation loss, PDE residual).

    Parameters
    ----------
    schedule_type : str
        Base schedule type
    total_epochs : int
        Total epochs
    adaptation_criterion : str, optional
        Metric to monitor ('val_loss', 'pde_residual', 'bc_loss')
    patience : int, optional
        Number of epochs to wait before adapting (default: 5)
    adaptation_factor : float, optional
        Factor to multiply lambda_data when adapting (default: 0.8)
    min_lambda_data : float, optional
        Minimum allowed lambda_data (default: 0.0)

    Examples
    --------
    >>> scheduler = AdaptiveCurriculumScheduler('exponential', total_epochs=100)
    >>> for epoch in range(100):
    ...     lambda_data = scheduler.get_lambda_data(epoch)
    ...     # Train model
    ...     val_loss = evaluate()
    ...     scheduler.step(epoch, val_loss)  # Adapt based on performance
    """

    def __init__(
        self,
        schedule_type: str,
        total_epochs: int,
        adaptation_criterion: str = 'val_loss',
        patience: int = 5,
        adaptation_factor: float = 0.8,
        min_lambda_data: float = 0.0,
        **kwargs
    ):
        super().__init__(schedule_type, total_epochs, **kwargs)
        self.adaptation_criterion = adaptation_criterion
        self.patience = patience
        self.adaptation_factor = adaptation_factor
        self.min_lambda_data = min_lambda_data

        # Tracking
        self.best_metric = float('inf')
        self.epochs_since_improvement = 0
        self.adaptations = []

    def step(self, epoch: int, metric_value: float) -> bool:
        """
        Update scheduler based on performance metric.

        Args:
            epoch: Current epoch
            metric_value: Current metric value (lower is better)

        Returns:
            adapted: True if schedule was adapted
        """
        adapted = False

        if metric_value < self.best_metric:
            self.best_metric = metric_value
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # Adapt if no improvement for patience epochs
        if self.epochs_since_improvement >= self.patience:
            # Reduce supervised weight more aggressively
            current_lambda = self.get_lambda_data(epoch)
            new_lambda = max(current_lambda * self.adaptation_factor, self.min_lambda_data)

            # Override the schedule for next epoch
            self.lambda_data_init = new_lambda

            self.adaptations.append({
                'epoch': epoch,
                'old_lambda': current_lambda,
                'new_lambda': new_lambda,
                'metric': metric_value
            })

            self.epochs_since_improvement = 0
            adapted = True

        return adapted


def get_recommended_schedule(
    total_epochs: int,
    problem_difficulty: str = 'medium'
) -> Dict:
    """
    Get recommended curriculum schedule based on problem characteristics.

    Args:
        total_epochs: Total training epochs
        problem_difficulty: 'easy', 'medium', 'hard'

    Returns:
        config: Recommended scheduler configuration
    """
    if problem_difficulty == 'easy':
        # Fast transition for easy problems
        return {
            'schedule_type': 'linear',
            'total_epochs': total_epochs,
            'lambda_data_init': 1.0,
            'lambda_data_final': 0.0,
            'warmup_epochs': int(0.1 * total_epochs),
            'freeze_epochs': 0
        }
    elif problem_difficulty == 'medium':
        # Moderate exponential decay
        return {
            'schedule_type': 'exponential',
            'total_epochs': total_epochs,
            'lambda_data_init': 1.0,
            'lambda_data_final': 0.01,
            'decay_rate': 3.0 / total_epochs,  # Reach ~0.05 at 100% epochs
            'warmup_epochs': int(0.1 * total_epochs),
            'freeze_epochs': int(0.05 * total_epochs)
        }
    elif problem_difficulty == 'hard':
        # Slow polynomial decay for hard problems
        return {
            'schedule_type': 'polynomial',
            'total_epochs': total_epochs,
            'lambda_data_init': 1.0,
            'lambda_data_final': 0.05,
            'polynomial_power': 3.0,
            'warmup_epochs': int(0.15 * total_epochs),
            'freeze_epochs': int(0.1 * total_epochs)
        }
    else:
        raise ValueError(f"Unknown difficulty: {problem_difficulty}")
