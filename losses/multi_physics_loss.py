"""
Multi-physics constraint losses for unsupervised PDE learning.

Implements:
- Boundary condition losses (Dirichlet, Neumann)
- Initial condition losses
- Conservation law losses (mass, momentum, energy)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from utils.graph_ops import compute_graph_gradient


class BoundaryConditionLoss(nn.Module):
    """
    Boundary condition loss for enforcing BC constraints.

    Supports:
    - Dirichlet BC: u = u_bc on boundary
    - Neumann BC: ∂u/∂n = g on boundary
    """

    def __init__(
        self,
        boundary_types: Dict[str, torch.Tensor],
        bc_values: Optional[Dict[str, torch.Tensor]] = None,
        use_soft_constraint: bool = True
    ):
        """
        Args:
            boundary_types: Dict with keys 'inlet', 'outlet', 'wall', 'symmetry'
                           Each value is a boolean mask [num_nodes]
            bc_values: Dict with boundary condition values (optional)
                      e.g., {'inlet': tensor([p_inlet]), 'outlet': tensor([p_outlet])}
            use_soft_constraint: If True, use MSE loss; if False, use hard constraint
        """
        super().__init__()
        self.boundary_types = boundary_types
        self.bc_values = bc_values if bc_values is not None else {}
        self.use_soft_constraint = use_soft_constraint

    def forward(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor,
        bc_values: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute boundary condition loss.

        Args:
            pred: Predicted field [num_nodes, 1] or [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            bc_values: Boundary condition values (overrides self.bc_values if provided)

        Returns:
            loss_bc: Total boundary condition loss (scalar)
            loss_dict: Dictionary with individual BC losses for each boundary type
        """
        if pred.dim() == 2:
            pred = pred.squeeze(-1)

        bc_vals = bc_values if bc_values is not None else self.bc_values
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        # Dirichlet BC for each boundary type
        for bc_type, mask in self.boundary_types.items():
            if mask.sum() > 0:  # If this boundary exists
                if bc_type in bc_vals:
                    # Prescribed value
                    bc_target = bc_vals[bc_type]
                    if isinstance(bc_target, (int, float)):
                        bc_target = torch.tensor(bc_target, device=pred.device)
                else:
                    # Default values based on boundary type
                    bc_target = self._get_default_bc_value(bc_type, pred)

                # MSE loss on boundary nodes
                pred_bc = pred[mask]
                if bc_target.dim() == 0:
                    bc_target = bc_target.expand_as(pred_bc)

                loss_bc_type = torch.mean((pred_bc - bc_target) ** 2)
                loss_dict[bc_type] = loss_bc_type
                total_loss = total_loss + loss_bc_type

        return total_loss, loss_dict

    def _get_default_bc_value(
        self,
        bc_type: str,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Get default boundary condition value based on type.

        Args:
            bc_type: Boundary type ('inlet', 'outlet', 'wall', 'symmetry')
            pred: Predicted field [num_nodes]

        Returns:
            bc_value: Default BC value (scalar or tensor)
        """
        device = pred.device

        if bc_type == 'inlet':
            # Default: slightly positive pressure
            return torch.tensor(0.1, device=device)
        elif bc_type == 'outlet':
            # Default: zero pressure (reference)
            return torch.tensor(0.0, device=device)
        elif bc_type == 'wall':
            # Default: zero gradient (Neumann)
            return torch.tensor(0.0, device=device)
        elif bc_type == 'symmetry':
            # Default: zero gradient
            return torch.tensor(0.0, device=device)
        else:
            return torch.tensor(0.0, device=device)

    def compute_neumann_loss(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_mask: torch.Tensor,
        target_gradient: float = 0.0
    ) -> torch.Tensor:
        """
        Compute Neumann boundary condition loss (gradient constraint).

        Args:
            pred: Predicted field [num_nodes, 1] or [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            boundary_mask: Boolean mask [num_nodes] for boundary nodes
            target_gradient: Target gradient value (default: 0 for zero-gradient BC)

        Returns:
            loss_neumann: Neumann BC loss (scalar)
        """
        if pred.dim() == 2:
            pred = pred.squeeze(-1)

        # Compute gradient at boundary nodes
        grad = compute_graph_gradient(pred, edge_index)  # [num_nodes, 3]
        grad_magnitude = torch.norm(grad, dim=1)  # [num_nodes]

        # Loss: gradient magnitude should match target
        grad_bc = grad_magnitude[boundary_mask]
        target = torch.tensor(target_gradient, device=pred.device).expand_as(grad_bc)

        loss_neumann = torch.mean((grad_bc - target) ** 2)

        return loss_neumann


class InitialConditionLoss(nn.Module):
    """
    Initial condition loss for time-dependent problems.

    Enforces: u(x, t=0) = u_0(x)
    """

    def __init__(
        self,
        ic_values: Optional[torch.Tensor] = None,
        use_zero_ic: bool = True
    ):
        """
        Args:
            ic_values: Initial condition values [num_nodes] (optional)
            use_zero_ic: If True and ic_values is None, use zero IC
        """
        super().__init__()
        self.ic_values = ic_values
        self.use_zero_ic = use_zero_ic

    def forward(
        self,
        pred: torch.Tensor,
        ic_mask: torch.Tensor,
        ic_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute initial condition loss.

        Args:
            pred: Predicted field [num_nodes, 1] or [num_nodes]
            ic_mask: Boolean mask [num_nodes] indicating IC nodes (usually all nodes at t=0)
            ic_values: Initial condition values [num_nodes] (overrides self.ic_values)

        Returns:
            loss_ic: Initial condition loss (scalar)
        """
        if pred.dim() == 2:
            pred = pred.squeeze(-1)

        ic_vals = ic_values if ic_values is not None else self.ic_values

        if ic_vals is None:
            if self.use_zero_ic:
                # Zero initial condition
                ic_vals = torch.zeros_like(pred)
            else:
                # No IC constraint
                return torch.tensor(0.0, device=pred.device)

        # MSE loss on IC nodes
        pred_ic = pred[ic_mask]
        target_ic = ic_vals[ic_mask] if ic_vals.dim() > 0 else ic_vals

        loss_ic = torch.mean((pred_ic - target_ic) ** 2)

        return loss_ic


class ConservationLoss(nn.Module):
    """
    Conservation law losses for physical constraints.

    Implements:
    - Mass conservation: ∇·u = 0 (incompressible flow)
    - Momentum conservation (optional)
    - Energy conservation (optional)
    """

    def __init__(
        self,
        conservation_type: str = 'mass',
        use_weighted_loss: bool = True
    ):
        """
        Args:
            conservation_type: Type of conservation law ('mass', 'momentum', 'energy')
            use_weighted_loss: If True, weight loss by cell volume
        """
        super().__init__()
        self.conservation_type = conservation_type
        self.use_weighted_loss = use_weighted_loss

    def forward(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute conservation law loss.

        Args:
            pred: Predicted field [num_nodes, 1] or [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            feats: Node features [num_nodes, num_feats] (for volume weighting)
            velocity: Velocity field [num_nodes, 3] (for momentum/energy conservation)

        Returns:
            loss_cons: Total conservation loss (scalar)
            loss_dict: Dictionary with individual conservation losses
        """
        if pred.dim() == 2:
            pred = pred.squeeze(-1)

        loss_dict = {}

        if self.conservation_type == 'mass':
            loss_mass = self.compute_mass_conservation(pred, edge_index, feats)
            loss_dict['mass'] = loss_mass
            total_loss = loss_mass

        elif self.conservation_type == 'momentum' and velocity is not None:
            loss_momentum = self.compute_momentum_conservation(pred, velocity, edge_index, feats)
            loss_dict['momentum'] = loss_momentum
            total_loss = loss_momentum

        elif self.conservation_type == 'energy' and velocity is not None:
            loss_energy = self.compute_energy_conservation(pred, velocity, edge_index, feats)
            loss_dict['energy'] = loss_energy
            total_loss = loss_energy

        else:
            total_loss = torch.tensor(0.0, device=pred.device)

        return total_loss, loss_dict

    def compute_mass_conservation(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor,
        feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mass conservation loss: ∇·u = 0.

        For pressure-based formulation, this becomes:
        ∇²p = 0 (Laplace equation in incompressible flow)

        Args:
            pred: Predicted pressure field [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            feats: Node features [num_nodes, num_feats] (for volume weighting)

        Returns:
            loss_mass: Mass conservation loss (scalar)
        """
        from utils.graph_ops import compute_graph_laplacian

        # Compute ∇²p (divergence of gradient)
        laplacian = compute_graph_laplacian(pred, edge_index, normalized=False)

        # Mass conservation: Laplacian should be zero
        if self.use_weighted_loss and feats is not None and feats.shape[1] > 7:
            # Weight by cell volume (feature column 7)
            volumes = feats[:, 7]
            volumes = volumes / (volumes.mean() + 1e-6)  # Normalize
            residual = laplacian ** 2 * volumes
        else:
            residual = laplacian ** 2

        loss_mass = torch.mean(residual)

        return loss_mass

    def compute_momentum_conservation(
        self,
        pred: torch.Tensor,
        velocity: torch.Tensor,
        edge_index: torch.Tensor,
        feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute momentum conservation loss.

        Navier-Stokes momentum equation (steady-state):
        ρ(u·∇)u = -∇p + μ∇²u + f

        Args:
            pred: Predicted pressure field [num_nodes]
            velocity: Velocity field [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            feats: Node features [num_nodes, num_feats]

        Returns:
            loss_momentum: Momentum conservation loss (scalar)
        """
        # Compute pressure gradient
        pressure_grad = compute_graph_gradient(pred, edge_index)  # [num_nodes, 3]

        # For simplified case, assume small Reynolds number (Stokes flow):
        # ∇p ≈ μ∇²u
        # Loss: pressure gradient should balance viscous forces

        # Placeholder: simplified momentum balance
        # In practice, this requires velocity field prediction or coupling
        residual = pressure_grad.norm(dim=1) ** 2

        loss_momentum = torch.mean(residual)

        return loss_momentum

    def compute_energy_conservation(
        self,
        pred: torch.Tensor,
        velocity: torch.Tensor,
        edge_index: torch.Tensor,
        feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy conservation loss.

        Energy equation (steady-state):
        ρcp(u·∇)T = k∇²T

        Args:
            pred: Predicted temperature field [num_nodes]
            velocity: Velocity field [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
            feats: Node features [num_nodes, num_feats]

        Returns:
            loss_energy: Energy conservation loss (scalar)
        """
        from utils.graph_ops import compute_graph_laplacian

        # Compute ∇²T (thermal diffusion)
        laplacian = compute_graph_laplacian(pred, edge_index, normalized=False)

        # For simplified case without velocity coupling:
        # Residual based on thermal diffusion balance
        residual = laplacian ** 2

        loss_energy = torch.mean(residual)

        return loss_energy


class MultiPhysicsLoss(nn.Module):
    """
    Combined multi-physics loss for unsupervised PDE learning.

    Integrates:
    - PDE residual loss
    - Boundary condition loss
    - Initial condition loss
    - Conservation law loss
    """

    def __init__(
        self,
        boundary_types: Dict[str, torch.Tensor],
        bc_values: Optional[Dict[str, torch.Tensor]] = None,
        ic_values: Optional[torch.Tensor] = None,
        conservation_type: str = 'mass',
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_cons: float = 1.0
    ):
        """
        Args:
            boundary_types: Dict with boundary masks
            bc_values: Boundary condition values
            ic_values: Initial condition values
            conservation_type: Conservation law type
            lambda_pde: Weight for PDE residual loss
            lambda_bc: Weight for boundary condition loss
            lambda_ic: Weight for initial condition loss
            lambda_cons: Weight for conservation loss
        """
        super().__init__()

        self.bc_loss = BoundaryConditionLoss(boundary_types, bc_values)
        self.ic_loss = InitialConditionLoss(ic_values)
        self.cons_loss = ConservationLoss(conservation_type)

        # Loss weights (can be adapted by hierarchical adaptive weighting)
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.lambda_cons = lambda_cons

    def forward(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor,
        feats: torch.Tensor,
        pde_residual: torch.Tensor,
        ic_mask: Optional[torch.Tensor] = None,
        bc_values: Optional[Dict[str, torch.Tensor]] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total multi-physics loss.

        Args:
            pred: Predicted field [num_nodes, 1] or [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            feats: Node features [num_nodes, num_feats]
            pde_residual: PDE residual [num_nodes]
            ic_mask: Initial condition mask [num_nodes] (optional)
            bc_values: Boundary condition values (optional)
            velocity: Velocity field [num_nodes, 3] (optional)

        Returns:
            loss_total: Total loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}

        # PDE residual loss
        loss_pde = torch.mean(pde_residual ** 2)
        loss_dict['pde'] = loss_pde

        # Boundary condition loss
        loss_bc, bc_dict = self.bc_loss(pred, edge_index, bc_values)
        loss_dict['bc'] = loss_bc
        loss_dict.update({f'bc_{k}': v for k, v in bc_dict.items()})

        # Initial condition loss (if applicable)
        if ic_mask is not None and ic_mask.sum() > 0:
            loss_ic = self.ic_loss(pred, ic_mask)
            loss_dict['ic'] = loss_ic
        else:
            loss_ic = torch.tensor(0.0, device=pred.device)
            loss_dict['ic'] = loss_ic

        # Conservation law loss
        loss_cons, cons_dict = self.cons_loss(pred, edge_index, feats, velocity)
        loss_dict['conservation'] = loss_cons
        loss_dict.update({f'cons_{k}': v for k, v in cons_dict.items()})

        # Total weighted loss
        loss_total = (
            self.lambda_pde * loss_pde +
            self.lambda_bc * loss_bc +
            self.lambda_ic * loss_ic +
            self.lambda_cons * loss_cons
        )

        loss_dict['total'] = loss_total

        return loss_total, loss_dict

    def update_weights(
        self,
        lambda_pde: Optional[float] = None,
        lambda_bc: Optional[float] = None,
        lambda_ic: Optional[float] = None,
        lambda_cons: Optional[float] = None
    ):
        """
        Update loss weights (for hierarchical adaptive weighting).

        Args:
            lambda_pde: New weight for PDE loss
            lambda_bc: New weight for BC loss
            lambda_ic: New weight for IC loss
            lambda_cons: New weight for conservation loss
        """
        if lambda_pde is not None:
            self.lambda_pde = lambda_pde
        if lambda_bc is not None:
            self.lambda_bc = lambda_bc
        if lambda_ic is not None:
            self.lambda_ic = lambda_ic
        if lambda_cons is not None:
            self.lambda_cons = lambda_cons
