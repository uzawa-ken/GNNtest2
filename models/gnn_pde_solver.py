"""
Integrated GNN-PDE Solver (Phase 6)

Unified model wrapper combining GNN architecture and PUP-HAW-U loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .sage_model import SimpleSAGE
from losses.integrated_loss import (
    PUPHAWLoss,
    PUPHAWUnsupervisedLoss,
    PUPHAWHybridLoss
)


class PUPHAWUnsupervised(nn.Module):
    """
    PUP-HAW-U: Physics-based Uncertainty Propagation with
    Hierarchical Adaptive Weighting - Unsupervised

    Integrated model for fully unsupervised PDE solving.

    Parameters
    ----------
    in_channels : int
        Input feature dimension
    hidden_channels : int, optional
        Hidden layer size (default: 64)
    num_layers : int, optional
        Number of GNN layers (default: 4)
    boundary_types : dict, optional
        Boundary condition types
    bc_values : dict, optional
        Boundary condition values
    conservation_type : str, optional
        Conservation law type (default: 'mass')
    mesh_quality_config : dict, optional
        Mesh quality weight configuration
    hierarchical_config : dict, optional
        Hierarchical adaptive configuration
    use_topology_propagation : bool, optional
        Use topology-aware weight propagation (default: True)

    Examples
    --------
    >>> model = PUPHAWUnsupervised(
    ...     in_channels=13,
    ...     boundary_types=boundary_types,
    ...     bc_values={'inlet': 0.1, 'outlet': 0.0}
    ... )
    >>> loss, info = model.compute_loss(x, edge_index, feats, A_csr, b, epoch)
    >>> loss.backward()
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        boundary_types: Optional[Dict] = None,
        bc_values: Optional[Dict] = None,
        ic_values: Optional[torch.Tensor] = None,
        conservation_type: str = 'mass',
        mesh_quality_config: Optional[Dict] = None,
        hierarchical_config: Optional[Dict] = None,
        use_topology_propagation: bool = True
    ):
        super().__init__()

        # GNN backbone
        self.gnn = SimpleSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )

        # Loss function
        self.loss_fn = PUPHAWUnsupervisedLoss(
            mesh_quality_config=mesh_quality_config,
            hierarchical_config=hierarchical_config,
            boundary_types=boundary_types,
            bc_values=bc_values,
            ic_values=ic_values,
            conservation_type=conservation_type,
            use_topology_propagation=use_topology_propagation
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            pred: Predicted solution [num_nodes]
        """
        return self.gnn(x, edge_index).squeeze()

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        feats: torch.Tensor,
        A_csr: Dict,
        b: torch.Tensor,
        epoch: int,
        ic_mask: Optional[torch.Tensor] = None,
        bc_values: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute unsupervised loss.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            feats: Node features for loss computation
            A_csr: System matrix in CSR format
            b: RHS vector [num_nodes]
            epoch: Current epoch
            ic_mask: Initial condition mask (optional)
            bc_values: BC values override (optional)

        Returns:
            loss: Total loss (scalar)
            info: Loss information dict
        """
        # Forward pass
        pred = self.forward(x, edge_index)

        # Compute loss (no supervised data)
        loss, info = self.loss_fn(
            pred=pred,
            feats=feats,
            A_csr=A_csr,
            b=b,
            edge_index=edge_index,
            epoch=epoch,
            ic_mask=ic_mask,
            bc_values=bc_values
        )

        return loss, info

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference mode prediction.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            pred: Predicted solution [num_nodes]
        """
        self.eval()
        with torch.no_grad():
            pred = self.forward(x, edge_index)
        return pred


class PUPHAW(nn.Module):
    """
    PUP-HAW: Supervised version with mesh quality weighting.

    Parameters
    ----------
    in_channels : int
        Input feature dimension
    hidden_channels : int, optional
        Hidden layer size (default: 64)
    num_layers : int, optional
        Number of GNN layers (default: 4)
    mesh_quality_config : dict, optional
        Mesh quality weight configuration
    hierarchical_config : dict, optional
        Hierarchical adaptive configuration
    use_topology_propagation : bool, optional
        Use topology-aware weight propagation (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        mesh_quality_config: Optional[Dict] = None,
        hierarchical_config: Optional[Dict] = None,
        use_topology_propagation: bool = True
    ):
        super().__init__()

        # GNN backbone
        self.gnn = SimpleSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )

        # Loss function (supervised)
        self.loss_fn = PUPHAWLoss(
            mesh_quality_config=mesh_quality_config,
            hierarchical_config=hierarchical_config,
            use_topology_propagation=use_topology_propagation
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.gnn(x, edge_index).squeeze()

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        feats: torch.Tensor,
        A_csr: Dict,
        b: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute supervised loss.

        Args:
            x: Node features
            edge_index: Edge connectivity
            target: Ground truth solution
            feats: Node features for loss
            A_csr: System matrix
            b: RHS vector
            epoch: Current epoch

        Returns:
            loss: Total loss
            info: Loss information
        """
        pred = self.forward(x, edge_index)

        loss, info = self.loss_fn(
            pred=pred,
            target=target,
            feats=feats,
            A_csr=A_csr,
            b=b,
            edge_index=edge_index,
            epoch=epoch
        )

        return loss, info

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            pred = self.forward(x, edge_index)
        return pred


class PUPHAWHybrid(nn.Module):
    """
    PUP-HAW Hybrid: Curriculum learning from supervised to unsupervised.

    Parameters
    ----------
    in_channels : int
        Input feature dimension
    hidden_channels : int, optional
        Hidden layer size (default: 64)
    num_layers : int, optional
        Number of GNN layers (default: 4)
    boundary_types : dict, optional
        Boundary condition types
    bc_values : dict, optional
        Boundary condition values
    conservation_type : str, optional
        Conservation law type
    mesh_quality_config : dict, optional
        Mesh quality configuration
    hierarchical_config : dict, optional
        Hierarchical adaptive configuration
    use_topology_propagation : bool, optional
        Use topology propagation (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        boundary_types: Optional[Dict] = None,
        bc_values: Optional[Dict] = None,
        conservation_type: str = 'mass',
        mesh_quality_config: Optional[Dict] = None,
        hierarchical_config: Optional[Dict] = None,
        use_topology_propagation: bool = True
    ):
        super().__init__()

        # GNN backbone
        self.gnn = SimpleSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )

        # Loss function (hybrid)
        self.loss_fn = PUPHAWHybridLoss(
            mesh_quality_config=mesh_quality_config,
            hierarchical_config=hierarchical_config,
            boundary_types=boundary_types,
            bc_values=bc_values,
            conservation_type=conservation_type,
            use_topology_propagation=use_topology_propagation
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.gnn(x, edge_index).squeeze()

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        feats: torch.Tensor,
        A_csr: Dict,
        b: torch.Tensor,
        epoch: int,
        lambda_data: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute hybrid loss with curriculum weighting.

        Args:
            x: Node features
            edge_index: Edge connectivity
            target: Ground truth solution
            feats: Node features for loss
            A_csr: System matrix
            b: RHS vector
            epoch: Current epoch
            lambda_data: Curriculum weight for supervised loss

        Returns:
            loss: Total loss
            info: Loss information
        """
        pred = self.forward(x, edge_index)

        loss, info = self.loss_fn(
            pred=pred,
            target=target,
            feats=feats,
            A_csr=A_csr,
            b=b,
            edge_index=edge_index,
            epoch=epoch,
            lambda_data=lambda_data
        )

        return loss, info

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            pred = self.forward(x, edge_index)
        return pred
