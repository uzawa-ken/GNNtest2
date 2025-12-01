"""
Mesh analysis utilities for boundary detection and mesh quality analysis.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List


def extract_boundary_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    coords: Optional[torch.Tensor] = None,
    method: str = 'degree'
) -> torch.Tensor:
    """
    Extract boundary nodes from graph structure.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Total number of nodes
        coords: Node coordinates [num_nodes, 3] (optional, for geometric detection)
        method: Detection method ('degree', 'geometric', 'hybrid')

    Returns:
        boundary_mask: Boolean mask [num_nodes] indicating boundary nodes

    Note:
        - 'degree': Detects nodes with lower degree (fewer neighbors)
        - 'geometric': Uses spatial coordinates to find domain boundaries
        - 'hybrid': Combines both methods
    """
    if method == 'degree':
        return _extract_boundary_by_degree(edge_index, num_nodes)
    elif method == 'geometric' and coords is not None:
        return _extract_boundary_by_geometry(coords)
    elif method == 'hybrid' and coords is not None:
        mask_degree = _extract_boundary_by_degree(edge_index, num_nodes)
        mask_geom = _extract_boundary_by_geometry(coords)
        return mask_degree | mask_geom
    else:
        # Default to degree-based method
        return _extract_boundary_by_degree(edge_index, num_nodes)


def _extract_boundary_by_degree(
    edge_index: torch.Tensor,
    num_nodes: int,
    threshold_percentile: float = 25.0
) -> torch.Tensor:
    """
    Detect boundary nodes by analyzing node degree distribution.

    Boundary nodes typically have fewer neighbors than interior nodes.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Total number of nodes
        threshold_percentile: Percentile threshold for low degree (default: 25%)

    Returns:
        boundary_mask: Boolean mask [num_nodes]
    """
    # Compute node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))

    # Nodes with degree below threshold are considered boundary
    threshold = torch.quantile(degrees.float(), threshold_percentile / 100.0)
    boundary_mask = degrees <= threshold

    return boundary_mask


def _extract_boundary_by_geometry(
    coords: torch.Tensor,
    epsilon: float = 1e-3
) -> torch.Tensor:
    """
    Detect boundary nodes by geometric analysis.

    Nodes near domain boundaries (min/max coordinates) are boundary nodes.

    Args:
        coords: Node coordinates [num_nodes, 3]
        epsilon: Tolerance for boundary detection (relative to domain size)

    Returns:
        boundary_mask: Boolean mask [num_nodes]
    """
    # Find domain bounds for each dimension
    min_coords = coords.min(dim=0).values
    max_coords = coords.max(dim=0).values
    domain_size = max_coords - min_coords

    # Detect nodes near boundaries
    boundary_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)

    for dim in range(coords.shape[1]):
        tol = epsilon * domain_size[dim]
        near_min = torch.abs(coords[:, dim] - min_coords[dim]) < tol
        near_max = torch.abs(coords[:, dim] - max_coords[dim]) < tol
        boundary_mask |= (near_min | near_max)

    return boundary_mask


def classify_boundary_types(
    edge_index: torch.Tensor,
    boundary_mask: torch.Tensor,
    coords: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Classify boundary nodes into different types (inlet, outlet, wall, etc.).

    Args:
        edge_index: Edge connectivity [2, num_edges]
        boundary_mask: Boolean mask [num_nodes] indicating boundary nodes
        coords: Node coordinates [num_nodes, 3] (optional)

    Returns:
        boundary_types: Dictionary with keys:
            - 'inlet': Mask for inlet boundary
            - 'outlet': Mask for outlet boundary
            - 'wall': Mask for wall boundary
            - 'symmetry': Mask for symmetry boundary

    Note:
        If coords is not provided, all boundary nodes are classified as 'wall'.
        With coords, we use geometric heuristics (e.g., x_min=inlet, x_max=outlet).
    """
    num_nodes = boundary_mask.shape[0]
    device = boundary_mask.device

    # Initialize all boundary types
    boundary_types = {
        'inlet': torch.zeros(num_nodes, dtype=torch.bool, device=device),
        'outlet': torch.zeros(num_nodes, dtype=torch.bool, device=device),
        'wall': torch.zeros(num_nodes, dtype=torch.bool, device=device),
        'symmetry': torch.zeros(num_nodes, dtype=torch.bool, device=device),
    }

    if coords is None:
        # Default: all boundaries are walls
        boundary_types['wall'] = boundary_mask
    else:
        # Geometric heuristic: assume flow direction along x-axis
        # x_min = inlet, x_max = outlet, others = wall
        x_coords = coords[:, 0]
        x_min = x_coords.min()
        x_max = x_coords.max()
        tol = 1e-3 * (x_max - x_min)

        at_x_min = torch.abs(x_coords - x_min) < tol
        at_x_max = torch.abs(x_coords - x_max) < tol

        boundary_types['inlet'] = boundary_mask & at_x_min
        boundary_types['outlet'] = boundary_mask & at_x_max
        boundary_types['wall'] = boundary_mask & ~at_x_min & ~at_x_max

    return boundary_types


def extract_initial_condition_nodes(
    time_index: int,
    num_nodes: int
) -> torch.Tensor:
    """
    Extract nodes where initial condition should be enforced.

    For time-dependent problems, returns all nodes at t=0.

    Args:
        time_index: Current time step index
        num_nodes: Total number of spatial nodes

    Returns:
        ic_mask: Boolean mask [num_nodes] indicating IC nodes
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if time_index == 0:
        # At t=0, all nodes should satisfy initial condition
        return torch.ones(num_nodes, dtype=torch.bool, device=device)
    else:
        # After t=0, initial condition is not enforced
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)


def compute_mesh_quality_metrics(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    feats: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive mesh quality metrics.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        coords: Node coordinates [num_nodes, 3]
        feats: Node features [num_nodes, num_feats] (optional, if already computed)

    Returns:
        metrics: Dictionary containing:
            - 'skewness': Cell skewness [num_nodes]
            - 'non_orthogonality': Non-orthogonality [num_nodes]
            - 'aspect_ratio': Aspect ratio [num_nodes]
            - 'volume': Cell volume [num_nodes]
            - 'quality_score': Overall quality score [num_nodes] (0=bad, 1=good)

    Note:
        If feats is provided and contains mesh quality columns, extract from there.
        Otherwise, compute approximate metrics from graph topology.
    """
    num_nodes = coords.shape[0]
    device = coords.device

    metrics = {}

    if feats is not None and feats.shape[1] >= 13:
        # Extract from features if available
        # Assuming feature columns: [x, y, z, p, skew, non_orth, ar, vol, ...]
        metrics['skewness'] = feats[:, 4] if feats.shape[1] > 4 else torch.zeros(num_nodes, device=device)
        metrics['non_orthogonality'] = feats[:, 5] if feats.shape[1] > 5 else torch.zeros(num_nodes, device=device)
        metrics['aspect_ratio'] = feats[:, 6] if feats.shape[1] > 6 else torch.zeros(num_nodes, device=device)
        metrics['volume'] = feats[:, 7] if feats.shape[1] > 7 else torch.ones(num_nodes, device=device)
    else:
        # Approximate metrics from topology
        degrees = torch.zeros(num_nodes, dtype=torch.float, device=device)
        degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))

        # Approximate quality based on degree uniformity
        mean_degree = degrees.mean()
        metrics['skewness'] = torch.abs(degrees - mean_degree) / (mean_degree + 1e-6)
        metrics['non_orthogonality'] = torch.zeros(num_nodes, device=device)
        metrics['aspect_ratio'] = torch.ones(num_nodes, device=device)
        metrics['volume'] = torch.ones(num_nodes, device=device)

    # Compute overall quality score (0=bad, 1=good)
    skew_score = 1.0 - torch.clamp(metrics['skewness'], 0, 1)
    orth_score = 1.0 - torch.clamp(metrics['non_orthogonality'], 0, 1)
    ar_score = 1.0 / (1.0 + metrics['aspect_ratio'])

    metrics['quality_score'] = (skew_score + orth_score + ar_score) / 3.0

    return metrics


def analyze_mesh_topology(
    edge_index: torch.Tensor,
    num_nodes: int
) -> Dict[str, any]:
    """
    Analyze graph topology statistics.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Total number of nodes

    Returns:
        stats: Dictionary containing:
            - 'num_nodes': Total nodes
            - 'num_edges': Total edges
            - 'avg_degree': Average node degree
            - 'min_degree': Minimum degree
            - 'max_degree': Maximum degree
            - 'boundary_ratio': Ratio of boundary nodes
    """
    # Compute degrees
    degrees = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))

    # Extract boundary
    boundary_mask = extract_boundary_nodes(edge_index, num_nodes, method='degree')

    stats = {
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'avg_degree': degrees.float().mean().item(),
        'min_degree': degrees.min().item(),
        'max_degree': degrees.max().item(),
        'boundary_ratio': boundary_mask.float().mean().item(),
    }

    return stats
