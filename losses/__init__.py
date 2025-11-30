"""
Losses module

1¢ph·√∑Â¡ÍÕó
"""

from .mesh_quality_weights import (
    build_w_pde_from_feats,
    get_mesh_quality_reference_values,
    get_mesh_quality_weight_coefficients
)

__all__ = [
    'build_w_pde_from_feats',
    'get_mesh_quality_reference_values',
    'get_mesh_quality_weight_coefficients'
]
