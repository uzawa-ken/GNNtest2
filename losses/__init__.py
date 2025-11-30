"""
Losses module

損失関数とメッシュ品質重み計算
"""

from .mesh_quality_weights import (
    build_w_pde_from_feats,
    get_mesh_quality_reference_values,
    get_mesh_quality_weight_coefficients
)

from .physics_based_weights import (
    PhysicsBasedMeshQualityWeight,
    LearnableMeshQualityWeight,
    compute_adaptive_reference_values
)

__all__ = [
    # ベースライン（既存実装）
    'build_w_pde_from_feats',
    'get_mesh_quality_reference_values',
    'get_mesh_quality_weight_coefficients',

    # Phase 2: 物理的不確実性伝播
    'PhysicsBasedMeshQualityWeight',
    'LearnableMeshQualityWeight',
    'compute_adaptive_reference_values'
]
