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

from .hierarchical_adaptive import (
    HierarchicalAdaptiveWeighting,
    MultiPhysicsHierarchicalAdaptiveWeighting,
    compute_gradient_statistics
)

from .integrated_loss import (
    PUPHAWLoss,
    PUPHAWUnsupervisedLoss,
    PUPHAWHybridLoss
)

from .multi_physics_loss import (
    BoundaryConditionLoss,
    InitialConditionLoss,
    ConservationLoss,
    MultiPhysicsLoss
)

__all__ = [
    # ベースライン（既存実装）
    'build_w_pde_from_feats',
    'get_mesh_quality_reference_values',
    'get_mesh_quality_weight_coefficients',

    # Phase 2: 物理的不確実性伝播
    'PhysicsBasedMeshQualityWeight',
    'LearnableMeshQualityWeight',
    'compute_adaptive_reference_values',

    # Phase 3: 階層的適応
    'HierarchicalAdaptiveWeighting',
    'MultiPhysicsHierarchicalAdaptiveWeighting',
    'compute_gradient_statistics',

    # Phase 4: マルチ物理制約
    'BoundaryConditionLoss',
    'InitialConditionLoss',
    'ConservationLoss',
    'MultiPhysicsLoss',

    # 統合損失関数
    'PUPHAWLoss',
    'PUPHAWUnsupervisedLoss',
    'PUPHAWHybridLoss'
]
