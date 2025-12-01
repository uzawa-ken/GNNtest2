"""
Utilities module

データローディングとユーティリティ関数
"""

from .data_loader import find_time_list, load_case_with_csr
from .sparse_ops import matvec_csr_torch
from .graph_ops import (
    compute_graph_laplacian,
    compute_graph_gradient,
    compute_gradient_magnitude,
    topology_aware_weight_propagation,
    compute_node_degree
)
from .mesh_analysis import (
    extract_boundary_nodes,
    classify_boundary_types,
    extract_initial_condition_nodes,
    compute_mesh_quality_metrics,
    analyze_mesh_topology
)
from .scheduling import (
    CurriculumScheduler,
    AdaptiveCurriculumScheduler,
    ScheduleType,
    get_recommended_schedule
)
from .monitoring import (
    TrainingMonitor,
    compare_schedules
)
from .evaluation import (
    PhysicsEvaluator,
    BaselineComparator,
    evaluate_model_comprehensive
)

__all__ = [
    'find_time_list',
    'load_case_with_csr',
    'matvec_csr_torch',
    'compute_graph_laplacian',
    'compute_graph_gradient',
    'compute_gradient_magnitude',
    'topology_aware_weight_propagation',
    'compute_node_degree',
    'extract_boundary_nodes',
    'classify_boundary_types',
    'extract_initial_condition_nodes',
    'compute_mesh_quality_metrics',
    'analyze_mesh_topology',
    'CurriculumScheduler',
    'AdaptiveCurriculumScheduler',
    'ScheduleType',
    'get_recommended_schedule',
    'TrainingMonitor',
    'compare_schedules',
    'PhysicsEvaluator',
    'BaselineComparator',
    'evaluate_model_comprehensive'
]
