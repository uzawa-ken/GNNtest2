#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated Loss Function

PUP-HAWの統合損失関数

Phase 2とPhase 3の全機能を統合：
- 物理的不確実性伝播に基づくメッシュ品質重み（Phase 2）
- トポロジー認識型重み伝播（Phase 2）
- 階層的適応重み付け（Phase 3）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_based_weights import PhysicsBasedMeshQualityWeight
from .hierarchical_adaptive import HierarchicalAdaptiveWeighting
from utils.graph_ops import topology_aware_weight_propagation
from utils.sparse_ops import matvec_csr_torch


class PUPHAWLoss(nn.Module):
    """
    Physics-based Uncertainty Propagation with
    Hierarchical Adaptive Weighting (PUP-HAW) Loss

    統合損失関数（教師あり学習版）

    Parameters
    ----------
    mesh_quality_config : dict, optional
        メッシュ品質重み設定
    hierarchical_config : dict, optional
        階層的適応設定
    use_topology_propagation : bool, optional
        トポロジー伝播を使用するか（デフォルト: True）
    topology_config : dict, optional
        トポロジー伝播設定

    Examples
    --------
    >>> loss_fn = PUPHAWLoss()
    >>> for epoch in range(num_epochs):
    ...     pred = model(x, edge_index)
    ...     loss, info = loss_fn(
    ...         pred, target, feats, A_csr, b, edge_index, epoch
    ...     )
    ...     loss.backward()
    """

    def __init__(
        self,
        mesh_quality_config=None,
        hierarchical_config=None,
        use_topology_propagation=True,
        topology_config=None
    ):
        super().__init__()

        # メッシュ品質重み計算（Phase 2）
        if mesh_quality_config is None:
            mesh_quality_config = {}

        self.mesh_quality_weight = PhysicsBasedMeshQualityWeight(
            **mesh_quality_config
        )

        # 階層的適応重み付け（Phase 3）
        if hierarchical_config is None:
            hierarchical_config = {}

        self.hierarchical_adaptive = HierarchicalAdaptiveWeighting(
            constraint_types=['data', 'pde'],
            **hierarchical_config
        )

        # トポロジー伝播設定
        self.use_topology_propagation = use_topology_propagation

        if topology_config is None:
            topology_config = {
                'num_hops': 2,
                'decay_factor': 0.5,
                'aggregation': 'max'
            }
        self.topology_config = topology_config

    def forward(
        self,
        pred,
        target,
        feats,
        A_csr,
        b,
        edge_index,
        epoch,
        return_gradients=False
    ):
        """
        損失を計算

        Parameters
        ----------
        pred : torch.Tensor, shape (num_nodes,)
            予測値
        target : torch.Tensor, shape (num_nodes,)
            正解データ
        feats : torch.Tensor, shape (num_nodes, num_features)
            ノード特徴量
        A_csr : dict
            CSR形式のシステム行列 {'row_ptr', 'col_ind', 'vals', 'row_idx'}
        b : torch.Tensor, shape (num_nodes,)
            右辺ベクトル
        edge_index : torch.Tensor, shape (2, num_edges)
            エッジリスト
        epoch : int
            現在のエポック
        return_gradients : bool, optional
            勾配を返すか（Level 2適応用）

        Returns
        -------
        loss : torch.Tensor
            総損失
        info : dict
            損失の詳細情報
        """
        # データ損失（MSE）
        loss_data = F.mse_loss(pred, target)

        # PDE残差損失（メッシュ品質重み付き）
        # Level 3: セル単位の物理的重み
        w_cell = self.mesh_quality_weight(feats, pred, edge_index)

        # トポロジー認識型伝播（オプション）
        if self.use_topology_propagation:
            w_cell = topology_aware_weight_propagation(
                w_cell,
                edge_index,
                **self.topology_config
            )

        # PDE残差: Ax̂ - b
        residual = matvec_csr_torch(
            A_csr['row_ptr'],
            A_csr['col_ind'],
            A_csr['vals'],
            A_csr['row_idx'],
            pred
        ) - b

        loss_pde = (w_cell * residual ** 2).mean()

        # Level 1: エポック単位の適応
        current_losses = {
            'data': loss_data.item(),
            'pde': loss_pde.item()
        }
        self.hierarchical_adaptive.update_level1_epoch(epoch, current_losses)

        # Level 2: バッチ単位の勾配調和（オプション）
        if return_gradients:
            # 勾配を計算
            grad_data = torch.autograd.grad(
                loss_data, pred, retain_graph=True, create_graph=False
            )[0]
            grad_pde = torch.autograd.grad(
                loss_pde, pred, retain_graph=True, create_graph=False
            )[0]

            gradients = {
                'data': grad_data.flatten(),
                'pde': grad_pde.flatten()
            }

            self.hierarchical_adaptive.update_level2_batch(gradients)

        # 最終的な重み（Level 1 & 2を統合）
        lambdas = self.hierarchical_adaptive.get_lambdas()

        # 総損失
        loss_total = lambdas['data'] * loss_data + lambdas['pde'] * loss_pde

        # 情報を返す
        info = {
            'loss_data': loss_data.item(),
            'loss_pde': loss_pde.item(),
            'loss_total': loss_total.item(),
            'lambda_data': lambdas['data'],
            'lambda_pde': lambdas['pde'],
            'w_cell_mean': w_cell.mean().item(),
            'w_cell_max': w_cell.max().item(),
            'w_cell_min': w_cell.min().item(),
            'stats': self.hierarchical_adaptive.get_stats()
        }

        return loss_total, info


class PUPHAWUnsupervisedLoss(nn.Module):
    """
    PUP-HAW Unsupervised Loss

    完全教師なし学習版（Phase 4-6で使用）

    教師データを使わず、物理制約のみで学習します。
    Phase 4のマルチ物理制約を統合しています。

    Parameters
    ----------
    mesh_quality_config : dict, optional
        メッシュ品質重み設定
    hierarchical_config : dict, optional
        階層的適応設定
    boundary_types : dict, optional
        境界条件タイプ情報 {'inlet': mask, 'outlet': mask, 'wall': mask, ...}
    bc_values : dict, optional
        境界条件値 {'inlet': value, 'outlet': value, ...}
    ic_values : torch.Tensor, optional
        初期条件値
    conservation_type : str, optional
        保存則タイプ ('mass', 'momentum', 'energy')

    Examples
    --------
    >>> from utils.mesh_analysis import extract_boundary_nodes, classify_boundary_types
    >>> boundary_mask = extract_boundary_nodes(edge_index, num_nodes)
    >>> boundary_types = classify_boundary_types(edge_index, boundary_mask)
    >>> bc_values = {'inlet': 0.1, 'outlet': 0.0}
    >>> loss_fn = PUPHAWUnsupervisedLoss(
    ...     boundary_types=boundary_types,
    ...     bc_values=bc_values
    ... )
    >>> for epoch in range(num_epochs):
    ...     pred = model(x, edge_index)
    ...     loss, info = loss_fn(
    ...         pred, feats, A_csr, b, edge_index, epoch
    ...     )
    ...     loss.backward()
    """

    def __init__(
        self,
        mesh_quality_config=None,
        hierarchical_config=None,
        boundary_types=None,
        bc_values=None,
        ic_values=None,
        conservation_type='mass',
        use_topology_propagation=True,
        topology_config=None
    ):
        super().__init__()

        # メッシュ品質重み（Phase 2）
        if mesh_quality_config is None:
            mesh_quality_config = {}

        self.mesh_quality_weight = PhysicsBasedMeshQualityWeight(
            **mesh_quality_config
        )

        # 階層的適応（Phase 3、マルチ物理制約版）
        from .hierarchical_adaptive import MultiPhysicsHierarchicalAdaptiveWeighting

        if hierarchical_config is None:
            hierarchical_config = {}

        self.hierarchical_adaptive = MultiPhysicsHierarchicalAdaptiveWeighting(
            **hierarchical_config
        )

        # Phase 4: マルチ物理制約損失
        from .multi_physics_loss import (
            BoundaryConditionLoss,
            InitialConditionLoss,
            ConservationLoss
        )

        # デフォルトの境界タイプ（空のマスク）
        if boundary_types is None:
            num_nodes = 1  # ダミー、実際は forward で更新
            device = torch.device('cpu')
            boundary_types = {
                'inlet': torch.zeros(num_nodes, dtype=torch.bool, device=device),
                'outlet': torch.zeros(num_nodes, dtype=torch.bool, device=device),
                'wall': torch.zeros(num_nodes, dtype=torch.bool, device=device),
                'symmetry': torch.zeros(num_nodes, dtype=torch.bool, device=device),
            }

        self.bc_loss = BoundaryConditionLoss(boundary_types, bc_values)
        self.ic_loss = InitialConditionLoss(ic_values)
        self.cons_loss = ConservationLoss(conservation_type)

        # トポロジー伝播
        self.use_topology_propagation = use_topology_propagation
        if topology_config is None:
            topology_config = {'num_hops': 2, 'decay_factor': 0.5}
        self.topology_config = topology_config

    def forward(
        self,
        pred,
        feats,
        A_csr,
        b,
        edge_index,
        epoch,
        ic_mask=None,
        bc_values=None,
        return_gradients=False
    ):
        """
        教師なし損失を計算（Phase 4統合版）

        Parameters
        ----------
        pred : torch.Tensor
            予測値（教師データなし）
        feats : torch.Tensor
            ノード特徴量
        A_csr : dict
            CSR行列
        b : torch.Tensor
            右辺ベクトル
        edge_index : torch.Tensor
            エッジリスト
        epoch : int
            エポック
        ic_mask : torch.Tensor, optional
            初期条件マスク [num_nodes]
        bc_values : dict, optional
            境界条件値 (オプション、デフォルト値を上書き)
        return_gradients : bool, optional
            勾配を返すか

        Returns
        -------
        loss : torch.Tensor
            総損失
        info : dict
            損失情報
        """
        # Level 3: セル単位のメッシュ品質重み（Phase 2）
        w_cell = self.mesh_quality_weight(feats, pred, edge_index)

        if self.use_topology_propagation:
            w_cell = topology_aware_weight_propagation(
                w_cell, edge_index, **self.topology_config
            )

        # PDE残差損失（重み付き）
        residual = matvec_csr_torch(
            A_csr['row_ptr'], A_csr['col_ind'],
            A_csr['vals'], A_csr['row_idx'], pred
        ) - b
        loss_pde = (w_cell * residual ** 2).mean()

        # Phase 4: マルチ物理制約損失

        # 境界条件損失（Dirichlet & Neumann）
        loss_bc, bc_dict = self.bc_loss(pred, edge_index, bc_values)

        # 初期条件損失
        if ic_mask is not None and ic_mask.sum() > 0:
            loss_ic = self.ic_loss(pred, ic_mask)
        else:
            loss_ic = torch.tensor(0.0, device=pred.device)

        # 保存則損失（質量保存など）
        loss_conservation, cons_dict = self.cons_loss(pred, edge_index, feats)

        # Level 1: エポック単位適応（Phase 3）
        current_losses = {
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item(),
            'conservation': loss_conservation.item()
        }
        self.hierarchical_adaptive.update_level1_epoch(epoch, current_losses)

        # Level 2: バッチ単位勾配調和（オプション）
        if return_gradients:
            # 各制約の勾配を計算
            gradients = {}
            for name, loss_val in [
                ('pde', loss_pde),
                ('bc', loss_bc),
                ('ic', loss_ic),
                ('conservation', loss_conservation)
            ]:
                if loss_val.requires_grad:
                    grad = torch.autograd.grad(
                        loss_val, pred, retain_graph=True, create_graph=False
                    )[0]
                    gradients[name] = grad.flatten()

            if len(gradients) > 1:
                self.hierarchical_adaptive.update_level2_batch(gradients)

        # 最終的な適応重み（Level 1 & 2統合）
        lambdas = self.hierarchical_adaptive.get_lambdas()

        # 総損失
        loss_total = (
            lambdas['pde'] * loss_pde +
            lambdas['bc'] * loss_bc +
            lambdas['ic'] * loss_ic +
            lambdas['conservation'] * loss_conservation
        )

        # 詳細情報
        info = {
            'loss_pde': loss_pde.item(),
            'loss_bc': loss_bc.item(),
            'loss_ic': loss_ic.item(),
            'loss_conservation': loss_conservation.item(),
            'loss_total': loss_total.item(),
            'lambdas': lambdas,
            'w_cell_mean': w_cell.mean().item(),
            'w_cell_max': w_cell.max().item(),
            'w_cell_min': w_cell.min().item(),
            'stats': self.hierarchical_adaptive.get_stats()
        }

        # 境界条件と保存則の詳細も追加
        info.update({f'bc_{k}': v.item() for k, v in bc_dict.items()})
        info.update({f'cons_{k}': v.item() for k, v in cons_dict.items()})

        return loss_total, info
