#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physics-Based Uncertainty Propagation Weights

物理的不確実性伝播に基づくメッシュ品質重み計算

新規性:
- メッシュ品質と解の曲率を結合した重み計算
- 誤差増幅理論に基づく物理的根拠
- トポロジー認識型の重み伝播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsBasedMeshQualityWeight(nn.Module):
    """
    物理的不確実性伝播に基づくメッシュ品質重み計算

    CFDの数値誤差理論より：
        ε_total ≈ ε_discretization + ε_mesh_quality × ||∇²φ||

    メッシュ品質の悪化は、解の曲率（2階微分）に比例して誤差を増幅します。
    この物理的メカニズムを重み計算に組み込みます。

    Parameters
    ----------
    ref_values : dict, optional
        基準値の辞書（デフォルト: CFD標準値）
    weight_coefficients : dict, optional
        重み係数の辞書（デフォルト: 既存実装と同じ）
    beta : float, optional
        べき乗の指数（デフォルト: 2.0）
    w_pde_max : float, optional
        重みの最大値（デフォルト: 20.0）
    use_curvature : bool, optional
        解の曲率を考慮するか（デフォルト: True）
    curvature_scale : float, optional
        曲率の影響スケール（デフォルト: 1.0）
    """

    def __init__(
        self,
        ref_values=None,
        weight_coefficients=None,
        beta=2.0,
        w_pde_max=20.0,
        use_curvature=True,
        curvature_scale=1.0
    ):
        super().__init__()

        # 基準値（Phase 2で適応的に決定予定）
        if ref_values is None:
            ref_values = {
                'skew': 0.5,       # 0.2 → 0.5（より現実的）
                'non_ortho': 50.0,  # 10.0 → 50.0
                'aspect': 5.0,
                'size_jump': 2.0    # 1.5 → 2.0
            }

        # 重み係数（Phase 2で学習可能パラメータ化予定）
        if weight_coefficients is None:
            weight_coefficients = {
                'skew': 1.0,
                'non_ortho': 0.5,
                'aspect': 0.5,
                'size_jump': 1.0
            }

        self.ref_values = ref_values
        self.weight_coefficients = weight_coefficients
        self.beta = beta
        self.w_pde_max = w_pde_max
        self.use_curvature = use_curvature
        self.curvature_scale = curvature_scale

    def compute_mesh_quality_factor(self, feats):
        """
        メッシュ品質による誤差増幅係数を計算

        Parameters
        ----------
        feats : torch.Tensor, shape (num_nodes, num_features)
            ノード特徴量

        Returns
        -------
        torch.Tensor, shape (num_nodes,)
            誤差増幅係数
        """
        # メトリクス抽出
        skew = feats[:, 5]
        non_ortho = feats[:, 6]
        aspect = feats[:, 7]
        size_jump = feats[:, 11]

        # 正規化（基準値で割る）
        q_skew = torch.clamp(skew / self.ref_values['skew'], 0.0, 5.0)
        q_non_ortho = torch.clamp(non_ortho / self.ref_values['non_ortho'], 0.0, 5.0)
        q_aspect = torch.clamp(aspect / self.ref_values['aspect'], 0.0, 5.0)
        q_size_jump = torch.clamp(size_jump / self.ref_values['size_jump'], 0.0, 5.0)

        # 誤差増幅係数（べき乗で非線形効果を表現）
        # メッシュ品質が悪いほど、指数関数的に誤差が増大
        amplification = (
            1.0
            + self.weight_coefficients['skew'] * torch.pow(
                torch.clamp(q_skew - 1.0, min=0.0), self.beta
            )
            + self.weight_coefficients['non_ortho'] * torch.pow(
                torch.clamp(q_non_ortho - 1.0, min=0.0), self.beta
            )
            + self.weight_coefficients['aspect'] * torch.pow(
                torch.clamp(q_aspect - 1.0, min=0.0), self.beta
            )
            + self.weight_coefficients['size_jump'] * torch.pow(
                torch.clamp(q_size_jump - 1.0, min=0.0), self.beta
            )
        )

        return amplification

    def forward(self, feats, pred, edge_index):
        """
        物理的不確実性伝播に基づく重みを計算

        Parameters
        ----------
        feats : torch.Tensor, shape (num_nodes, num_features)
            ノード特徴量
        pred : torch.Tensor, shape (num_nodes,)
            予測値（解の場）
        edge_index : torch.Tensor, shape (2, num_edges)
            エッジリスト

        Returns
        -------
        torch.Tensor, shape (num_nodes,)
            各ノードの重み [1.0, w_pde_max]
        """
        from utils.graph_ops import compute_graph_laplacian

        # メッシュ品質による誤差増幅係数
        amplification = self.compute_mesh_quality_factor(feats)

        if self.use_curvature and pred is not None:
            # 解の曲率（ラプラシアン）を計算
            laplacian = compute_graph_laplacian(pred, edge_index)
            curvature = torch.abs(laplacian)

            # 物理的不確実性重み
            # 高曲率（複雑な解） × 低品質（誤差増幅大）= 高重み
            uncertainty_weight = 1.0 + amplification * (
                1.0 + self.curvature_scale * curvature
            )
        else:
            # 曲率を使わない場合（ベースライン）
            uncertainty_weight = amplification

        # 正規化（統計的安定性のため）
        w_normalized = uncertainty_weight / (uncertainty_weight.mean() + 1e-6)

        # 最終的な重み（クリッピング）
        w_final = torch.clamp(w_normalized, 1.0, self.w_pde_max)

        return w_final


def compute_adaptive_reference_values(feats_all_cases, percentile=75):
    """
    データセット全体の統計から基準値を適応的に決定

    Parameters
    ----------
    feats_all_cases : torch.Tensor or list of torch.Tensor
        全ケースの特徴量
    percentile : float, optional
        基準値を決定するパーセンタイル（デフォルト: 75）

    Returns
    -------
    dict
        適応的基準値の辞書

    Examples
    --------
    >>> feats_list = [torch.randn(100, 13) for _ in range(10)]
    >>> ref_values = compute_adaptive_reference_values(feats_list)
    >>> print(ref_values)
    {'skew': 0.45, 'non_ortho': 48.2, ...}
    """
    # 全ケースを結合
    if isinstance(feats_all_cases, list):
        feats_all = torch.cat(feats_all_cases, dim=0)
    else:
        feats_all = feats_all_cases

    # 各メトリクスのパーセンタイル
    skew_all = feats_all[:, 5]
    non_ortho_all = feats_all[:, 6]
    aspect_all = feats_all[:, 7]
    size_jump_all = feats_all[:, 11]

    # パーセンタイル計算
    skew_ref = torch.quantile(skew_all, percentile / 100.0).item()
    non_ortho_ref = torch.quantile(non_ortho_all, percentile / 100.0).item()
    aspect_ref = torch.quantile(aspect_all, percentile / 100.0).item()
    size_jump_ref = torch.quantile(size_jump_all, percentile / 100.0).item()

    # CFD標準との比較（安全のため）
    import numpy as np
    cfd_standards = {
        'skew': 0.5,
        'non_ortho': 50.0,
        'aspect': 5.0,
        'size_jump': 2.0
    }

    # 標準値より厳しくならないように制限
    return {
        'skew': min(skew_ref, cfd_standards['skew']),
        'non_ortho': min(non_ortho_ref, cfd_standards['non_ortho']),
        'aspect': aspect_ref,  # アスペクト比は固定
        'size_jump': size_jump_ref
    }


class LearnableMeshQualityWeight(nn.Module):
    """
    学習可能なメッシュ品質重みパラメータ

    新規性: 重み係数とべき乗の指数を学習可能パラメータ化

    Parameters
    ----------
    ref_values : dict, optional
        基準値（固定または適応的）
    init_alpha : dict, optional
        重み係数の初期値
    init_beta : float, optional
        べき乗指数の初期値
    init_w_max : float, optional
        最大重みの初期値
    """

    def __init__(
        self,
        ref_values=None,
        init_alpha=None,
        init_beta=2.0,
        init_w_max=20.0
    ):
        super().__init__()

        if ref_values is None:
            ref_values = {
                'skew': 0.5,
                'non_ortho': 50.0,
                'aspect': 5.0,
                'size_jump': 2.0
            }
        self.ref_values = ref_values

        # 学習可能な重み係数（softplus経由で正の値を保証）
        if init_alpha is None:
            init_alpha = {'skew': 1.0, 'non_ortho': 0.5, 'aspect': 0.5, 'size_jump': 1.0}

        self.alpha_skew_logit = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_alpha['skew']))
        )
        self.alpha_nonorth_logit = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_alpha['non_ortho']))
        )
        self.alpha_aspect_logit = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_alpha['aspect']))
        )
        self.alpha_sizejump_logit = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_alpha['size_jump']))
        )

        # 学習可能なべき乗指数（1.0〜3.0の範囲）
        self.beta_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid経由

        # 学習可能な最大重み（1.0〜20.0の範囲）
        self.w_max_logit = nn.Parameter(torch.tensor(3.0))  # sigmoid経由

    @staticmethod
    def _inverse_softplus(x, beta=1.0):
        """softplusの逆関数（初期化用）"""
        import numpy as np
        return np.log(np.exp(x * beta) - 1.0) / beta

    def get_alpha(self):
        """学習可能な重み係数を取得（正の値を保証）"""
        return {
            'skew': F.softplus(self.alpha_skew_logit),
            'non_ortho': F.softplus(self.alpha_nonorth_logit),
            'aspect': F.softplus(self.alpha_aspect_logit),
            'size_jump': F.softplus(self.alpha_sizejump_logit)
        }

    def get_beta(self):
        """学習可能なべき乗指数を取得（1.0〜3.0）"""
        return 1.0 + 2.0 * torch.sigmoid(self.beta_logit)

    def get_w_max(self):
        """学習可能な最大重みを取得（1.0〜20.0）"""
        return 1.0 + 19.0 * torch.sigmoid(self.w_max_logit)

    def forward(self, feats, pred=None, edge_index=None):
        """
        学習可能パラメータを使った重み計算

        Parameters
        ----------
        feats : torch.Tensor
            ノード特徴量
        pred : torch.Tensor, optional
            予測値（曲率計算用）
        edge_index : torch.Tensor, optional
            エッジリスト（曲率計算用）

        Returns
        -------
        torch.Tensor
            重み
        """
        # 学習可能パラメータを取得
        alpha = self.get_alpha()
        beta = self.get_beta()
        w_max = self.get_w_max()

        # PhysicsBasedMeshQualityWeightを使用
        weight_calculator = PhysicsBasedMeshQualityWeight(
            ref_values=self.ref_values,
            weight_coefficients=alpha,
            beta=beta.item(),
            w_pde_max=w_max.item(),
            use_curvature=(pred is not None and edge_index is not None)
        )

        return weight_calculator(feats, pred, edge_index)
