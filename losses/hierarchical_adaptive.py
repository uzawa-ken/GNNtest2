#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hierarchical Adaptive Weighting

階層的適応重み付けシステム

3層（または5層）の適応機構：
- Level 0: データ適応型基準値（前処理、Phase 2で実装済み）
- Level 1: エポック単位の大域的適応（このモジュール）
- Level 2: バッチ単位の勾配調和（このモジュール）
- Level 3: セル単位の物理的重み（Phase 2で実装済み）

新規性:
- 複数の時間スケールでの適応
- 減衰率不均衡の自動検出と調整
- 勾配競合の解決
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


class HierarchicalAdaptiveWeighting:
    """
    階層的適応重み付けシステム

    異なる時間スケールで動作する3層の適応機構を統合します。

    Parameters
    ----------
    constraint_types : list of str
        制約の種類（例: ['data', 'pde'] または ['pde', 'bc', 'ic', 'conservation']）
    init_lambdas : dict, optional
        各制約の初期重み
    adaptation_rate : float, optional
        適応速度（デフォルト: 0.1）
    history_window : int, optional
        履歴ウィンドウサイズ（デフォルト: 10）

    Examples
    --------
    >>> haw = HierarchicalAdaptiveWeighting(['data', 'pde'])
    >>> # 訓練ループ内で
    >>> lambdas = haw.get_lambdas()
    >>> loss = lambdas['data'] * loss_data + lambdas['pde'] * loss_pde
    """

    def __init__(
        self,
        constraint_types=['data', 'pde'],
        init_lambdas=None,
        adaptation_rate=0.1,
        history_window=10
    ):
        self.constraint_types = constraint_types

        # Level 1: エポック単位の大域的重み
        if init_lambdas is None:
            init_lambdas = {ct: 1.0 for ct in constraint_types}
        self.lambdas_global = {ct: init_lambdas.get(ct, 1.0) for ct in constraint_types}

        # Level 2: バッチ単位の調整係数
        self.alpha_batch = {ct: 1.0 for ct in constraint_types}

        # 損失履歴（Level 1用）
        self.loss_history = {ct: [] for ct in constraint_types}

        # パラメータ
        self.adaptation_rate = adaptation_rate
        self.history_window = history_window

        # 統計情報
        self.stats = {
            'epoch': 0,
            'adaptations': 0,
            'gradient_conflicts': 0
        }

    def update_level1_epoch(self, epoch, current_losses):
        """
        Level 1: エポック単位の大域的適応

        各制約の減衰率を監視し、不均衡を検出して重みを調整します。

        新規性:
        - 減衰率不均衡の検出（既存研究では未使用）
        - 自動的な重みバランシング

        Parameters
        ----------
        epoch : int
            現在のエポック
        current_losses : dict
            現在の各制約の損失値

        Notes
        -----
        減衰率不均衡の検出:
            decay_rate_i = (loss_old - loss_new) / loss_old

        不均衡条件:
            imbalance = decay_rate_i / decay_rate_j
            if imbalance > 2.0 or imbalance < 0.5:
                → 重みを調整
        """
        # 損失履歴を更新
        for ct in self.constraint_types:
            if ct in current_losses:
                self.loss_history[ct].append(current_losses[ct])

        # 履歴が十分にあれば適応
        if epoch >= self.history_window:
            decay_rates = {}

            for ct in self.constraint_types:
                history = self.loss_history[ct]
                if len(history) >= self.history_window:
                    # 過去history_windowエポックの減衰率
                    old_loss = history[-self.history_window]
                    new_loss = history[-1]

                    if old_loss > 1e-8:
                        decay_rate = (old_loss - new_loss) / old_loss
                        decay_rates[ct] = max(decay_rate, 0.0)  # 負の減衰率は0に
                    else:
                        decay_rates[ct] = 0.0

            # 減衰率の不均衡を検出
            if len(decay_rates) >= 2:
                min_decay = min(decay_rates.values()) + 1e-8
                max_decay = max(decay_rates.values())

                for ct, decay in decay_rates.items():
                    # 相対的な減衰率
                    relative_decay = decay / min_decay

                    if relative_decay < 0.5:  # 減衰が遅い（学習が遅れている）
                        # この制約の重みを増やす
                        self.lambdas_global[ct] *= (1.0 + self.adaptation_rate)
                        self.stats['adaptations'] += 1

                    elif relative_decay > 2.0:  # 減衰が速い（過学習の可能性）
                        # この制約の重みを減らす
                        self.lambdas_global[ct] *= (1.0 - self.adaptation_rate * 0.5)
                        self.stats['adaptations'] += 1

                # 正規化（合計を一定に保つ）
                total = sum(self.lambdas_global.values())
                for ct in self.lambdas_global.keys():
                    self.lambdas_global[ct] /= total

        self.stats['epoch'] = epoch

    def update_level2_batch(self, gradients):
        """
        Level 2: バッチ単位の勾配調和

        複数の制約からの勾配が競合している場合に調整します。

        新規性:
        - N個の制約間での勾配競合解決
        - コサイン類似度に基づく調和

        Parameters
        ----------
        gradients : dict of torch.Tensor
            各制約の勾配（1次元に平坦化）

        Notes
        -----
        勾配競合の検出:
            cos_sim = <grad_i, grad_j> / (||grad_i|| × ||grad_j||)
            if cos_sim < 0.2:
                → 競合あり、大きい方を抑制
        """
        if len(gradients) < 2:
            return  # 制約が1つしかない場合はスキップ

        # 勾配のノルムを計算
        grad_norms = {ct: g.norm().item() for ct, g in gradients.items()}

        # ペアワイズのコサイン類似度を計算
        conflicts = []
        grad_keys = list(gradients.keys())

        for i, ct1 in enumerate(grad_keys):
            for ct2 in grad_keys[i+1:]:
                # コサイン類似度
                g1 = gradients[ct1]
                g2 = gradients[ct2]

                cos_sim = F.cosine_similarity(
                    g1.unsqueeze(0),
                    g2.unsqueeze(0),
                    dim=1
                ).item()

                if cos_sim < 0.2:  # 競合検出（閾値は調整可能）
                    conflicts.append((ct1, ct2, cos_sim, grad_norms[ct1], grad_norms[ct2]))
                    self.stats['gradient_conflicts'] += 1

        # 競合がある場合、大きい勾配を抑制
        for ct1, ct2, sim, norm1, norm2 in conflicts:
            if norm1 > norm2:
                # ct1の勾配が大きい → 抑制
                self.alpha_batch[ct1] *= 0.9
            else:
                # ct2の勾配が大きい → 抑制
                self.alpha_batch[ct2] *= 0.9

        # alpha_batchを緩やかに1.0に戻す（指数移動平均）
        for ct in self.alpha_batch.keys():
            self.alpha_batch[ct] = 0.9 * self.alpha_batch[ct] + 0.1 * 1.0

    def get_lambdas(self):
        """
        現在の重みを取得（Level 1とLevel 2を統合）

        Returns
        -------
        dict
            各制約の最終的な重み

        Notes
        -----
        最終的な重み:
            lambda_final = lambda_global × alpha_batch
        """
        return {
            ct: self.lambdas_global[ct] * self.alpha_batch[ct]
            for ct in self.constraint_types
        }

    def get_stats(self):
        """
        統計情報を取得

        Returns
        -------
        dict
            統計情報
        """
        return self.stats.copy()

    def reset_batch_alphas(self):
        """バッチ単位の調整係数をリセット"""
        self.alpha_batch = {ct: 1.0 for ct in self.constraint_types}


class MultiPhysicsHierarchicalAdaptiveWeighting(HierarchicalAdaptiveWeighting):
    """
    マルチ物理制約用の階層的適応重み付け

    完全教師なし学習（Phase 6）で使用する拡張版です。
    PDE、境界条件、初期条件、保存則などの複数の物理制約を扱います。

    Parameters
    ----------
    constraint_types : list of str, optional
        制約の種類（デフォルト: ['pde', 'bc', 'ic', 'conservation']）
    init_lambdas : dict, optional
        各制約の初期重み

    Examples
    --------
    >>> mp_haw = MultiPhysicsHierarchicalAdaptiveWeighting(
    ...     constraint_types=['pde', 'bc', 'ic', 'conservation']
    ... )
    >>> # 教師なし学習ループ内で
    >>> lambdas = mp_haw.get_lambdas()
    >>> loss = (lambdas['pde'] * loss_pde +
    ...         lambdas['bc'] * loss_bc +
    ...         lambdas['ic'] * loss_ic +
    ...         lambdas['conservation'] * loss_conservation)
    """

    def __init__(
        self,
        constraint_types=None,
        init_lambdas=None,
        **kwargs
    ):
        if constraint_types is None:
            # 完全教師なし学習用のデフォルト
            constraint_types = ['pde', 'bc', 'ic', 'conservation']

        if init_lambdas is None:
            # 各制約の初期重み（調整可能）
            init_lambdas = {
                'pde': 1.0,
                'bc': 1.0,
                'ic': 1.0,
                'conservation': 0.5
            }

        super().__init__(
            constraint_types=constraint_types,
            init_lambdas=init_lambdas,
            **kwargs
        )


def compute_gradient_statistics(gradients):
    """
    勾配の統計情報を計算（デバッグ・分析用）

    Parameters
    ----------
    gradients : dict of torch.Tensor
        各制約の勾配

    Returns
    -------
    dict
        統計情報（ノルム、平均、標準偏差など）
    """
    stats = {}

    for ct, grad in gradients.items():
        stats[ct] = {
            'norm': grad.norm().item(),
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.max().item(),
            'min': grad.min().item()
        }

    # ペアワイズのコサイン類似度
    grad_keys = list(gradients.keys())
    cosine_sims = {}

    for i, ct1 in enumerate(grad_keys):
        for ct2 in grad_keys[i+1:]:
            cos_sim = F.cosine_similarity(
                gradients[ct1].unsqueeze(0),
                gradients[ct2].unsqueeze(0),
                dim=1
            ).item()
            cosine_sims[f'{ct1}_vs_{ct2}'] = cos_sim

    stats['cosine_similarities'] = cosine_sims

    return stats
