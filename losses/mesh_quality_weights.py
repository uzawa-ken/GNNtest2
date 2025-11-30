#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mesh Quality Weighting

メッシュ品質メトリクスに基づくPDE残差損失の重み計算
"""

import numpy as np


def build_w_pde_from_feats(feats_np: np.ndarray, w_pde_max: float = 20.0) -> np.ndarray:
    """
    メッシュ品質に基づくPDE損失の重みを計算

    4つのメッシュ品質メトリクス（歪度、非直交性、アスペクト比、サイズジャンプ）
    に基づいて、各セルのPDE残差損失に対する重みを計算します。
    品質が低いセルほど高い重みが割り当てられ、物理的整合性がより重視されます。

    Parameters
    ----------
    feats_np : np.ndarray, shape (nCells, 13)
        セル特徴量
        - feats_np[:, 5]: 歪度（Skewness）
        - feats_np[:, 6]: 非直交性（Non-orthogonality）
        - feats_np[:, 7]: アスペクト比（Aspect ratio）
        - feats_np[:, 11]: サイズジャンプ（Size jump）

    w_pde_max : float, optional
        重みの最大値（デフォルト: 20.0）

    Returns
    -------
    np.ndarray, shape (nCells,)
        各セルのPDE損失重み [1.0, w_pde_max] の範囲

    Notes
    -----
    重み計算式:
        1. 正規化: q_i = clip(metric_i / REF_i, 0.0, 5.0)
        2. 線形結合:
           w_raw = 1.0 + 1.0 × (q_skew - 1.0)
                       + 0.5 × (q_non_ortho - 1.0)
                       + 0.5 × (q_aspect - 1.0)
                       + 1.0 × (q_sizeJump - 1.0)
        3. クリッピング: w = clip(w_raw, 1.0, w_pde_max)

    基準値（REF）:
        - SKEW_REF      = 0.2
        - NONORTH_REF   = 10.0
        - ASPECT_REF    = 5.0
        - SIZEJUMP_REF  = 1.5

    Examples
    --------
    >>> feats = np.random.rand(100, 13).astype(np.float32)
    >>> weights = build_w_pde_from_feats(feats)
    >>> print(weights.shape)  # (100,)
    >>> print(weights.min(), weights.max())  # 範囲: [1.0, 20.0]
    """
    # メトリクス抽出
    skew      = feats_np[:, 5]
    non_ortho = feats_np[:, 6]
    aspect    = feats_np[:, 7]
    size_jump = feats_np[:, 11]

    # 基準値（現在のベースライン実装）
    # 注: Phase 2で適応的基準値に更新予定
    SKEW_REF      = 0.2
    NONORTH_REF   = 10.0
    ASPECT_REF    = 5.0
    SIZEJUMP_REF  = 1.5

    # 正規化（0.0〜5.0の範囲にクリップ）
    q_skew      = np.clip(skew      / (SKEW_REF + 1e-12),     0.0, 5.0)
    q_non_ortho = np.clip(non_ortho / (NONORTH_REF + 1e-12),  0.0, 5.0)
    q_aspect    = np.clip(aspect    / (ASPECT_REF + 1e-12),   0.0, 5.0)
    q_sizeJump  = np.clip(size_jump / (SIZEJUMP_REF + 1e-12), 0.0, 5.0)

    # 生の重み計算（線形結合）
    # 重み係数: skew=1.0, non_ortho=0.5, aspect=0.5, sizeJump=1.0
    w_raw = (
        1.0
        + 1.0 * (q_skew      - 1.0)
        + 0.5 * (q_non_ortho - 1.0)
        + 0.5 * (q_aspect    - 1.0)
        + 1.0 * (q_sizeJump  - 1.0)
    )

    # 最終的な重み（1.0〜w_pde_maxの範囲にクリップ）
    w_clipped = np.clip(w_raw, 1.0, w_pde_max)

    return w_clipped.astype(np.float32)


def get_mesh_quality_reference_values():
    """
    メッシュ品質の基準値を取得

    現在のベースライン実装で使用されている固定基準値を返します。
    Phase 2で適応的基準値に更新予定。

    Returns
    -------
    dict
        基準値の辞書
        - 'skew': 歪度の基準値
        - 'non_ortho': 非直交性の基準値
        - 'aspect': アスペクト比の基準値
        - 'size_jump': サイズジャンプの基準値
    """
    return {
        'skew': 0.2,
        'non_ortho': 10.0,
        'aspect': 5.0,
        'size_jump': 1.5
    }


def get_mesh_quality_weight_coefficients():
    """
    メッシュ品質重み係数を取得

    現在のベースライン実装で使用されている固定重み係数を返します。
    Phase 2で学習可能パラメータに更新予定。

    Returns
    -------
    dict
        重み係数の辞書
        - 'skew': 歪度の係数
        - 'non_ortho': 非直交性の係数
        - 'aspect': アスペクト比の係数
        - 'size_jump': サイズジャンプの係数
    """
    return {
        'skew': 1.0,
        'non_ortho': 0.5,
        'aspect': 0.5,
        'size_jump': 1.0
    }
