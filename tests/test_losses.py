#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Losses Module

損失関数モジュールのユニットテスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from losses import (
    build_w_pde_from_feats,
    get_mesh_quality_reference_values,
    get_mesh_quality_weight_coefficients
)


def test_mesh_quality_import():
    """メッシュ品質関数がインポートできるか"""
    assert build_w_pde_from_feats is not None
    assert get_mesh_quality_reference_values is not None
    assert get_mesh_quality_weight_coefficients is not None


def test_get_reference_values():
    """基準値が取得できるか"""
    ref_values = get_mesh_quality_reference_values()

    assert 'skew' in ref_values
    assert 'non_ortho' in ref_values
    assert 'aspect' in ref_values
    assert 'size_jump' in ref_values

    assert ref_values['skew'] == 0.2
    assert ref_values['non_ortho'] == 10.0
    assert ref_values['aspect'] == 5.0
    assert ref_values['size_jump'] == 1.5


def test_get_weight_coefficients():
    """重み係数が取得できるか"""
    coeffs = get_mesh_quality_weight_coefficients()

    assert 'skew' in coeffs
    assert 'non_ortho' in coeffs
    assert 'aspect' in coeffs
    assert 'size_jump' in coeffs

    assert coeffs['skew'] == 1.0
    assert coeffs['non_ortho'] == 0.5
    assert coeffs['aspect'] == 0.5
    assert coeffs['size_jump'] == 1.0


def test_build_w_pde_basic():
    """基本的な重み計算が動作するか"""
    # ダミーの特徴量（100セル、13特徴）
    feats_np = np.random.rand(100, 13).astype(np.float32)

    # メッシュ品質メトリクスを設定
    feats_np[:, 5] = 0.2   # skew
    feats_np[:, 6] = 10.0  # non_ortho
    feats_np[:, 7] = 5.0   # aspect
    feats_np[:, 11] = 1.5  # size_jump

    weights = build_w_pde_from_feats(feats_np, w_pde_max=20.0)

    # 出力形状の確認
    assert weights.shape == (100,)

    # 重みの範囲確認（1.0〜20.0）
    assert np.all(weights >= 1.0)
    assert np.all(weights <= 20.0)


def test_build_w_pde_good_quality():
    """良好なメッシュ品質で重みが小さいか"""
    feats_np = np.zeros((10, 13), dtype=np.float32)

    # 良好なメッシュ品質（基準値以下）
    feats_np[:, 5] = 0.1   # skew < 0.2
    feats_np[:, 6] = 5.0   # non_ortho < 10.0
    feats_np[:, 7] = 3.0   # aspect < 5.0
    feats_np[:, 11] = 1.0  # size_jump < 1.5

    weights = build_w_pde_from_feats(feats_np)

    # 良好なメッシュなので重みは1.0に近い
    assert np.all(weights >= 1.0)
    assert np.all(weights < 5.0)


def test_build_w_pde_poor_quality():
    """低品質なメッシュで重みが大きいか"""
    feats_np = np.zeros((10, 13), dtype=np.float32)

    # 低品質なメッシュ（基準値を大きく超える）
    feats_np[:, 5] = 2.0   # skew >> 0.2
    feats_np[:, 6] = 50.0  # non_ortho >> 10.0
    feats_np[:, 7] = 20.0  # aspect >> 5.0
    feats_np[:, 11] = 10.0 # size_jump >> 1.5

    weights = build_w_pde_from_feats(feats_np, w_pde_max=20.0)

    # 低品質なメッシュなので重みは大きい
    assert np.all(weights > 10.0)
    assert np.all(weights <= 20.0)


def test_build_w_pde_dtype():
    """出力データ型が正しいか"""
    feats_np = np.random.rand(50, 13).astype(np.float32)
    weights = build_w_pde_from_feats(feats_np)

    assert weights.dtype == np.float32


def test_build_w_pde_custom_max():
    """カスタムw_pde_maxが機能するか"""
    feats_np = np.zeros((10, 13), dtype=np.float32)
    feats_np[:, 5] = 10.0  # 非常に高い歪度

    weights = build_w_pde_from_feats(feats_np, w_pde_max=10.0)

    # 最大値が指定した値以下
    assert np.all(weights <= 10.0)


if __name__ == '__main__':
    print("Running losses tests...")

    test_mesh_quality_import()
    print("✓ Import test passed")

    test_get_reference_values()
    print("✓ Reference values test passed")

    test_get_weight_coefficients()
    print("✓ Weight coefficients test passed")

    test_build_w_pde_basic()
    print("✓ Basic weight calculation test passed")

    test_build_w_pde_good_quality()
    print("✓ Good quality mesh test passed")

    test_build_w_pde_poor_quality()
    print("✓ Poor quality mesh test passed")

    test_build_w_pde_dtype()
    print("✓ Data type test passed")

    test_build_w_pde_custom_max()
    print("✓ Custom w_pde_max test passed")

    print("\n✅ All losses tests passed!")
