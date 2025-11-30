#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Utils Module

ユーティリティモジュールのユニットテスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pytest
from utils import matvec_csr_torch


def test_matvec_csr_import():
    """matvec_csr_torchがインポートできるか"""
    assert matvec_csr_torch is not None


def test_matvec_csr_simple():
    """CSR行列ベクトル積が正しく計算されるか（簡単なケース）"""
    # 2x2 行列 A = [[1, 2], [3, 0]]
    # CSR形式: vals=[1, 2, 3], col_ind=[0, 1, 0], row_idx=[0, 0, 1]

    row_ptr = torch.tensor([0, 2, 3], dtype=torch.long)
    col_ind = torch.tensor([0, 1, 0], dtype=torch.long)
    vals = torch.tensor([1.0, 2.0, 3.0])
    row_idx = torch.tensor([0, 0, 1], dtype=torch.long)

    x = torch.tensor([1.0, 2.0])

    y = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x)

    # 期待される結果: [1*1 + 2*2, 3*1 + 0*2] = [5, 3]
    expected = torch.tensor([5.0, 3.0])

    assert torch.allclose(y, expected)


def test_matvec_csr_identity():
    """単位行列での動作確認"""
    # 3x3 単位行列
    n = 3
    row_ptr = torch.arange(0, n + 1, dtype=torch.long)
    col_ind = torch.arange(0, n, dtype=torch.long)
    vals = torch.ones(n)
    row_idx = torch.arange(0, n, dtype=torch.long)

    x = torch.tensor([1.0, 2.0, 3.0])

    y = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x)

    # 単位行列なので x と同じ
    assert torch.allclose(y, x)


def test_matvec_csr_zero():
    """ゼロ行列での動作確認"""
    # 3x3 ゼロ行列（非ゼロ要素なし）
    row_ptr = torch.zeros(4, dtype=torch.long)
    col_ind = torch.tensor([], dtype=torch.long)
    vals = torch.tensor([])
    row_idx = torch.tensor([], dtype=torch.long)

    x = torch.tensor([1.0, 2.0, 3.0])

    y = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x)

    # ゼロ行列なので全て0
    expected = torch.zeros(3)
    assert torch.allclose(y, expected)


def test_matvec_csr_gradient():
    """勾配が計算できるか"""
    row_ptr = torch.tensor([0, 2, 3], dtype=torch.long)
    col_ind = torch.tensor([0, 1, 0], dtype=torch.long)
    vals = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    row_idx = torch.tensor([0, 0, 1], dtype=torch.long)

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    y = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x)
    loss = y.sum()
    loss.backward()

    # 勾配が計算されているか
    assert x.grad is not None
    assert vals.grad is not None


if __name__ == '__main__':
    print("Running utils tests...")

    test_matvec_csr_import()
    print("✓ Import test passed")

    test_matvec_csr_simple()
    print("✓ Simple CSR test passed")

    test_matvec_csr_identity()
    print("✓ Identity matrix test passed")

    test_matvec_csr_zero()
    print("✓ Zero matrix test passed")

    test_matvec_csr_gradient()
    print("✓ Gradient test passed")

    print("\n✅ All utils tests passed!")
