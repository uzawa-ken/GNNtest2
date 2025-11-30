#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Models Module

モデルモジュールのユニットテスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from models import SimpleSAGE


def test_simple_sage_import():
    """SimpleSAGEモデルがインポートできるか"""
    assert SimpleSAGE is not None


def test_simple_sage_initialization():
    """SimpleSAGEモデルが初期化できるか"""
    model = SimpleSAGE(in_channels=13, hidden_channels=64, num_layers=4)

    assert model.in_channels == 13
    assert model.hidden_channels == 64
    assert model.num_layers == 4
    assert len(model.convs) == 4


def test_simple_sage_forward():
    """SimpleSAGEの順伝播が正しく動作するか"""
    model = SimpleSAGE(in_channels=13, hidden_channels=64, num_layers=4)

    # ダミーデータ作成
    num_nodes = 100
    num_edges = 200
    x = torch.randn(num_nodes, 13)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 順伝播
    output = model(x, edge_index)

    # 出力形状の確認
    assert output.shape == (num_nodes,)

    # NaNやInfがないか確認
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_simple_sage_different_sizes():
    """異なるサイズの入力で動作するか"""
    model = SimpleSAGE(in_channels=13)

    # 小規模
    x_small = torch.randn(10, 13)
    edge_small = torch.randint(0, 10, (2, 20))
    out_small = model(x_small, edge_small)
    assert out_small.shape == (10,)

    # 大規模
    x_large = torch.randn(1000, 13)
    edge_large = torch.randint(0, 1000, (2, 2000))
    out_large = model(x_large, edge_large)
    assert out_large.shape == (1000,)


def test_simple_sage_gradient():
    """勾配が計算できるか"""
    model = SimpleSAGE(in_channels=13)

    x = torch.randn(50, 13, requires_grad=True)
    edge_index = torch.randint(0, 50, (2, 100))

    output = model(x, edge_index)
    loss = output.sum()
    loss.backward()

    # 勾配が計算されているか
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    # pytestがない場合の簡易実行
    print("Running model tests...")

    test_simple_sage_import()
    print("✓ Import test passed")

    test_simple_sage_initialization()
    print("✓ Initialization test passed")

    test_simple_sage_forward()
    print("✓ Forward test passed")

    test_simple_sage_different_sizes()
    print("✓ Different sizes test passed")

    test_simple_sage_gradient()
    print("✓ Gradient test passed")

    print("\n✅ All model tests passed!")
