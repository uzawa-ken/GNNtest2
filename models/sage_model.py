#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleSAGE GNN Model

GraphSAGEベースのシンプルなGNNモデル
圧力場予測のための4層アーキテクチャ
"""

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise RuntimeError(
        "torch_geometric がインストールされていません。"
        "pip install torch-geometric などでインストールしてください。"
    )


class SimpleSAGE(nn.Module):
    """
    SimpleSAGE Graph Neural Network

    4層のGraphSAGEアーキテクチャ
    - 入力層: in_channels → hidden_channels
    - 隠れ層: (hidden_channels → hidden_channels) × (num_layers - 2)
    - 出力層: hidden_channels → 1 (圧力予測値)

    Parameters
    ----------
    in_channels : int
        入力特徴量の次元数（デフォルト: 13）
    hidden_channels : int, optional
        隠れ層のチャンネル数（デフォルト: 64）
    num_layers : int, optional
        総レイヤー数（デフォルト: 4）
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.convs = nn.ModuleList()

        # 入力層
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # 隠れ層
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # 出力層
        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            ノード特徴量 [num_nodes, in_channels]
        edge_index : torch.Tensor
            エッジインデックス [2, num_edges]

        Returns
        -------
        torch.Tensor
            予測値 [num_nodes]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # 出力層以外はReLU活性化
            if i != len(self.convs) - 1:
                x = nn.functional.relu(x)

        # [num_nodes, 1] → [num_nodes] に変形
        return x.view(-1)

    def __repr__(self):
        return (f'SimpleSAGE(in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers})')
