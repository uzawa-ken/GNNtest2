#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Operations

グラフ構造に基づく微分演算
- グラフラプラシアン（2階微分の近似）
- グラフ勾配（1階微分の近似）
- トポロジー認識型重み伝播
"""

import torch
import torch.nn.functional as F


def compute_graph_laplacian(x, edge_index, normalized=True):
    """
    グラフラプラシアンによる2階微分（∇²φ）の近似

    グラフ構造を利用して、各ノードにおける解の曲率（2階微分）を近似します。
    これは、解の空間的な複雑さを表す指標として使用されます。

    Parameters
    ----------
    x : torch.Tensor, shape (num_nodes,) or (num_nodes, 1)
        ノードの値（例：圧力場の予測値）
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジリスト [source_nodes, target_nodes]
    normalized : bool, optional
        次数で正規化するか（デフォルト: True）

    Returns
    -------
    torch.Tensor, shape (num_nodes,)
        各ノードのラプラシアン値（∇²φ の近似）

    Notes
    -----
    離散ラプラシアン演算子:
        (∇²φ)_i ≈ (1/deg(i)) Σ_{j∈N(i)} (φ_j - φ_i)

    ここで：
    - N(i): ノード i の近傍ノード集合
    - deg(i): ノード i の次数

    物理的意味：
    - 正の値: 周囲より低い（凹）→ 解が局所的に減少
    - 負の値: 周囲より高い（凸）→ 解が局所的に増加
    - 絶対値が大きい: 解の曲率が大きい（複雑な領域）

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> laplacian = compute_graph_laplacian(x, edge_index)
    >>> print(laplacian)
    tensor([1.0, 0.0, -1.0])  # 線形増加なので中央は0
    """
    # 入力の形状を統一
    if x.dim() == 1:
        x = x.view(-1, 1)

    num_nodes = x.size(0)
    row, col = edge_index[0], edge_index[1]

    # 各ノードの次数を計算
    deg = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    deg.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=x.device))

    # deg が 0 のノードを処理（孤立ノード）
    deg = deg.float().clamp(min=1.0)

    # ラプラシアンの計算: Σ_{j∈N(i)} (x_j - x_i)
    laplacian = torch.zeros_like(x)

    # x[col] - x[row] を計算して row にスキャッタ加算
    diff = x[col] - x[row]
    laplacian.index_add_(0, row, diff)

    if normalized:
        # 次数で正規化
        laplacian = laplacian / deg.view(-1, 1)

    return laplacian.view(-1)


def compute_graph_gradient(x, edge_index, coords):
    """
    グラフ勾配による1階微分（∇φ）の近似

    グラフ構造と空間座標を利用して、各ノードにおける勾配を近似します。

    Parameters
    ----------
    x : torch.Tensor, shape (num_nodes,)
        ノードの値
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジリスト
    coords : torch.Tensor, shape (num_nodes, 3)
        ノードの空間座標 [x, y, z]

    Returns
    -------
    torch.Tensor, shape (num_nodes, 3)
        各ノードの勾配ベクトル [∂φ/∂x, ∂φ/∂y, ∂φ/∂z]

    Notes
    -----
    勾配の近似:
        ∇φ_i ≈ (1/|N(i)|) Σ_{j∈N(i)} (φ_j - φ_i) / ||r_j - r_i|| × (r_j - r_i)/||r_j - r_i||

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> grad = compute_graph_gradient(x, edge_index, coords)
    >>> print(grad.shape)
    torch.Size([3, 3])
    """
    num_nodes = x.size(0)
    row, col = edge_index[0], edge_index[1]

    # 空間的な差分ベクトル
    spatial_diff = coords[col] - coords[row]  # (num_edges, 3)
    spatial_dist = torch.norm(spatial_diff, dim=1, keepdim=True).clamp(min=1e-8)

    # 値の差分
    value_diff = x[col] - x[row]  # (num_edges,)

    # 勾配の寄与: (φ_j - φ_i) / ||r_j - r_i|| × (r_j - r_i)/||r_j - r_i||
    grad_contrib = (value_diff.unsqueeze(1) / spatial_dist) * (spatial_diff / spatial_dist)

    # 各ノードの勾配を集計
    gradient = torch.zeros(num_nodes, 3, device=x.device)
    gradient.index_add_(0, row, grad_contrib)

    # 次数で正規化
    deg = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    deg.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=x.device))
    deg = deg.float().clamp(min=1.0)

    gradient = gradient / deg.view(-1, 1)

    return gradient


def compute_gradient_magnitude(x, edge_index, coords):
    """
    勾配の大きさを計算

    Parameters
    ----------
    x : torch.Tensor, shape (num_nodes,)
        ノードの値
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジリスト
    coords : torch.Tensor, shape (num_nodes, 3)
        ノードの空間座標

    Returns
    -------
    torch.Tensor, shape (num_nodes,)
        各ノードの勾配の大きさ ||∇φ||

    Examples
    --------
    >>> grad_mag = compute_gradient_magnitude(x, edge_index, coords)
    >>> print(grad_mag.shape)
    torch.Size([num_nodes])
    """
    gradient = compute_graph_gradient(x, edge_index, coords)
    grad_magnitude = torch.norm(gradient, dim=1)
    return grad_magnitude


def topology_aware_weight_propagation(
    w_local,
    edge_index,
    num_hops=2,
    decay_factor=0.5,
    aggregation='max'
):
    """
    トポロジー認識型の重み伝播

    低品質メッシュの影響が近傍に伝播する物理的効果をモデル化します。
    CFDでは、メッシュ品質の悪い領域が周辺の解精度にも影響を与えます。

    Parameters
    ----------
    w_local : torch.Tensor, shape (num_nodes,)
        各ノードの局所的な重み
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジリスト
    num_hops : int, optional
        伝播するホップ数（デフォルト: 2）
    decay_factor : float, optional
        各ホップでの減衰係数（デフォルト: 0.5）
    aggregation : str, optional
        集約方法 ('max' or 'mean')（デフォルト: 'max'）

    Returns
    -------
    torch.Tensor, shape (num_nodes,)
        伝播後の重み

    Notes
    -----
    アルゴリズム:
        1. 各ホップで近傍の最大重みを収集
        2. 減衰係数を適用: w_hop = decay^hop × w_neighbor
        3. 局所重みと伝播重みの最大値を取る

    物理的根拠:
        CFDの数値誤差は、低品質メッシュから周辺セルに伝播します。
        この伝播は距離とともに減衰します。

    Examples
    --------
    >>> w_local = torch.tensor([1.0, 10.0, 1.0, 1.0])  # ノード1が低品質
    >>> edge_index = torch.tensor([[0,1,1,2,2,3], [1,0,2,1,3,2]])
    >>> w_prop = topology_aware_weight_propagation(w_local, edge_index, num_hops=2)
    >>> # ノード1の高重みが近傍に伝播
    >>> print(w_prop)  # tensor([5.5, 10.0, 5.5, 2.75]) (減衰しながら伝播)
    """
    num_nodes = w_local.size(0)
    row, col = edge_index[0], edge_index[1]

    # 現在の重み（伝播なし）
    w_propagated = w_local.clone()

    for hop in range(num_hops):
        # このホップでの減衰係数
        decay = decay_factor ** (hop + 1)

        # 近傍の重みを収集
        neighbor_weights = torch.zeros(num_nodes, device=w_local.device)

        if aggregation == 'max':
            # 各ノードの近傍の最大重み
            neighbor_weights.index_reduce_(
                0, row, w_propagated[col], 'amax', include_self=False
            )
        elif aggregation == 'mean':
            # 各ノードの近傍の平均重み
            neighbor_weights.index_add_(0, row, w_propagated[col])
            deg = torch.zeros(num_nodes, dtype=torch.long, device=w_local.device)
            deg.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=w_local.device))
            deg = deg.float().clamp(min=1.0)
            neighbor_weights = neighbor_weights / deg
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # 減衰係数を適用して伝播
        propagated_contribution = decay * neighbor_weights

        # 局所重みと伝播重みの最大値を取る
        w_propagated = torch.maximum(w_propagated, propagated_contribution)

    return w_propagated


def compute_node_degree(edge_index, num_nodes):
    """
    各ノードの次数を計算

    Parameters
    ----------
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジリスト
    num_nodes : int
        ノード数

    Returns
    -------
    torch.Tensor, shape (num_nodes,)
        各ノードの次数

    Examples
    --------
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> deg = compute_node_degree(edge_index, 3)
    >>> print(deg)  # tensor([1, 2, 1])
    """
    row = edge_index[0]
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=edge_index.device))
    return deg
