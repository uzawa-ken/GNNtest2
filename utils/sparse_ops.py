#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparse Matrix Operations

CSR形式の疎行列演算
"""

import torch


def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    """
    CSR形式の疎行列とベクトルの積を計算

    y = A @ x を CSR (Compressed Sparse Row) 形式で効率的に計算します。

    Parameters
    ----------
    row_ptr : torch.Tensor
        CSR形式の行ポインタ [nRows+1]
        （この実装では使用されていません）
    col_ind : torch.Tensor
        CSR形式の列インデックス [nnz]
    vals : torch.Tensor
        CSR形式の非ゼロ要素値 [nnz]
    row_idx : torch.Tensor
        各非ゼロ要素の行インデックス [nnz]
    x : torch.Tensor
        入力ベクトル [nCols]

    Returns
    -------
    torch.Tensor
        結果ベクトル y = A @ x [nRows]

    Notes
    -----
    この実装は row_idx を使用した scatter-add による計算を行います。

    計算式:
        y[row_idx[k]] += vals[k] * x[col_ind[k]]  for all k in [0, nnz)

    Examples
    --------
    >>> # 2x2 行列 A = [[1, 2], [3, 0]] の CSR 表現
    >>> row_ptr = torch.tensor([0, 2, 3], dtype=torch.long)
    >>> col_ind = torch.tensor([0, 1, 0], dtype=torch.long)
    >>> vals = torch.tensor([1.0, 2.0, 3.0])
    >>> row_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    >>> x = torch.tensor([1.0, 2.0])
    >>> y = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x)
    >>> print(y)  # tensor([5., 3.])  # [1*1 + 2*2, 3*1 + 0*2]
    """
    # 結果ベクトルを初期化
    y = torch.zeros_like(x)

    # vals[k] * x[col_ind[k]] を y[row_idx[k]] に加算
    # index_add_ はインプレース演算で効率的
    y.index_add_(0, row_idx, vals * x[col_ind])

    return y
