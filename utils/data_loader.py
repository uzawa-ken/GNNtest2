#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Loader Utilities

CFDシミュレーションデータのローディング
- pEqn_*_rank*.dat: セル特徴量とグラフ構造
- x_*_rank*.dat: 正解データ
- A_csr_*.dat: システム行列（CSR形式）
"""

import os
import numpy as np


# utils/data_loader.py の find_time_list() 関数
# processor*/gnn 構造に対応しているか確認

# もし並列計算結果の場合、以下のように修正が必要かもしれません：
def find_time_list(case_dir):
    """
    Find available time steps in case directory.
    Handles both serial (gnn/) and parallel (processor*/gnn/) structures.
    """
    case_path = Path(case_dir)
    time_dirs = set()
    
    # Serial case: gnn/ directory
    gnn_dir = case_path / 'gnn'
    if gnn_dir.exists():
        for f in gnn_dir.glob('pEqn_*'):
            time_str = f.name.split('_')[1]
            time_dirs.add(time_str)
    
    # Parallel case: processor*/gnn/ directories
    for proc_dir in case_path.glob('processor*/gnn'):
        for f in proc_dir.glob('pEqn_*'):
            time_str = f.name.split('_')[1]
            time_dirs.add(time_str)
    
    return sorted(time_dirs, key=lambda x: float(x))



def load_case_with_csr(data_dir: str, time_str: str, rank_str: str):
    """
    単一タイムステップのCFDデータをロード

    以下のファイルから情報を読み込む：
    - pEqn_*.dat: セル特徴量、グラフ構造、右辺ベクトルb
    - x_*.dat: 正解データ（圧力場）
    - A_csr_*.dat: システム行列A（CSR形式）

    Parameters
    ----------
    data_dir : str
        データディレクトリのパス
    time_str : str
        タイムステップ文字列（例: "0.001"）
    rank_str : str
        MPIランク識別子（例: "7"）

    Returns
    -------
    dict
        以下のキーを持つ辞書：
        - time : str
            タイムステップ
        - feats_np : np.ndarray, shape (nCells, 13)
            セル特徴量（座標、メッシュ品質メトリクス等）
        - edge_index_np : np.ndarray, shape (2, num_edges)
            グラフのエッジリスト
        - x_true_np : np.ndarray, shape (nCells,)
            正解データ（圧力値）
        - b_np : np.ndarray, shape (nCells,)
            右辺ベクトル
        - row_ptr_np : np.ndarray, shape (nRows+1,)
            CSR行列の行ポインタ
        - col_ind_np : np.ndarray, shape (nnz,)
            CSR行列の列インデックス
        - vals_np : np.ndarray, shape (nnz,)
            CSR行列の非ゼロ要素値
        - row_idx_np : np.ndarray, shape (nnz,)
            各非ゼロ要素の行インデックス

    Raises
    ------
    FileNotFoundError
        必要なファイルが見つからない場合
    RuntimeError
        ファイル形式が不正な場合

    Examples
    --------
    >>> data = load_case_with_csr("./gnn", "0.001", "7")
    >>> print(data["feats_np"].shape)  # (nCells, 13)
    >>> print(data["x_true_np"].shape)  # (nCells,)
    """
    p_path   = os.path.join(data_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
    csr_path = os.path.join(data_dir, f"A_csr_{time_str}.dat")

    # ファイル存在確認
    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(csr_path):
        raise FileNotFoundError(csr_path)

    # ========== pEqn_*.dat の読み込み ==========
    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # ヘッダー解析
    try:
        header_nc = lines[0].split()
        header_nf = lines[1].split()
        assert header_nc[0] == "nCells"
        assert header_nf[0] == "nFaces"
        nCells = int(header_nc[1])
    except Exception as e:
        raise RuntimeError(f"nCells/nFaces ヘッダの解釈に失敗しました: {p_path}\n{e}")

    # セクション位置の特定
    try:
        idx_cells = next(i for i, ln in enumerate(lines) if ln.startswith("CELLS"))
        idx_edges = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES"))
    except StopIteration:
        raise RuntimeError(f"CELLS/EDGES セクションが見つかりません: {p_path}")

    idx_wall = None
    for i, ln in enumerate(lines):
        if ln.startswith("WALL_FACES"):
            idx_wall = i
            break
    if idx_wall is None:
        idx_wall = len(lines)

    cell_lines = lines[idx_cells + 1: idx_edges]
    edge_lines = lines[idx_edges + 1: idx_wall]

    if len(cell_lines) != nCells:
        print(f"[WARN] nCells={nCells} と CELLS 行数={len(cell_lines)} が異なります (time={time_str}).")

    # セル特徴量の読み込み
    feats_np = np.zeros((len(cell_lines), 13), dtype=np.float32)
    b_np     = np.zeros(len(cell_lines), dtype=np.float32)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 14:
            raise RuntimeError(f"CELLS 行の列数が足りません: {ln}")

        cell_id = int(parts[0])
        xcoord  = float(parts[1])
        ycoord  = float(parts[2])
        zcoord  = float(parts[3])
        diag    = float(parts[4])
        b_val   = float(parts[5])
        skew    = float(parts[6])
        non_ortho  = float(parts[7])
        aspect     = float(parts[8])
        diag_con   = float(parts[9])
        V          = float(parts[10])
        h          = float(parts[11])
        size_jump  = float(parts[12])
        Co         = float(parts[13])

        if not (0 <= cell_id < len(cell_lines)):
            raise RuntimeError(f"cell_id の範囲がおかしいです: {cell_id}")

        feats_np[cell_id, :] = np.array(
            [
                xcoord, ycoord, zcoord,
                diag, b_val, skew, non_ortho, aspect,
                diag_con, V, h, size_jump, Co
            ],
            dtype=np.float32
        )
        b_np[cell_id] = b_val

    # エッジリストの読み込み
    e_src = []
    e_dst = []
    for ln in edge_lines:
        parts = ln.split()
        if parts[0] == "WALL_FACES":
            break
        if len(parts) != 5:
            raise RuntimeError(f"EDGES 行の列数が 5 ではありません: {ln}")

        lower = int(parts[1])
        upper = int(parts[2])
        if not (0 <= lower < len(cell_lines) and 0 <= upper < len(cell_lines)):
            raise RuntimeError(f"lower/upper の cell index が範囲外です: {ln}")

        # 双方向エッジ
        e_src.append(lower)
        e_dst.append(upper)
        e_src.append(upper)
        e_dst.append(lower)

    edge_index_np = np.vstack([
        np.array(e_src, dtype=np.int64),
        np.array(e_dst, dtype=np.int64)
    ])

    # ========== x_*.dat の読み込み（正解データ） ==========
    x_true_np = np.zeros(len(cell_lines), dtype=np.float32)
    with open(x_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 2:
                raise RuntimeError(f"x_*.dat の行形式が想定外です: {ln}")

            cid = int(parts[0])
            val = float(parts[1])
            if not (0 <= cid < len(cell_lines)):
                raise RuntimeError(f"x_*.dat の cell id が範囲外です: {cid}")
            x_true_np[cid] = val

    # ========== A_csr_*.dat の読み込み（システム行列） ==========
    with open(csr_path, "r") as f:
        csr_lines = [ln.strip() for ln in f if ln.strip()]

    # ヘッダー解析
    try:
        h0 = csr_lines[0].split()
        h1 = csr_lines[1].split()
        h2 = csr_lines[2].split()
        assert h0[0] == "nRows"
        assert h1[0] == "nCols"
        assert h2[0] == "nnz"
        nRows = int(h0[1])
        nCols = int(h1[1])
        nnz   = int(h2[1])
    except Exception as e:
        raise RuntimeError(f"A_csr_{time_str}.dat のヘッダ解釈に失敗しました: {csr_path}\n{e}")

    if nRows != nCells:
        print(f"[WARN] CSR nRows={nRows} と pEqn nCells={nCells} が異なります (time={time_str}).")

    # セクション位置の特定
    try:
        idx_rowptr = next(i for i, ln in enumerate(csr_lines) if ln.startswith("ROW_PTR"))
        idx_colind = next(i for i, ln in enumerate(csr_lines) if ln.startswith("COL_IND"))
        idx_vals   = next(i for i, ln in enumerate(csr_lines) if ln.startswith("VALUES"))
    except StopIteration:
        raise RuntimeError(f"ROW_PTR/COL_IND/VALUES が見つかりません: {csr_path}")

    row_ptr_str = csr_lines[idx_rowptr + 1].split()
    col_ind_str = csr_lines[idx_colind + 1].split()
    vals_str    = csr_lines[idx_vals + 1].split()

    # 長さ検証
    if len(row_ptr_str) != nRows + 1:
        raise RuntimeError(
            f"ROW_PTR の長さが nRows+1 と一致しません: len={len(row_ptr_str)}, nRows={nRows}"
        )
    if len(col_ind_str) != nnz:
        raise RuntimeError(
            f"COL_IND の長さが nnz と一致しません: len={len(col_ind_str)}, nnz={nnz}"
        )
    if len(vals_str) != nnz:
        raise RuntimeError(
            f"VALUES の長さが nnz と一致しません: len={len(vals_str)}, nnz={nnz}"
        )

    # NumPy配列に変換
    row_ptr_np = np.array(row_ptr_str, dtype=np.int64)
    col_ind_np = np.array(col_ind_str, dtype=np.int64)
    vals_np    = np.array(vals_str,    dtype=np.float32)

    # 行インデックスの生成（PyTorchでの疎行列演算用）
    row_idx_np = np.empty(nnz, dtype=np.int64)
    for i in range(nRows):
        start = row_ptr_np[i]
        end   = row_ptr_np[i+1]
        row_idx_np[start:end] = i

    return {
        "time": time_str,
        "feats_np": feats_np,
        "edge_index_np": edge_index_np,
        "x_true_np": x_true_np,
        "b_np": b_np,
        "row_ptr_np": row_ptr_np,
        "col_ind_np": col_ind_np,
        "vals_np": vals_np,
        "row_idx_np": row_idx_np,
    }
