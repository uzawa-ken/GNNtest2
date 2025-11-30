#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gnn_auto_trainval_pde_weighted.py

- DATA_DIR 内から自動的に pEqn_*_rank{RANK_STR}.dat を走査し、
  TIME_LIST を最大 MAX_NUM_CASES 件まで自動生成。
- その TIME_LIST を train/val に分割して学習。
- 損失は data loss + mesh-quality-weighted PDE loss。

ベースライン:
    LAMBDA_DATA   = 0.1
    LAMBDA_PDE    = 1e-4
    TRAIN_FRACTION= 0.8 (全ケースのうち 80% を train, 残りを val)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise RuntimeError(
        "torch_geometric がインストールされていません。"
        "pip install torch-geometric などでインストールしてください。"
    )


# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------

DATA_DIR       = "./gnn"
RANK_STR       = "7"
NUM_EPOCHS     = 1000
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
MAX_NUM_CASES  = 100   # 自動検出した time のうち先頭 MAX_NUM_CASES 件を使用
TRAIN_FRACTION = 0.8   # 全ケースのうち train に使う割合

LAMBDA_DATA = 0.1
LAMBDA_PDE  = 1e-4

W_PDE_MAX = 20.0  # w_pde の最大値


# ------------------------------------------------------------
# ユーティリティ: time list 自動検出
# ------------------------------------------------------------

def find_time_list(data_dir: str, rank_str: str):
    times = []
    for fn in os.listdir(data_dir):
        if not fn.startswith("pEqn_"):
            continue
        if not fn.endswith(f"_rank{rank_str}.dat"):
            continue

        core = fn[len("pEqn_") : -len(f"_rank{rank_str}.dat")]
        time_str = core

        x_path   = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
        csr_path = os.path.join(data_dir, f"A_csr_{time_str}.dat")
        if os.path.exists(x_path) and os.path.exists(csr_path):
            times.append(time_str)

    times = sorted(set(times), key=lambda s: float(s))
    return times


# ------------------------------------------------------------
# pEqn + CSR + x_true 読み込み
# ------------------------------------------------------------

def load_case_with_csr(data_dir: str, time_str: str, rank_str: str):
    p_path   = os.path.join(data_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
    csr_path = os.path.join(data_dir, f"A_csr_{time_str}.dat")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(csr_path):
        raise FileNotFoundError(csr_path)

    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    try:
        header_nc = lines[0].split()
        header_nf = lines[1].split()
        assert header_nc[0] == "nCells"
        assert header_nf[0] == "nFaces"
        nCells = int(header_nc[1])
    except Exception as e:
        raise RuntimeError(f"nCells/nFaces ヘッダの解釈に失敗しました: {p_path}\n{e}")

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

        e_src.append(lower)
        e_dst.append(upper)
        e_src.append(upper)
        e_dst.append(lower)

    edge_index_np = np.vstack([
        np.array(e_src, dtype=np.int64),
        np.array(e_dst, dtype=np.int64)
    ])

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

    with open(csr_path, "r") as f:
        csr_lines = [ln.strip() for ln in f if ln.strip()]

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

    try:
        idx_rowptr = next(i for i, ln in enumerate(csr_lines) if ln.startswith("ROW_PTR"))
        idx_colind = next(i for i, ln in enumerate(csr_lines) if ln.startswith("COL_IND"))
        idx_vals   = next(i for i, ln in enumerate(csr_lines) if ln.startswith("VALUES"))
    except StopIteration:
        raise RuntimeError(f"ROW_PTR/COL_IND/VALUES が見つかりません: {csr_path}")

    row_ptr_str = csr_lines[idx_rowptr + 1].split()
    col_ind_str = csr_lines[idx_colind + 1].split()
    vals_str    = csr_lines[idx_vals + 1].split()

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

    row_ptr_np = np.array(row_ptr_str, dtype=np.int64)
    col_ind_np = np.array(col_ind_str, dtype=np.int64)
    vals_np    = np.array(vals_str,    dtype=np.float32)

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


# ------------------------------------------------------------
# GNN
# ------------------------------------------------------------

class SimpleSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = nn.functional.relu(x)
        return x.view(-1)


# ------------------------------------------------------------
# CSR Ax
# ------------------------------------------------------------

def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    y = torch.zeros_like(x)
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y


# ------------------------------------------------------------
# メッシュ品質 w_pde
# ------------------------------------------------------------

def build_w_pde_from_feats(feats_np: np.ndarray) -> np.ndarray:
    skew      = feats_np[:, 5]
    non_ortho = feats_np[:, 6]
    aspect    = feats_np[:, 7]
    size_jump = feats_np[:, 11]

    SKEW_REF      = 0.2
    NONORTH_REF   = 10.0
    ASPECT_REF    = 5.0
    SIZEJUMP_REF  = 1.5

    q_skew      = np.clip(skew      / (SKEW_REF + 1e-12),     0.0, 5.0)
    q_non_ortho = np.clip(non_ortho / (NONORTH_REF + 1e-12),  0.0, 5.0)
    q_aspect    = np.clip(aspect    / (ASPECT_REF + 1e-12),   0.0, 5.0)
    q_sizeJump  = np.clip(size_jump / (SIZEJUMP_REF + 1e-12), 0.0, 5.0)

    w_raw = (
        1.0
        + 1.0 * (q_skew      - 1.0)
        + 0.5 * (q_non_ortho - 1.0)
        + 0.5 * (q_aspect    - 1.0)
        + 1.0 * (q_sizeJump  - 1.0)
    )

    w_clipped = np.clip(w_raw, 1.0, W_PDE_MAX)
    return w_clipped.astype(np.float32)


# ------------------------------------------------------------
# raw_case → torch case への変換ヘルパ
# ------------------------------------------------------------

def convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device):
    feats_np  = rc["feats_np"]
    x_true_np = rc["x_true_np"]

    feats_norm     = (feats_np  - feat_mean) / feat_std
    x_true_norm_np = (x_true_np - x_mean)   / x_std

    w_pde_np = build_w_pde_from_feats(feats_np)

    feats       = torch.from_numpy(feats_norm).float().to(device)
    edge_index  = torch.from_numpy(rc["edge_index_np"]).long().to(device)
    x_true      = torch.from_numpy(x_true_np).float().to(device)
    x_true_norm = torch.from_numpy(x_true_norm_np).float().to(device)

    b       = torch.from_numpy(rc["b_np"]).float().to(device)
    row_ptr = torch.from_numpy(rc["row_ptr_np"]).long().to(device)
    col_ind = torch.from_numpy(rc["col_ind_np"]).long().to(device)
    vals    = torch.from_numpy(rc["vals_np"]).float().to(device)
    row_idx = torch.from_numpy(rc["row_idx_np"]).long().to(device)

    w_pde = torch.from_numpy(w_pde_np).float().to(device)

    return {
        "time": rc["time"],
        "feats": feats,
        "edge_index": edge_index,
        "x_true": x_true,
        "x_true_norm": x_true_norm,
        "b": b,
        "row_ptr": row_ptr,
        "col_ind": col_ind,
        "vals": vals,
        "row_idx": row_idx,
        "w_pde": w_pde,
        "w_pde_np": w_pde_np,  # 統計用
    }


# ------------------------------------------------------------
# メイン: train/val 分離版
# ------------------------------------------------------------

def train_gnn_auto_trainval_pde_weighted(data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # --- time list 検出 & 分割 ---
    all_times = find_time_list(data_dir, RANK_STR)
    if not all_times:
        raise RuntimeError(
            f"{data_dir} 内に pEqn_*_rank{RANK_STR}.dat / x_* / A_csr_* が見つかりませんでした。"
        )

    all_times = all_times[:MAX_NUM_CASES]
    n_total = len(all_times)
    n_train = max(1, int(n_total * TRAIN_FRACTION))
    n_val   = n_total - n_train

    time_train = all_times[:n_train]
    time_val   = all_times[n_train:]

    print(f"[INFO] 検出された time 数 (使用分) = {n_total}")
    print(f"[INFO] train: {n_train} cases, val: {n_val} cases (TRAIN_FRACTION={TRAIN_FRACTION})")
    print("=== 使用する train ケース (time, rank) ===")
    for t in time_train:
        print(f"  time={t}, rank={RANK_STR}")
    print("=== 使用する val ケース (time, rank) ===")
    if time_val:
        for t in time_val:
            print(f"  time={t}, rank={RANK_STR}")
    else:
        print("  (val ケースなし)")
    print("===========================================")

    # --- raw ケース読み込み（train + val 両方） ---
    raw_cases_train = []
    raw_cases_val   = []

    train_set = set(time_train)
    for t in all_times:
        print(f"[LOAD] time={t}, rank={RANK_STR} のグラフ+PDE情報を読み込み中...")
        rc = load_case_with_csr(data_dir, t, RANK_STR)
        if t in train_set:
            raw_cases_train.append(rc)
        else:
            raw_cases_val.append(rc)

    # 一貫性チェック
    nCells0 = raw_cases_train[0]["feats_np"].shape[0]
    nFeat   = raw_cases_train[0]["feats_np"].shape[1]
    for rc in raw_cases_train + raw_cases_val:
        if rc["feats_np"].shape[0] != nCells0 or rc["feats_np"].shape[1] != nFeat:
            raise RuntimeError("全ケースで nCells/nFeatures が一致していません。")

    print(f"[INFO] nCells (1 ケース目) = {nCells0}, nFeatures = {nFeat}")

    # --- グローバル正規化: train+val 全体で統計を取る ---
    all_feats = np.concatenate(
        [rc["feats_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )
    all_xtrue = np.concatenate(
        [rc["x_true_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-12

    x_mean = all_xtrue.mean()
    x_std  = all_xtrue.std() + 1e-12

    print(
        f"[INFO] x_true (all train+val cases): "
        f"min={all_xtrue.min():.3e}, max={all_xtrue.max():.3e}, mean={x_mean:.3e}"
    )

    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

    # --- torch ケース化 & w_pde 統計 ---
    cases_train = []
    cases_val   = []
    w_all_list  = []

    for rc in raw_cases_train:
        cs = convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device)
        cases_train.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    for rc in raw_cases_val:
        cs = convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device)
        cases_val.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    w_all = np.concatenate(w_all_list, axis=0)
    w_min  = float(w_all.min())
    w_max  = float(w_all.max())
    w_mean = float(w_all.mean())
    p90    = float(np.percentile(w_all, 90))
    p99    = float(np.percentile(w_all, 99))
    print("=== w_pde (mesh-quality weights) statistics over all train+val cases ===")
    print(f"  min   = {w_min:.3e}")
    print(f"  mean  = {w_mean:.3e}")
    print(f"  max   = {w_max:.3e}")
    print(f"  p90   = {p90:.3e}")
    print(f"  p99   = {p99:.3e}")
    print("==========================================================================")

    num_train = len(cases_train)
    num_val   = len(cases_val)

    # --- モデル定義 ---
    model = SimpleSAGE(in_channels=nFeat, hidden_channels=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_data = nn.MSELoss()

    print("=== Training start (data loss + weighted PDE loss, train/val split) ===")

    # --- 学習ループ ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        total_data_loss = 0.0
        total_pde_loss  = 0.0
        sum_rel_err_tr  = 0.0
        sum_R_pred_tr   = 0.0

        # -------- train で勾配計算 --------
        for cs in cases_train:
            feats       = cs["feats"]
            edge_index  = cs["edge_index"]
            x_true      = cs["x_true"]
            x_true_norm = cs["x_true_norm"]
            b           = cs["b"]
            row_ptr     = cs["row_ptr"]
            col_ind     = cs["col_ind"]
            vals        = cs["vals"]
            row_idx     = cs["row_idx"]
            w_pde       = cs["w_pde"]

            x_pred_norm = model(feats, edge_index)
            data_loss_case = criterion_data(x_pred_norm, x_true_norm)

            x_pred = x_pred_norm * x_std_t + x_mean_t

            Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r  = Ax - b

            sqrt_w = torch.sqrt(w_pde)
            wr = sqrt_w * r
            wb = sqrt_w * b
            norm_wr = torch.norm(wr)
            norm_wb = torch.norm(wb) + 1e-12
            R_pred = norm_wr / norm_wb
            pde_loss_case = R_pred * R_pred

            total_data_loss = total_data_loss + data_loss_case
            total_pde_loss  = total_pde_loss  + pde_loss_case

            with torch.no_grad():
                rel_err_case = torch.norm(x_pred.detach() - x_true) / (torch.norm(x_true) + 1e-12)
                sum_rel_err_tr += rel_err_case.item()
                sum_R_pred_tr  += R_pred.detach().item()

        total_data_loss = total_data_loss / num_train
        total_pde_loss  = total_pde_loss  / num_train
        loss = LAMBDA_DATA * total_data_loss + LAMBDA_PDE * total_pde_loss

        loss.backward()
        optimizer.step()

        # --- ロギング（train + val） ---
        if epoch % 50 == 0 or epoch == 1:
            avg_rel_err_tr = sum_rel_err_tr / num_train
            avg_R_pred_tr  = sum_R_pred_tr / num_train

            avg_rel_err_val = None
            avg_R_pred_val  = None
            if num_val > 0:
                model.eval()
                sum_rel_err_val = 0.0
                sum_R_pred_val  = 0.0
                with torch.no_grad():
                    for cs in cases_val:
                        feats      = cs["feats"]
                        edge_index = cs["edge_index"]
                        x_true     = cs["x_true"]
                        b          = cs["b"]
                        row_ptr    = cs["row_ptr"]
                        col_ind    = cs["col_ind"]
                        vals       = cs["vals"]
                        row_idx    = cs["row_idx"]
                        w_pde      = cs["w_pde"]

                        x_pred_norm = model(feats, edge_index)
                        x_pred = x_pred_norm * x_std_t + x_mean_t

                        diff = x_pred - x_true
                        rel_err = torch.norm(diff) / (torch.norm(x_true) + 1e-12)

                        Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                        r  = Ax - b
                        sqrt_w = torch.sqrt(w_pde)
                        wr = sqrt_w * r
                        wb = sqrt_w * b
                        norm_wr = torch.norm(wr)
                        norm_wb = torch.norm(wb) + 1e-12
                        R_pred = norm_wr / norm_wb

                        sum_rel_err_val += rel_err.item()
                        sum_R_pred_val  += R_pred.item()

                avg_rel_err_val = sum_rel_err_val / num_val
                avg_R_pred_val  = sum_R_pred_val / num_val

            log = (
                f"[Epoch {epoch:5d}] loss={loss.item():.4e}, "
                f"data_loss={LAMBDA_DATA * total_data_loss:.4e}, "
                f"PDE_loss={LAMBDA_PDE * total_pde_loss:.4e}, "
                f"rel_err_train(avg)={avg_rel_err_tr:.4e}, "
                f"R_pred_train(avg)={avg_R_pred_tr:.4e}"
            )
            if avg_rel_err_val is not None:
                log += (
                    f", rel_err_val(avg)={avg_rel_err_val:.4e}, "
                    f"R_pred_val(avg)={avg_R_pred_val:.4e}"
                )
            print(log)

    # --- 最終評価 ---
    print("\n=== Final diagnostics (train cases) ===")
    model.eval()
    for cs in cases_train:
        time_str   = cs["time"]
        feats      = cs["feats"]
        edge_index = cs["edge_index"]
        x_true     = cs["x_true"]
        b          = cs["b"]
        row_ptr    = cs["row_ptr"]
        col_ind    = cs["col_ind"]
        vals       = cs["vals"]
        row_idx    = cs["row_idx"]
        w_pde      = cs["w_pde"]

        with torch.no_grad():
            x_pred_norm = model(feats, edge_index)
            x_pred = x_pred_norm * x_std_t + x_mean_t
            diff = x_pred - x_true
            rel_err = torch.norm(diff) / (torch.norm(x_true) + 1e-12)

            Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r  = Ax - b
            sqrt_w = torch.sqrt(w_pde)
            wr = sqrt_w * r
            wb = sqrt_w * b
            norm_wr = torch.norm(wr)
            norm_wb = torch.norm(wb) + 1e-12
            R_pred = norm_wr / norm_wb

        print(
            f"  [train] Case (time={time_str}, rank={RANK_STR}): "
            f"rel_err = {rel_err.item():.4e}, R_pred = {R_pred.item():.4e}"
        )

        x_pred_np = x_pred.cpu().numpy().reshape(-1)
        out_path = os.path.join(DATA_DIR, f"x_pred_train_{time_str}_rank{RANK_STR}.dat")
        with open(out_path, "w") as f:
            for i, val in enumerate(x_pred_np):
                f.write(f"{i} {val:.9e}\n")
        print(f"    [INFO] train x_pred を {out_path} に書き出しました。")

    if num_val > 0:
        print("\n=== Final diagnostics (val cases) ===")
        for cs in cases_val:
            time_str   = cs["time"]
            feats      = cs["feats"]
            edge_index = cs["edge_index"]
            x_true     = cs["x_true"]
            b          = cs["b"]
            row_ptr    = cs["row_ptr"]
            col_ind    = cs["col_ind"]
            vals       = cs["vals"]
            row_idx    = cs["row_idx"]
            w_pde      = cs["w_pde"]

            with torch.no_grad():
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t
                diff = x_pred - x_true
                rel_err = torch.norm(diff) / (torch.norm(x_true) + 1e-12)

                Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                r  = Ax - b
                sqrt_w = torch.sqrt(w_pde)
                wr = sqrt_w * r
                wb = sqrt_w * b
                norm_wr = torch.norm(wr)
                norm_wb = torch.norm(wb) + 1e-12
                R_pred = norm_wr / norm_wb

            print(
                f"  [val]   Case (time={time_str}, rank={RANK_STR}): "
                f"rel_err = {rel_err.item():.4e}, R_pred = {R_pred.item():.4e}"
            )

            x_pred_np = x_pred.cpu().numpy().reshape(-1)
            out_path = os.path.join(DATA_DIR, f"x_pred_val_{time_str}_rank{RANK_STR}.dat")
            with open(out_path, "w") as f:
                for i, val in enumerate(x_pred_np):
                    f.write(f"{i} {val:.9e}\n")
            print(f"    [INFO] val x_pred を {out_path} に書き出しました。")


if __name__ == "__main__":
    train_gnn_auto_trainval_pde_weighted(DATA_DIR)

