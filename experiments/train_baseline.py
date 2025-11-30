#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Training Script

新しいモジュール構造を使った訓練スクリプト（ベースライン）

このスクリプトは、リファクタリング後のモジュールを使用して、
既存の train_gnn_auto_trainval_pde_weighted.py と同等の機能を提供します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import SimpleSAGE
from utils import find_time_list, load_case_with_csr, matvec_csr_torch
from losses import build_w_pde_from_feats
from config import get_default_config


def convert_case_to_torch(raw_case, device='cpu'):
    """
    NumPy形式のケースデータをPyTorchテンソルに変換

    Parameters
    ----------
    raw_case : dict
        load_case_with_csr() から返されたデータ
    device : str, optional
        使用デバイス ('cpu' or 'cuda')

    Returns
    -------
    dict
        PyTorchテンソルに変換されたデータ
    """
    return {
        'time': raw_case['time'],
        'feats': torch.from_numpy(raw_case['feats_np']).to(device),
        'edge_index': torch.from_numpy(raw_case['edge_index_np']).to(device),
        'x_true': torch.from_numpy(raw_case['x_true_np']).to(device),
        'b': torch.from_numpy(raw_case['b_np']).to(device),
        'row_ptr': torch.from_numpy(raw_case['row_ptr_np']).to(device),
        'col_ind': torch.from_numpy(raw_case['col_ind_np']).to(device),
        'vals': torch.from_numpy(raw_case['vals_np']).to(device),
        'row_idx': torch.from_numpy(raw_case['row_idx_np']).to(device),
    }


def normalize_features(cases):
    """
    全ケースの特徴量をグローバルに正規化

    Parameters
    ----------
    cases : list of dict
        変換済みケースのリスト

    Returns
    -------
    tuple
        (mean, std) 正規化パラメータ
    """
    all_feats = torch.cat([c['feats'] for c in cases], dim=0)
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0) + 1e-8

    # 各ケースの特徴量を正規化
    for case in cases:
        case['feats'] = (case['feats'] - mean) / std

    return mean, std


def train_epoch(model, optimizer, train_cases, config, device='cpu'):
    """
    1エポックの訓練

    Parameters
    ----------
    model : SimpleSAGE
        訓練するモデル
    optimizer : torch.optim.Optimizer
        オプティマイザ
    train_cases : list of dict
        訓練ケースのリスト
    config : Config
        設定
    device : str, optional
        デバイス

    Returns
    -------
    dict
        損失の統計情報
    """
    model.train()

    total_data_loss = 0.0
    total_pde_loss = 0.0
    total_loss = 0.0

    for case in train_cases:
        optimizer.zero_grad()

        # 順伝播
        pred = model(case['feats'], case['edge_index'])

        # データ損失（MSE）
        data_loss = F.mse_loss(pred, case['x_true'])

        # PDE残差損失（メッシュ品質重み付き）
        # Ax̂ - b を計算
        residual = matvec_csr_torch(
            case['row_ptr'],
            case['col_ind'],
            case['vals'],
            case['row_idx'],
            pred
        ) - case['b']

        # メッシュ品質による重み
        feats_np = case['feats'].cpu().numpy()
        w_pde = torch.from_numpy(
            build_w_pde_from_feats(feats_np, config.mesh_quality.w_pde_max)
        ).to(device)

        pde_loss = (w_pde * residual ** 2).mean()

        # 総損失
        loss = (config.training.lambda_data * data_loss +
                config.training.lambda_pde * pde_loss)

        # 逆伝播
        loss.backward()
        optimizer.step()

        # 統計
        total_data_loss += data_loss.item()
        total_pde_loss += pde_loss.item()
        total_loss += loss.item()

    n = len(train_cases)
    return {
        'data_loss': total_data_loss / n,
        'pde_loss': total_pde_loss / n,
        'total_loss': total_loss / n
    }


def evaluate(model, val_cases, config, device='cpu'):
    """
    検証セットで評価

    Parameters
    ----------
    model : SimpleSAGE
        評価するモデル
    val_cases : list of dict
        検証ケースのリスト
    config : Config
        設定
    device : str, optional
        デバイス

    Returns
    -------
    dict
        評価メトリクス
    """
    model.eval()

    total_data_loss = 0.0
    total_pde_loss = 0.0

    with torch.no_grad():
        for case in val_cases:
            pred = model(case['feats'], case['edge_index'])

            # データ損失
            data_loss = F.mse_loss(pred, case['x_true'])

            # PDE残差損失
            residual = matvec_csr_torch(
                case['row_ptr'],
                case['col_ind'],
                case['vals'],
                case['row_idx'],
                pred
            ) - case['b']

            feats_np = case['feats'].cpu().numpy()
            w_pde = torch.from_numpy(
                build_w_pde_from_feats(feats_np, config.mesh_quality.w_pde_max)
            ).to(device)

            pde_loss = (w_pde * residual ** 2).mean()

            total_data_loss += data_loss.item()
            total_pde_loss += pde_loss.item()

    n = len(val_cases)
    return {
        'data_loss': total_data_loss / n,
        'pde_loss': total_pde_loss / n
    }


def main():
    """メイン訓練ループ"""
    print("="*60)
    print("Baseline Training - Using New Module Structure")
    print("="*60)

    # 設定の読み込み
    config = get_default_config()

    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # データの読み込み
    print(f"\nLoading data from {config.data.data_dir}...")
    time_list = find_time_list(config.data.data_dir, config.data.rank_str)

    if not time_list:
        print(f"Error: No data found in {config.data.data_dir}")
        return

    time_list = time_list[:config.data.max_num_cases]
    print(f"Found {len(time_list)} timesteps")

    # 全ケースをロード
    print("\nLoading all cases...")
    raw_cases = []
    for time_str in time_list:
        case = load_case_with_csr(
            config.data.data_dir,
            time_str,
            config.data.rank_str
        )
        raw_cases.append(case)

    # PyTorchテンソルに変換
    all_cases = [convert_case_to_torch(c, device) for c in raw_cases]

    # 特徴量の正規化
    print("Normalizing features...")
    mean, std = normalize_features(all_cases)

    # 訓練/検証分割
    n_train = int(len(all_cases) * config.data.train_fraction)
    train_cases = all_cases[:n_train]
    val_cases = all_cases[n_train:]

    print(f"\nTrain cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")

    # モデルの作成
    print("\nCreating model...")
    model = SimpleSAGE(
        in_channels=config.model.in_channels,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers
    ).to(device)

    print(f"Model: {model}")

    # オプティマイザ
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # 訓練ループ
    print(f"\nTraining for {config.training.num_epochs} epochs...")
    print("-"*60)

    for epoch in range(config.training.num_epochs):
        # 訓練
        train_metrics = train_epoch(model, optimizer, train_cases, config, device)

        # 定期的に評価
        if (epoch + 1) % 100 == 0 or epoch == 0:
            val_metrics = evaluate(model, val_cases, config, device)

            print(f"Epoch {epoch+1:4d}/{config.training.num_epochs} | "
                  f"Train Loss: {train_metrics['total_loss']:.6f} "
                  f"(Data: {train_metrics['data_loss']:.6f}, "
                  f"PDE: {train_metrics['pde_loss']:.6f}) | "
                  f"Val Loss: "
                  f"(Data: {val_metrics['data_loss']:.6f}, "
                  f"PDE: {val_metrics['pde_loss']:.6f})")

    print("-"*60)
    print("Training completed!")

    # 最終評価
    print("\nFinal evaluation...")
    final_val_metrics = evaluate(model, val_cases, config, device)
    print(f"Validation Data Loss: {final_val_metrics['data_loss']:.6f}")
    print(f"Validation PDE Loss: {final_val_metrics['pde_loss']:.6f}")

    print("\n✅ Baseline training finished successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
