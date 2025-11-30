#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Configuration

プロジェクト全体の基本設定パラメータ
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """データ関連の設定"""

    # データディレクトリ
    data_dir: str = "./gnn"

    # MPIランク識別子
    rank_str: str = "7"

    # 使用する最大ケース数
    max_num_cases: int = 100

    # 訓練/検証データの分割比率
    train_fraction: float = 0.8


@dataclass
class ModelConfig:
    """モデル関連の設定"""

    # 入力特徴量の次元数
    in_channels: int = 13

    # 隠れ層のチャンネル数
    hidden_channels: int = 64

    # 総レイヤー数
    num_layers: int = 4


@dataclass
class TrainingConfig:
    """訓練関連の設定"""

    # エポック数
    num_epochs: int = 1000

    # 学習率
    learning_rate: float = 1e-3

    # L2正則化係数
    weight_decay: float = 1e-5

    # データ損失の重み
    lambda_data: float = 0.1

    # PDE残差損失の重み
    lambda_pde: float = 1e-4


@dataclass
class MeshQualityConfig:
    """メッシュ品質重み関連の設定"""

    # PDE損失重みの最大値
    w_pde_max: float = 20.0

    # 基準値（現在のベースライン）
    # Phase 2で適応的基準値に更新予定
    skew_ref: float = 0.2
    nonorth_ref: float = 10.0
    aspect_ref: float = 5.0
    sizejump_ref: float = 1.5

    # 重み係数（現在のベースライン）
    # Phase 2で学習可能パラメータに更新予定
    alpha_skew: float = 1.0
    alpha_nonorth: float = 0.5
    alpha_aspect: float = 0.5
    alpha_sizejump: float = 1.0


@dataclass
class Config:
    """全体設定"""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    mesh_quality: MeshQualityConfig = MeshQualityConfig()

    # 実験名（オプション）
    experiment_name: Optional[str] = None

    # 乱数シード（オプション）
    random_seed: Optional[int] = None


def get_default_config() -> Config:
    """
    デフォルト設定を取得

    Returns
    -------
    Config
        デフォルト設定オブジェクト

    Examples
    --------
    >>> config = get_default_config()
    >>> print(config.training.num_epochs)
    1000
    >>> print(config.data.data_dir)
    ./gnn
    """
    return Config()


def create_config(
    data_dir: str = "./gnn",
    num_epochs: int = 1000,
    learning_rate: float = 1e-3,
    **kwargs
) -> Config:
    """
    カスタム設定を作成

    Parameters
    ----------
    data_dir : str, optional
        データディレクトリ
    num_epochs : int, optional
        訓練エポック数
    learning_rate : float, optional
        学習率
    **kwargs
        その他の設定パラメータ

    Returns
    -------
    Config
        カスタム設定オブジェクト

    Examples
    --------
    >>> config = create_config(
    ...     data_dir="./my_data",
    ...     num_epochs=500,
    ...     learning_rate=5e-4
    ... )
    >>> print(config.training.num_epochs)
    500
    """
    config = get_default_config()

    # データ設定
    config.data.data_dir = data_dir

    # 訓練設定
    config.training.num_epochs = num_epochs
    config.training.learning_rate = learning_rate

    # その他のパラメータを更新
    for key, value in kwargs.items():
        # ネストされた設定を更新
        if '.' in key:
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)

    return config
