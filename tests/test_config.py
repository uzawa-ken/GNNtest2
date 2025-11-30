#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Config Module

設定モジュールのユニットテスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    MeshQualityConfig,
    get_default_config,
    create_config
)


def test_config_import():
    """設定クラスがインポートできるか"""
    assert Config is not None
    assert DataConfig is not None
    assert ModelConfig is not None
    assert TrainingConfig is not None
    assert MeshQualityConfig is not None


def test_default_config():
    """デフォルト設定が取得できるか"""
    config = get_default_config()

    assert isinstance(config, Config)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.mesh_quality, MeshQualityConfig)


def test_data_config_defaults():
    """データ設定のデフォルト値が正しいか"""
    config = get_default_config()

    assert config.data.data_dir == "./gnn"
    assert config.data.rank_str == "7"
    assert config.data.max_num_cases == 100
    assert config.data.train_fraction == 0.8


def test_model_config_defaults():
    """モデル設定のデフォルト値が正しいか"""
    config = get_default_config()

    assert config.model.in_channels == 13
    assert config.model.hidden_channels == 64
    assert config.model.num_layers == 4


def test_training_config_defaults():
    """訓練設定のデフォルト値が正しいか"""
    config = get_default_config()

    assert config.training.num_epochs == 1000
    assert config.training.learning_rate == 1e-3
    assert config.training.weight_decay == 1e-5
    assert config.training.lambda_data == 0.1
    assert config.training.lambda_pde == 1e-4


def test_mesh_quality_config_defaults():
    """メッシュ品質設定のデフォルト値が正しいか"""
    config = get_default_config()

    assert config.mesh_quality.w_pde_max == 20.0
    assert config.mesh_quality.skew_ref == 0.2
    assert config.mesh_quality.nonorth_ref == 10.0
    assert config.mesh_quality.aspect_ref == 5.0
    assert config.mesh_quality.sizejump_ref == 1.5


def test_create_custom_config():
    """カスタム設定が作成できるか"""
    config = create_config(
        data_dir="./my_data",
        num_epochs=500,
        learning_rate=5e-4
    )

    assert config.data.data_dir == "./my_data"
    assert config.training.num_epochs == 500
    assert config.training.learning_rate == 5e-4


def test_config_modification():
    """設定を変更できるか"""
    config = get_default_config()

    # 値を変更
    config.data.max_num_cases = 200
    config.training.num_epochs = 2000

    assert config.data.max_num_cases == 200
    assert config.training.num_epochs == 2000


if __name__ == '__main__':
    print("Running config tests...")

    test_config_import()
    print("✓ Import test passed")

    test_default_config()
    print("✓ Default config test passed")

    test_data_config_defaults()
    print("✓ Data config defaults test passed")

    test_model_config_defaults()
    print("✓ Model config defaults test passed")

    test_training_config_defaults()
    print("✓ Training config defaults test passed")

    test_mesh_quality_config_defaults()
    print("✓ Mesh quality config defaults test passed")

    test_create_custom_config()
    print("✓ Custom config test passed")

    test_config_modification()
    print("✓ Config modification test passed")

    print("\n✅ All config tests passed!")
