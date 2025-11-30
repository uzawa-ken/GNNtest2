# GNN for PDE Solving

A Graph Neural Network (GNN) implementation for solving Partial Differential Equations (PDEs) from Computational Fluid Dynamics (CFD) simulations, with a focus on mesh-quality-weighted loss functions.

## Overview

This project trains a GraphSAGE neural network to predict solutions to pressure equations (pEqn) from CFD simulations. The model combines data-driven learning with physics-informed constraints, using mesh quality metrics to weight the importance of physical consistency across different regions of the computational domain.

## Features

- **Graph Neural Network Architecture**: 4-layer GraphSAGE model for pressure field prediction
- **Physics-Informed Learning**: Dual loss function combining data loss and PDE residual loss
- **Mesh Quality Weighting**: Automatically weights loss based on mesh quality metrics (skewness, non-orthogonality, aspect ratio, size jump)
- **Automatic Train/Val Split**: Configurable data splitting with automatic timestep discovery
- **Sparse Matrix Operations**: Efficient CSR format handling for large-scale CFD data

## Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric
- NumPy

## Installation

```bash
# Install PyTorch (visit pytorch.org for system-specific installation)
pip install torch

# Install PyTorch Geometric
pip install torch-geometric

# Install NumPy
pip install numpy
```

## Data Format

The script expects data files in the `./gnn/` directory with the following naming convention:

```
gnn/
├── pEqn_{timestep}_rank{rank}.dat    # Cell features and graph structure
├── x_{timestep}_rank{rank}.dat       # Ground truth solutions
└── A_csr_{timestep}.dat              # System matrix in CSR format
```

### Input Features (13 per cell)

1. Spatial coordinates (x, y, z)
2. Mesh quality metrics:
   - Skewness
   - Non-orthogonality
   - Aspect ratio
   - Size jump
3. Cell volume
4. Characteristic length
5. Courant number
6. Diagonal values
7. RHS vector

## Configuration

Key parameters can be modified in the script (lines 34-48):

```python
DATA_DIR       = "./gnn"      # Data directory path
RANK_STR       = "7"          # MPI rank identifier
NUM_EPOCHS     = 1000         # Training epochs
LR             = 1e-3         # Learning rate
WEIGHT_DECAY   = 1e-5         # L2 regularization
MAX_NUM_CASES  = 100          # Maximum timesteps to use
TRAIN_FRACTION = 0.8          # Train/validation split ratio
LAMBDA_DATA    = 0.1          # Weight for data loss
LAMBDA_PDE     = 1e-4         # Weight for PDE residual loss
W_PDE_MAX      = 20.0         # Maximum mesh quality weight
```

## Usage

```bash
python train_gnn_auto_trainval_pde_weighted.py
```

The script will:
1. Automatically discover available timestep data files
2. Split data into training and validation sets (80/20 by default)
3. Train the GNN model for the specified number of epochs
4. Generate prediction files for both training and validation cases

## Output

The script generates prediction files in the data directory:

```
x_pred_train_{timestep}_rank{rank}.dat  # Training set predictions
x_pred_val_{timestep}_rank{rank}.dat    # Validation set predictions
```

## Model Architecture

**SimpleSAGE Network:**
- Input layer: 13 features → 64 channels
- Hidden layers: 3 × (64 → 64 channels with ReLU activation)
- Output layer: 64 channels → 1 (pressure prediction)

## Loss Function

The model uses a combined loss function:

```
Total Loss = λ_data × Data Loss + λ_pde × PDE Residual Loss
```

- **Data Loss (MSE)**: Matches predictions to ground truth values
- **PDE Residual Loss**: Ensures physical consistency weighted by mesh quality
  - Higher weights (up to W_PDE_MAX) are applied to cells with poor mesh quality
  - Mesh quality metrics: skewness, non-orthogonality, aspect ratio, size jump

## Mesh Quality Weighting

The PDE loss is weighted based on four mesh quality metrics:

| Metric | Reference Value |
|--------|----------------|
| Skewness | 0.2 |
| Non-orthogonality | 10.0 |
| Aspect Ratio | 5.0 |
| Size Jump | 1.5 |

Weights range from 1.0 (good quality) to W_PDE_MAX=20.0 (poor quality), ensuring the model pays more attention to physically consistent predictions in challenging regions.

## Technical Details

- **Graph Construction**: Automatically extracted from sparse matrix structure
- **Normalization**: Global z-score normalization across all cases
- **Optimizer**: Adam with learning rate 1e-3 and weight decay 1e-5
- **Training Duration**: 1000 epochs (configurable)

## Project Structure

```
GNNtest2/
├── train_gnn_auto_trainval_pde_weighted.py  # Main training script
├── README.md                                 # This file
└── gnn/                                      # Data directory (not included)
```

## License

This project is provided as-is for research and educational purposes.

---

# GNN for PDE Solving (日本語)

計算流体力学（CFD）シミュレーションの偏微分方程式（PDE）を解くためのグラフニューラルネットワーク（GNN）実装です。メッシュ品質による重み付け損失関数を使用しています。

## 概要

このプロジェクトは、CFDシミュレーションから得られる圧力方程式（pEqn）の解を予測するGraphSAGEニューラルネットワークを訓練します。モデルは、データ駆動学習と物理学に基づく制約を組み合わせ、メッシュ品質メトリクスを使用して計算領域の異なる領域における物理的整合性の重要度を重み付けします。

## 主な機能

- **グラフニューラルネットワークアーキテクチャ**: 圧力場予測のための4層GraphSAGEモデル
- **物理学に基づく学習**: データ損失とPDE残差損失を組み合わせた二重損失関数
- **メッシュ品質による重み付け**: メッシュ品質メトリクス（歪度、非直交性、アスペクト比、サイズジャンプ）に基づいて損失を自動的に重み付け
- **自動訓練/検証分割**: 自動タイムステップ検出による設定可能なデータ分割
- **疎行列演算**: 大規模CFDデータの効率的なCSR形式処理

## 必要要件

- Python 3.x
- PyTorch
- PyTorch Geometric
- NumPy

## 使用方法

```bash
python train_gnn_auto_trainval_pde_weighted.py
```

スクリプトは以下を実行します：
1. 利用可能なタイムステップデータファイルを自動検出
2. データを訓練セットと検証セットに分割（デフォルト80/20）
3. 指定されたエポック数でGNNモデルを訓練
4. 訓練セットと検証セットの両方について予測ファイルを生成

## モデルアーキテクチャ

**SimpleSAGEネットワーク:**
- 入力層: 13特徴 → 64チャンネル
- 隠れ層: 3 × (64 → 64チャンネル、ReLU活性化)
- 出力層: 64チャンネル → 1（圧力予測）

## 損失関数

モデルは組み合わせ損失関数を使用します：

```
総損失 = λ_data × データ損失 + λ_pde × PDE残差損失
```

- **データ損失（MSE）**: 予測値を真値に一致させる
- **PDE残差損失**: メッシュ品質で重み付けされた物理的整合性を保証
  - メッシュ品質が低いセルには高い重み（最大W_PDE_MAX）を適用
  - メッシュ品質メトリクス: 歪度、非直交性、アスペクト比、サイズジャンプ

## ライセンス

本プロジェクトは研究および教育目的として提供されています。
