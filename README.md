# PUP-HAW-U: Physics-based Uncertainty Propagation with Hierarchical Adaptive Weighting (Unsupervised)

計算流体力学（CFD）シミュレーションの偏微分方程式（PDE）を解くためのグラフニューラルネットワーク（GNN）実装です。メッシュ品質による適応的重み付けと完全教師なし学習をサポートしています。

## 概要

このプロジェクトは、CFDシミュレーションから得られる圧力方程式（pEqn）の解を予測するGraphSAGEニューラルネットワークを訓練します。モデルは、データ駆動学習と物理学に基づく制約を組み合わせ、メッシュ品質メトリクスを使用して計算領域の異なる領域における物理的整合性の重要度を重み付けします。

**実装済みのフェーズ:**
- **Phase 1**: Baseline SimpleSAGE model
- **Phase 2**: Physics-based uncertainty propagation with mesh quality weighting
- **Phase 3**: Hierarchical adaptive weighting (Level 0-2)
- **Phase 4**: Multi-physics constraints (PDE + BC + IC + Conservation laws)
- **Phase 5**: Hybrid learning with curriculum scheduling
- **Phase 6**: Fully unsupervised learning
- **Phase 7**: Experimental framework and ablation study

詳細は [PAPER_OUTLINE.md](./PAPER_OUTLINE.md) を参照してください。

---

## 🚀 クイックスタート / Quick Start

### データローディングの検証

まず、CFDデータが正しくロードできるか確認します：

```bash
cd /path/to/GNNtest2
python test_data_loading.py
```

**成功時の出力:**
```
============================================================
Testing Original Data Format
============================================================
✓ Found 100 time steps
✓ Data loaded successfully!
  Features shape: (4800, 13)
  Solution shape: (4800,)
============================================================
✓ DATA LOADING TEST PASSED!
============================================================
```

### Phase 1 Baseline の実行

データローディングが成功したら、Phase 1のベースラインモデルを訓練します：

```bash
cd experiments
python train_baseline.py --data_dir ../../cylinder/work/data/gnn --rank_str 0 --epochs 10
```

---

## ⚠️ トラブルシューティング / Troubleshooting

### エラー1: "NameError" または "find_time_list() missing required argument"

**症状:**
```python
Traceback (most recent call last):
  File "test_data_loading.py", line 15, in <module>
    time_list = find_time_list(data_dir)
TypeError: find_time_list() missing 1 required positional argument: 'rank_str'
```

**原因:** データローダーの関数は `rank_str` パラメータを必要とします。

**解決策:** 最新の `test_data_loading.py` を使用してください。このスクリプトはランクを自動検出します。

---

### エラー2: "No time steps found!"

**原因:** ファイル命名規則が期待される形式と異なる

**確認事項:**
1. ファイル名が `pEqn_{time}_rank{rank}.dat` の形式か確認
2. 同じディレクトリに `x_{time}_rank{rank}.dat` と `A_csr_{time}.dat` が存在するか確認

```bash
# データディレクトリを確認
ls -l your_data_directory/ | grep -E "(pEqn|x_|A_csr)"

# 期待される出力例:
# pEqn_0.001_rank0.dat
# x_0.001_rank0.dat
# A_csr_0.001.dat
```

**ランク番号の特定:**
```bash
# ファイル名からランク番号を確認
ls your_data_directory/pEqn_*_rank*.dat | head -1
# 例: pEqn_0.001_rank0.dat → rank_str = "0"
```

---

### エラー3: 並列計算データ（processor*/gnn/）の統合

CFDソルバーを並列実行した場合、データが `processor0/gnn/`, `processor1/gnn/`, ... のように分散している可能性があります。

**方法1: シンボリックリンクで統合（推奨）**
```bash
mkdir -p merged_data
ln -s ../processor0/gnn/* merged_data/
# または、特定のrank のみ
ln -s ../processor0/gnn/pEqn_*_rank0.dat merged_data/
ln -s ../processor0/gnn/x_*_rank0.dat merged_data/
ln -s ../processor0/gnn/A_csr_*.dat merged_data/
```

**方法2: データをコピー**
```bash
mkdir -p merged_data
cp processor*/gnn/*_rank0.dat merged_data/
cp processor*/gnn/A_csr_*.dat merged_data/
```

その後、`merged_data/` を `--data_dir` として指定します。

---

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

## インストール

```bash
# PyTorchのインストール（システムに応じた詳細はpytorch.orgを参照）
pip install torch

# PyTorch Geometricのインストール
pip install torch-geometric

# NumPyのインストール
pip install numpy
```

## データフォーマット

スクリプトは`./gnn/`ディレクトリ内に以下の命名規則に従ったデータファイルを必要とします：

```
gnn/
├── pEqn_{timestep}_rank{rank}.dat    # セル特徴量とグラフ構造
├── x_{timestep}_rank{rank}.dat       # 正解データ
└── A_csr_{timestep}.dat              # CSR形式のシステム行列
```

### 入力特徴量（セルあたり13個）

1. 空間座標（x, y, z）
2. メッシュ品質メトリクス：
   - 歪度（Skewness）
   - 非直交性（Non-orthogonality）
   - アスペクト比（Aspect ratio）
   - サイズジャンプ（Size jump）
3. セル体積
4. 特性長さ
5. クーラン数
6. 対角値
7. 右辺ベクトル

## 設定

主要なパラメータはスクリプト内（34-48行目）で変更できます：

```python
DATA_DIR       = "./gnn"      # データディレクトリのパス
RANK_STR       = "7"          # MPIランク識別子
NUM_EPOCHS     = 1000         # 訓練エポック数
LR             = 1e-3         # 学習率
WEIGHT_DECAY   = 1e-5         # L2正則化
MAX_NUM_CASES  = 100          # 使用する最大タイムステップ数
TRAIN_FRACTION = 0.8          # 訓練/検証データの分割比率
LAMBDA_DATA    = 0.1          # データ損失の重み
LAMBDA_PDE     = 1e-4         # PDE残差損失の重み
W_PDE_MAX      = 20.0         # メッシュ品質重みの最大値
```

## 使用方法

```bash
python train_gnn_auto_trainval_pde_weighted.py
```

スクリプトは以下を実行します：
1. 利用可能なタイムステップデータファイルを自動検出
2. データを訓練セットと検証セットに分割（デフォルト80/20）
3. 指定されたエポック数でGNNモデルを訓練
4. 訓練セットと検証セットの両方について予測ファイルを生成

## 出力

スクリプトはデータディレクトリ内に以下の予測ファイルを生成します：

```
x_pred_train_{timestep}_rank{rank}.dat  # 訓練セットの予測結果
x_pred_val_{timestep}_rank{rank}.dat    # 検証セットの予測結果
```

## モデルアーキテクチャ

**SimpleSAGEネットワーク:**
- 入力層: 13特徴 → 64チャンネル
- 隠れ層: 3層 × (64 → 64チャンネル、ReLU活性化関数)
- 出力層: 64チャンネル → 1（圧力予測値）

## 損失関数

モデルは以下の組み合わせ損失関数を使用します：

### 総損失

```
ℒ_total = λ_data × ℒ_data + λ_pde × ℒ_pde
```

ここで、λ_data = 0.1、λ_pde = 1e-4 がデフォルト値です。

### データ損失（MSE）

予測値 *x̂* を正解データ *x* に一致させるための損失：

```
ℒ_data = (1/N) Σᵢ (x̂ᵢ - xᵢ)²
```

ここで、*N* はセル数、*i* はセルのインデックスです。

### PDE残差損失

物理的整合性を保証するため、システム行列 *A* を用いた残差を最小化します：

```
ℒ_pde = (1/N) Σᵢ wᵢ × (Ax̂ - b)ᵢ²
```

ここで：
- *A*: システム行列（CSR形式の疎行列）
- *b*: 右辺ベクトル
- *wᵢ*: セル *i* のメッシュ品質に基づく重み（詳細は下記参照）
- *Ax̂ - b*: PDE残差ベクトル

メッシュ品質が低いセルには高い重み *wᵢ*（最大W_PDE_MAX=20.0）が適用され、物理法則の満足度がより重視されます。

## メッシュ品質による重み付け

PDE損失は以下の4つのメッシュ品質メトリクスに基づいて重み付けされます：

| メトリクス | 基準値 | 重み係数 |
|-----------|-------|---------|
| 歪度（Skewness） | 0.2 | 1.0 |
| 非直交性（Non-orthogonality） | 10.0 | 0.5 |
| アスペクト比（Aspect Ratio） | 5.0 | 0.5 |
| サイズジャンプ（Size Jump） | 1.5 | 1.0 |

### 重み計算式

各セルのPDE損失の重み *w* は以下の手順で計算されます：

**1. 正規化されたメッシュ品質指標の計算:**

各メトリクスを基準値で正規化し、0.0から5.0の範囲にクリップします：

```
q_skew      = clip(skew / 0.2,  0.0, 5.0)
q_non_ortho = clip(non_ortho / 10.0, 0.0, 5.0)
q_aspect    = clip(aspect / 5.0, 0.0, 5.0)
q_sizeJump  = clip(size_jump / 1.5, 0.0, 5.0)
```

**2. 生の重みの計算:**

正規化された指標を重み係数で加重平均します：

```
w_raw = 1.0 + 1.0 × (q_skew - 1.0)
            + 0.5 × (q_non_ortho - 1.0)
            + 0.5 × (q_aspect - 1.0)
            + 1.0 × (q_sizeJump - 1.0)
```

**3. 最終的な重みの決定:**

生の重みを1.0からW_PDE_MAX（=20.0）の範囲にクリップします：

```
w = clip(w_raw, 1.0, 20.0)
```

この計算により、メッシュ品質が基準値を超えるほど重みが大きくなり、品質の低いセルにおいて物理的整合性（PDE残差の最小化）がより重要視されます。歪度とサイズジャンプには重み係数1.0、非直交性とアスペクト比には重み係数0.5が適用されます。

## 技術的詳細

- **グラフ構造**: 疎行列構造から自動的に抽出
- **正規化**: 全ケースにわたるグローバルz-score正規化
- **最適化手法**: Adam（学習率1e-3、重み減衰1e-5）
- **訓練期間**: 1000エポック（設定変更可能）

## プロジェクト構成

```
GNNtest2/
├── train_gnn_auto_trainval_pde_weighted.py  # メイン訓練スクリプト
├── README.md                                 # このファイル
└── gnn/                                      # データディレクトリ（含まれていません）
```

## ライセンス

本プロジェクトは研究および教育目的として提供されています。
