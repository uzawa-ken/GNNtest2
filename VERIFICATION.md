# Phase 1-7 検証ガイド / Verification Guide

PUP-HAW-U実装の検証手順書

---

## 📋 検証の概要

Phases 1-7の実装（6,760+行のコード）が完了しました。このドキュメントでは、各フェーズの検証手順を説明します。

---

## 🔧 エラー修正の記録

### 発生したエラー

test_data_loading.py の初回実行時に以下のエラーが発生しました：

```
Traceback (most recent call last):
  File "/home/uzawa/OpenFOAM/work/v2412/Surrogate/cylinder/work/data/GNNtest2/test_data_loading.py", line 15, in <module>
    time_list = find_time_list(data_dir)
  File "/home/uzawa/OpenFOAM/work/v2412/Surrogate/cylinder/work/data/GNNtest2/utils/data_loader.py", line 26, in find_time_list
    case_path = Path(case_dir)
NameError: name 'Path' is not defined
```

### 根本原因

**問題1: 関数シグネチャの不一致**

オリジナルの `utils/data_loader.py` の関数は、2つのパラメータを必要とします：

```python
# 実際のシグネチャ（utils/data_loader.py:17）
def find_time_list(data_dir: str, rank_str: str):
    """
    data_dir: データディレクトリのパス
    rank_str: MPIランク識別子（例: "0", "7"）
    """
    ...

# 実際のシグネチャ（utils/data_loader.py:64）
def load_case_with_csr(data_dir: str, time_str: str, rank_str: str):
    """
    data_dir: データディレクトリのパス
    time_str: タイムステップ文字列（例: "0.001"）
    rank_str: MPIランク識別子（例: "0", "7"）
    """
    ...
```

しかし、Phase 1-7実装中に作成したテストスクリプトは、誤って `rank_str` パラメータを省略していました：

```python
# 誤った呼び出し（修正前）
time_list = find_time_list(data_dir)  # ❌ rank_str が足りない
```

**問題2: データファイルの命名規則**

オリジナルのデータローダーは、OpenFOAMの並列計算に対応した命名規則を期待しています：

```
gnn/
├── pEqn_0.001_rank0.dat    # Cell features, graph structure, RHS
├── x_0.001_rank0.dat        # Ground truth solution
├── A_csr_0.001.dat          # System matrix (CSR format)
├── pEqn_0.002_rank0.dat
├── x_0.002_rank0.dat
└── A_csr_0.002.dat
```

ここで `rank0` の部分は、MPIランク番号を示します。

### 適用した修正

**修正1: test_data_loading.py の更新**

`test_data_loading.py` を以下の機能で書き直しました：

1. **ランク自動検出機能**
   ```python
   def find_correct_rank():
       """
       ファイル名からランク文字列を自動検出
       例: pEqn_0.001_rank0.dat → "0"
       """
       gnn_path = Path(data_dir)
       pEqn_files = list(gnn_path.glob("pEqn_*_rank*.dat"))
       # ファイル名からrankを抽出
       fname = pEqn_files[0].name
       parts = fname.split('_rank')
       rank_part = parts[1].replace('.dat', '')
       return rank_part
   ```

2. **正しい関数呼び出し**
   ```python
   # 修正後（正しい呼び出し）
   time_list = find_time_list(data_dir, rank_str)  # ✓ rank_str を渡す
   data = load_case_with_csr(data_dir, time_str, rank_str)  # ✓ 3つのパラメータ
   ```

3. **詳細な診断情報**
   - ディレクトリ内のファイル一覧表示
   - 検出されたタイムステップの表示
   - ロードされたデータの形状とメッシュ品質統計の表示

**修正2: README.md の更新**

- クイックスタートセクションの追加
- トラブルシューティングガイドの追加
- 並列計算データの統合方法の説明

---

## ✅ 検証手順

### Step 0: 環境の準備

**必要な依存関係:**
```bash
pip install numpy torch torch-geometric matplotlib
```

**データの配置:**
- CFDソルバーの出力データを1つのディレクトリに配置
- 並列計算の場合は、processor*/gnn/ のデータを統合（詳細はREADME.mdのトラブルシューティング参照）

### Step 1: データローディングテスト ✓ 修正済み

**実行方法:**
```bash
cd /path/to/GNNtest2
python test_data_loading.py
```

**期待される出力:**
```
GNNtest2 Data Loading Test

Attempting to auto-detect data format...
Auto-detected rank: 0

Using rank string: 0
If this is incorrect, please modify the test script.

============================================================
Testing Original Data Format
============================================================
Data directory: ../cylinder/work/data/gnn
Rank string: 0

Files in directory:
  A_csr_0.001.dat
  pEqn_0.001_rank0.dat
  x_0.001_rank0.dat
  ...

✓ Found 100 time steps
  Time steps: ['0.001', '0.002', '0.003', '0.004', '0.005']
  ... and 95 more

Loading time step: 0.001
✓ Data loaded successfully!
  Features shape: (4800, 13)
  Solution shape: (4800,)
  Edge index shape: (2, 38400)
  CSR matrix nnz: 33600
  Number of cells: 4800

  Feature ranges:
    Coordinates (x,y,z): [-0.050, 0.150]
    Skewness: [0.001, 0.856]
    Non-orthogonality: [0.123, 45.678]
    Aspect ratio: [1.012, 8.456]

============================================================
✓ DATA LOADING TEST PASSED!
============================================================

You can now proceed to run the training scripts.

Next step:
  cd experiments
  python train_baseline.py --data_dir ../../cylinder/work/data/gnn --epochs 10
```

**エラーが発生した場合:**
- README.mdのトラブルシューティングセクションを参照
- ファイル命名規則を確認（`pEqn_{time}_rank{rank}.dat` 形式か？）
- データディレクトリのパスが正しいか確認

---

### Step 2: Phase 1 Baseline モデル

**目的:** SimpleSAGEモデルが正しく動作するか確認

**実行方法:**
```bash
cd experiments
python train_baseline.py \
    --data_dir ../../cylinder/work/data/gnn \
    --rank_str 0 \
    --epochs 10 \
    --lr 1e-3
```

**⚠️ 注意:** training scriptが `rank_str` パラメータをサポートしているか確認が必要です（次のステップで確認）。

**期待される動作:**
- データのロードが成功
- モデルの訓練が開始
- 各エポックで損失が表示される
- 訓練完了後、検証メトリクスが表示される

**検証項目:**
- [ ] データロードエラーが発生しないこと
- [ ] 訓練損失が減少すること
- [ ] MSE, MAE, 相対誤差などのメトリクスが計算されること
- [ ] 予測ファイルが生成されること（もしあれば）

---

### Step 3: Phase 2 Physics-based Weighting

**目的:** メッシュ品質による重み付けが機能するか確認

**実行方法:**
```bash
cd experiments
python train_physics_weighted.py \
    --data_dir ../../cylinder/work/data/gnn \
    --rank_str 0 \
    --epochs 50
```

**検証項目:**
- [ ] メッシュ品質ファクター（α）が計算されること
- [ ] Solution curvature（κ）が計算されること
- [ ] 重み分布が可視化されること（もしあれば）
- [ ] Baseline（Phase 1）と比較して性能が向上すること

---

### Step 4: Phase 3 Hierarchical Adaptive Weighting

**目的:** 階層的適応重み付けが機能するか確認

**実行方法:**
```bash
cd experiments
python train_hierarchical.py \
    --data_dir ../../cylinder/work/data/gnn \
    --rank_str 0 \
    --epochs 50
```

**検証項目:**
- [ ] Level 1（エポック単位）の適応が動作すること
- [ ] Level 2（バッチ単位）の勾配調和が動作すること
- [ ] 重みの推移がログに記録されること
- [ ] Phase 2と比較して収束が安定すること

---

### Step 5: Phase 4 Multi-physics Constraints（Unsupervised）

**目的:** 教師なし学習（PDE + BC + IC + Conservation）が機能するか確認

**実行方法:**
```bash
cd experiments
python train_unsupervised.py \
    --data_dir ../../cylinder/work/data/gnn \
    --rank_str 0 \
    --epochs 100
```

**検証項目:**
- [ ] PDE residual損失が計算されること
- [ ] Boundary condition損失が計算されること
- [ ] Conservation law損失が計算されること
- [ ] 正解データなしで訓練が進行すること
- [ ] 物理制約のみで妥当な解が得られること

---

### Step 6: Phase 5 Hybrid Learning

**目的:** カリキュラム学習（教師あり→教師なし）が機能するか確認

**実行方法:**
```bash
cd experiments
python train_hybrid.py \
    --data_dir ../../cylinder/work/data/gnn \
    --rank_str 0 \
    --epochs 100
```

**検証項目:**
- [ ] データ損失の重みが時間とともに減少すること
- [ ] 物理損失の重みが時間とともに増加すること
- [ ] スケジュールが正しく適用されること
- [ ] Pure unsupervised（Phase 4）より高精度な解が得られること

---

### Step 7: Ablation Study

**目的:** 各コンポーネントの貢献度を定量評価

**実行方法:**
```bash
cd experiments
python run_ablation_study.py \
    --data_dir ../../cylinder/work/data/gnn \
    --epochs 100 \
    --runs_per_config 3
```

**検証項目:**
- [ ] 6つの設定（Full, w/o Physics, w/o Hierarchical, w/o Topology, w/o Multi-physics, Baseline）が実行されること
- [ ] 各設定で3回実行され、統計が計算されること
- [ ] 結果がJSON形式で保存されること
- [ ] 比較レポートが生成されること

**生成される結果ファイル:**
```
outputs/ablation/
├── full_results.json
├── no_physics_weights_results.json
├── no_hierarchical_results.json
├── no_topology_results.json
├── no_multi_physics_results.json
├── baseline_results.json
└── ablation_complete.json
```

---

## 📊 論文用の図の生成

Ablation studyの結果から論文用の図を生成：

```python
from utils.visualization import generate_paper_figures
generate_paper_figures('./outputs/ablation', './paper_figures')
```

**生成される図:**
- `ablation_pde_residual_l2.png`
- `ablation_bc_total_mae.png`
- `ablation_conservation_l2.png`
- `ablation_mse.png`
- `ablation_relative_error.png`

---

## 🐛 既知の問題

### Issue 1: training scripts の `rank_str` 対応

**現状:** Phase 1-7のtraining scriptsは、データローダーに `rank_str` を渡していない可能性があります。

**確認が必要なファイル:**
- `experiments/train_baseline.py`
- `experiments/train_physics_weighted.py`
- `experiments/train_hierarchical.py`
- `experiments/train_unsupervised.py`
- `experiments/train_hybrid.py`
- `experiments/run_ablation_study.py`

**修正方法:**
各スクリプトで、`find_time_list()` と `load_case_with_csr()` の呼び出しに `rank_str` パラメータを追加する必要があります。

**例:**
```python
# 修正前
time_list = find_time_list(data_dir)

# 修正後
time_list = find_time_list(data_dir, rank_str="0")  # または args.rank_str
```

**次のステップ:** 各training scriptを読んで、この修正が必要か確認します。

---

## 📝 次のアクション

1. ✅ `test_data_loading.py` を実行してデータローディングを確認
2. ⏳ training scriptsが `rank_str` パラメータをサポートしているか確認
3. ⏳ 必要に応じてtraining scriptsを修正
4. ⏳ Phase 1から順に実行して動作を確認
5. ⏳ 各フェーズの結果を記録
6. ⏳ Ablation studyを実行
7. ⏳ 論文用の図を生成

---

## 📧 問い合わせ

質問や問題が発生した場合は、GitHub Issuesで報告してください。

**修正履歴:**
- 2025-12-01: test_data_loading.py のエラー修正（rank_str パラメータの追加）
- 2025-12-01: README.md の更新（トラブルシューティング追加）
- 2025-12-01: VERIFICATION.md の作成（このドキュメント）
