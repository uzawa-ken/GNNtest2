# PUP-HAW-U 実装計画
**Physics-based Uncertainty Propagation with Hierarchical Adaptive Weighting - Unsupervised**

## 目標
現在の教師あり学習モデルから、完全教師なし学習のPUP-HAW-Uモデルへ段階的に移行する。

---

## プロジェクト構成（最終形）

```
GNNtest2/
├── train_gnn_auto_trainval_pde_weighted.py  # 既存（Phase 1で段階的に改修）
├── models/
│   ├── __init__.py
│   ├── sage_model.py                         # SimpleSAGEモデル（既存から移動）
│   └── gnn_pde_solver.py                     # 新規：PUP-HAW-U統合モデル
├── losses/
│   ├── __init__.py
│   ├── base_loss.py                          # 基底クラス
│   ├── mesh_quality_weights.py               # Phase 2: 物理的不確実性伝播
│   ├── hierarchical_adaptive.py              # Phase 3: 階層的適応機構
│   └── multi_physics_loss.py                 # Phase 4: マルチ物理制約
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                        # データローディング（既存から移動）
│   ├── graph_ops.py                          # グラフ演算（ラプラシアンなど）
│   └── mesh_analysis.py                      # メッシュ品質分析
├── config/
│   ├── __init__.py
│   ├── base_config.py                        # 基本設定
│   └── experiment_configs.py                 # 実験設定
├── experiments/
│   ├── train_supervised.py                   # Phase 1: 既存（改良版）
│   ├── train_hybrid.py                       # Phase 5: ハイブリッド学習
│   └── train_unsupervised.py                 # Phase 6: 完全教師なし
├── tests/
│   ├── test_mesh_weights.py
│   ├── test_hierarchical_adaptive.py
│   └── test_multi_physics.py
├── notebooks/
│   └── visualization.ipynb                   # 結果可視化
├── README.md                                 # 既存
└── IMPLEMENTATION_PLAN.md                    # このファイル
```

---

## Phase 1: 基礎インフラ構築（1-2週間）

### 目標
- 既存コードのリファクタリング
- モジュール化された構造への移行
- テスト環境の構築

### タスク

#### 1.1 プロジェクト構造の作成
```bash
mkdir -p models losses utils config experiments tests notebooks
touch models/__init__.py losses/__init__.py utils/__init__.py config/__init__.py
```

#### 1.2 既存コードのモジュール分割

**タスク 1.2.1: モデルの分離**
- [ ] `SimpleSAGE` を `models/sage_model.py` に移動
- [ ] モデルの柔軟性向上（設定ファイルから読み込み）

**タスク 1.2.2: データローダーの分離**
- [ ] `find_time_list()`, `load_case_with_csr()` を `utils/data_loader.py` に移動
- [ ] データローディングのクラス化

**タスク 1.2.3: 損失関数の分離**
- [ ] `build_w_pde_from_feats()` を `losses/mesh_quality_weights.py` に移動
- [ ] 基底損失クラス `losses/base_loss.py` を作成

**タスク 1.2.4: 設定管理**
- [ ] `config/base_config.py` を作成
- [ ] ハードコードされた定数を設定ファイルに移動

#### 1.3 テスト環境の構築

**タスク 1.3.1: ユニットテストの作成**
- [ ] `tests/test_data_loader.py`: データローディングのテスト
- [ ] `tests/test_mesh_weights.py`: メッシュ品質重みのテスト
- [ ] `tests/test_model.py`: モデルの forward/backward テスト

**タスク 1.3.2: 統合テスト**
- [ ] 既存機能が正しく動作することを確認
- [ ] リファクタリング後の結果が元のコードと一致することを検証

### 成果物
- ✅ モジュール化されたコードベース
- ✅ 全てのユニットテストが通過
- ✅ 既存の訓練スクリプトが正常動作

### マイルストーン
**M1.1**: モジュール分割完了、テスト通過

---

## Phase 2: 物理的不確実性伝播の実装（2週間）

### 目標
- メッシュ品質と解の曲率を結合した重み計算
- グラフラプラシアンによる2階微分近似
- トポロジー認識型重み伝播

### タスク

#### 2.1 グラフ演算の実装

**タスク 2.1.1: `utils/graph_ops.py` の作成**
```python
# 実装する関数:
- compute_graph_laplacian()     # ∇²φ の近似
- compute_gradient()            # ∇φ の近似
- propagate_weights()           # トポロジー認識伝播
```

**タスク 2.1.2: テストの作成**
- [ ] `tests/test_graph_ops.py`
- [ ] 既知の解析解との比較テスト

#### 2.2 物理的不確実性伝播重みの実装

**タスク 2.2.1: `losses/mesh_quality_weights.py` の拡張**
```python
class MeshQualityWeight:
    def __init__(self, config):
        pass

    def compute_basic_weights(feats):
        """既存の線形重み（ベースライン）"""
        pass

    def compute_physics_based_weights(feats, pred, edges):
        """新規：物理的不確実性伝播重み"""
        pass
```

**タスク 2.2.2: 誤差増幅係数の実装**
- [ ] メッシュ品質メトリクスの2次効果
- [ ] 解の曲率との結合
- [ ] パラメータチューニング機能

#### 2.3 トポロジー認識伝播の実装

**タスク 2.3.1: 重み伝播アルゴリズム**
```python
def topology_aware_weight_propagation(w_local, edges, num_hops=2):
    """近傍への重み伝播"""
    pass
```

**タスク 2.3.2: パラメータスタディ**
- [ ] num_hops の影響を分析（1, 2, 3）
- [ ] 減衰係数の最適化

### 成果物
- ✅ `utils/graph_ops.py` 実装完了
- ✅ 物理的不確実性伝播重みが動作
- ✅ トポロジー認識伝播が実装され検証済み

### マイルストーン
**M2.1**: グラフ演算実装完了
**M2.2**: 物理的重み計算が既存重みより高精度

---

## Phase 3: 階層的適応機構の実装（2-3週間）

### 目標
- Level 1: エポック単位の大域的適応
- Level 2: バッチ単位の勾配調和
- Level 3: セル単位の物理的重み（Phase 2で実装済み）

### タスク

#### 3.1 Level 1: エポック単位適応

**タスク 3.1.1: `losses/hierarchical_adaptive.py` の作成**
```python
class HierarchicalAdaptiveWeighting:
    def __init__(self, constraint_types):
        self.lambdas = {}  # 各損失の重み
        self.loss_history = {}

    def update_level1_epoch(self, epoch, loss_history):
        """減衰率不均衡に基づく調整"""
        pass
```

**タスク 3.1.2: 減衰率モニタリング**
- [ ] 各損失の減衰率計算
- [ ] 不均衡検出アルゴリズム
- [ ] 適応的重み更新

**タスク 3.1.3: 可視化**
- [ ] 訓練中の重み変化をプロット
- [ ] 減衰率の時系列プロット

#### 3.2 Level 2: バッチ単位勾配調和

**タスク 3.2.1: 勾配競合検出**
```python
def update_level2_batch(self, gradients):
    """コサイン類似度に基づく調和"""
    pass
```

**タスク 3.2.2: 勾配スケーリング**
- [ ] コサイン類似度の計算
- [ ] 競合時の調整係数
- [ ] スムーズな遷移

#### 3.3 統合と検証

**タスク 3.3.1: 3層統合**
```python
def get_final_weights(self, w_cell_physics):
    """Level 1-3 を統合した最終重み"""
    return {
        'lambda_data': self.lambda_data_global * self.alpha_batch,
        'lambda_pde': self.lambda_pde_global * self.alpha_batch,
        'w_pde_cell': w_cell_physics
    }
```

**タスク 3.3.2: テストとデバッグ**
- [ ] `tests/test_hierarchical_adaptive.py`
- [ ] 各レベルの独立動作確認
- [ ] 統合動作確認

### 成果物
- ✅ 3層階層的適応機構が動作
- ✅ 訓練安定性の向上を確認
- ✅ 適応過程の可視化

### マイルストーン
**M3.1**: Level 1 実装完了
**M3.2**: Level 2 実装完了
**M3.3**: 統合テスト通過、訓練時間短縮を確認

---

## Phase 4: マルチ物理制約の実装（2-3週間）

### 目標
- 境界条件損失の実装
- 初期条件損失の実装
- 保存則損失の実装
- マルチ制約間の階層的適応

### タスク

#### 4.1 境界条件の実装

**タスク 4.1.1: 境界情報の抽出**
```python
# utils/mesh_analysis.py
def extract_boundary_nodes(edges, feats):
    """グラフ構造から境界ノードを抽出"""
    pass
```

**タスク 4.1.2: 境界条件損失**
```python
# losses/multi_physics_loss.py
class BoundaryConditionLoss:
    def __init__(self, boundary_info):
        pass

    def compute_dirichlet_loss(self, pred):
        """ディリクレ境界条件"""
        pass

    def compute_neumann_loss(self, pred, edges):
        """ノイマン境界条件（勾配）"""
        pass
```

#### 4.2 初期条件の実装

**タスク 4.2.1: 時系列データの処理**
- [ ] タイムステップ間の関係を定義
- [ ] 初期条件の指定方法

**タスク 4.2.2: 初期条件損失**
```python
class InitialConditionLoss:
    def compute(self, pred_t0, ic_values):
        pass
```

#### 4.3 保存則の実装

**タスク 4.3.1: 質量保存則**
```python
class ConservationLoss:
    def compute_mass_conservation(self, pred, edges):
        """∇·u = 0"""
        pass
```

**タスク 4.3.2: その他の保存則**
- [ ] 運動量保存（必要に応じて）
- [ ] エネルギー保存（必要に応じて）

#### 4.4 マルチ制約階層適応

**タスク 4.4.1: `HierarchicalAdaptiveWeighting` の拡張**
```python
# 2つの損失 → N個の物理制約へ拡張
constraint_types = ['pde', 'bc', 'ic', 'conservation']
```

**タスク 4.4.2: N個の制約間での勾配調和**
- [ ] ペアワイズ競合検出
- [ ] マルチ制約バランシング

### 成果物
- ✅ 境界条件損失が動作
- ✅ 保存則損失が動作
- ✅ マルチ制約適応が機能

### マイルストーン
**M4.1**: 境界条件実装完了
**M4.2**: 保存則実装完了
**M4.3**: マルチ制約適応動作確認

---

## Phase 5: ハイブリッド学習（1-2週間）

### 目標
- 教師ありから教師なしへの段階的移行
- カリキュラム学習の実装

### タスク

#### 5.1 ハイブリッド損失の実装

**タスク 5.1.1: `experiments/train_hybrid.py` の作成**
```python
# 教師データの重みを徐々に減少
lambda_data(epoch) = lambda_data_init * exp(-alpha * epoch)
```

**タスク 5.1.2: スケジューリング**
- [ ] 線形減衰
- [ ] 指数減衰
- [ ] ステップ減衰

#### 5.2 検証

**タスク 5.2.1: 移行過程のモニタリング**
- [ ] 各エポックでの精度追跡
- [ ] 物理制約満足度の追跡

**タスク 5.2.2: 最適なスケジュールの決定**
- [ ] 複数のスケジュールを試行
- [ ] 最良の移行曲線を特定

### 成果物
- ✅ ハイブリッド学習スクリプト
- ✅ 最適な移行スケジュール

### マイルストーン
**M5.1**: ハイブリッド学習が安定動作

---

## Phase 6: 完全教師なし学習（2週間）

### 目標
- PUP-HAW-U の完全実装
- 教師データなしでの訓練

### タスク

#### 6.1 教師なし訓練スクリプト

**タスク 6.1.1: `experiments/train_unsupervised.py` の作成**
```python
# 完全教師なし
loss_total = (
    lambda_pde * loss_pde +
    lambda_bc * loss_bc +
    lambda_ic * loss_ic +
    lambda_cons * loss_conservation
)
```

**タスク 6.1.2: 統合モデルの作成**
```python
# models/gnn_pde_solver.py
class PUPHAWUnsupervised(nn.Module):
    """PUP-HAW-U 統合モデル"""
    pass
```

#### 6.2 検証と評価

**タスク 6.2.1: 評価メトリクス**
- [ ] PDE残差の評価
- [ ] 境界条件エラー
- [ ] 保存則の満足度
- [ ] （参考）教師データとの比較

**タスク 6.2.2: ベースライン比較**
- [ ] 既存の教師あり学習
- [ ] 固定重み教師なし学習
- [ ] PUP-HAW-U

### 成果物
- ✅ 完全教師なし学習が動作
- ✅ ベースラインを上回る性能

### マイルストーン
**M6.1**: 教師なし学習が収束
**M6.2**: 性能評価完了

---

## Phase 7: 実験・論文化（3-4週間）

### 目標
- 包括的な実験
- 論文執筆

### タスク

#### 7.1 アブレーションスタディ

**タスク 7.1.1: 各コンポーネントの寄与度**
- [ ] w/o 物理的不確実性伝播
- [ ] w/o 階層的適応
- [ ] w/o トポロジー伝播
- [ ] w/o マルチ制約適応

#### 7.2 ベースライン比較実験

**タスク 7.2.1: 既存手法との比較**
- [ ] 固定重みPINNs
- [ ] 勾配アライメント法
- [ ] lbPINNs

**タスク 7.2.2: メッシュ品質依存性**
- [ ] 異なる品質のメッシュでの性能比較
- [ ] 低品質メッシュでの優位性実証

#### 7.3 可視化

**タスク 7.3.1: 結果可視化**
- [ ] 予測場のプロット
- [ ] 重み分布の可視化
- [ ] 訓練過程のアニメーション

**タスク 7.3.2: `notebooks/visualization.ipynb`**
- [ ] インタラクティブな可視化
- [ ] 論文用の図表生成

#### 7.4 論文執筆

**タスク 7.4.1: 論文構成**
- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Methodology (PUP-HAW-U)
- [ ] Experiments
- [ ] Results & Discussion
- [ ] Conclusion

### マイルストーン
**M7.1**: 全実験完了
**M7.2**: 論文初稿完成

---

## スケジュール概要

| Phase | 期間 | 主要マイルストーン |
|-------|------|------------------|
| Phase 1: 基礎インフラ | 1-2週 | M1.1: モジュール化完了 |
| Phase 2: 物理的不確実性伝播 | 2週 | M2.2: 新重み計算が高精度 |
| Phase 3: 階層的適応 | 2-3週 | M3.3: 訓練時間短縮確認 |
| Phase 4: マルチ物理制約 | 2-3週 | M4.3: マルチ制約動作 |
| Phase 5: ハイブリッド学習 | 1-2週 | M5.1: 安定移行確認 |
| Phase 6: 完全教師なし | 2週 | M6.2: 性能評価完了 |
| Phase 7: 実験・論文 | 3-4週 | M7.2: 論文初稿完成 |
| **合計** | **13-18週（3-4.5ヶ月）** | |

---

## リスク管理

### リスク 1: 教師なし学習の収束不安定
**対策**:
- Phase 5でハイブリッド学習を十分に検証
- カリキュラム学習の最適化

### リスク 2: 計算コストの増大
**対策**:
- グラフ演算の効率化
- バッチ処理の最適化
- GPUメモリ管理

### リスク 3: 新規手法の効果が限定的
**対策**:
- 各Phaseでベースラインと比較
- 効果が薄い場合は方針転換

---

## 次のステップ

1. **Phase 1 開始の準備**
   - プロジェクト構造の作成
   - 既存コードのバックアップ

2. **開発環境の整備**
   - テストフレームワークのセットアップ
   - バージョン管理の確認

3. **Phase 1 タスク開始**
   - タスク 1.1: プロジェクト構造作成
   - タスク 1.2.1: モデル分離

---

## 参考資料

### 実装参考
- PyTorch Geometric Documentation
- Physics-Informed Neural Networks (PINNs) tutorials

### 論文参考
- Adaptive loss weighting for PINNs
- Multi-task learning in neural networks
- CFD mesh quality metrics

---

**作成日**: 2025-11-30
**最終更新**: 2025-11-30
