# BirdCLEF+ 2026 EoS-3 コード解析レポート

**タグ:** `#BirdCLEF2026` `#アンサンブル` `#Perch` `#ProtoSSM` `#SED` `#知識蒸留` `#GeMFreq` `#StateSpaceModel` `#後処理最適化` `#LB0.949`

---

## 1. コードの概要表

| 項目 | 内容 |
|------|------|
| **コード名** | BirdCLEF+ 2026 EoS-3 (Ensemble of Solutions 3) |
| **日付** | 2026-05-17（最終実行） |
| **主要アルゴリズム** | Perch v2 ONNX + ProtoSSM + ResidualSSM + Distilled-SED (EfficientNet-B0) + rank_aware_scaling |
| **CVスコア** | OOF macro-AUC = 0.9924（Model_10内部） |
| **LBスコア** | **0.949**（v16: Model_3×0.026 + Model_10×0.974） |
| **参考リンク** | [nina2025/birdclef-2026-eos-3](https://www.kaggle.com/code/nina2025/birdclef-2026-eos-3) |

---

## 2. コード手順解説

### 全体構造

```
EoS-3 Notebook
├── [Cell 1-2]  パラメータ設定 (solutions dict)
│               アンサンブル対象モデルとウェイトの定義
├── [Cell 3]    Model_2: Distilled-SED (Tucker Arrants, LB=0.917)
│               ※現在の設定では非アクティブ
├── [Cell 4-7]  Model_4: ProtoSSM + ResSSM (Mattia Angeli, LB=0.935)
│               ※現在の設定では非アクティブ
├── [Cell 8-11] Model_6x: ONNX Perch & SED Blend (Raunak Dey, LB=0.946)
│               ※現在の設定では非アクティブ
├── [Cell 12]   Model_10: Karnakbayev PowerOptimization (LB=0.949) ← メイン
│               内部でProtoSSM + SED xBlend + rank_aware_scaling(power=0.6)
├── [Cell 26]   アンサンブル関数定義 (direct / rank_1 / add_safe)
├── [Cell 27]   ブレンド方式選択
└── [Cell 29]   submission.csv 出力
```

**現在の設定（v16）:**
- Model_3 (LB=0.928) × 0.026 + Model_10 (LB=0.949) × 0.974 → **LB 0.949**

---

### 手順の箇条書き

#### Step 1: パラメータ設定 (`solutions` dict)
- `type_add`: ブレンド方式 (`'direct'` / `'rank.1'` / `'add_safe'`)
- `Models`: 各モデルの設定（CSV名・ウェイト・xSED比率・LB）
- **差異点**: ウェイト・方式を1セルで一元管理し、v番号で実験履歴が追跡可能

#### Step 2: Model_2 — Distilled-SED（条件付き実行）
- EfficientNet-B0バックボーン (`tf_efficientnet_b0.ns_jft_in1k`)
- Perch v2 ONNX を「教師」として使い **知識蒸留**（MSEロス、alpha=1.0）
- **GeMFreq Pooling**: 学習可能なp（初期値3.0）で周波数方向のGeneralized Mean Pool
- AttBlock（attention × classification の積でクリップ予測）
- MixUp (`alpha=0.4`) + FocalSC-MixUp + SpecAugment
- 5-Fold Stratified KFold、希少種のアップサンプリング（MIN_SAMPLE=20）
- **差異点**: Perch蒸留ヘッドを持つSEDモデルが他SEDと異なる特徴

#### Step 3: Model_4 — ProtoSSM + ResidualSSM（条件付き実行）
- Perch v2（ONNX or TF SavedModel）で1536次元埋め込みを取得
- **ProtoSSM**: 埋め込み→SSMによるプロトタイプ学習（BCEFocal + MSE蒸留ロス）
  - focal_gamma=2.5, label_smoothing=0.03, SWA（swa_start_frac=0.65）
- **ResidualSSM**: ProtoSSM予測の残差補正（d_model=128, d_state=16, 2層）
- **MLP Probes**: PCA 64次元 + MLP(256,128) で各クラスの補助予測
- **後処理パイプライン**:
  1. `temporal_smooth` (alpha=0.25)
  2. `apply_prior` (サイト×時刻prior、lambda=0.4)
  3. `file_confidence_scale` (alpha=0.15)
  4. `rank_aware_scaling` (power=0.4)
  5. `adaptive_delta_smooth` (base_alpha=0.20)
  6. 分類群別温度スケーリング (Insecta=1.15, Amphibia=1.10)
  7. `calibrate_and_optimize_thresholds` (Isotonic Regression + F1最適化)
- **差異点**: 地理・時刻情報を活用したpriorsは他モデルにない特徴

#### Step 4: Model_10 — Karnakbayev PowerOptimization（メインモデル）
- Model_4と同じProtoSSM + ResidualSSMパイプラインをベースにした **exp019**
- **exp017差分**: `lambda_prior: 0.4 → 0.5`（priorsの影響を強化）
- **exp019差分**: `rank_aware_scaling(power: 0.5 → 0.6)`（パワー最適化）
- **xSED Rank Blend**: ProtoSSM 60% + Distilled-SED 40% をランクブレンド
- **Sonotype mirroring**: 10列に鳴き型コピー適用
- **希少種 Adaptive Thresholding**: 44種に適応的閾値処理
- 実行ログ: ProtoSSM 13.7s、ResidualSSM 1.8s、合計~1.4分
- **差異点**: `rank_aware_scaling` の `power` パラメータを0.1単位で段階的に最適化

#### Step 5: アンサンブル処理
- **`direct`**: 加重平均ブレンド（確率値の直接合成）
- **`rank_1`**: 百分位ランクに変換してからウェイト付き合成（分布の違いを吸収）
- **`add_safe`** (Pilkwang Kim式): 入力検証付き加重平均（NaN/範囲チェック含む）
- 現在: direct blend, Model_3×0.031 + Model_10×0.969

---

## 3. 実行ログの分析

```
ONNX Runtime available                         → ONNX Perchが利用可能
Global random seed set to 4                    → 再現性確保
MODE = submit                                  → 推論専用モード（学習なし）
Classes: 234 | Fully-labeled files: 59         → 234種、59ファイルが完全ラベル済み
Full-file windows: 708 | Active classes: 71    → 学習窓数708、実際に出現71種
Using ONNX Perch: perch_v2_no_dft.onnx        → DFT不要の高速版ONNXを使用
Mapped: 203 / 234 species have a Perch logit  → 31種はマッピング不可
Unmapped: 31 | Proxy: 3 | No signal: 28       → 28種はシグナルなし（MLP probeで対処）
ProtoSSM training: 13.7s                       → 高速学習
Embedding PCA variance retained: 81.47%       → 64次元PCAで81%の情報保持
Trained 58 MLP probes                         → 58種のMLP補助分類器
[LB0.948] Per-class first-pass weights: mapped=0.60 unmapped=0.35
                                               → マッピング済み種に高いウェイト
Calibrated 31 classes | Mean threshold: 0.463 → 平均閾値0.463（0.5より低い）
correction_weight=0.10  OOF macro-AUC=0.99241 → ResidualSSMのベスト補正重み
Best correction_weight=0.10                   → わずかな補正が最適
SED folds loaded: 5 folds                     → 5-Fold SEDモデル
Executing xSED rank blend (0.60 Proto/0.40 SED) → 最終ブレンド比率
Sonotype mirroring applied to 10 columns      → 10種に鳴き型ミラーリング
Adaptive thresholding applied to 44 rare species → 44希少種に特別処理
```

### ログから読み取れる重要な知見

1. **マッピング率の問題**: 234種中31種（13%）はPerchに直接マッピングできず、精度の上限が制約される
2. **ResidualSSM補正は最小限が最適**: correction_weight=0.10でAUC=0.99241、値を上げても改善しない → ProtoSSMがすでに良好な予測を出している
3. **閾値の低下**: 平均閾値0.463 < 0.5 → モデルは控えめな確率を出力する傾向あり、閾値を下げることで感度が上がる
4. **PCA情報保持率**: 64次元で81.47%の情報保持 → 残り18.53%に有用な情報が含まれる可能性

---

## 4. 精度向上に貢献している特徴

### 4-1. Perch v2 事前学習モデルの活用
- Google DeepMindが数百万件の鳥の音声で学習した1536次元埋め込み
- 204クラス以上の直接マッピングにより、少ないデータでも高精度を達成
- ONNX版（no-DFT）により推論速度が大幅改善

### 4-2. ProtoSSM（状態空間モデルによるプロトタイプ学習）
- 時系列の12窓（60秒）を一括処理するSSMアーキテクチャ
- 各クラスのプロトタイプを学習し、テスト時の埋め込みと比較
- BCEFocal (gamma=2.5) で希少クラスの学習を強化

### 4-3. 地理・時刻Priorsの導入
- サイト別・時刻別の出現確率をlogit空間で適用
- パンタナール湿地の生態的知識（どこに何が住んでいるか）をモデルに組み込む
- `lambda_prior=0.5`（exp019での最適値）で効果を最大化

### 4-4. rank_aware_scaling (power=0.6)
- 各ファイルの最大スコアをべき乗してスケールファクターに使用
- 高信頼度の予測をさらに強調し、低信頼度を相対的に下げる
- `power: 0.5 → 0.6`の変更でLB 0.948 → 0.949に改善

### 4-5. xSED Rank Blend (60:40)
- ProtoSSMとDistilled-SEDを確率ではなく**ランク**で合成
- 両モデルの出力スケールの違いを吸収しながら相補的な情報を融合
- GeMFreq pooling付きSEDは周波数方向の特徴抽出に強み

### 4-6. 知識蒸留 (Distilled-SED)
- EfficientNet-B0がPerch v2の1536次元埋め込みを模倣
- Perchの鳥類音響知識をコンパクトなCNNに転移
- MSEロスによる蒸留でSED本来のBCEロスを補完

### 4-7. 多段階後処理パイプライン
1. 時間的スムージング → 短期的ノイズ除去
2. サイト/時刻prior → 生態的知識の組み込み
3. file_confidence_scale → ファイル全体の信頼度補正
4. rank_aware_scaling → 高信頼予測の強調
5. adaptive_delta_smooth → 信頼度に応じた適応的スムージング
6. 分類群別温度スケーリング → Insecta/Amphibiaの調整
7. Isotonic Regression + F1最適化閾値 → クラス別最適閾値

### 4-8. Sonotype Mirroring
- 鳴き声のパターン（ソノタイプ）が類似する種間で確率を伝播
- Perchがマッピングできない種のシグナルを補完

---

## 5. LB 0.950 に向けた精度向上策

### 短期（±0.001改善、1実験）

| 策 | 根拠 | 期待効果 |
|----|------|----------|
| **lambda_prior を 0.5 → 0.55 に調整** | exp019でlambda=0.5が0.4より優れたことを確認済み。次の刻みで探索 | +0.001 |
| **rank_aware_scaling power=0.6 → 0.65** | 単調増加が確認されており次の刻みを試す価値あり | +0.001 |
| **xSED blend比率: 0.60/0.40 → 0.55/0.45** | SEDの比率を上げるとSEDの強みが活きる可能性 | ±0.001 |
| **Model_3ウェイト: 0.026 → 0.030** | 0.928モデルが少量でも多様性提供。最適点を細かく探索 | ±0.001 |

### 中期（新機能追加、+0.002〜+0.005）

#### A. Perchマッピング外の28種への対処強化
- 現在: 28種がシグナルなし（MLP probe任せ）
- **策**: 属レベルではなく科レベルでのProxy Mapping拡張
- Insecta/Amphibiaは異なるアプローチが必要（音声特性が大きく異なる）

#### B. 追加SEDモデルのアンサンブル
- yukiZ (LB=0.928) のPerch+ProtoSSM+ResSSMを低ウェイトで追加
- 多様性（異なるアーキテクチャ）でアンサンブル効果を向上
- **現在v16 = [Model_3×0.026 + Model_10×0.974]**
- **提案 = [Model_3×0.02 + yukiZ_SED×0.02 + Model_10×0.96]**

#### C. グリッドサーチによる後処理パラメータの最適化
```python
for power in [0.55, 0.60, 0.65, 0.70]:
    for lambda_prior in [0.45, 0.50, 0.55, 0.60]:
        # OOF AUCで評価
```
- OOF AUCで全パラメータ組み合わせを評価し最適解を探索

#### D. adaptive_delta_smooth の base_alpha チューニング
- 現在 base_alpha=0.20 は未調整（デフォルト値）
- 0.10〜0.30 の範囲でグリッドサーチ

### 長期（アーキテクチャ変更、+0.005以上）

#### E. EfficientNet-B0 → B4/B5 へのスケールアップ
- Distilled-SEDのバックボーンを大きくして特徴抽出能力を向上
- PERCH_EMBED_DIM=1536 への蒸留精度も改善

#### F. マルチスケールメルスペクトログラム
- N_MELS: 256単一 → [128, 256, 512] の3スケールを並列処理
- より広い周波数帯域での特徴学習

#### G. 学習データ拡張
- xeno-canto等からPantanal近辺の種の追加データを取得
- 特にマッピング外28種のデータを重点収集

#### H. TTA（Test Time Augmentation）
- 複数の時間シフトや音量変化でテスト推論を行い平均
- 現在の推論はTTAなし → +0.003〜+0.005の余地

---

## 6. 実験管理の知見

このコードの特徴的な実験管理手法：

1. **単一スカラー変化の原則**: 1実験で1パラメータのみ変更し因果関係を明確化
2. **決定ルールの明記**: 次のexp番号での行動方針を事前に定義
3. **バージョン履歴のLB**: v1〜v21の全バージョンのLBを表形式で管理
4. **国際コラボレーション**: 韓国・日本・ロシア・パキスタン等のKagglerの成果を統合

---

*分析日: 2026-05-18 | 対象ノートブック: birdclef2026eos3.ipynb | LBスコア: 0.949*
