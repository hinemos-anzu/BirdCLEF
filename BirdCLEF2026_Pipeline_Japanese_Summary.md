# BirdCLEF 2026 パイプライン 日本語解説ドキュメント

**タイトル:** BirdCLEF 2026: ONNX Perch + Sequence Modeling & SED Blend

**タグ:** `ONNX` `Perch v2` `SSM` `SelectiveSSM` `Cross-Attention` `SWA` `TTA` `MLP` `PCA` `IsotonicCalibration` `RankBlend` `BirdCLEF` `音声分類` `音響モデリング` `種分類`

---

## 1. タイトルとタグ

**コード名:** BirdCLEF 2026: ONNX Perch + Sequence Modeling & SED Blend
**LBスコア:** 0.946

**主要タグ:**
- `ONNX Perch v2` — Googleの鳥声識別モデルをONNX形式で高速化
- `LightProtoSSM` — SelectiveSSM + Cross-Attention による時系列モデリング
- `ResidualSSM` — 残差誤差補正モジュール
- `VectorizedMLPProbes` — ベクトル化MLP推論
- `IsotonicCalibration` — 単調増加制約によるキャリブレーション
- `Rank-Blend` — パーセンタイルランクによる2ブランチ合成
- `SWA` (Stochastic Weight Averaging) — 汎化性能向上
- `TTA` (Test-Time Augmentation) — 訓練データへの非対称適用

---

## 2. サマリーテーブル

| 項目 | 内容 |
|------|------|
| コード名 | BirdCLEF 2026: ONNX Perch + Sequence Modeling & SED Blend |
| 日付 | 2026年 BirdCLEF コンペ期間 |
| 主要アルゴリズム | ONNX Perch v2, LightProtoSSM (SelectiveSSM + Cross-Attention), ResidualSSM, VectorizedMLPProbes, IsotonicCalibration, Rank-Blend |
| CVスコア | Submit modeのため非実行（train modeでGroupKFold 3〜5分割） |
| LBスコア | **0.946** |
| 参考リンク | Vyanktesh Dwivedi (base notebook), Tucker Arrants (bc2026-distilled-sed-public), Jaejohn (perch-meta), Rishikesh Jani (perch-onnx-for-birdclef-2026), Ashok205 (tf-wheels), Google Bird Vocalization Classifier Perch v2 |

---

## 3. コード解説

### 3.1 全体構造の概要

このノートブックは11ステージからなる完全パイプラインで構成されている。各ステージは明確な役割分担を持ち、音響特徴量抽出から最終予測生成まで一貫して処理する。

```
全体フロー:
1. 環境セットアップ
        ↓
2. データ・ラベル準備
        ↓
3. Perch バックボーン (ONNX)
        ↓
4. 推論キャッシュ構築
        ↓
5. バリデーション・後処理ヘルパー定義
        ↓
6. Prior確率テーブル (Bayesian Shrinkage)
        ↓
7. MLP Probeブランチ
        ↓
8. LightProtoSSM訓練
        ↓
9. ResidualSSM訓練
        ↓
10. Distilled SED分岐
        ↓
11. Rank Blend + 後処理 → 最終予測
```

**各ステージの詳細:**

1. **環境セットアップ** — ONNX Runtime + TF 2.20 インストール。`pip install onnxruntime` とホイールファイルによるTensorFlowインストールを実行
2. **データ・ラベル準備** — `taxonomy.csv` から234クラスを読み込み。完全ラベル付きファイル59件を特定
3. **Perch バックボーン** — ONNX優先でPerch v2を読み込み（TFより150倍高速）。234種中203種をマッピング
4. **推論キャッシュ構築** — 12窓×60秒の窓分割でスコア行列 `scores(708, 234)` と埋め込み行列 `embs(708, 1536)` を構築
5. **バリデーション・後処理ヘルパー定義** — cmAP計算、閾値探索、スムージング関数などを定義
6. **Prior確率テーブル** — site/hour/joint の3層ベイズ収縮で事前確率を計算
7. **MLP Probeブランチ** — PCA 64次元に圧縮後、クラスごとにMLP(128, 64)を訓練。`VectorizedMLPProbes` でバッチ行列積による高速化
8. **LightProtoSSM訓練** — 双方向SelectiveSSM + Cross-Attention + SWA を組み合わせた時系列モデル
9. **ResidualSSM訓練** — 第1パス（ProtoSSM）の系統的誤差を残差予測で補正
10. **Distilled SED分岐** — Tucker Arrantsが公開したONNX形式のfold別SEDモデルを読み込み
11. **Rank Blend + 後処理** — パーセンタイルランク(60/40)で2ブランチを合成し、継続性ゲート・希少種抑制・Sonotype Mirroringを適用

---

### 3.2 ベースラインとの差別化ポイント

- **ONNX Perch:** TF実装と比較してCPU推論が150倍高速。4スレッド最適化で59ファイルを135.3秒でキャッシュ構築。TFのオーバーヘッドを排除し、より多くの前処理・後処理に時間を割当可能

- **属レベル Proxy マッピング:** 234種中31種がPerch未対応だが、そのうち3種は属レベルのシグナルを補完的に使用（残り28種はシグナルなし）。属内の類似音響特性を活用することでカバレッジを向上

- **SelectiveSSM (Mambaライク):** 選択的状態空間モデル（d_state=16）により、時系列上の重要な情報を選択的に保持・伝播。単純な線形RNNと異なり、入力依存のゲーティングで鳴き声の有無を動的に判断

- **Cross-Attention (12窓間の全域アテンション):** 2ヘッド・2層のCross-Attentionにより、ある窓での鳴き声を他の窓の文脈で評価。例えば窓6で鳴き声があった場合、窓1〜12の全コンテキストを参照して判断

- **プロトタイプ初期化:** 各クラスの正例埋め込みの平均でプロトタイプを初期化（ランダム初期化より収束が速く、特に少数例クラスで安定）。負例との距離を学習する metric learning 的アプローチ

- **融合α (per-class mixing weight):** クラスごとに学習可能なPerchスコア/SSMスコアの混合比αを保持。特定クラスでPerchが強い・弱いという差異を自動的に学習

- **SWA (Stochastic Weight Averaging):** epoch全体の65%以降に `swa_lr=4e-4` でSWAを開始。損失曲面の平坦な極小点を探索することで汎化性能を向上。単一の最終epochよりもロバストな重みを生成

- **TTA非対称設計:** 訓練データには5シフト（0, ±1, ±2フレーム）でTTAを適用するが、テストデータには意図的に非適用。これによりキャリブレーション最適化を訓練側で行い、テストは1回推論で安定させる

- **ResidualSSM:** 第1パス（LightProtoSSM）の予測値と正解ラベルの残差を入力として、第2パスのSSMが系統的誤差を補正。アンサンブルではなく直列的な誤差補正チェーン

- **VectorizedMLPProbes:** PyTorchのバッチ行列積を使い、クラスごとのMLPを10〜50倍高速化。全58クラスのMLPを一括推論することで、ループオーバーヘッドを排除

- **PCA 64次元:** 1536次元の埋め込みを64次元に圧縮（分散保持率81.47%）。次元削減によりMLP過学習を抑制し、少数例クラスでの汎化を改善

- **ファイル信頼度スケーリング:** 各ファイルのtop-2クラスの平均スコアを `power=0.4` でスケール。信頼度が低いファイルの予測全体を抑制することで偽陽性を削減

- **ランク認識スケーリング:** ファイル内の最大スコアの `power=0.4` 乗でスコアを正規化。ファイル間のスコール尺度のばらつきを統一

- **適応型デルタスムージング:** 確信度 `conf` に応じて `α = 0.20 × (1 - conf)` を動的調整。確信度が高いほどスムージングを弱め、鋭いピークを保護する

- **アイソトニック校正 + 閾値最適化:** 31クラスに対してIsotonic Regressionで確率を校正し、F1スコアを最大化する閾値を探索（平均閾値0.469、範囲[0.25, 0.50]）

- **Rank Blend (60/40):** パーセンタイルランクに変換した上でProtoSSMブランチ60% + SEDブランチ40%を合成。生スコールの尺度差を吸収しつつ、両ブランチの相補的情報を統合

- **偽陽性ゲート:** ProtoSSMスコアが高い かつ SEDスコアが低い場合に予測を抑制。一方のブランチだけが高い場合の偽陽性を防ぐ交差検証ゲート

- **t分布カーネル:** ±3窓・35秒コンテキストで t分布カーネルによる継続性スムージングを適用。単発の雑音ピークを抑制しつつ、本物の鳴き声の持続を保護

- **Sonotype Mirroring:** 視覚的（音響的）に類似した種グループ（10カラム）を `max-pool` で統一。同じ音を異なる種名で予測した場合に整合性を保つ

- **希少種適応閾値:** Amphibia/Mammalia/Reptilia に対し、クラス平均スコア + 0.05 以下の場合にスコアを0.9倍抑制。生態的に稀な種の偽陽性を抑制

---

## 4. ログ結果の分析

### 実際のログ出力と解釈

| ログエントリ | 意味・解釈 |
|-------------|-----------|
| `ONNX Runtime installed, TF 2.20 installed, ONNX Runtime available` | 環境構築成功。ONNX RuntimeとTensorFlow 2.20が共存しており、ONNX推論が優先される |
| `MODE = submit` | 提出モードで実行中。OOF評価をスキップし、全訓練データで学習する |
| `CFG: n_epochs=40, patience=8, oof_n_splits=3, mlp_max_iter=200` | Submit modeのためエポック数は40（train modeでは倍以上になる場合もある）、Early Stoppingのpatience=8、MLP最大反復200回 |
| `Classes: 234 \| Fully-labeled files: 59` | 競技データは234クラス。そのうち完全なラベル付きが59ファイルのみ（残りはweakラベルまたはラベルなし）。少数のデータで234クラスを学習する難しさを示す |
| `Full-file windows: 708 \| Active classes: 71` | 59ファイル×12窓=708窓。実際に正例が存在するクラスは71のみ（残り163クラスは訓練例なし） |
| `Using ONNX Perch (150x faster)` | TF実装でなくONNX実装を選択。CPU推論で150倍高速化を確認 |
| `Mapped: 203/234 species` | 234クラス中203クラスがPerch v2のクラス定義にマッピング済み（86.8%カバレッジ） |
| `Unmapped: 31, with genus proxy: 3, still without signal: 28` | 31クラスが未マッピング。うち3クラスは属レベルプロキシで補完、残り28クラスはシグナルなし（事前確率のみで予測） |
| `Cache built from scratch: 135.3s for 59 files` | ONNXで59ファイルのキャッシュ構築に135.3秒。TFなら約5.6時間かかる計算 |
| `scores=(708,234) embs=(708,1536)` | 708窓×234クラスのスコア行列と、708窓×1536次元の埋め込み行列が構築済み |
| `Submit mode: skipping OOF evaluation` | 提出モードでは交差検証をスキップ。全データで学習して最大性能を引き出す |
| `No hidden test → dry-run on 20 train files, Test scores: (240,234)` | テストファイルが見つからないため、訓練ファイル20件でドライランを実施。240窓×234クラスの結果 |
| `ProtoSSM training: 13.6s` | LightProtoSSMの訓練が13.6秒で完了。軽量設計の恩恵 |
| `Embedding: (708,1536) → PCA:(708,64), variance=81.47%` | 1536次元を64次元に削減。81.47%の分散を保持（情報損失は18.53%） |
| `Training MLP probes for 58 species` | 正例が存在する58クラスに対してMLP Probeを訓練（71 active - 13 too-few-examples = 58） |
| `Trained 58 MLP probes` | 58クラスすべてのMLPが正常に収束 |
| `Calibrated 31 classes, Mean threshold: 0.469, Range:[0.25, 0.50]` | 31クラスでIsotonicキャリブレーション + F1最適閾値を適用。平均閾値0.469は0.5より若干低く、再現率重視の傾向を示す |
| `ResidualSSM training: 1.7s` | 第2パスの残差補正モデルが1.7秒で訓練完了。軽量な補正モジュール |
| `Sonotype mirroring: 10 columns` | 10グループの類似種ペアに対してSonotype Mirroringを適用 |
| `Adaptive thresholding: 44 rare species` | 44種（Amphibia/Mammalia/Reptiliaの希少種）に適応型閾値を適用 |
| `Dry-run detected: aligning with sample_submission.csv` | ドライランを検知し、提出フォーマットに合わせて出力を整形 |

### パフォーマンスへの影響分析

- **訓練時間の効率性:** キャッシュ構築135.3秒 + ProtoSSM 13.6秒 + ResidualSSM 1.7秒という短時間で主要なモデルが構築される。Kaggleの時間制限内に収まる設計
- **データ効率:** 59ファイル・71クラスの限られたデータから、キャリブレーション・プロキシマッピング・ベイズ事前確率を組み合わせて234クラス対応を実現
- **ドライラン整合性:** テストデータがない場合の安全なフォールバックを実装し、提出フォーマットを常に保証

---

## 5. 精度向上への寄与機能の説明

### 各機能がなぜ精度向上に貢献するか

**ONNX高速化**
- 推論速度150倍向上により、同じ時間制限内でより多くのエポック数・より複雑な後処理が実行可能になる。キャッシュ構築が135秒で済むため、試行錯誤のイテレーション速度が劇的に向上

**SSM時系列モデリング**
- 音声データは時間的連続性を持つ（鳥は連続して鳴く）。SelectiveSSMは過去・未来の窓情報を双方向に統合し、単一窓の判断では誤りやすい一時的ノイズと本物の鳴き声を区別できる

**Cross-Attention（全域文脈）**
- 12窓全体をグローバルに参照することで、「この窓で鳴いているのは、全体の文脈から見て本物か」を判断できる。局所的なSSMだけでは捉えられない長距離依存性を補完

**プロトタイプ初期化**
- 正例埋め込みの平均からスタートすることで、ランダム初期化より大幅に収束が速くなる。特に訓練例が少ない希少種で、少ない反復で安定したプロトタイプを確立できる

**SWA（確率的重み平均化）**
- 通常の確率的勾配降下法は鋭い極小点に収束しがちで汎化性能が低い。SWAは複数のチェックポイントを平均することで、より平坦な損失曲面の解を見つけ、未知テストデータへの汎化を改善

**TTA非対称設計**
- 訓練データへのTTAはアーキテクチャのキャリブレーション最適化に使用し、テストへの不適用で推論の一貫性を保つ。テスト時のTTAは実行時間増加とノイズ混入のリスクがあり、意図的に回避

**ResidualSSM（残差誤差補正）**
- 第1パスのProtoSSMが系統的に過小/過大評価するクラスを、第2パスが学習して補正。アンサンブルではなく直列補正により、同じ誤りの繰り返しを防ぐ

**Isotonicキャリブレーション**
- 過信（高スコアが実際の陽性率より高い）や過小信頼（低スコアが実際の陽性率より低い）を単調増加制約で補正。cmAPのようなランキングベース指標でも、確率のキャリブレーションは閾値選択に影響する

**事前確率テーブル（site/hour）**
- 生態学的知識（「この場所のこの時間帯にこの種が出現する確率」）をベイズ的に組み込む。モデルが純粋に音響特徴だけで判断できないケース（背景雑音が多い等）で、事前確率が正しい種を選択する助けになる

**Rank Blend（60/40）**
- ProtoSSM（時系列的整合性を考慮）とSED（スペクトログラム特徴に強い）は相補的な情報源。パーセンタイルランクに変換することでスコア尺度の差を吸収し、両者の長所を統合。どちらか片方だけより高いcmAPが期待できる

**ファイル信頼度スケーリング**
- 全窓で低スコアのファイルは、そのファイル自体の音質が悪い可能性が高い。ファイルレベルで信頼度を計算し、信頼度の低いファイルの予測全体を抑制することで偽陽性を削減

**適応型スムージング**
- 確信度が低い時ほど強くスムージングし、確信度が高い時は弱くする動的α設計。固定αのスムージングと異なり、真のピークを保護しつつノイズを抑制

**Sonotype Mirroring**
- 音響的に類似した種（ソノタイプが同一）は、モデルが混同しやすい。最大プーリングにより、グループ内のいずれかが高スコアならグループ全体に伝播させ、種名の曖昧さによる失点を防ぐ

---

## 6. さらなる改善提案

### コンペ仕様制約（必須前提条件）

> **BirdCLEF 2026 コード要件**（2026年5月時点）
> - **CPU Notebook: 実行時間 90分以内**（厳守）
> - **GPU Notebook: 事実上無効**（提出可能だが実行時間1分のみ）
> - **インターネットアクセス: 無効**
> - 自由・公開の外部データ（事前学習済みモデル含む）は使用可
> - 提出ファイル名は `submission.csv` 固定

**制約の実践的意味:**
- GPU加速・外部API呼び出し・実行中のデータダウンロードは一切不可
- 追加モデル・外部データはすべて **Kaggle Dataset として事前アタッチ**が必要
- 全処理（Perch推論・モデル訓練・後処理・ブレンド）を90分=5400秒以内に完了させなければならない

### 現行パイプラインの推定実行時間内訳

| 処理ステップ | 訓練ファイル(59件) | テストファイル(推定) | 備考 |
|---|---|---|---|
| Perch ONNX キャッシュ構築 | 135s | 〜150〜600s | テスト規模次第 |
| ProtoSSM 訓練 | 13.6s | — | |
| MLP Probe 訓練 | 〜60s | — | 58クラス分 |
| ResidualSSM 訓練 | 1.7s | — | |
| SED ONNX 推論 | — | 〜600〜1800s | fold数×ファイル数 |
| 後処理・ブレンド | 〜10s | — | |
| **合計（推定）** | | **〜15〜45分** | 90分制限に対して余裕あり |

ONNXによる150倍高速化がなければ、Perch推論だけで制限時間を超過する。

---

### 具体的な改善提案（制約準拠）

1. **エポック数の最適化（時間計測ベース）**
   - 現在: submit modeで `n_epochs=40`（ProtoSSM 13.6秒）
   - 提案: 実測でCPU時間を計測しながら `n_epochs=60〜80` まで増加。SWA収束がより安定する
   - 制約注意: GPUは使えないため、増加幅はCPUでの実測時間を基準に決める。60エポックでも推定20秒程度で完了するはず
   - 期待効果: SWAの収束改善、よりフラットな損失曲面の解

2. **Cross-Attentionの軽量拡張**
   - 現在: 2ヘッド・2層
   - 提案: 4ヘッド・2層（層数はそのままでヘッド数増加のみ）+ `attention_dropout=0.1`。3層化はCPU推論時間増加が大きいため保留
   - 制約注意: 1層追加でProtoSSM訓練時間が約1.5倍になる可能性。時間測定後に判断
   - 期待効果: 窓間の細かい依存性の捕捉。Dropoutによる過学習抑制

3. **公開SED fold数の増加（Kaggle Dataset経由）**
   - 現在: Tucker Arrants公開のONNX SEDを使用
   - 提案: 他の公開ONNX foldsをKaggle Datasetとして追加アタッチ。**インターネット無効のため実行中のダウンロードは不可** — 事前にDataset登録が必須
   - 制約注意: fold数増加は推論時間に直結。5-fold追加で推論時間が最大5倍になりうる。全体90分制限内かを事前試算すること
   - 期待効果: SEDブランチの安定性・多様性向上

4. **種ごとの温度スケーリング最適化**
   - 現在: Amphibia/Mammalia/Reptiliaに一律0.9倍、他に一律1.10倍
   - 提案: 訓練データのOOF予測から各クラスのECE（Expected Calibration Error）を計算し、クラスごとに温度を最適化。**追加モデル不要・時間増加ほぼゼロ**
   - 制約注意: なし（既存処理の後処理改善のみ）
   - 期待効果: 希少クラスの偽陽性/偽陰性バランスの個別最適化

5. **Pseudo-labeling（事前計算をDatasetとして添付）**
   - 提案: ローカル環境（時間無制限）で訓練データ全体のPerch推論＋高信頼度予測を計算し、疑似ラベルをKaggle Datasetとして事前アタッチ。Notebook内ではDatasetを読み込むだけ
   - 制約注意: **Notebook実行中のデータ生成・学習は時間制限内**だが、事前計算済みDatasetの読み込みは高速。インターネット無効のため、Notebook内でのXeno-cantoアクセスは不可
   - 期待効果: 訓練例ゼロの28クラス（シグナルなし）の予測品質向上

6. **3次元Prior（site × hour × 季節）の事前計算**
   - 現在: site/hour の2層ベイズ収縮（Notebook内でリアルタイム計算）
   - 提案: ローカルで月次・季節次元を加えた3次元Priorを計算し、npzファイルとしてKaggle Datasetに添付。Notebook内は読み込みのみ
   - 制約注意: なし（読み込みコストは無視できる）
   - 期待効果: 渡り鳥の季節的出現パターンを反映し、生態的に不在の種の偽陽性を抑制

7. **SED側のガウシアンスムージングσの最適化**
   - 現在: `gaussian_filter1d(sigma=0.65)`
   - 提案: OOFでσ = [0.5, 0.65, 0.8, 1.0] をグリッドサーチし最適値を固定。**追加時間コストほぼゼロ**
   - 期待効果: SED分岐の時間方向平滑化の最適化

8. **ResidualSSM correction_weightの動的最適化**
   - 現在: `correction_weight=0.30` 固定
   - 提案: OOF残差のL2ノルムに基づき、クラスごとにcorrection_weightを動的スケール
   - 制約注意: なし（Notebook内計算のみ、推定追加時間 < 5秒）
   - 期待効果: 第1パスの誤差が大きいクラスに対する補正を強化

9. **ランクブレンド比率のOOF最適化**
   - 現在: ProtoSSM 60% / SED 40% 固定
   - 提案: 訓練データでProtoSSM OOF と SED OOFを生成し、クラスグループ（Aves/Amphibia/Insecta）ごとに最適ブレンド比をグリッドサーチ
   - 制約注意: なし（既存の後処理改善のみ）
   - 期待効果: 分類群ごとの得意/不得意に応じた最適ブレンドで全体AUC向上

10. **外部事前学習済みPerch埋め込みキャッシュのDataset添付**
    - 現在: Notebook実行時にPerch推論キャッシュを135秒かけて構築
    - 提案: 訓練ファイル全体の埋め込みキャッシュをローカルで事前計算し、Kaggle Datasetとして添付。Notebook内では読み込みのみ（コード内の外部キャッシュ参照ロジックは既に実装済み）
    - 制約注意: **この手法はすでにコード内に実装済み**（`EXTERNAL_CACHE_DIRS` 参照）。Dataset添付さえすれば即効果あり
    - 期待効果: Perch推論135秒を削減し、その分をモデル訓練や後処理チューニングに充当

---

## 付録A: コンペ制約チェックリスト

| 制約 | 現行コードの対応状況 |
|------|---------------------|
| CPU ≤ 90分 | ✅ ONNX最適化で推定15〜45分以内 |
| GPU無効 | ✅ CPUのみで完結（PyTorchもCPUモード） |
| インターネット無効 | ✅ すべてKaggle Dataset/Model経由 |
| 外部データ使用 | ✅ 公開DatasetのみアタッチでOK |
| submission.csv | ✅ 最終出力ファイル名として設定済み |
| OOF評価の有無 | ⚠️ submit modeではスキップ（train modeで検証） |

---

## 付録B: 主要ハイパーパラメータ一覧

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `n_epochs` | 40 (submit) | ProtoSSM訓練エポック数 |
| `patience` | 8 | Early Stopping閾値 |
| `oof_n_splits` | 3 | GroupKFold分割数 |
| `mlp_max_iter` | 200 | MLP最大反復数 |
| `d_state` | 16 | SelectiveSSMの状態次元 |
| `n_heads` | 2 | Cross-Attentionヘッド数 |
| `n_attn_layers` | 2 | Cross-Attentionレイヤー数 |
| `swa_lr` | 4e-4 | SWA学習率 |
| `swa_start_frac` | 0.65 | SWA開始割合（epoch全体の65%） |
| `pca_dim` | 64 | PCA圧縮後の次元数 |
| `pca_variance` | 81.47% | 保持分散率 |
| `rank_blend_ratio` | 60/40 | ProtoSSM/SEDブレンド比 |
| `file_conf_power` | 0.4 | ファイル信頼度スケーリングの指数 |
| `smooth_alpha_base` | 0.20 | 適応型スムージングの基底α |
| `mean_threshold` | 0.469 | キャリブレーション後の平均閾値 |
| `threshold_range` | [0.25, 0.50] | 閾値の範囲 |
| `rare_suppression` | 0.9× | 希少種スコア抑制倍率 |
| `tta_shifts` | 0, ±1, ±2 | TTA適用シフト数（訓練のみ） |
| `context_windows` | ±3 (35秒) | t分布カーネルのコンテキスト範囲 |
| `sonotype_groups` | 10 | Sonotype Mirroringグループ数 |
| `calibrated_classes` | 31 | Isotonicキャリブレーション対象クラス数 |
| `total_classes` | 234 | 全クラス数 |
| `active_classes` | 71 | 訓練例がある有効クラス数 |
| `mlp_probe_classes` | 58 | MLP Probe訓練クラス数 |
| `fully_labeled_files` | 59 | 完全ラベル付きファイル数 |
| `total_windows` | 708 | 全推論窓数（59ファイル×12窓） |
| `emb_dim` | 1536 | Perch埋め込み次元 |
| `mapped_species` | 203/234 | Perchマッピング済み種数 |
| `genus_proxy` | 3 | 属レベルプロキシ補完種数 |
| `cache_build_time` | 135.3s | ONNXキャッシュ構築時間 |
| `protossm_train_time` | 13.6s | ProtoSSM訓練時間 |
| `residualssm_train_time` | 1.7s | ResidualSSM訓練時間 |
| `LB_score` | **0.946** | Leaderboardスコア |
