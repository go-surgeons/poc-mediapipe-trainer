# Surgical Skill Scorer — 設計ドキュメント

- **Status:** Draft (brainstorming 完了、未承認)
- **Date:** 2026-05-03
- **Owner:** Shingo Noguchi (`go-surgeons`)
- **関連リポジトリ:** `poc-mediapipe`, `poc-mediapipe-trainer` (本リポジトリ)
- **対応する将来リポジトリ:** `surgical-skill-scorer` (新設予定)

---

## 1. ゴールと非ゴール

### ゴール (v1)

外科医向けトレーニング動画 (laparoscopic box trainer での縫合・結紮) を入力すると、

1. **解釈可能なスコア** (総合スコア + 特徴ごとの寄与度) を返し、
2. **指導医のレビュー支援** (タイムライン上に異常区間をハイライト) を提供し、
3. **指導医からのフィードバックを次サイクルの学習に取り込む**

ようなシステムを構築する。

### ユースケース (v1 で対応する)

- **B. 練習直後のリプレイ採点**: 1 セッション終了 → 数十秒以内にサーバ側でスコア表示
- **C. 指導医レビュー支援**: 録画をアップロードし、指導医がスコア + 注釈付きタイムラインを見ながらフィードバック

### 非ゴール (v1)

- **A. リアルタイム (< 1s) フィードバック** — v2 で MediaPipe on-device 路線に降ろす予定
- **D. 客観試験・認定スコア** — 統計的妥当性検証が別プロジェクト規模になるため
- 暗黙のスキル評価 (組織愛護、力加減等。力センサ前提のため別系統)
- 複数タスクをまたいだ統合スコア化 (タスクごと別評価)
- 公開モデルの配布 (ライセンス・学術的検証が別途必要)

---

## 2. 用語

| 用語 | 意味 |
|---|---|
| Trial | 1 回の練習セッション (= 1 動画) |
| Pair | スコア学習用の 2 trial 比較ラベル |
| Stage 1 | 動画 → per-frame の鉗子・針・糸検出 (perception) |
| Stage 2 | per-frame 検出 → trial レベルの解釈可能特徴ベクトル |
| Stage 3 | 特徴ベクトル → 潜在スコア (Bradley-Terry / RankNet) |
| Reviewer | pair 投票を担当する指導医・エキスパート |
| BT | Bradley-Terry モデル (pairwise 比較から潜在スコアを推定) |
| GRS | Global Rating Scale (modified OSATS の合計値、外部妥当性検証用) |
| OOD | Out-of-Distribution (学習分布から外れた trial) |

---

## 3. 前提と制約

### ドメイン

- **対象**: laparoscopic box trainer (普通の腹腔鏡鉗子 + 物理ボックス) での suturing / knot tying
- da Vinci ロボット手術器ではない (= JIGSAWS 互換にしない)
- 患者映像は扱わない (倫理ハードル低)

### データ前提

- **公開データ**: PETRAW, ROSMA, EndoVis 系。Skill ラベルは付いていないため **perception の事前学習にのみ** 用いる
- **自施設動画**: 現状 数十 trial (suturing/knot tying)。継続的に増えていく前提
- **教師信号**: pairwise 比較ラベル (Bradley-Terry で潜在スコア化)。指導医アノテータの稼働は v1 開始時点では未確定

### コールドスタートの方針

絶対値スコア (modified OSATS / GRS) を直接予測するルートは採らない。理由:

- 公開データに skill ラベルがほぼ無い
- 絶対値スコアは reviewer 間一致率が pairwise より低い
- pairwise なら少ない比較数からでも BT で潜在スコアを推定できる

代わりに、ラベルがゼロの状態から始めるために **proxy ブートストラップ** (§7) を使う。

---

## 4. システム全体構成

```
┌─────────────────────── オフライン (学習側) ──────────────────────┐

  公開データ (PETRAW / ROSMA / EndoVis)         自施設 box trainer 動画
       │                                                │
       ▼                                                ▼
   [Stage 1 事前学習]                          [動画 ingestion + メタ管理]
       │                                                │
       └── 重み転移 ──► [Stage 1 ファインチューン]
                                                        │
                                                        ▼
                                          [Stage 2 特徴抽出 (バッチ)]
                                                        │
                                       ┌────────────────┴───────────────┐
                                       ▼                                ▼
                              [アノテーション UI]              [特徴 DB (parquet)]
                              pairwise 比較投票                          │
                                       │                                ▼
                                       ▼                  [Stage 3 学習: BT + RankNet]
                              [比較ラベル DB]                            │
                                       │                                │
                                       └─────────┬──────────────────────┘
                                                 ▼
                                       [Active Learning Sampler]

└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────── オンライン (推論 + UX) ──────────────────────┐

   練習者 → 動画アップロード
                │
                ▼
      [推論サーバ (FastAPI)]
       Stage 1 → Stage 2 → Stage 3
                │
                ▼
      [指導医レビュー UI (Next.js)]
       スコア + 寄与度 + タイムラインハイライト

   指導医のコメント・修正・新規 pair → アノテーション DB に逆流

└──────────────────────────────────────────────────────────────────────┘
```

### 設計上のキー判断

- **3 段は明示的なファイル契約で結合**: 動画 → JSONL (per-frame) → JSON (per-trial) → モデル重み。各段が独立に再学習・差し替え可能
- **アノテーション UI 自体がフィードバックループの中核**: pair 投票 → 比較 DB → AL → 次の pair 提示の閉じた輪
- **Stage 1 は MediaPipe ImageSegmenter / Object Detector の延長**: 既存 `train_oxford_pet.ipynb` の作法を継承し、将来の on-device 化 (v2) の道を残す
- **推論はバッチで OK**: B/C 用途なので低レイテンシ要件なし

---

## 5. コンポーネント詳細

### 5.1 Stage 1: Perception

| 項目 | 内容 |
|---|---|
| 入力 | MP4/MOV (固定 30fps、240p〜720p 想定) |
| 出力 | per-frame JSONL: `{frame, ts_ms, instruments: [...]}` |
| クラス | `left_grasper`, `right_grasper`, `needle`, `thread`, `pad` |
| モデル | MediaPipe Object Detector + ImageSegmenter (主) / SAM2 (アノテーション補助) |
| 事前学習 | PETRAW/ROSMA で instrument detection を pretrain |
| ファインチューン | 自施設動画 ~200–500 frame を SAM2 半自動アノテーション → 学習 |

**評価指標**: mAP@0.5 ≥ 0.7、keypoint PCK の hold-out 評価

### 5.2 Stage 2: Feature Extraction

per-trial の解釈可能特徴ベクトル `x ∈ ℝ^d` (d = 15-25) を出す純粋関数。学習なし、ユニットテスト可。

**特徴グループ**:

- **時間系**: 完了時間、idle time (鉗子静止 ≥0.5s 合計)、stitch あたり所要時間の分散
- **運動系**: 鉗子先端 path length、平均速度、**spectral arc length** (滑らかさ指標)、jerk RMS
- **両手協調**: 左右同時稼働率、片手 idle 率、bimanual cross-correlation
- **タスク特異**: regrasp 回数、針落下、糸絡まり (簡易ヒューリスティクス)
- **品質**: `coverage_ratio` (検出が連続して取れた割合)

**根拠**: ICSAD/ROVIMAS 由来の運動指標 + JIGSAWS modified OSATS の "Time and Motion / Flow / Instrument Handling" 軸への対応

### 5.3 Stage 3: Scoring Head

- **入力**: 標準化された特徴ベクトル `x`
- **構造**: 小 MLP (隠れ 32-64 / 2 層) または線形 + monotonic constraints
- **損失**: pairwise margin ranking
  ```
  L = max(0, m - (s(x_winner) - s(x_loser)))
  ```
  + tie 項 + reviewer 一致率による per-pair 重み
- **出力**: 潜在スコア `s(x) ∈ ℝ` + 特徴寄与度 (linear なら係数、MLP なら Integrated Gradients)
- **校正**: 学習後に percentile スケール化して UI 表示
- **reviewer バイアス**: BT に reviewer fixed effect を組み込み (`s(a) - s(b) + bias_reviewer`)

### 5.4 アノテーション UI

最小機能:

- 2 動画 side-by-side 同期再生 (任意倍速、A/B シーク)
- ボタン: `Left better` / `Right better` / `Tie` / `Skip (不適切ペア)`
- (オプション) 決定理由の OSATS 軸チェックボックス
- セッション再開・1 比較 30〜60 秒目標
- バックエンド: 比較記録 + Bradley-Terry online 更新 + Active Learning キュー

スタック: Next.js + Postgres + FastAPI

### 5.5 推論サーバ

- FastAPI 1 エンドポイント `POST /score (video_id)` → 非同期ジョブ → 完了後 webhook or polling
- 中間生成物 (per-frame JSONL, feature parquet) を保存して再利用
- GPU は Stage 1 のみ必要。Stage 2/3 は CPU で十分

### 5.6 指導医レビュー UI

- 動画再生 + タイムライン上に特徴オーバーレイ (jerk スパイク、idle 区間)
- スコアカード: 総合スコア + 寄与の高い特徴 top 3 / bottom 3
- アクション: 「この trial を pair queue に投入」「コメント」「別 trial と pairwise 比較」
- 出力 (コメント・新規 pair) は比較ラベル DB に逆流

---

## 6. データ契約 (スキーマ & ファイル形式)

### 6.1 ストレージレイアウト

```
s3://surgical-trainer/
├── raw_videos/{trial_id}.mp4
├── meta/trials.parquet
├── perception/{trial_id}.jsonl.gz          # Stage 1 出力
├── features/{trial_id}.json                # Stage 2 出力
├── annotations/
│   ├── pairs.parquet
│   └── reviewer_metadata.parquet
└── models/
    ├── stage1/{version}/...
    └── stage3/{version}/scorer.pkl
```

### 6.2 主要スキーマ

**`trials.parquet`**

```
trial_id (str, ULID)  surgeon_id (str)  task (enum: suturing|knot_tying)
recorded_at (ts)      fps (int)         duration_s (float)
device (str)          camera_pose (str) consent_status (enum)
status (enum: raw|scorable|quarantine|ingest_failed|unreliable)
```

**`perception/{trial_id}.jsonl.gz`** (1 行 = 1 フレーム)

```json
{"frame": 1234, "ts_ms": 41133,
 "instruments": [
   {"class": "left_grasper",  "bbox": [x,y,w,h], "keypoints": [[tip_x,tip_y,conf], [shaft_x,shaft_y,conf]], "mask_rle": "..."},
   {"class": "right_grasper", "bbox": ..., "keypoints": ...},
   {"class": "needle",        "bbox": ..., "tip": [x,y,conf]}
 ]}
```

**`features/{trial_id}.json`**

```json
{"trial_id": "...", "schema_version": 1, "features": {
   "completion_time_s": 142.3,
   "left_path_length_px": 18920,
   "right_path_length_px": 16240,
   "spectral_arc_length_left": -2.31,
   "jerk_rms_right": 0.18,
   "idle_time_ratio": 0.12,
   "regrasp_count": 7,
   "bimanual_xcorr": 0.41,
   "needle_drops": 1,
   "coverage_ratio": 0.97
 }}
```

**`annotations/pairs.parquet`**

```
pair_id (ULID)  trial_a (str)  trial_b (str)  reviewer_id (str)
preference (enum: A|B|TIE|SKIP)  confidence (1-5, optional)
osats_axes (json, optional)
created_at (ts)  active_learning_score (float)
```

### 6.3 パイプラインのトリガ

```
新規動画アップロード
  → trial 行を trials.parquet に追加 (status=raw)
  → ジョブキュー に "perception" 投入

"perception" 完了
  → perception/{trial_id}.jsonl.gz 生成
  → ジョブキュー に "features" 投入

"features" 完了
  → features/{trial_id}.json 生成
  → trial.status = "scorable"
  → スコアリング & Active Learning 候補に追加

"score" リクエスト
  → 最新 stage3 モデルで s(x) を計算 → API 返却

"retrain" (週次 cron or 手動)
  → annotations が N 件溜まったら再学習
  → models/stage3/{new_version}/ に出力
```

実装は最初は Prefect + S3 トリガで十分。Kubernetes は不要。

### 6.4 API 契約 (推論サーバ)

```
POST /score
  body:  {trial_id: "..."}
  response:
  {
    "trial_id": "...",
    "model_version": "stage1@v3,stage3@v7",
    "score": 67.2,
    "raw_latent": -0.31,
    "feature_breakdown": [
       {"name": "regrasp_count",      "value": 7,   "contribution": -0.18},
       {"name": "spectral_arc_length","value": -2.3,"contribution": +0.12}
    ],
    "highlights": [
       {"t_start": 32.1, "t_end": 38.4, "type": "long_idle", "severity": 0.7},
       {"t_start": 71.0, "t_end": 71.5, "type": "needle_drop","severity": 1.0}
    ]
  }

POST /pairs/next?reviewer_id=...   # Active Learning が選ぶ次のペア
POST /pairs/vote                    # 投票登録
```

### 6.5 バージョニング

- 各 Stage は独立した semver: `stage1@v3.1.0`, `stage3@v7.0.2`
- 推論結果には常に両バージョンを記録
- 退役モデルは `models/archive/` に残し、過去スコアの再現性を確保

---

## 7. アノテーション戦略 & フィードバックループ

### 7.1 コールドスタート (ラベルゼロから v0 立ち上げ)

ラベルが揃うまで開発を止めない。

1. Stage 2 の特徴のうち客観性の高い 3〜5 個を選ぶ (完了時間、regrasp 回数、idle 比、spectral arc length)
2. 文献の慣行値で加重和を取り、ブートストラップ・スコア `s_0(x)` とする
3. v0 でランキングを生成し、Active Learning の "種" にする
4. 最初の数十 pair が集まり次第 BT/RankNet で `s_1` を学習し、`s_0` を捨てる

### 7.2 Active Learning サンプラ

レビュー対象 pair の優先度:

```
priority(a, b) =
   α · uncertainty(a, b)              # |s(a) - s(b)| が小さいほど高
 + β · model_disagreement(a, b)       # 直近 N モデル間でスコア順位が割れる
 + γ · coverage_gap(a, b)             # 既存 BT graph での最短経路長
 - δ · redundancy(a, b)               # 多数決が固まっている pair を抑制
 + ε · reviewer_specialty_match(...)
```

α, β, γ は運用で調整。1 ペア = 1 単位の効用最大化が目標。

### 7.3 信頼性とノイズ処理

- **inter-rater reliability**: pair 全体の ~10% を 2 名以上に重複レビュー → Cohen's κ / Kendall τ を週次計算
- **reviewer bias 補正**: BT に reviewer fixed effect を組み込み、きつい/甘い採点者を吸収
- **逸脱検出**: 1 reviewer の投票だけが majority と乖離 → フラグ → 再キャリブレーション会議
- **decay**: 古いラベルは指数減衰 (半減期 90 日)。スコア基準のドリフトに対応

### 7.4 4 つのフィードバック経路

```
                   ┌───────────────────────────────────────────────┐
                   │                                               │
                   ▼                                               │
   [Active Learning] ──選ぶ──► [指導医アノテーション UI]           │
                                       │                           │
                                       │ 投票・OSATS 軸タグ        │
                                       ▼                           │
                                [pairs.parquet]                    │
                                       │                           │
                                       │ N 件溜まる or 週次         │
                                       ▼                           │
                                [Stage 3 再学習]                   │
                                       │                           │
                                       ▼                           │
                                [新 scorer デプロイ]               │
                                       │                           │
                  ┌────────────────────┴────────────────────┐      │
                  ▼                                         ▼      │
        [推論 API: 既存 trial を再採点]         [指導医レビュー UI]│
                                                            │      │
                                                            │ 修正・新規 pair
                                                            └──────┘
```

1. **明示**: アノテーション UI での pair 投票 (主経路)
2. **暗黙**: コーチ UI で「これと比べたい」操作 → 新 pair 化
3. **修正**: スコアへの「不当」フラグ → corrective pair (隣接 percentile trial との比較) 自動生成
4. **新規データ**: 新 trial が増える → 自動 ingest → AL 候補追加

### 7.5 再学習のトリガ

- 時間: 週次 cron
- 量: 新規 pair ≥ 50 / 累積 ≥ 200 で前回比増加
- 品質: hold-out Kendall τ が前モデル未満ならデプロイ拒否
- ロールバック: 直近 1 モデル分は shadow で常時稼働

### 7.6 評価指標 (運用ダッシュボード)

| 層 | 指標 | 目標 |
|---|---|---|
| Stage 1 | mAP@0.5, keypoint PCK | mAP ≥ 0.7 |
| Stage 2 | 特徴の test-retest 相関 | r ≥ 0.95 |
| Stage 3 | hold-out pair の Kendall τ, BT 対数尤度 | τ ≥ 0.6 |
| 全体 | 指導医 GRS との Spearman 相関 (年 1 回) | ρ ≥ 0.7 |
| 運用 | 平均 pair レビュー時間, reviewer κ | κ ≥ 0.6 |

---

## 8. ツール・環境・リポジトリ構成

### 8.1 リポジトリ配置

```
go-surgeons/
├── poc-mediapipe/                ※既存:  ブラウザ UI / MediaPipe 推論
├── poc-mediapipe-trainer/        ※既存:  Stage 1 学習ノートブック (拡張)
└── surgical-skill-scorer/        ※新規:  Stage 2/3 + UI + API
```

`poc-mediapipe-trainer` の拡張: 現状の Oxford Pet Notebook を pretraining デモとして残し、`train_instrument_detector.ipynb` を新規追加 (PETRAW で pretrain → 自施設データで fine-tune)。`.tflite` を Release に上げる流儀は継承。

### 8.2 `surgical-skill-scorer` レイアウト (推奨)

```
surgical-skill-scorer/
├── packages/
│   ├── perception/         # Stage 1 推論ラッパ (MediaPipe Tasks SDK)
│   ├── features/           # Stage 2 純粋関数群 + テスト
│   ├── scorer/             # Stage 3 学習・推論
│   ├── api/                # FastAPI: /score, /pairs/*, /trials/*
│   └── jobs/               # Prefect: ingest → perception → features → score
├── apps/
│   ├── annotation-ui/      # Next.js: pair 投票 UI
│   └── coach-ui/           # Next.js: レビュー UI
├── infra/
│   ├── docker-compose.yml  # ローカル: postgres + minio + api + workers
│   └── terraform/          # 本番用 (後回し可)
└── docs/superpowers/specs/
```

monorepo: pnpm workspaces + Python は uv。アプリ 2 つは `poc-mediapipe` の design system を共有。

### 8.3 環境

| 用途 | 環境 |
|---|---|
| Stage 1 pretrain | Colab Pro+ (A100) または GCP `g2-standard-8 + L4` |
| Stage 1 fine-tune | T4/L4 で十分 |
| Stage 2 特徴抽出 | CPU、CI で回せる |
| Stage 3 学習 | ノート PC で十分 |
| 実験トラッキング | Weights & Biases (または MLflow) |
| データバージョン管理 | DVC (S3 backend、`trials.parquet` と pair ラベル対象) |

### 8.4 データインフラ

- 動画ストア: S3 (本番) / MinIO (ローカル)
- メタ DB: Postgres
- ジョブキュー: Prefect (可視化 + Python ネイティブ + 小規模で運用負荷低)

### 8.5 CI / CD

- GitHub Actions:
  - Stage 2 関数群の pytest
  - API contract テスト (OpenAPI スキーマ生成 + mock client)
  - データ契約テスト (pandera)
  - Stage 1/3 学習は手動トリガ (`workflow_dispatch`) → W&B
- モデルデプロイ: GitHub Release + S3 同期 (現状の `.tflite` リリース流儀に揃える)

### 8.6 開発・ローカル起動

```sh
make dev      # docker-compose up: postgres + minio + api + perception worker + 2 UI
make seed     # サンプル trial 5 件 + 種ペア 10 件
make annotate # http://localhost:3001/annotate
make review   # http://localhost:3002/review
```

「指導医に最初の pair を投票してもらう」までを **15 分で立ち上げ** が目標。

### 8.7 プライバシー / 倫理

box trainer は患者映像なしで IRB ハードルは低いが:

- 顔・氏名・施設名の映り込みは consent_status で管理。UI 表示は trial 番号のみ
- レビュー UI からの動画ダウンロード禁止 (HLS expiring URL)
- pair 投票履歴は reviewer_id をハッシュ化、bias 推定にだけ使う
- 公開データ (PETRAW 等) のライセンスを README に明示
- 自施設データで学習したモデルを **公開モデルとして配布しない** ポリシー

---

## 9. エラー処理 & エッジケース

### 9.1 Stage 1 (Perception)

| 事象 | 検出 | 対応 |
|---|---|---|
| 鉗子が画面外に長時間 | per-frame 検出の途切れ閾値 (連続 ≥ 2.0s) | feature 計算で除外し `coverage_ratio` を特徴に追加。0.7 未満で `unreliable` フラグ |
| 手袋と鉗子の色が類似 | confidence 分布の二極化 | 学習側に hard negative を追加。短期はクラス別 conf 閾値 |
| カメラブレ・露出変化 | 明度急変、conf 急落 | preprocessing でガンマ補正・stabilization。再生不能なものは ingestion 時 reject |
| FPS / 解像度想定外 | trial メタ生成時の validate | `status="quarantine"` + 再アップロード要求 |

### 9.2 Stage 2 (Features)

| 事象 | 対応 |
|---|---|
| 左右トラッキング ID swap | 最初の数フレームで「画面左半分にいる方を left」と確定。連続性チェックで後検出補正 |
| trial 長すぎ/短すぎ | 妥当範囲 (30〜600s) 外を `outlier`、scorer に流さず UI で警告 |
| 計算バグ後方互換 | 出力に `schema_version` を持たせ、不一致なら自動再生成 |
| 数値発散 (NaN, Inf) | finite check + clip して `_clipped` フラグを残す |

### 9.3 Stage 3 (Scoring)

| 事象 | 検出 | 対応 |
|---|---|---|
| BT グラフの分断 | NetworkX で connected components | AL に「橋渡しペア」(島間で最も effective な pair) を強制挿入 |
| OOD trial | 特徴空間の Mahalanobis 距離 / Isolation Forest | スコアは返すが UI に「OOD: 参考値」バッジ。次回再学習で優先 |
| reviewer 系統的バイアス | reviewer fixed effect 推定 + 時系列ドリフト | 大きすぎる場合は当該 reviewer の新規ラベルを一時除外 + キャリブレーション会議 |
| ラベル汚染 (誤投票) | hold-out τ の急落 | 直近 N 日のラベルを除外して再学習 → 改善しなければロールバック |
| gaming (特徴最適化) | 特徴周辺分布の急変、特に `idle_time_ratio` 極端低下 | 定期的な人手レビューと突き合わせ。順位制にしない等の運用設計でも緩和 |

### 9.4 運用層 / データパイプライン

| 事象 | 対応 |
|---|---|
| Prefect ジョブ詰まり | 並列度上限 + DLQ。失敗 trial は `status=ingest_failed` で UI 表示 |
| 競合する再学習 | Postgres advisory lock で 1 本に絞る |
| デプロイ後の劣化 | shadow mode で N 日並走 → hold-out τ で昇格判定。劣化時は自動巻き戻し |
| Active Learning starvation | reviewer 別 backlog ダッシュボード + リマインダ。閾値超えたら easy pair モードに切替 |
| consent 取り消し | 動画削除、特徴・スコア保持で trial_id 匿名化、ラベルは保持しつつ pair から該当 trial を除外して再学習 |

### 9.5 コーチ UI

| 事象 | 対応 |
|---|---|
| スコアに納得しない | 「不当」フラグ → corrective pair 自動生成 → 次回再学習へ |
| 説明できない特徴 | 寄与度 top 3 だけ表示。ホバーで定義文 |
| ネットワーク不安定 | Optimistic UI + IndexedDB バッファ → 復帰時 flush |

---

## 10. テスト & 評価戦略

| 層 | 種別 | 内容 |
|---|---|---|
| Stage 1 | unit/integration | hold-out frame で mAP@0.5・PCK。SAM2 擬似 GT を含むサニティセット |
| Stage 2 | unit (純粋関数) | 既知軌道 (合成) で feature が解析解と一致。同一動画 2 回計算で完全一致 (test-retest) |
| Stage 3 | hold-out pair | Kendall τ, BT log-likelihood, 校正曲線。Reviewer-leave-one-out |
| 契約 | schema | parquet / API レスポンスを pandera + OpenAPI で CI 検査 |
| E2E | fixture trial | 1 動画から `/score` 完了まで。スコア帯域 (±5pt) アサーション |
| UI | Playwright | アノテーション UI 投票フロー / 復帰時 flush。コーチ UI の再生・修正 pair 投入 |
| ループ品質 | 週次ダッシュボード | reviewer κ, hold-out τ, OOD 検出率, AL カバレッジ (BT グラフ直径) |

---

## 11. v1 ロードマップ (12 週)

| 週 | マイルストーン |
|---|---|
| 1–2 | Infra (S3 / Postgres / Prefect / docker-compose) + `poc-mediapipe-trainer` に instrument detector ノートブック追加 |
| 3–4 | 自施設動画 ingestion + Stage 1 fine-tune (SAM2 半自動アノテーション) |
| 5–6 | Stage 2 純粋関数群 + proxy `s_0` ブートストラップで全 trial に仮スコア |
| 7–8 | アノテーション UI ローンチ + 1〜2 名指導医 onboarding + 種ペア 50 件収集 |
| 9–10 | Stage 3 (BT + RankNet) 初学習 + コーチ UI v0 |
| 11–12 | レトロ会、Active Learning 重み調整、評価ダッシュボード、v1 リリース |

### v1 終了条件

- 動画 ≥ 50 trial
- Pair ラベル ≥ 200
- Stage 1 mAP ≥ 0.7
- Stage 3 hold-out Kendall τ ≥ 0.6
- 指導医 ≥ 2 名でレトロを 1 回完了し、フィードバック反映済み
- コーチ UI を試験運用で 1 拠点に出せる状態

---

## 12. オープン課題 / 将来検討

### 12.1 v1 開始時に未確定

- 指導医アノテータの稼働 (人数・週あたり時間) — 確保次第、ロードマップの 7-8 週を調整
- 撮影体制の統一 (カメラ角度、解像度、照明) — 標準化ガイドを v1 序盤に作成必要
- 公開データのライセンス確認 (PETRAW / ROSMA / EndoVis 各 challenge) — 商用利用可否を ingest 前に明確化

### 12.2 v2 候補

- リアルタイム (< 1s) フィードバック (MediaPipe on-device 路線、案 1 の A 用途)
- 案 3 (Hybrid): 案 2 の Stage 3 に SSL embedding を追加してマルチタスク化
- 力センサ連携: 組織愛護や力加減の暗黙スキル評価
- 多施設展開: 撮影プロトコルの正規化、施設間バイアス補正
- タスク追加: peg transfer, pattern cutting 等への横展開

### 12.3 検証が必要な仮定

- pairwise 比較ラベルが modified OSATS 絶対値ラベルと十分相関する (Spearman ρ ≥ 0.7) — 年 1 回の外部妥当性検証で確認
- 数十 trial + 数百 pair で意味のある潜在スコアが学習できる — v1 中盤の中間評価で確認、足りなければ AL 重みと特徴セットを調整

---

## 13. 関連ドキュメント

- 既存設計: `poc-mediapipe` 側 `docs/superpowers/specs/2026-05-02-custom-segmenter-trainer-design.md`
- 既存ノートブック: `train_oxford_pet.ipynb` (Stage 1 拡張のベース)
- 公開データセット:
  - PETRAW (peg transfer workflow)
  - ROSMA (robotic surgical maneuvers)
  - EndoVis 系 (instrument segmentation 等)
