# poc-mediapipe-trainer

`poc-mediapipe` の補助。自前で学習した Image Segmenter `.tflite` をブラウザに乗せるまでの導通を見るためのトレーニング・パイプライン。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<ORG>/poc-mediapipe-trainer/blob/main/train_oxford_pet.ipynb)

## 何をするか

Oxford-IIIT Pet（3 クラス trimap）で MobileNetV2 ベースの小さな U-Net を学習し、`tflite-support` で MediaPipe ImageSegmenter 用メタデータ付きの `.tflite` を出力する。Python Tasks Vision SDK で sanity 推論まで通したのち、ブラウザ側 (`poc-mediapipe`) に渡して導通確認する。

## 使い方

1. 上の "Open in Colab" バッジを開く
2. ランタイム → ランタイムのタイプを変更 → ハードウェア アクセラレータを **GPU (T4)** に
3. 全セルを順に実行（合計 10〜15 分）
4. セル 8 で `oxford_pet_unet.tflite` がローカルにダウンロードされる
5. 下記「Release フロー」で GH Release に upload

## Release フロー

```sh
# Colab からダウンロードした oxford_pet_unet.tflite と同じディレクトリで:
gh release create v0.1.0 \
  --title "v0.1.0 — initial Oxford-IIIT Pet U-Net" \
  --notes "PoC 用。精度は最低限。" \
  oxford_pet_unet.tflite
```

公開後の URL パターン:

```
https://github.com/<ORG>/poc-mediapipe-trainer/releases/download/v0.1.0/oxford_pet_unet.tflite
```

これを `poc-mediapipe` の `scripts/download-models.sh` に追記する。

## 非ゴール

- 任意の精度水準の達成
- 任意データセットへの一般化（Oxford-IIIT Pet 固定）
- CI 自動再学習
- PyTorch / `ai-edge-torch` 経由

## 関連

- 対応アプリ: `poc-mediapipe`
- 設計: `poc-mediapipe` 側 `docs/superpowers/specs/2026-05-02-custom-segmenter-trainer-design.md`
