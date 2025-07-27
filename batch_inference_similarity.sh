#!/bin/bash

# --- ユーザー設定項目 ---

# テストしたい類似度のリスト（スペース区切りで記述）
SIMILARITY_RATES=(0.5 1.0 1.5)

# .pthファイルが格納されているディレクトリ
MODEL_DIR="runs/ITA_finetune_1"

# 元音声ファイルのパス
SOURCE_AUDIO="input/input.wav"

# 参照音声ファイルのパス
TARGET_AUDIO="dataset/ITA-corpus/zundamon/recitation001.wav"

# すべての出力の親ディレクトリ
BASE_OUTPUT_DIR="output/similarity_tests"
# --- 設定はここまで ---


# スクリプト本体
echo "種類度ごとの一括推論を開始します..."
echo "テストする類似度: ${SIMILARITY_RATES[@]}"
echo "----------------------------------"

# 類似度の値ごとにループ
for rate in "${SIMILARITY_RATES[@]}"; do
  # この類似度用の出力ディレクトリパスを作成 (例: output/similarity_tests/rate_0.7)
  CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/rate_${rate}"
  mkdir -p "$CURRENT_OUTPUT_DIR"

  echo ""
  echo "===== 類似度: $rate の処理を開始 (出力先: $CURRENT_OUTPUT_DIR) ====="

  # MODEL_DIR 内のすべての .pth ファイルに対してループ処理
  for pth_file in "$MODEL_DIR"/CFM_*.pth; do
    # ファイルが存在するか確認
    if [ -f "$pth_file" ]; then
      echo "  >> 処理中のモデル: $(basename "$pth_file")"

      # 推論スクリプトを実行
      python inference_v2.py \
        --cfm-checkpoint-path "$pth_file" \
        --source "$SOURCE_AUDIO" \
        --target "$TARGET_AUDIO" \
        --output "$CURRENT_OUTPUT_DIR" \
        --similarity-cfg-rate "$rate" # 類似度を引数として追加

      echo "  ✅ 完了"
    fi
  done
  echo "===== 類似度: $rate の処理が完了 ====="
done

echo ""
echo "すべての処理が完了しました。"