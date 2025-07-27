#!/bin/bash

# --- ユーザー設定項目 ---

# 使用する重みファイル（.pth）のパスをここで固定します
MODEL_PTH="./logs/ITA_finetune_1/CFM_epoch_00129_step_21000.pth"

# 参照音声ファイルがまとめて入っているディレクトリ
# このディレクトリ内のすべての.wavファイルがテスト対象になります
TARGET_VOICE_DIR="./dataset/ITA-corpus/zundamon"

# 使用する類似度（この値で固定してテストします）
SIMILARITY_RATE=0.7

# 元音声ファイルのパス
SOURCE_AUDIO="input/input.wav"

# すべての出力の親ディレクトリ
BASE_OUTPUT_DIR="output/target_voice_ID_tests"
# --- 設定はここまで ---


# スクリプト本体
echo "参照音声ごとの一括推論を開始します（モデル固定）..."
echo "使用モデル: $MODEL_PTH"
echo "参照音声ディレクトリ: $TARGET_VOICE_DIR"
echo "----------------------------------"

# 参照音声ディレクトリ内のすべての.wavファイルに対してループ
for target_file in "$TARGET_VOICE_DIR"/*.wav; do
  # ファイルが存在するか確認
  if [ -f "$target_file" ]; then

    echo ""
    echo "===== 参照音声: $target_name の処理を開始 (出力先: $CURRENT_OUTPUT_DIR) ====="

    # 推論スクリプトを実行（モデルのループはなし）
    python inference_v2.py \
      --cfm-checkpoint-path "$MODEL_PTH" \
      --source "$SOURCE_AUDIO" \
      --target "$target_file" \
      --output "$BASE_OUTPUT_DIR" \
      --similarity-cfg-rate "$SIMILARITY_RATE"

    echo "  ✅ 完了"

    echo "===== 参照音声: $target_name の処理が完了 ====="
  fi
done

echo ""
echo "すべての処理が完了しました。"