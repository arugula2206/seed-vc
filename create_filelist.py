import os

# 音声ファイルが含まれるディレクトリのリスト
dataset_dirs = [
    'dataset/jvs_ver1',
    'dataset/ITA-corpus'
]

# 検索する音声ファイルの拡張子
audio_extensions = {'.wav'}

# 出力するファイルリストの名前
output_filename = 'my_dataset.txt'

# 見つかったファイルパスを格納するリスト
file_paths = []

print("Searching for audio files... 🔎")

# 各ディレクトリを再帰的に探索
for directory in dataset_dirs:
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found, skipping: {directory}")
        continue
        
    for root, _, files in os.walk(directory):
        for file in files:
            # ファイルの拡張子が音声ファイルのものかチェック
            if any(file.endswith(ext) for ext in audio_extensions):
                # ファイルパスをリストに追加
                full_path = os.path.join(root, file)
                # パスの区切り文字をスラッシュに統一（OS間の互換性のため）
                file_paths.append(full_path.replace('\\', '/'))

# ファイルリストをテキストファイルに書き出す
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        for path in file_paths:
            f.write(path + '\n')
    print(f"✅ Successfully created '{output_filename}' with {len(file_paths)} audio files.")
except IOError as e:
    print(f"❌ Error writing to file: {e}")