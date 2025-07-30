# ベースイメージとしてPyTorch公式イメージを指定
# -develタグはビルドツールが含まれているため、このまま使用します
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV_NAME=speech_env

# --- 修正箇所 1: 必要なシステムライブラリの集約と追加 ---
# ビルドツール(build-essential)とPython3.10で分離されたdistutils(python3-distutils)を追加
RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    libsndfile1 \
    ffmpeg \
    git \
    build-essential \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# ベースのconda環境ではなく、専用の環境を作成して依存関係を分離
RUN conda create -n ${CONDA_ENV_NAME} python=3.10 -y

# Conda環境をアクティブにするための設定
RUN conda init bash
SHELL ["conda", "run", "-n", "speech_env", "/bin/bash", "-c"]
RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc


# --- ライブラリのインストール ---

# 1. CondaでMFAと、その日本語サポートパッケージをインストール
RUN conda install -c conda-forge montreal-forced-aligner spacy sudachipy sudachidict-core -y

# 2. MFAがインストールした可能性のあるPyTorchを強制的にアンインストール
RUN pip uninstall -y torch torchaudio torchvision

# 3. 安定したバージョンのPyTorchをインストール (変更なし)
RUN pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 4. ESPnetと関連ライブラリをインストール (変更なし)
RUN pip install "espnet[recipes]"
RUN pip install espnet_model_zoo
RUN pip install sparc
RUN pip install "transformers==4.25.1" sentencepiece
RUN pip install torchcrepe

# 5. 依存関係の競合を解決 (変更なし)
RUN pip install "numpy<1.24" "importlib-metadata<5.0"

# --- 修正箇所 2: FastSpeech2の依存ライブラリをインストール ---
# 修正したrequirements.txtをコピーしてインストール
COPY./FastSpeech2_P2E/requirements.txt /app/FastSpeech2_P2E/requirements.txt
RUN pip install --no-cache-dir -r /app/FastSpeech2_P2E/requirements.txt


# --- モデルのダウンロード ---
# MFAの日本語モデルをダウンロード
RUN mfa model download dictionary japanese_mfa
RUN mfa model download acoustic japanese_mfa

# 作業ディレクトリの設定
WORKDIR /app

# コンテナ起動時のデフォルトコマンド
CMD [ "bash" ]