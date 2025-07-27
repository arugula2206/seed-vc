# ベースイメージとしてNVIDIA公式のCUDA 12.4対応イメージを使用
# 開発に必要なツールキットが含まれるdevel版を選択
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# APTパッケージインストール時の対話を無効化
ENV DEBIAN_FRONTEND=noninteractive

# システムの依存関係をインストール
# Python、Git、音声処理用のffmpeg、Soundfileライブラリ用のlibsndfile1をインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# pythonコマンドでpython3が実行されるようにシンボリックリンクを作成
RUN ln -s /usr/bin/python3 /usr/bin/python

# 作業ディレクトリを設定
WORKDIR /app

COPY requirements.txt .

# PyTorchをCUDA 12.1対応版でインストール
# requirements.txtに記載のtorch==2.2.2はCUDA 12.1と互換性があります。
# ホストのCUDA Driverが12.4であれば問題なく動作します。
RUN pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# requirements.txtからPyTorch関連の記述を削除し、残りのパッケージをインストール
# これにより、CPU版のPyTorchが意図せずインストールされるのを防ぎます。
RUN sed -i '/^torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]