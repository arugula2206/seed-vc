# seed-vc

# 環境構築

## github

[https://github.com/Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc)

## Dockerfile

```docker
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
```

イメージを作成

```bash
docker build -t seed-vc .
```

以下を実行して、コンテナを作成

```bash
docker run --name seed-vc -itd \
-v /mnt/sda1/Dataset/jvs_ver1:/app/dataset/jvs_ver1 \
-v /mnt/sda1/Dataset/ITA-corpus:/app/dataset/ITA-corpus \
-v /mnt/sda1/logs/seed-vc:/app/logs \
-v /home/arugula/Work/seed-vc:/app/ \
-p 8080:8080 \
--gpus all --shm-size=8g seed-vc
```

# ファインチューニング

~~2つのデータセットを１つのディレクトリ内に配置しているように見せるためにシンボリックリンクを利用~~

シンボリックリンクではうまくいかなかったため、コンテナ起動時のマウントを2つのデータセットのみに絞ることで対応

```bash
ln -s ../dataset/jvs_ver1 DUMMY/jvs_ver1
ln -s ../dataset/ITA-corpus/ DUMMY/ITA-corpus
```

## v2モデルのファインチューニング

以下を実行

```bash
nohup accelerate launch train_v2.py \
--config ./configs/v2/vc_wrapper.yaml \
--dataset-dir ./dataset \
--run-name ITA_finetune_1 \
--batch-size 16 \
--max-steps 21000 \
--max-epochs 1000 \
--save-every 500 \
--num-workers 0 \
--train-cfm \
> train.log 2>&1 &
```

## v1モデル（リアルタイム向け）のファインチューニング

以下を実行

```bash
nohup python train.py \
--config ./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml \
--dataset-dir ./dataset/ITA-corpus \
--run-name ITA_for_realtime \
--batch-size 16 \
--max-steps 21000 \
--max-epochs 1000 \
--save-every 500 \
--num-workers 0 \
> ft_for_realtime.log 2>&1 &
```

## 推論

公式の推論スクリプトで推論するには、以下を実行

```bash
python inference_v2.py \
**--cfm-checkpoint-path logs/ITA_finetune_1/CFM_epoch_00129_step_21000.pth \**
--source input/input.wav \
--target dataset/ITA-corpus/zundamon/recitation001.wav \
--output output
```

ディレクトリ内の全ての音声を変換するには、以下の自作スクリプトを実行
（変換に適している参照話者IDを調べるために利用）

```bash
nohup python batch_inference_v2.py \
--cfm-checkpoint-path logs/ITA_finetune_1/CFM_epoch_00129_step_21000.pth \
--source input/input.wav \
--target dataset/ITA-corpus \
--output output/target_voice_ID_tests \
> batch_inference.log 2>&1 &
```

v1モデルの推論には、以下の公式推論スクリプトを実行

```bash
python inference.py --checkpoint ./checkpoints/ITA_for_realtime/DiT_epoch_00234_step_19000.pth --config ./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml --diffusion_steps 10 --source ./input.wav  --target ./recitation146.wav  --output ./output
```

# 通信部分

自宅など研究室外の環境で実験する場合、以下の手順で行う

1. SSH接続で研究室PC（サーバ）に接続
2. 8080番ポート（コンテナ作成時に指定したポートと同じ）を利用するために、
ポートフォワーディングを行う
VSCodeだと簡単に設定できるため、おすすめ
3. クライアント側で指定するIPアドレスは”10.**.**.**”のように1010.で始まるものを利用すること
理由は研究室ネットワークの外から接続しているため(SSH接続時と同じアドレスのはず）
4. コンテナを起動後、研究室PCでサーバを、自宅PCでクライアントを起動すれば接続完了

## サーバの起動

```bash
python server_seedvc.py
```

各モデルの読み込みとウォームアップが完了すると、通信受付を開始

## クライアントの起動

```bash
python client_utterance.py
```

はじめに参照話者を番号で選択
以降、発話ごとにサーバへ送信
参照話者を変えたいときは、”c”キーを押すと最初と同じ設定画面になるので、番号を指定

# リアルタイム推論（未実装）

```bash
**apt update**
```

```bash
 apt-get install python3-tk
```

```bash
apt install libportaudio2
```