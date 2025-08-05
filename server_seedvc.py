# server_seedvc.py (修正後)

import socket
import argparse
import numpy as np
import torch
import struct
import time
import threading

# 変換エンジンをインポート
import seedvc_converter as converter

# VAD（発話検出）関連
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    VAD_ENABLED = True
    print("Silero VADモデルの読み込みに成功しました。")
    import torchaudio
except Exception as e:
    VAD_ENABLED = False
    print(f"警告: Silero VADモデルの読み込みに失敗しました。: {e}")

# --- ネットワーク設定 ---
HOST = '0.0.0.0'
PORT = 8080

# --- ▼▼▼ 話者定義 ▼▼▼ ---
# ここに使用したい話者の名前とWAVファイルのパスを追加・編集してください
TARGET_SPEAKERS = {
    "zundamon": "./dataset/ITA-corpus/zundamon/recitation146.wav",
    "methane": "./dataset/ITA-corpus/methane/recitation105.wav",
    "sora": "./dataset/ITA-corpus/sora/recitation032.wav",
    "itako": "./dataset/ITA-corpus/itako/recitation001.wav"
}
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def handle_client(conn, addr, args):
    """クライアントからの音声データを一括で受信し、変換して返送する"""
    print(f"\nクライアントが接続しました: {addr}")
    try:
        with conn:
            # --- ▼▼▼ 話者指定を受信する処理を追加 ▼▼▼ ---
            speaker_len_data = conn.recv(1)
            if not speaker_len_data: return
            speaker_len = struct.unpack('>B', speaker_len_data)[0]
            speaker_name_data = conn.recv(speaker_len)
            speaker_name = speaker_name_data.decode('utf-8')
            print(f"クライアントが話者 '{speaker_name}' を指定しました。")
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # データ受信
            len_data = conn.recv(4)
            if not len_data: return
            input_len = struct.unpack('>I', len_data)[0]
            input_data = b''
            while len(input_data) < input_len:
                packet = conn.recv(4096)
                if not packet: break
                input_data += packet
            
            print(f"音声受信完了。変換処理を開始します...")
            
            # (VAD処理は変更なし)
            is_speech = False
            if VAD_ENABLED:
                try:
                    input_wave_tensor = torch.from_numpy(np.frombuffer(input_data, dtype=np.int16)).float() / 32768.0
                    resampler = torchaudio.transforms.Resample(orig_freq=args.client_sample_rate, new_freq=16000)
                    resampled_tensor = resampler(input_wave_tensor)
                    speech_timestamps = get_speech_timestamps(resampled_tensor, vad_model, sampling_rate=16000)
                    if speech_timestamps: is_speech = True; print(f"VAD: 発話を検出しました。")
                    else: print("VAD: 発話を検出できませんでした。変換をスキップします。")
                except Exception as e: is_speech = True; print(f"VAD処理中にエラーが発生しました: {e}")
            else: is_speech = True

            if is_speech:
                # --- ▼▼▼ 変換時に話者名を渡すように変更 ▼▼▼ ---
                processed_bytes = converter.convert_voice(input_data, speaker_name, args)
                # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                
                print(f"処理完了。クライアントに送信します... (サイズ: {len(processed_bytes)} バイト)")
                conn.sendall(struct.pack('>I', len(processed_bytes)))
                conn.sendall(processed_bytes)
            else:
                conn.sendall(struct.pack('>I', 0))

            print("送信完了。")
    except Exception as e:
        print(f"クライアント {addr} との通信中にエラーが発生しました: {e}")
    finally:
        print(f"クライアント {addr} との接続処理を終了します。")

def start_server(args):
    # (ウォームアップ処理は変更なし)
    if args.warmup > 0:
        print(f"AIモデルを{args.warmup}回ウォームアップしています...")
        dummy_wav_bytes = (np.random.randn(48000) * 10000).astype(np.int16).tobytes()
        # 最初の話者でウォームアップ
        first_speaker = list(args.targets.keys())[0]
        for i in range(args.warmup):
            print(f"  ウォームアップ実行中... ({i+1}/{args.warmup})")
            _ = converter.convert_voice(dummy_wav_bytes, first_speaker, args)
        print("ウォームアップ完了。")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        
        print(f"\nサーバーが {HOST}:{PORT} で待機中です。(停止するには Ctrl+C を押してください)")
        
        try:
            while True:
                conn, addr = s.accept()
                # クライアントごとにスレッドを立てて並列処理
                threading.Thread(target=handle_client, args=(conn, addr, args), daemon=True).start()
        except KeyboardInterrupt:
            print("\n停止信号を受信しました。サーバーをシャットダウンします。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="発話単位の声質変換サーバー (seed-vc)")
    
    # --- seed-vc用引数 ---
    parser.add_argument("--config", type=str, default="./logs/ITA_finetune_1/vc_wrapper.yaml", help="seed-vcのラッパー設定ファイル")
    # --- ▼▼▼ 単一のtarget引数を削除し、targetsを追加 ▼▼▼ ---
    parser.add_argument('--targets', type=dict, default=TARGET_SPEAKERS, help='話者名と参照音声パスの辞書')
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    parser.add_argument("--ar_checkpoint", type=str, default=None, help="（任意）ARモデルのチェックポイントファイルへのパス")
    parser.add_argument("--cfm_checkpoint", type=str, default="./logs/ITA_finetune_1/CFM_epoch_00129_step_21000.pth", help="（任意）CFMモデルのチェックポイントファイルへのパス")
    
    # (変換パラメータ、サーバー設定は変更なし)
    parser.add_argument("--diffusion_steps", type=int, default=20, help="Diffusionステップ数")
    parser.add_argument("--length_adjust", type=float, default=1.0, help="長さ調整")
    parser.add_argument("--intelligibility_cfg_rate", type=float, default=0.7, help="明瞭性CFGレート")
    parser.add_argument("--similarity_cfg_rate", type=float, default=0.7, help="類似性CFGレート")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-pサンプリング")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="繰り返しペナルティ")
    parser.add_argument("--client_sample_rate", type=int, default=48000, help="クライアントが使用するサンプリングレート")
    parser.add_argument("--warmup", type=int, default=50, help="ウォームアップの回数 (0で無効化)")
    
    args = parser.parse_args()

    converter.initialize_models(args)
    start_server(args)
    print("サーバープログラムを終了します。")