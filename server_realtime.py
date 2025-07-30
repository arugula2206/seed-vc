# server_realtime_modular.py

import socket
import argparse
import numpy as np
import time
import torch
import torchaudio

# 作成した変換エンジンをインポート
import realtime_converter as converter

# --- VADモデルの読み込み ---
try:
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    VAD_ENABLED = True
    print("Silero VADモデルの読み込みに成功しました。")
except Exception as e:
    VAD_ENABLED = False
    print(f"警告: Silero VADモデルの読み込みに失敗しました。VADは無効化されます。: {e}")

# --- ネットワーク設定など ---
HOST = '0.0.0.0'
PORT = 8080
CHUNK = 4096
OVERLAP_SAMPLES = 256

def handle_client_connection(conn, addr, args):
    print(f"\nクライアント接続処理を開始: {addr}")
    try:
        with conn:
            conn.settimeout(1.0)
            data_buffer, previous_processed_wave = b'', np.zeros(CHUNK // 2, dtype=np.float32)
            hanning_window = np.hanning(OVERLAP_SAMPLES * 2).astype(np.float32)
            fade_out, fade_in = hanning_window[OVERLAP_SAMPLES:], hanning_window[:OVERLAP_SAMPLES]
            idle_timeout_counter, MAX_IDLE_TIMEOUTS = 0, 5

            while True:
                try:
                    data = conn.recv(CHUNK)
                    if not data: break
                    idle_timeout_counter = 0
                    data_buffer += data
                    
                    while len(data_buffer) >= CHUNK:
                        process_chunk = data_buffer[:CHUNK]
                        data_buffer = data_buffer[CHUNK:]
                        
                        is_speech = False
                        if VAD_ENABLED:
                            chunk_tensor = torch.from_numpy(np.frombuffer(process_chunk, dtype=np.int16)).float() / 32768.0
                            resampler_16k = torchaudio.transforms.Resample(orig_freq=args.client_sample_rate, new_freq=16000)
                            resampled_tensor = resampler_16k(chunk_tensor)
                            if vad_model(resampled_tensor, 16000).item() > args.vad_threshold:
                                is_speech = True
                        else:
                            is_speech = True

                        processed_bytes = converter.convert_chunk(process_chunk, args.client_sample_rate) if is_speech else process_chunk
                        
                        current_wave = np.frombuffer(processed_bytes, dtype=np.int16).astype(np.float32)
                        blended_part = (previous_processed_wave[-OVERLAP_SAMPLES:] * fade_out) + (current_wave[:OVERLAP_SAMPLES] * fade_in)
                        output_wave = np.concatenate((previous_processed_wave[:-OVERLAP_SAMPLES], blended_part))
                        conn.sendall(output_wave.astype(np.int16).tobytes())
                        previous_processed_wave = current_wave
                
                except socket.timeout:
                    idle_timeout_counter += 1
                    if idle_timeout_counter >= MAX_IDLE_TIMEOUTS: break
                    continue
    finally:
        print(f"クライアント {addr} との接続処理を終了します。")

def start_server(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT)); s.listen(); s.settimeout(1.0)
        print(f"サーバーが {HOST}:{PORT} で待機中です...")
        try:
            while True:
                try:
                    conn, addr = s.accept()
                    handle_client_connection(conn, addr, args)
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\nサーバーをシャットダウンします。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="リアルタイム声質変換サーバー (Modularized)")
    parser.add_argument("--config", type=str, default="./logs/ITA_finetune_1/vc_wrapper.yaml", help="モデルのYAML設定ファイル")
    parser.add_argument("--cfm_checkpoint_path", type=str, default="./logs/ITA_finetune_1/CFM_epoch_00129_step_21000.pth", help="CFMモデルの.pthファイル")
    parser.add_argument("--tgtwav", type=str, default="./dataset/ITA-corpus/zundamon/recitation146.wav", help="目標話者のWAVファイル")
    parser.add_argument("--device", type=str, default="cuda", help="使用デバイス (cuda or cpu)")
    parser.add_argument("--vad_threshold", type=float, default=0.5, help="VADの発話検出のしきい値")
    parser.add_argument("--client_sample_rate", type=int, default=48000, help="クライアントが使用するサンプリングレート")
    args = parser.parse_args()

    try:
        converter.initialize_models(args)
        start_server(args)
    except Exception as e:
        print(f"起動中にエラーが発生しました: {e}")
    
    print("サーバープログラムを終了します。")