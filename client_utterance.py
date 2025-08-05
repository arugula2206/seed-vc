# client_utterance.py (sounddevice版)

import socket
import sounddevice as sd
import time
import argparse
import sys
import numpy as np
import struct
import queue

# --- ▼▼▼ 設定 ▼▼▼ ---
# 使用するデバイス名を部分的に指定してください (例: "Focusrite", "MacBook Pro Microphone")
# 空白のままにすると、OSのデフォルトデバイスが使用されます。
INPUT_DEVICE_NAME = "マイク (3- HyperX QuadCast S)"
OUTPUT_DEVICE_NAME = "Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)"

# サーバー設定
SERVER_IP = 'localhost'
SERVER_PORT = 8080

# 音声設定
SAMPLING_RATE = 48000
CHANNELS = 1
DTYPE = 'int16'
CHUNK = 1024

# 発話検出（VAD）設定
VAD_THRESHOLD = 300  # 環境に合わせて調整してください
SILENCE_CHUNKS = int(1.0 * SAMPLING_RATE / CHUNK)
MAX_RECORD_CHUNKS = int(10 * SAMPLING_RATE / CHUNK)
# --- ▲▲▲ 設定ここまで ▲▲▲

# 録音データを保持するキュー
q = queue.Queue()

def find_device_id(name, kind):
    """デバイス名（部分一致）からWASAPIデバイスIDを検索する"""
    if name == "":
        return None # デフォルトデバイスを使用
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name in device['name'] and device[f'max_{kind}_channels'] > 0 and device['hostapi'] == 2: # WASAPI上のデバイスを使用
            print(f"'{name}' に一致するWASAPIの{kind}デバイスが見つかりました: {device['name']} (ID: {i})")
            return i
    raise ValueError(f"'{name}' に一致するWASAPIの{kind}デバイスが見つかりませんでした。")

def audio_callback(indata, frames, time, status):
    """マイクからの入力をキューに入れるコールバック関数"""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def main():
    try:
        # デバイスIDを検索
        input_device_id = find_device_id(INPUT_DEVICE_NAME, 'input')
        output_device_id = find_device_id(OUTPUT_DEVICE_NAME, 'output')

        print("\nクライアント起動完了。Ctrl+Cで終了します。")

        # マイクからの入力ストリームを開始
        with sd.InputStream(samplerate=SAMPLING_RATE, device=input_device_id,
                            channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
            
            while True:
                print("\n-----------------------------------------")
                print("🎤 発話の開始を待っています...")
                
                # キューをクリア
                while not q.empty():
                    q.get()

                # 発話開始を待つ
                while True:
                    data = q.get()
                    rms = np.sqrt(np.mean(np.square(data.astype(np.float64))))
                    if rms > VAD_THRESHOLD:
                        break
                
                print("🔥 発話を検知しました！ 録音中...")
                frames = [data]
                silent_count = 0
                while True:
                    data = q.get()
                    frames.append(data)
                    rms = np.sqrt(np.mean(np.square(data.astype(np.float64))))
                    
                    if rms < VAD_THRESHOLD:
                        silent_count += 1
                    else:
                        silent_count = 0
                    
                    if silent_count > SILENCE_CHUNKS or len(frames) > MAX_RECORD_CHUNKS:
                        break
                
                recorded_data = np.concatenate(frames).tobytes()
                
                try:
                    print(f"💬 録音終了。サーバーに接続して変換します...")
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((SERVER_IP, SERVER_PORT))
                        s.sendall(struct.pack('>I', len(recorded_data)))
                        s.sendall(recorded_data)
                        print("🔊 音声データをサーバーに送信しました。")
                        
                        response_len_data = s.recv(4)
                        if not response_len_data:
                            print("サーバーから応答がありません。終了します。")
                            break

                        response_len = struct.unpack('>I', response_len_data)[0]

                        if response_len > 0:
                            print(f"✅ 変換済みデータ({response_len}バイト)を受信します。")
                            converted_data_bytes = b''
                            while len(converted_data_bytes) < response_len:
                                packet = s.recv(4096)
                                if not packet: break
                                converted_data_bytes += packet
                            
                            print("🎶 変換後の音声を再生します...")
                            converted_data_np = np.frombuffer(converted_data_bytes, dtype=DTYPE)
                            sd.play(converted_data_np, samplerate=SAMPLING_RATE, device=output_device_id)
                            sd.wait() # 再生が完了するまで待つ
                        else:
                            print("🔇 サーバーから再生不要の信号を受信しました。次の発話に移ります。")
                
                except (ConnectionRefusedError, ConnectionResetError, socket.error) as e:
                    print(f"\n[エラー] サーバーとの接続が失われました: {e}")
                    print("サーバーが停止したため、クライアントを終了します。")
                    break
                
    except KeyboardInterrupt:
        print("\nCtrl+Cを検知しました。終了します。")
    except Exception as e:
        print(f"\n[エラー] 予期せぬエラーが発生しました: {e}")
    finally:
        print("クライアントを終了しました。")

if __name__ == '__main__':
    # 利用可能なデバイス一覧を表示する機能
    parser = argparse.ArgumentParser(description="発話単位で音声を変換するクライアント (sounddevice版)")
    parser.add_argument('--list-devices', action='store_true', help='利用可能なオーディオデバイスの一覧を表示して終了します。')
    args = parser.parse_args()

    if args.list_devices:
        print("利用可能なオーディオデバイス:")
        print(sd.query_devices())
        sys.exit()
    
    main()
