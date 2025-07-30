# client_realtime.py

import socket
import sounddevice as sd
import numpy as np
import sys
import queue
import threading

# --- ▼▼▼ 設定 ▼▼▼ ---
# VoiceMeeter Bananaで設定したデバイス名に合わせてください
INPUT_DEVICE_NAME = ""  # 通常のマイクを指定
OUTPUT_DEVICE_NAME = "" # 通常のヘッドフォンを指定

# サーバー設定
SERVER_IP = 'localhost'
SERVER_PORT = 8080

# 音声設定（サーバーと完全に一致させる必要があります）
SAMPLING_RATE = 48000
CHANNELS = 1
DTYPE = 'int16'
# サーバー側のVADに合わせてチャンクサイズを3072バイトに設定
CHUNK = 3072
# --- ▲▲▲ 設定ここまで ▲▲▲

# マイクからの音声とサーバーからの音声を一時的に保持するキュー
mic_queue = queue.Queue()
server_queue = queue.Queue()

stop_event = threading.Event()

def mic_callback(indata, frames, time, status):
    """マイクからの入力をキューに入れる"""
    if status:
        print(status, file=sys.stderr)
    mic_queue.put(indata.copy())

def network_thread(sock):
    """マイクのキューから音声を取り出し、サーバーに送信し、受信した音声を別のキューに入れる"""
    while not stop_event.is_set():
        try:
            # マイクの音声チャンクをサーバーに送信
            mic_data = mic_queue.get(timeout=1)
            sock.sendall(mic_data.tobytes())
            
            # サーバーから変換済み音声チャンクを受信
            processed_bytes = sock.recv(CHUNK)
            if not processed_bytes:
                print("サーバーとの接続が切れました。")
                break
            
            processed_data = np.frombuffer(processed_bytes, dtype=DTYPE).reshape(-1, CHANNELS)
            server_queue.put(processed_data)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"ネットワークスレッドでエラーが発生しました: {e}")
            break
    stop_event.set()

def find_device_id(name, kind):
    """デバイス名（部分一致）からデバイスIDを検索する"""
    if not name: return None
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name in device['name'] and device[f'max_{kind}_channels'] > 0:
            print(f"'{name}' に一致する{kind}デバイスが見つかりました: {device['name']} (ID: {i})")
            return i
    print(f"警告: '{name}' に一致する{kind}デバイスが見つかりませんでした。デフォルトデバイスを使用します。")
    return None

def main():
    input_device_id = find_device_id(INPUT_DEVICE_NAME, 'input')
    output_device_id = find_device_id(OUTPUT_DEVICE_NAME, 'output')

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"{SERVER_IP}:{SERVER_PORT} に接続しています...")
            s.connect((SERVER_IP, SERVER_PORT))
            print("サーバーに接続しました。")

            # ネットワーク処理用のスレッドを開始
            net_thread = threading.Thread(target=network_thread, args=(s,))
            net_thread.start()

            # 入出力ストリームを開始
            with sd.InputStream(samplerate=SAMPLING_RATE, device=input_device_id, channels=CHANNELS, dtype=DTYPE, callback=mic_callback, blocksize=CHUNK//2), \
                 sd.OutputStream(samplerate=SAMPLING_RATE, device=output_device_id, channels=CHANNELS, dtype=DTYPE) as ostream:
                
                print("\nリアルタイム声質変換を開始します。Ctrl+Cで終了します。")
                
                # サーバーからの音声を再生し続ける
                while not stop_event.is_set():
                    try:
                        processed_data = server_queue.get(timeout=1)
                        ostream.write(processed_data)
                    except queue.Empty:
                        continue
                        
    except KeyboardInterrupt:
        print("\nCtrl+Cを検知しました。終了します。")
    except Exception as e:
        print(f"\n[エラー] 予期せぬエラーが発生しました: {e}")
    finally:
        stop_event.set()
        if 'net_thread' in locals() and net_thread.is_alive():
            net_thread.join()
        print("クライアントを終了しました。")

if __name__ == '__main__':
    main()