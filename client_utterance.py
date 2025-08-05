# client_utterance.py (ノンブロッキング入力対応版)

import socket
import sounddevice as sd
import time
import argparse
import sys
import numpy as np
import struct
import queue
import threading
import os # OS判別のためにインポート

# --- ▼▼▼ 設定 ▼▼▼ ---
INPUT_DEVICE_NAME = "マイク (3- HyperX QuadCast S)"
OUTPUT_DEVICE_NAME = "Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)"
SERVER_IP = 'localhost'
SERVER_PORT = 8080
SAMPLING_RATE = 48000
CHANNELS = 1
DTYPE = 'int16'
CHUNK = 1024
VAD_THRESHOLD = 300
SILENCE_CHUNKS = int(0.5 * SAMPLING_RATE / CHUNK)
MAX_RECORD_CHUNKS = int(10 * SAMPLING_RATE / CHUNK)
# --- ▲▲▲ 設定ここまで ▲▲▲

# --- ▼▼▼ 話者リストを更新 ▼▼▼ ---
SPEAKERS = ["zundamon", "methane", "sora", "itako"]
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

q = queue.Queue()
stop_event = threading.Event()
change_speaker_request = threading.Event()

def find_device_id(name, kind):
    if name == "": return None
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name in device['name'] and device[f'max_{kind}_channels'] > 0:
            print(f"'{name}' に一致する{kind}デバイスが見つかりました: {device['name']} (ID: {i})")
            return i
    raise ValueError(f"'{name}' に一致する{kind}デバイスが見つかりませんでした。")

def audio_callback(indata, frames, time, status):
    if status: print(status, file=sys.stderr)
    q.put(indata.copy())

def command_listener():
    """OSを判別し、適切なノンブロッキング入力でコマンドを監視するスレッド"""
    print("\n'c'キーで話者変更 | Ctrl+Cで終了")
    if os.name == 'nt': # Windowsの場合
        import msvcrt
        while not stop_event.is_set():
            if msvcrt.kbhit():
                try:
                    char = msvcrt.getch().decode('utf-8').lower()
                    if char == 'c':
                        print("\n[INFO] 話者変更リクエストを受け付けました。")
                        change_speaker_request.set()
                except UnicodeDecodeError: pass
            time.sleep(0.1)
    else: # Linux, macOSの場合
        import select
        import tty
        import termios
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char.lower() == 'c':
                        print("\n[INFO] 話者変更リクエストを受け付けました。")
                        change_speaker_request.set()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def select_speaker():
    """話者を選択するUIを表示し、選択された話者名を返す"""
    print("\n--- 変換したい話者を選択してください ---")
    for i, name in enumerate(SPEAKERS):
        print(f"  [{i+1}] {name}")
    
    speaker_name = ""
    while not speaker_name:
        try:
            choice = int(input("番号を入力してください: "))
            if 1 <= choice <= len(SPEAKERS):
                speaker_name = SPEAKERS[choice - 1]
            else:
                print("無効な番号です。もう一度入力してください。")
        except ValueError:
            print("数字を入力してください。")
    print(f"-> '{speaker_name}' を選択しました。")
    return speaker_name

def main():
    try:
        input_device_id = find_device_id(INPUT_DEVICE_NAME, 'input')
        output_device_id = find_device_id(OUTPUT_DEVICE_NAME, 'output')
        current_speaker = select_speaker()

        print("\nクライアント起動完了。")
        print("話者を変更したい場合は 'c' キーを押してください。(Enter不要)")
        print("プログラムを終了するには Ctrl+C を押してください。")

        cmd_thread = threading.Thread(target=command_listener, daemon=True)
        cmd_thread.start()

        with sd.InputStream(samplerate=SAMPLING_RATE, device=input_device_id,
                            channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
            while not stop_event.is_set():
                if change_speaker_request.is_set():
                    current_speaker = select_speaker()
                    change_speaker_request.clear()

                print("\n-----------------------------------------")
                print(f"[現在の話者: {current_speaker}] 発話の開始を待っています...")
                
                while not q.empty(): q.get()

                while True:
                    if change_speaker_request.is_set(): break
                    data = q.get()
                    if np.sqrt(np.mean(np.square(data.astype(np.float64)))) > VAD_THRESHOLD:
                        break
                
                if change_speaker_request.is_set(): continue

                print("発話を検知しました！ 録音中...")
                frames = [data]
                silent_count = 0
                while True:
                    data = q.get()
                    frames.append(data)
                    if np.sqrt(np.mean(np.square(data.astype(np.float64)))) < VAD_THRESHOLD:
                        silent_count += 1
                    else:
                        silent_count = 0
                    if silent_count > SILENCE_CHUNKS or len(frames) > MAX_RECORD_CHUNKS:
                        break
                
                recorded_data = np.concatenate(frames).tobytes()
                
                try:
                    print(f"録音終了。サーバーに接続して変換します...")
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((SERVER_IP, SERVER_PORT))
                        speaker_name_bytes = current_speaker.encode('utf-8')
                        s.sendall(struct.pack('>B', len(speaker_name_bytes)))
                        s.sendall(speaker_name_bytes)
                        s.sendall(struct.pack('>I', len(recorded_data)))
                        s.sendall(recorded_data)
                        
                        response_len_data = s.recv(4)
                        if not response_len_data: break
                        response_len = struct.unpack('>I', response_len_data)[0]

                        if response_len > 0:
                            converted_data_bytes = b''
                            while len(converted_data_bytes) < response_len:
                                packet = s.recv(4096)
                                if not packet: break
                                converted_data_bytes += packet
                            
                            print("変換後の音声を再生します...")
                            converted_data_np = np.frombuffer(converted_data_bytes, dtype=DTYPE)
                            sd.play(converted_data_np, samplerate=SAMPLING_RATE, device=output_device_id)
                            sd.wait()
                        else:
                            print("サーバーから再生不要の信号を受信しました。")
                
                except (ConnectionRefusedError, ConnectionResetError, socket.error) as e:
                    print(f"\n[エラー] サーバーとの接続が失われました: {e}")
                    break
                
    except KeyboardInterrupt:
        print("\nCtrl+Cを検知しました。終了します。")
    except Exception as e:
        print(f"\n[エラー] 予期せぬエラーが発生しました: {e}")
    finally:
        stop_event.set()
        print("クライアントを終了しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="発話単位で音声を変換するクライアント (sounddevice版)")
    parser.add_argument('--list-devices', action='store_true', help='利用可能なオーディオデバイスの一覧を表示して終了します。')
    args = parser.parse_args()
    if args.list_devices:
        print("利用可能なオーディオデバイス:"); print(sd.query_devices()); sys.exit()
    main()