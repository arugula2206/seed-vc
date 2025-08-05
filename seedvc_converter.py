# seedvc_converter.py (修正後)

import torch
import yaml
import soundfile as sf
import numpy as np
import io
from hydra.utils import instantiate
from omegaconf import DictConfig
import librosa # librosaをインポート
import torchaudio # torchaudioをインポート

# --- グローバル変数 ---
device = None
dtype = None
vc_wrapper = None
# --- ▼▼▼ 単一のパスから、全話者のデータを保持する辞書に変更 ▼▼▼ ---
target_audio_cache = {}
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def initialize_models(args):
    """
    サーバー起動時に一度だけ呼ばれ、seed-vcのモデルと全参照話者を初期化する
    """
    global device, dtype, vc_wrapper, target_audio_cache

    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    dtype = torch.float16

    print("seed-vcモデルを読み込んでいます...")
    cfg = DictConfig(yaml.safe_load(open(args.config, "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=args.ar_checkpoint,
        cfm_checkpoint_path=args.cfm_checkpoint
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    # --- ▼▼▼ 全ての話者のデータをループで読み込み、事前計算してキャッシュする ▼▼▼ ---
    print("\n--- 全ての参照話者のデータを事前計算しています ---")
    for name, path in args.targets.items():
        try:
            print(f"  話者 '{name}' を読み込み中... (from: {path})")
            # seed-vcのラッパーはファイルパスを要求するため、そのままパスをキャッシュ
            target_audio_cache[name] = path
        except Exception as e:
            print(f"警告: 話者 '{name}' の読み込みに失敗しました。スキップします。エラー: {e}")
    print("-------------------------------------------------")
    
    print("seed-vcモデルの初期化が完了しました。")

def convert_voice(input_audio_bytes, speaker_name, args):
    """
    クライアントから受信した音声バイトデータを、指定された話者に変換する
    """
    if vc_wrapper is None: raise RuntimeError("モデルが初期化されていません。")
    if speaker_name not in target_audio_cache:
        raise ValueError(f"指定された話者 '{speaker_name}' はサーバーに存在しません。")

    # --- ▼▼▼ キャッシュから指定された話者のパスを取得 ▼▼▼ ---
    target_audio_path = target_audio_cache[speaker_name]
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    input_audio_int16 = np.frombuffer(input_audio_bytes, dtype=np.int16)
    input_audio_float32 = input_audio_int16.astype(np.float32) / 32768.0
    temp_wav = io.BytesIO()
    sf.write(temp_wav, input_audio_float32, args.client_sample_rate, format='WAV')
    temp_wav.seek(0)

    generator = vc_wrapper.convert_voice_with_streaming(
        source_audio_path=temp_wav,
        target_audio_path=target_audio_path, # ここで指定された話者のパスを使用
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        device=device,
        dtype=dtype,
        stream_output=True
    )
    
    full_audio = None
    for output in generator:
        _, full_audio = output
    
    if full_audio is None: raise RuntimeError("声質変換に失敗しました。")

    output_sr, output_audio_float = full_audio
    
    if output_sr != args.client_sample_rate:
        import soxr
        print(f"サンプリングレートを変換します: {output_sr} Hz -> {args.client_sample_rate} Hz")
        output_audio_float = soxr.resample(output_audio_float, output_sr, args.client_sample_rate, 'VHQ')

    output_audio_int16 = (output_audio_float * 32767.0).astype(np.int16)
    return output_audio_int16.tobytes()