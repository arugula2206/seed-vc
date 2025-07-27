import torch
import yaml
import soundfile as sf
import numpy as np
import io
from hydra.utils import instantiate
from omegaconf import DictConfig

# --- グローバル変数 ---
device = None
dtype = None
vc_wrapper = None
target_audio_path = None # 目標話者のパスを保持

def initialize_models(args):
    """
    サーバー起動時に一度だけ呼ばれ、seed-vcのモデルを初期化する
    """
    global device, dtype, vc_wrapper, target_audio_path

    # デバイスとデータ型を設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16

    # モデルのラッパーをロード
    print("seed-vcモデルを読み込んでいます...")
    cfg = DictConfig(yaml.safe_load(open(args.config, "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=args.ar_checkpoint,
        cfm_checkpoint_path=args.cfm_checkpoint
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()

    # キャッシュのセットアップ
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    # 目標話者のパスを保存
    target_audio_path = args.target
    print(f"目標話者として {target_audio_path} を設定しました。")
    print("seed-vcモデルの初期化が完了しました。")

def convert_voice(input_audio_bytes, args):
    """
    クライアントから受信した音声バイトデータを変換する
    """
    if vc_wrapper is None:
        raise RuntimeError("モデルが初期化されていません。initialize_modelsを先に呼び出してください。")

    # 1. バイトデータをNumPy配列に変換
    # クライアントは16bit整数で送信してくる
    input_audio_int16 = np.frombuffer(input_audio_bytes, dtype=np.int16)
    # モデルはfloat32を想定しているため変換
    input_audio_float32 = input_audio_int16.astype(np.float32) / 32768.0

    # 2. NumPy配列を一時的なWAVファイルとしてメモリ上に作成
    #    seed-vcのラッパーはファイルパスを要求するため
    temp_wav = io.BytesIO()
    sf.write(temp_wav, input_audio_float32, args.client_sample_rate, format='WAV')
    temp_wav.seek(0)

    # 3. 声質変換を実行
    #    ストリーミングジェネレータから最終的な音声を取得
    generator = vc_wrapper.convert_voice_with_streaming(
        source_audio_path=temp_wav,
        target_audio_path=target_audio_path,
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
    
    if full_audio is None:
        raise RuntimeError("声質変換に失敗しました。")

    # 4. 変換後の音声データをクライアント向けの形式に変換
    output_sr, output_audio_float = full_audio
    
    # クライアントのサンプリングレートと異なる場合はリサンプリング
    if output_sr != args.client_sample_rate:
        import soxr
        print(f"サンプリングレートを変換します: {output_sr} Hz -> {args.client_sample_rate} Hz")
        output_audio_float = soxr.resample(output_audio_float, output_sr, args.client_sample_rate, 'VHQ')

    # float32 -> int16 のバイトデータに変換
    output_audio_int16 = (output_audio_float * 32767.0).astype(np.int16)
    return output_audio_int16.tobytes()
