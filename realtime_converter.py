# rt_converter.py

import torch
import yaml
import torchaudio
from omegaconf import DictConfig
import hydra
from accelerate import Accelerator

# グローバル変数としてモデルと設定を保持
model = None
config = None
ref_mel = None
device = None

def initialize_models(args):
    """
    サーバー起動時に一度だけ呼ばれ、seed-vcのモデルを初期化する
    """
    global model, config, ref_mel, device

    print("声質変換モデルを初期化しています...")
    try:
        device = torch.device(args.device)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        cfg = DictConfig(config)
        accelerator = Accelerator()
        model = hydra.utils.instantiate(cfg)
        
        print("チェックポイントを読み込んでいます...")
        cfm_checkpoint = torch.load(args.cfm_checkpoint_path, map_location=device)
        model.cfm.load_state_dict(cfm_checkpoint['net']['cfm'])
        model.cfm_length_regulator.load_state_dict(cfm_checkpoint['net']['length_regulator'])
        
        model = accelerator.prepare(model)
        model.eval()

        print("目標話者のメルスペクトログラムを計算しています...")
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['sr'], 
            n_fft=config['mel_fn']['n_fft'],
            win_length=config['mel_fn']['win_size'], 
            hop_length=config['mel_fn']['hop_size'],
            n_mels=config['mel_fn']['num_mels'], 
            f_min=config['mel_fn']['fmin'], 
            f_max=config['mel_fn']['fmax']
        ).to(device)
        
        ref_wave, sr = torchaudio.load(args.tgtwav)
        ref_wave = torchaudio.functional.resample(ref_wave, sr, config['sr'])
        ref_mel = mel_spectrogram_transform(ref_wave.to(device))

        print("モデルの初期化が完了しました。")

    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        raise

def convert_chunk(chunk_bytes, client_sr=48000):
    """
    音声データのチャンク（バイト列）を変換する
    """
    if model is None:
        raise RuntimeError("モデルが初期化されていません。")
    
    with torch.no_grad():
        # バイトデータをfloat32のTensorに変換
        source_wave = torch.from_numpy(
            np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        ).unsqueeze(0).to(device)
        
        # モデルのサンプリングレートに合わせてリサンプリング
        source_wave = torchaudio.functional.resample(source_wave, client_sr, config['sr'])

        # 声質変換を実行
        output_wave_tensor, _ = model.voice_conversion_v2(
            source_wave,
            ref_mel.to(device)
        )

        # float32のTensorをint16のバイトデータに変換
        output_wave_int16 = (output_wave_tensor.squeeze().cpu().numpy() * 32768.0).astype(np.int16)
        return output_wave_int16.tobytes()